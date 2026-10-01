# Business Entity Resolution — Amazon ML Challenge 2026

This pipeline links every Source 1 business to its matching Source 2 / Source 3 records and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

```text
normalize -> sparse TF-IDF retrieval (4 routes, per country and source)
          -> learned LightGBM filter (last blocking stage: candidate_pairs.tsv, ~18 per S1)
          -> pair features + rerankers: xlm-roberta-base (all pairs), bge-reranker-v2-m3 (uncertain 5%),
             Qwen3-Reranker-4B + LoRA (hardest <=700k pairs), each trained in two folds so scores stay out-of-sample
          -> LightGBM stage 1 (5-fold OOF) -> stage 2 (context features) -> one-owner -> expected-F0.5 decoding
```

Every train S1 entity goes through the same retrieval, filter and scoring as test, so competition features have the same distribution in training and test. The matcher learns from 30% of train entities, and the other 70% train the rerankers. The 10% listed in `work_ids_ce10.parquet` are held out of both.

## Environment

- **Pipeline** (CPU): Python 3.12, `pip install -r requirements.txt`.
- **Rerankers** (GPU): `torch`, `transformers`, `peft`, `scikit-learn`, `pandas`, `pyarrow`. Versions tested locally are in `requirements-gpu.txt`. The final run used the Colab G4 runtime's preinstalled `torch`/`transformers` plus `peft` 0.20.0.
- `BER_CE_PYTHON` must point to the interpreter that has the GPU packages. It can be the same interpreter.

Models (all Apache-2.0 or MIT, about 5.8B parameters in total, counted conservatively):
- `xlm-roberta-base` (278M) and `BAAI/bge-reranker-v2-m3` (568M): two fold copies of each.
- `Qwen/Qwen3-Reranker-4B` (4.0B): one base model with two LoRA adapters.

Models are downloaded from the Hugging Face Hub at training time. No external data is used.

## Data and folders

`<project>` below is the folder that contains `code/`: in the submission zip, the zip root. Place the organizers' `Datasets/` folder there (so the TSVs are at `<project>/Datasets/student_resource/dataset/{train,test}/`), or set `BER_DATA_DIR`.

| Variable | Meaning | Default |
|---|---|---|
| `BER_DATA_DIR` | Folder with `train/` and `test/` TSVs | `<project>/Datasets/student_resource/dataset` |
| `BER_WORK_DIR` | Intermediate files (TSVs are cached there as Parquet on first read) | `<project>/work` |
| `BER_OUTPUT_DIR` | The two submission TSVs and `test_scores.parquet` | `<project>/output` |
| `BER_THREADS` | CPU threads | cores − 2 |

Copy `work_ids_ce10.parquet` (shipped in this folder) to `$BER_WORK_DIR/cand/`. It lists the 10% of train S1 entities kept out of the matcher. It was made by `python -m src.candidates train 0.2` followed by `python -m src.candidates train-ce 0.1`, which are deterministic but also run retrieval for those subsets (about 1 hour).

## Reproduce the final submission

These are the settings of the final run, on a machine with 48 cores, 176 GB RAM and one 96 GB GPU:

```bash
export BER_K=20 BER_CAND_BUDGET=16 BER_MODEL_FRAC=0.3 BER_CE_PYTHON=$(which python)
python -m src.pipeline_v3 prepare candidates filter ce_export      # ~65 min: normalize, retrieve every entity, filter
python -m src.pipeline_v3 features                                 # ~20 min, CPU (can run beside the next line)
python -m src.pipeline_v4 ce_data ce_train ce_score                # ~2.2 h, GPU: base cross-encoder, two folds
python -m src.pipeline_v4 ce_data_large ce_train ce_large          # ~1 h, GPU: large cross-encoder, two folds
python -m src.pipeline_v4 llm_data llm_train llm_score             # ~1.8 h, GPU: Qwen3-Reranker-4B + LoRA, two folds
BER_STAGE_ALL=1 python -m src.pipeline_v4 stage1 stage2 predict    # ~45 min: matcher on 100% of train entities
# post-processing (CPU, ~30 min):
python -m src.bridge $BER_WORK_DIR/v3/val_pairs.parquet $BER_OUTPUT_DIR/test_scores.parquet $BER_OUTPUT_DIR bridged/
python -m src.france_suffix $BER_OUTPUT_DIR/test_scores.parquet bridged/ suffixed/
python -m src.france_swap $BER_OUTPUT_DIR/test_scores.parquet suffixed/ final/
```

- `stage2` prints the validation macro F0.5 (overall, India, US) and writes `$BER_WORK_DIR/v3/validation.json`.
- `predict` writes `matching_results.tsv`, `candidate_pairs.tsv` and `test_scores.parquet` to `$BER_OUTPUT_DIR`.
- `BER_STAGE_ALL=1`: every train entity already has out-of-sample reranker scores (the two-fold scheme), so the LightGBM matcher learns from all of them instead of the 30% sample.
- **`src.bridge`** recovers true matches that retrieval never paired with their S1 entity.
  - It joins every unclaimed record to records already assigned with p ≥ 0.995, using four keys: sorted name words, sorted address words, sorted phonetic name words, and house numbers plus the first street word.
  - A LightGBM verifier decides each proposal. It is trained on the stage-2 validation entities (India/US, labeled), and its threshold is chosen there (+0.0010 macro F0.5 on validation).
  - Rescued pairs are added to both output files.
- **`src.france_suffix`** (France only) claims unclaimed records at their S1's exact address that copy the name as "{first word} {legal form} {suffix}", e.g. "Marc Club SAS" → "Marc SAS Développement".
  - The data uses such suffixes evenly, yet the matcher accepted "Cie"/"Services" copies (97%) and rejected "Groupe"/"Développement" ones (9%).
  - The same shape is 99% true in US training data.
- **`src.france_swap`** (France only) claims unclaimed near-copies at the S1's exact address, where only one S1 sits at that address and the whole address agrees closely.
  - A near-copy is the S1 name plus added words ("… Développement", "… & Fils"), or with one word misspelled ("VPAJ Residence" for "VPJ Residence").
  - `--swaps` also claims copies with one real word replaced. It is **not** used in the final run: on the leaderboard it cost −0.0029, because in France such copies are mostly a second business at the same address.
- `final/` holds the submitted `matching_results.tsv` and `candidate_pairs.tsv`.

On Colab, the same steps run unattended from two notebooks:
- `colab/run_v4.ipynb`: everything up to the large cross-encoder, with checkpoints on Drive.
- `colab/run_v5.ipynb`: the LLM stage and the matcher.

`colab/build_notebook.py` rebuilds them from `src/`, embedding the code. With `BER_V4_GPUS=N`, the fold models train side by side and scoring is split across N GPUs.

Check the output with the organizers' validator:

```bash
python ../../Datasets/student_resource/utils/validate_submission.py -m $BER_OUTPUT_DIR/matching_results.tsv \
    -c $BER_OUTPUT_DIR/candidate_pairs.tsv -t $BER_DATA_DIR/test
```

`python -m src.redecode SCORES BASE_DIR OUT_DIR COUNTRY...` re-decodes saved test probabilities for the given countries and keeps the other rows of a base submission. With no changes it reproduces `predict` byte for byte.

Tests: `python -m pytest tests`. `BER_SMOKE=1` runs every step on a tiny subset (use a small reranker, e.g. `BER_V4_LLM_NAME=Qwen/Qwen3-Reranker-0.6B`).

## Modules

| Module | Role |
|---|---|
| `data.py` | Reads the TSVs (tab-separated, quoting off, all text) and caches them as Parquet |
| `translit.py` | Transliterates nine Indic scripts to Latin, tuned for English words written in Indic script |
| `normalize.py` | Name and address normalization: legal forms, abbreviations, state/region codes, house numbers, phonetic keys |
| `prepare.py` | Normalizes every record in parallel and caches the result |
| `retrieve.py` | Per-country sparse TF-IDF top-k search: name@place, address, joint name+address, hard-record name 3-grams |
| `candidates.py` | Runs retrieval per country and source, stores pairs with integer ids |
| `features.py`, `train.py` | Retrieval-context and pair-similarity features, LightGBM helpers |
| `context.py` | Stage-2 context features from stage-1 probabilities over the candidate graph |
| `cross_encoder.py` | Cross-encoder fine-tuning and scoring (GPU) |
| `llm_reranker.py` | Qwen3-Reranker LoRA fine-tuning and scoring (GPU) |
| `pipeline_v3.py` | Normalize → retrieve → filter → features → stages 1–2 → predict, for every train and test entity |
| `pipeline_v4.py` | Two-fold rerankers (base, large, LLM) on the pipeline's own candidates, multi-GPU scheduling |
| `decode.py` | One-owner resolution, expected-F0.5 and threshold decoders |
| `redecode.py` | Re-decoding saved test probabilities |
| `bridge.py` | Sibling-bridge rescue of records retrieval missed, with a validated verifier |
| `france_suffix.py` | France-only claim of suffix-copy records at the S1's exact address |
| `france_swap.py` | France-only claim of near-copy records (added words, misspellings) at the S1's exact address |
| `metric.py`, `submission.py` | Official metric, oracle ceiling, output writer |

The remaining modules (`train2.py`, `predict.py`, `ce_data.py`, `filtering.py`, `filter_sweep.py`, `eval_retrieval.py`, `eda.py`, `bench_translit.py`) belong to the earlier 20%-sample pipeline (v1/v2) and to analysis. They are kept for reference and are not used by the final run.

# Business Entity Resolution at Scale — Amazon ML Challenge 2026

**Team Claudes Plan** — Ahaan Verma, Vishwaswarup Rath, Daksh Goel, Shivam Pawar (Manipal University Jaipur)

Link every business in a clean reference list to its noisy copies in two other sources: about 1.7 million test businesses and 10 million candidate records, across India, the US and a third country, **France, that never appears in training**. The final submission scored **0.9798 macro F0.5** on the public leaderboard (final rank **≈ 1100**), and **0.9870** on 441k held-out India/US entities.

The field was strong: around rank 500 the board sat at 0.983, and the leaders were above 0.99. This repository records what we built and what the gap to the top taught us, not a winning entry.

This repository holds the full pipeline, the notebooks that ran it, the methodology write-up, and the analysis behind every decision, including the ideas that failed.

---

## The problem

| | |
|---|---|
| **Input** | Source 1 (S1): one record per business. Sources 2 and 3 (S2/S3): noisy copies of those businesses plus distractors. Each record is a name, an address and a country. |
| **Output** | For every S1 business, the list of its S2/S3 records. An empty list means "no match". |
| **Metric** | F0.5 computed **per S1 entity**, then averaged. Precision counts twice as much as recall. A singleton scores 1 only if its list is empty. |
| **Scale** | Train: 2.2M S1 / 5.0M S2 / 5.3M S3 (India and US only). Test: 1.73M / 4.89M / 5.08M (India 47%, US 38%, **France 15%**, unseen). |
| **Noise** | Typos and OCR slips, legal-form variants ("Pvt Ltd", "[L.L.C.]", "S.A.R.L."), reordered fields, domain-style names ("psportive.com"), initials ("SF"), invented alias names at the same address, zero-padded house numbers ("0338"), and Hindi transliterations of English words in nine Indic scripts. |
| **Rules** | No external data or lookups. Models must be MIT/Apache-2.0, ≤ 8B parameters. The submitted code must reproduce the output. |

## Results

| Version | What changed | Validation (India/US) | Public leaderboard |
|---|---|---|---|
| v1 | LightGBM matcher on a 20% train sample | 0.962 | 0.944 |
| v2 | + fine-tuned cross-encoder | 0.974 | 0.962 |
| v3 | **Every train entity processed exactly like test**, 4th search route, large cross-encoder | 0.9853 | 0.9773 |
| v4 | Cross-encoders retrained on our own candidates in two folds | 0.9860 | 0.9772 |
| v5 + post | + Qwen3-Reranker-4B (LoRA), **sibling bridge**, **France suffix rule** | 0.9866 | 0.9790 |
| **v7 (final)** | Matcher on **100% of train**, stronger bridge, France near-copy rule | **0.9870** | **0.9798** |
| *rejected* | v7 + French word-swap claims | – | 0.9769 |

France (no labels) was measured on the leaderboard itself (see below). It went from about **0.93** to about **0.942**, while India/US sit at about 0.985.

## Model metrics

**Cross-encoders** (held-out pairs from entities each fold model never trained on; two folds per model):

| Model | Scored on | Training pairs / fold | Held-out AUC | Held-out log-loss | Training time / fold |
|---|---|---|---|---|---|
| `xlm-roberta-base`, fold 0 | all candidate pairs | 2.91M | 0.99961 | 0.0203 | 23 min |
| `xlm-roberta-base`, fold 1 | all candidate pairs | 2.91M | 0.99957 | 0.0210 | 23 min |
| `bge-reranker-v2-m3`, fold 0 | the uncertain 5% (base p in 0.02–0.98) | 1.16M | 0.9798 | 0.1547 | 26 min |
| `bge-reranker-v2-m3`, fold 1 | the uncertain 5% | 1.16M | 0.9792 | 0.1556 | 26 min |
| `Qwen3-Reranker-4B` + LoRA (r=16), 2 folds | the hardest ≤700k pairs | 60k | not recorded | not recorded | 39 min for both folds |

The large model's AUC is lower because it only sees pairs the base model already found hard. The LLM's held-out numbers were printed to a Colab log that wasn't saved; its effect is measured through the matcher below.

**Matcher** (macro F0.5 on held-out India/US entities, after one-owner resolution and decoding):

| Version | Stage 1 | Stage 2 (final) | Singleton accuracy | India | US |
|---|---|---|---|---|---|
| v3 | 0.9840 | 0.9853 | 0.993 | 0.9829 | 0.9869 |
| v4 | 0.9847 | 0.9860 | 0.994 | 0.9832 | 0.9875 |
| v5 (+ LLM feature) | 0.9847 | 0.9860 | 0.994 | 0.9833 | 0.9875 |
| v6 (100% of train, same entities as v5) | 0.9848 | 0.9863 | 0.994 | 0.9836 | 0.9881 |
| v7 = v6 + sibling bridge | – | **0.9870** | – | – | – |

**Candidate generation:** filter keeps 15.2 candidates per entity on validation, with an oracle ceiling of **0.9948** (unfiltered retrieval: 112 per entity, ceiling 0.9953). On test, the submitted `candidate_pairs.tsv` holds 31.6M pairs, **18.25 per S1 entity**.

**France** (no labels): about 0.93 for v3/v4, about 0.942 for v7, estimated from leaderboard probes (see below).

## How it works

```text
             ┌───────────────────────────────── candidate generation ─────────────────────────────────┐
S1, S2, S3 → normalize → sparse TF-IDF retrieval (4 routes, per country) → learned LightGBM filter (~18/entity)
             └────────────────────────────────────────────────────────────────────────────────────────┘
          → rerankers: xlm-roberta-base (all pairs) → bge-reranker-v2-m3 (uncertain 5%) → Qwen3-Reranker-4B (hardest ≤700k)
          → LightGBM stage 1 (≈50 features) → stage 2 (competition/context features) → one record, one owner
          → per-entity expected-F0.5 decoding → sibling bridge → France copy rules → matching_results.tsv
```

1. **Normalization** (`normalize.py`, `translit.py`). Our own transliterator for nine Indic scripts, which beat `anyascii` and `indic-transliteration` on 9,971 true pairs. It also does:
   - A consonant-skeleton phonetic key ("sunrise properties" ≈ "sanraij propartij").
   - Legal-form extraction and abbreviation expansion ("St" → street or saint by position).
   - State, region and department codes.
   - House numbers without zero padding.
2. **Retrieve wide, filter small.** Four sparse TF-IDF routes per country and source:
   - name tied to place
   - address
   - name + address
   - a character-3-gram name route over records with no address, domain names or Indic names, the places our miss analysis found most misses
   A LightGBM filter then keeps about 18 candidates per business. That filtered list is the submitted `candidate_pairs.tsv`, since the organizers rank smaller candidate sets higher. A perfect matcher on those lists would score 0.9948.
3. **Rerankers, each trained in two folds**, so every score the matcher sees is out-of-sample:
   - `xlm-roberta-base` (278M) scores every pair.
   - `bge-reranker-v2-m3` (568M) rescores the 5% of pairs the first model is unsure about.
   - `Qwen3-Reranker-4B` with LoRA (4.0B) rescores the hardest ≤700k pairs.
   - About 5.8B parameters in total, counted conservatively.
4. **Two-stage LightGBM.** String, phonetic, address, number, retrieval-rank and reranker features. Stage 2 then adds what competing S1 claimants say about each record.
5. **Exact-metric decoding.** Each S2/S3 record goes to one owner. For each business we then keep the top-k candidates, with k chosen to maximize *expected* F0.5 (a Poisson-binomial over the pair probabilities), empty lists included.
6. **Post-processing:**
   - **Sibling bridge** (`bridge.py`). A record that retrieval never paired with its business can still look almost identical to *another* record we matched confidently. We join unclaimed records to such records on four keys: sorted name words, sorted address words, phonetic name, and house number + street. A LightGBM verifier trained on held-out labeled entities decides each link: **+0.0010 F0.5 at 95% precision**.
   - **France copy rules** (`france_suffix.py`, `france_swap.py`), below.

## What we learned

- **Train the way you test.** v1/v2 built training candidates for only 20% of train businesses, so each record faced about 4 competing businesses in training but about 12 in test. Every competition feature shifted, and the leaderboard trailed validation by up to 0.018. Running the full population through the pipeline (v3) closed the India/US gap completely.
- **Out-of-sample stacking unlocks all the data.** Training the rerankers in two folds made every train entity's scores honest. The LightGBM matcher could then learn from 100% of train instead of 30% (+0.0005).
- **An unseen country fails differently.** Every model variant agreed on 96% of French predictions, so France's loss was upstream, not in the final judge:
  - 57% of French candidate lists were full (against 4% in the US), crowded by same-name businesses, so alias, domain and initials copies were pushed out. The bridge brought them back.
  - The matcher had learned from US data that "Group" marks a look-alike business. It accepted 97% of French "…SAS Services" copies but only 9% of "…SAS Groupe" copies, although the data uses both suffixes equally often. A rule restoring that symmetry lifted France.
- **Test rules on the slice they affect.** Two submissions that differ only in French rows measure a France change exactly, with no labels needed. A rule that is right 90–94% of the time in India/US (claiming same-address copies with one word replaced) *lost* 0.0029 in France, where those records are usually a second business at the same address.
- **Bigger isn't automatically better.** The 4B LLM reranker was neutral on validation. The gains came from recall and from understanding how the data was generated.
- **What the top teams (0.99+) likely had:** near-100% candidate recall (our lists capped a perfect matcher at 0.9948) and matching built around the generator's copy operations, which transfers to a new country.

## Leaderboard history

| Score | Submission |
|---|---|
| 0.9440 | v1 |
| 0.9621 | v2 |
| 0.9773 | v3 |
| 0.9772 | v4 |
| 0.9790 | v5 + bridge + France suffix |
| **0.9798** | **v7 (final)** |
| 0.9769 | v7 + French word swaps (rejected) |

Final public-leaderboard rank: **≈ 1100** of a very large field.

## Repository layout

```text
code/business_entity_resolution/   the pipeline (see its README for exact reproduction steps)
  src/                             normalization, retrieval, features, rerankers, pipeline_v3/v4, decoding, bridge, France rules
  colab/                           notebooks that ran v3–v6 on a Colab G4 (built by build_notebook.py, code embedded)
  tests/                           71 unit tests (normalization, features, metric, decoding, submission format)
docs/
  Documentation_template.md        the methodology document submitted to the organizers
  FRANCE_CHALLENGES.md             the France investigation: measurements, ruled-out hypotheses
  FRANCE_IDEAS.md                  experiment proposals from a second review
  er-game-plan.html                the illustrated build plan
WINNING_STRATEGY.md                the strategy rulebook with every measured decision, day by day
scripts/make_package.py            builds the organizers' submission zip
LICENSE                            MIT
```

## Reproducing

The competition data is **not included**; it belongs to the organizers. With the dataset under `Datasets/student_resource/dataset/`, follow [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md). In short:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt                 # pipeline (CPU)
pip install -r requirements-gpu.txt             # rerankers (GPU)
export BER_K=20 BER_CAND_BUDGET=16 BER_MODEL_FRAC=0.3
python -m src.pipeline_v3 prepare candidates filter ce_export features
python -m src.pipeline_v4 ce_data ce_train ce_score ce_data_large ce_train ce_large llm_data llm_train llm_score
BER_STAGE_ALL=1 python -m src.pipeline_v4 stage1 stage2 predict
python -m src.bridge ... && python -m src.france_suffix ... && python -m src.france_swap ...
```

The final run used a Colab G4 runtime (48 cores, 176 GB RAM, one RTX PRO 6000 96 GB): about 3.7 h for search, filter and cross-encoders, 2.2 h for the LLM, 1.3 h for the matcher, then about 40 min of CPU post-processing. `python -m pytest tests` runs the unit tests.

## Models and licenses

| Model | License | Parameters | Role |
|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | MIT | 278M | cross-encoder on every candidate pair |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M | cross-encoder on uncertain pairs |
| `Qwen/Qwen3-Reranker-4B` + LoRA | Apache-2.0 | 4.0B | reranker on the hardest pairs |
| LightGBM | MIT | – | candidate filter, matcher stages, bridge verifier |

## License

Code: [MIT](LICENSE). The competition data and organizer documents are not part of this repository and remain the property of the organizers.

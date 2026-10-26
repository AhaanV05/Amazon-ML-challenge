# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Claudes Plan  
**Team Members:** Ahaan Verma, Vishwaswarup Rath, Daksh Goel, Shivam Pawar (Manipal University Jaipur)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We retrieve wide with per-country sparse TF-IDF search, then a learned filter cuts each Source 1 entity's list to about 18 candidates. A stack of three fine-tuned rerankers scores the pairs: `xlm-roberta-base`, `bge-reranker-v2-m3`, and `Qwen3-Reranker-4B` with LoRA. The larger models score only the pairs the smaller ones are unsure about. Their scores, together with about 50 string and retrieval features, feed a two-stage LightGBM matcher trained on all train entities. Each entity's match list is then chosen to maximize its expected F0.5.

Two post-processing steps follow:
- **Sibling bridge.** It recovers records retrieval never paired with their business, by linking them to that business's confidently matched records. A verifier validated on labeled entities decides each link.
- **France-specific rule.** It fixes one systematic error on the unseen country.

The main design choice is that every training entity goes through exactly the same retrieval, filtering and competition as a test entity, and every stacked score is out-of-sample.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Size and shape.** Train has 2.2M / 5.0M / 5.3M records in Sources 1 / 2 / 3; test has 1.73M / 4.89M / 5.08M (India 47%, US 38%, France 15% of test S1).
  - An S1 business has on average 3.46 true matches.
  - 5.6% are singletons.
  - About 26% of S2/S3 records are distractors that belong to no S1 entity.
  - No S2/S3 record belongs to two S1 entities.
  - Country agrees on 100% of true pairs.
  - India and US have the same cluster-size distribution, which suggests one generator for all countries.
- **Signal strength.** Address is the strongest single signal (AUC 0.95 against hard negatives). Names are weaker, especially in India (0.72–0.75), where some names are scrambled or written in one of nine Indic scripts (about 9% of names). Postcodes appear in only 12% (US) and 1% (India) of addresses.
- **Noise patterns.**
  - Typos and OCR-like substitutions ("Char1otte").
  - Legal-form variants ("Pvt Ltd", "[L.L.C.]", "S.A.R.L.").
  - Reordered name and address parts.
  - Domain-style names ("wilfordhancock.com").
  - Trade-name prefixes ("X trading as Y", "née").
  - Zero-padded house numbers ("0338").
  - State names versus codes, and missing address fields.
  - Hindi transliterations of English words ("sanraij propartij" = "sunrise properties").
- **France is test-only**, with no French training data. French copies of one business use either the region or one of its departments ("Bordeaux, Nouvelle-Aquitaine" vs "Bordeaux, Gironde"). Several businesses often share one street, or one name with different house numbers.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: blocking (wide sparse retrieval, then a learned filter), then a cascade of neural rerankers stacked into a two-stage gradient-boosted matcher, then set-level decoding under a one-owner constraint.

**Core Innovation:**
- **Population-consistent training.** An early version trained on candidates for only 20% of train entities, so each S2/S3 record faced about 4 competing S1 entities in training but about 12 in test, which shifted every competition feature.
  - The public leaderboard showed a gap of 0.02 against validation because of this.
  - Since v3, every train S1 entity goes through retrieval, the filter and scoring exactly like test. The matcher then learns from 30% of them, and the rest serve as real competition.
- **Out-of-sample stacking at scale.** The rerankers are fine-tuned on the ~20M labeled candidate pairs of non-matcher entities in two folds, so every score the matcher sees is out-of-sample.
- **Exact-metric decoding.** For each entity we choose the match list that maximizes its expected F0.5 under the pair probabilities (via a Poisson-binomial distribution over the number of true matches), including the empty list for singletons.

---

## 3. Candidate Generation (Blocking)

**Normalization** (`src/normalize.py`, `src/translit.py`), with rules taken from noise seen in train:
- Our own transliterator for nine Indic scripts. It beat `anyascii` and `indic-transliteration` on 9,971 true pairs.
- A consonant-skeleton phonetic key that equates English and Hindi-transliterated spellings.
- Legal-form extraction (including French forms), domain unwrapping, "M/s" removal and leet-digit repair.
- Address abbreviation expansion ("St" becomes "street" or "saint" by position; "R." becomes "rue").
- State, region and department canonicalization (US, India, France).
- House-number extraction without leading zeros, and postcode extraction.

**Retrieval** (`src/retrieve.py`, `src/candidates.py`): sparse TF-IDF top-k search per country and source (S1 → S2 and S1 → S3), with a 1% common-word cutoff. Routes:

| Route | Key | Top-k |
|---|---|---|
| place | name tokens joined to place tokens | 20 |
| address | normalized address | 20 |
| joint | name and address together | 20 |
| hard | name character 3-grams, searched only against records with no address, a domain-style name or an Indic-script name | 10 |

- The hard route targets the records where our search-miss analysis found most misses: 36% of misses have no address, against 4% of found pairs.
- Retrieval yields about 111 pairs per S1 entity and is not exported.

**Learned filter**, the last blocking stage (`pipeline_v3.step_filter`):
- A LightGBM model scores every retrieved pair from retrieval-context features: route scores and ranks, gaps to the best pair, and the number of competing queries.
- It keeps pairs with probability ≥ τ, at most r per S2/S3 record and at most k_max per S1 entity and source.
- (τ, r, k_max) = (0.001, 99, 15) is picked by a sweep on validation entities for the best ceiling under a budget of ≤16 candidates per entity.

- **Blocking keys used:** normalized name tokens tied to place, normalized address, name+address, and name character 3-grams (per country, per source).
- **Candidate pairs generated:** 31,571,667 test pairs from the learned filter (**18.2 per S1 entity**, from about 192M retrieved pairs), which are the pairs the rerankers and the LightGBM matcher score. Train gets the same treatment: 33.6M pairs.
  - The sibling bridge (section 4) then adds 40,991 records to the S1 entities it links them to, in both output files.
  - These were not in the filtered set, and a separate verifier scores them rather than the matcher.
  - The submitted `candidate_pairs.tsv` therefore holds **31,612,658 pairs (18.25 per S1)**, and every matched ID in `matching_results.tsv` appears in it.
- **How we ensured true matches were not lost:**
  - We measured the oracle ceiling at every stage on held-out train entities (the macro F0.5 of a perfect matcher restricted to the candidates).
  - Top-20 retrieval alone reached 0.9928. A study of 40,000 train entities showed that 92% of misses were outranked, not lexically absent. That led to the hard route and the leading-zero fix, which raised the ceiling to 0.9953.
  - After the filter the ceiling is **0.9948 at 15.2 candidates per entity** on validation.

---

## 4. Matching Model

**Features used** (`src/features.py`, `src/context.py`):
- **Name features:** RapidFuzz ratio, partial ratio, token-sort and token-set ratios; Jaro-Winkler; token Jaccard; phonetic-key ratio and token-set similarity; first-token equality; token counts and lengths; legal-form agreement state; domain-style flag; Indic-script flag.
- **Address features:** ratio, partial, token-sort and token-set ratios; token Jaccard; house-number Jaccard; postcode agreement state; empty-address and Indic-address flags; lengths.
- **Retrieval context:** per-route score and rank (place, address, joint, hard), the number of routes that found the pair, rank within the S1 entity's list and within the record's list, gaps to the best pair on each side, and the number of competing queries.
- **Neural scores:**
  - `ce_logit` from a fine-tuned `xlm-roberta-base` cross-encoder (MIT, 278M), on every pair.
  - `ce_large_logit` from a fine-tuned `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M), on the 5% of pairs where the base probability is between 0.02 and 0.98.
  - `llm_logit` from `Qwen/Qwen3-Reranker-4B` (Apache-2.0, 4.0B) with LoRA (r = 16, all attention and MLP projections). It covers the most uncertain pairs (large-model probability 0.05–0.95, at most 700k pairs), and the score is logit(yes) − logit(no) at the answer position.
  - Each model is trained twice, on two halves of the non-matcher entities. A pair is always scored by the model that did not train on its entity, and test pairs by fold 0.
- **Stage-2 context** (from out-of-fold stage-1 probabilities over the whole candidate graph):
  - The entity's top probabilities, sum and rank.
  - The best competing S1 claimant for the same record, and the gap to it.
  - Evidence from the other source.
  - Counts of confident pairs.

**Model type:** LightGBM in two stages.
- Stage 1 is trained on all 33.6M candidate pairs of all 2.2M train entities, with 5-fold out-of-fold predictions grouped by entity.
  - Earlier versions used only a 30% sample. Once the two-fold rerankers made every train entity's scores out-of-sample, the full population became usable, which gained +0.0005 on the same validation entities.
- Stage 2 adds the context features and uses fold 4 for validation.

**Post-processing** (`src/bridge.py`, `src/france_suffix.py`):
- **Sibling bridge.**
  - Records no entity claimed are joined to records already matched with p ≥ 0.995. The join uses sorted name words, sorted address words, sorted phonetic name words, or house numbers plus the first street word.
  - A LightGBM verifier decides each proposal. It sees string features against both the S1 record and the linking sibling, the number of linking siblings, and the number of competing owners.
  - It is trained and thresholded on the labeled validation entities, with the rest of the train population treated as claimed.
  - Validation: +0.0010 macro F0.5, 95% precision of added pairs.
  - Examples: domain-style copies ("psportive.com"), initials ("SF" for Sacrees Foyer), alias names at the business's address, and typo copies.
- **France suffix copies.** The data writes some copies of a business as "{first word} {legal form} {suffix}" at the same address, with the suffix drawn evenly from Cie, Services, Groupe, Développement, France, & Fils and & Associés.
  - The matcher, trained on US text where "Group"/"Holdings" mark look-alike businesses, accepted 97% of the Cie/Services copies but only 9–35% of the others.
  - We claim such copies when no entity claimed them and exactly one S1 sits at that exact address. The same shape is 99% true in US training data.
- **France near-copies.** Under the same conditions, plus close agreement of the whole address, we claim records that keep the S1 name and add words ("Club de Master Développement"), or misspell one word ("NXAU Societe" for "NXA Societe").
  - In India/US validation, near-copies at the S1's exact address with no other S1 there are true matches 90–94% of the time.
- The neural models total about **5.8B parameters**, counted conservatively: the two fold copies of both cross-encoders counted separately (2 × 278M + 2 × 568M), plus one 4.0B Qwen3 base with its two LoRA adapters. That is under the 8B limit, and all licenses are MIT or Apache-2.0.

**Threshold selection method:**
- No global threshold. After a one-owner pass (each S2/S3 record goes to its most probable S1 claimant, since no record belongs to two entities in train), each entity keeps the top-k of its candidates, with k chosen to maximize the expected F0.5.
- Validation compares this decoder with the best global threshold, and the better one is used. The expected-F0.5 decoder won: 0.9860 vs 0.9856 for v4.
- The pair probabilities are well calibrated. Refitting a logistic correction on validation changed F0.5 by less than 0.0001.

---

## 5. Results & Error Analysis

| Version | What changed | Validation macro F0.5 (India / US) | Public leaderboard |
|---|---|---|---|
| v1 | LightGBM stages 1–2 on a 20% train sample | 0.962 | 0.944 |
| v2 | + cross-encoder | 0.974 | 0.962 |
| v3 | Every train entity processed like test; hard route; large cross-encoder | 0.9853 (0.9829 / 0.9869) | 0.9773 |
| v4 | Both cross-encoders retrained on our own candidates in two folds; French region map | 0.9860 (0.9834 / 0.9877) | 0.9772 |
| v5 | + Qwen3-Reranker-4B (LoRA) on the hardest pairs | 0.9860 (0.9833 / 0.9875) | – |
| v5 + bridge + France suffix | Post-processing | +0.0008 from the bridge | 0.9790 |
| v6 | Matcher trained on 100% of train entities | 0.9863 (0.9836 / 0.9881)¹ | – |
| **v7 (final)** | v6 + bridge (four keys) + France suffix + France near-copies | 0.9870 (with bridge) | **0.9798** (final rank ≈ 1100) |
| v7 + French word swaps (rejected) | Also claims French copies with one real word replaced | – | 0.9769 |

¹ Scored on v5's validation entities, for a like-for-like comparison.

- **F_0.5 Score (macro):** **0.9870** on 441,486 held-out train entities (fold 4 of all train entities, India and US): the v6 matcher scores 0.9860, and the bridge adds +0.0010. The retrieval and filter ceiling is 0.9948, and singleton accuracy is 0.994.
- **France.** France has no training data, so it has no validation.
  - We emptied the predictions of a random 10% of French entities in one leaderboard submission and measured the drop. From that, France scored about **0.93** (v3/v4), against about 0.985 for India and the US.
  - Adding the bridge and the suffix rule raised France to about **0.940**: the leaderboard gained +0.0017, of which India/US validation explains +0.0007. The final v7 implies France ≈ 0.942.
  - One France hypothesis failed on the leaderboard. Same-address French copies with one real word replaced ("Institut de la Jeanne" → "Maternelle de la Jeanne") are true 90–94% of the time in India/US. Claiming them in France cost −0.0029 overall (about −0.019 on France). In France they are mostly a second business at the same address, so the final submission leaves them alone.
  - The model's own probabilities predict 0.985 for France, so French errors are confident ones.
  - Measured against true US matches, our French predictions are too strict:
    - 75% have an exactly equal normalized name (55% in true US matches).
    - Half as many have an empty address.
    - A third as many have a house-number conflict.
  - The noisier true French matches are therefore being missed.
- **Common false positives (wrong merges):**
  - 77–79% of validation false positives with probability above 0.5 are records **with an empty address** whose name matches, where only the name can decide.
  - The rest are sibling businesses, i.e. the same name at a neighbouring house number, or a different generic word at the same address.
  - In France we also see records claimed by several same-name S1 businesses. In 0.1% of French entities the model prefers a claimant with a conflicting house number over one with the exact number.
- **Common false negatives (missed matches):**
  - Records never retrieved, which cost 0.5% of F0.5 on validation. They are mostly records with no address, domain-style names, and Indic-script names of common businesses ("baba eksaports").
  - Heavily noised matches the matcher rejects: typos in several fields at once, or a trade-name or alias record whose name shares nothing with the S1 name.
  - In France, same-name records without an address (probabilities 0.05–0.5) are the most visible group of unclaimed records.

---

## 6. Conclusion

Most of our progress came from treating the problem as a measurable pipeline rather than a single model:
- Oracle ceilings at every stage.
- Training candidates built exactly like test (+0.015 on the leaderboard).
- Out-of-sample stacking of larger rerankers only where the smaller ones are unsure.
- A decoder that optimizes F0.5 per entity directly.

The main lesson is generalization to an unseen country. India and the US match their validation closely on the leaderboard, while France loses about 0.05 through confident mistakes on noisy matches. Country-agnostic evidence (house numbers, sibling businesses, missing addresses) matters more than extra capacity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: all source in `src/`, run instructions in `README.md`, pinned packages in `requirements.txt`, and Colab notebooks in `colab/`, generated by `colab/build_notebook.py`.

| Entry point | What it does |
|---|---|
| `python -m src.data` | Converts the TSVs to Parquet once. |
| `python -m src.candidates train-ce 0.1` | Fixes the 10% of train entities held out from the matcher (`work_ids_ce10.parquet`). |
| `python -m src.pipeline_v3 prepare candidates filter ce_export features` | Normalization, retrieval for every train and test entity, the learned filter (writes the candidate set), reranker inputs, and pair features. |
| `python -m src.pipeline_v4 ce_data ce_train ce_score ce_data_large ce_train ce_large` | Both cross-encoders in two folds (GPU). |
| `python -m src.pipeline_v4 llm_data llm_train llm_score` | LLM reranker (GPU). |
| `BER_STAGE_ALL=1 python -m src.pipeline_v4 stage1 stage2 predict` | LightGBM stages 1–2 on all train entities, decoding, both TSVs, and `test_scores.parquet`. |
| `python -m src.bridge ...`, then `src.france_suffix`, then `src.france_swap` | Post-processing; writes the final TSVs (see README). |

On one Colab G4 machine (48 cores, 176 GB RAM, RTX PRO 6000 96 GB):
- `colab/run_v4.ipynb` ran search, filter and cross-encoders (3.7 h).
- `colab/run_v5.ipynb` ran the LLM stage (2.2 h).
- `colab/run_v6.ipynb` ran the matcher on all entities (1.3 h).
- The post-processing runs on a laptop CPU (~40 min).

Other modules:

| Module | Role |
|---|---|
| `normalize.py`, `translit.py`, `prepare.py` | Record normalization |
| `retrieve.py`, `candidates.py` | Search |
| `train.py`, `features.py`, `context.py`, `lgbutil.py` | Features and LightGBM |
| `cross_encoder.py`, `llm_reranker.py` | Rerankers |
| `decode.py` | One-owner, expected-F0.5 and threshold decoders |
| `submission.py`, `metric.py` | Output writer and metric |
| `redecode.py` | Re-decodes saved test probabilities |

### B. Additional Results

- **Cross-encoders** (held-out pairs, v4):
  - Base: AUC 0.9996, log-loss 0.020.
  - Large, on the uncertain band it was trained on: AUC 0.980, log-loss 0.155.
  - LLM (Qwen3-Reranker-4B with LoRA, on the ≤700k hardest pairs): neutral on validation. With it the stack scores 0.98597, against 0.98599 without it. Its effect on France is unmeasured.
- **Candidate-set size vs ceiling** (filter sweep on validation): 15.2 candidates per entity keep a 0.9948 ceiling; the unfiltered retrieval (112 per entity) is at 0.9953.
- **Stage-1 → stage-2 gain** (v4 validation): 0.9846 → 0.9860 with expected-F decoding. Singleton accuracy rises from 0.975 to 0.994, because context features catch records claimed more strongly elsewhere.

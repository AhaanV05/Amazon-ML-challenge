# France: why our entity-resolution pipeline scores ~0.93 there, and what we know

This briefing is for someone joining with no prior context. We want concrete, label-free ideas to raise our France score, ranked by expected gain against the time we have left. Everything below is measured unless marked as a hypothesis.

Repo root: `C:\Users\ahaan\CODE\Amazon ML challenge`. Pipeline code: `code/business_entity_resolution/src/`.

---

## 1. The task and the rules

- **Task (Amazon ML Challenge 2026).** Link every Source 1 (S1) business record to all Source 2 / Source 3 (S2/S3) records of the same business. Submit one row per S1 entity with a comma-separated list of S2/S3 ids; an empty list means "no match".
- **Metric.** Macro F0.5, computed per S1 entity and averaged over all S1 entities. A singleton (an entity with no true matches) scores 1 for an empty prediction and 0 otherwise. Precision weighs twice as much as recall.
- **Data.**
  - Train covers India and US only.
  - Test adds France, which appears nowhere in training and has no labels.
  - Test S1: India 809,986 (46.75%), US 663,106 (38.27%), **France 259,452 (14.97%)**.
- **Rules.**
  - Models must be MIT or Apache-2.0 licensed, at most 8B parameters. We already use about 5.8B (see section 3), so only about 2B is left.
  - No external data or lookups: no geocoding, no registries, no internet data.
  - The final zip must reproduce `matching_results.tsv` from the packaged code.
  - The final ranking uses the private leaderboard (the part of the test set not shown during the challenge); the public leaderboard is a subset.
- **Time and budget, as of 12:49 IST on 27 Sep.**
  - Deadline 23:59 IST today.
  - **2 leaderboard submissions left.** The last should go in by about 21:00–22:00.
  - Compute: a Colab G4 runtime (48 cores, 176 GB RAM, RTX PRO 6000 with 96 GB) on pay-as-you-go units, plus a Windows laptop with an 8 GB RTX GPU.
  - v5 (below) occupies the G4 until about 16:00 IST.

## 2. Data facts

- **Sizes.** Train has 2.2M / 5.0M / 5.3M records (S1 / S2 / S3). Test has 1.73M / 4.89M / 5.08M.
- **Records.** Each record is `entity_id`, `business_name`, `business_address`, `country`.
- **Match structure (India/US train, identical in both countries):**
  - 3.46 true matches per S1 entity on average.
  - 5.6% singletons; 26% of entities have 5 or more matches.
  - About 26% of S2/S3 records are distractors that belong to no S1 entity.
  - No S2/S3 record belongs to two S1 entities.
  - Country is identical on all true pairs.
  - The distributions being identical across India and the US suggests one generator for every country, so France probably has the same cluster sizes. That is an assumption, not a measurement.
- **Records per S1 in test:** France 5.53, India 5.82, US 5.76. In train: 4.68.
- **Noise seen in train:** typos and OCR-like digits, legal-form variants, reordered parts, domain-style names ("x.com"), "X trading as Y", "née", zero-padded house numbers ("0338"), state name vs code, missing addresses, Hindi transliterations of English words.

### What French records look like (test only)

- **Only three regions:** Hauts-de-France (Lille, Roubaix, Tourcoing, Dunkerque, Calais), Nouvelle-Aquitaine (Bordeaux, Mérignac, Pessac, La Teste-de-Buch, Lège-Cap-Ferret) and Pays de la Loire (Nantes, Saint-Nazaire, Saint-Herblain, Pornic).
- **Names are often "{City} {generic word} {legal form}"**, for example "Lille Compagnie SARL", "Nantes Club SAS", "Pessac Ateliers EURL", "Dunkerque Ecole SASU". Others follow "{word} {word} ({legal})", like "Petite Services (France) EURL". Legal forms: SARL, SAS, SASU, EURL, SA, SCI, SNC.
- **Many distinct S1 businesses share one name in a city**, e.g. four different S1 entities named "Lille Compagnie SARL" at different addresses. 47% of French S1 entities share their normalized name with another S1 (India 53%, US 38%).
- **Several distinct S1 businesses share one address**: 11.8% of French S1 addresses are shared (India 4.9%, US 4.0%). For example, four S1 entities sit at "49 b Digue de Mer, Dunkerque".
- **Address variants:**
  - Region vs department: "Bordeaux, Nouvelle-Aquitaine" vs "Bordeaux, Gironde"; Nord and Pas-de-Calais for Hauts-de-France; Loire-Atlantique for Pays de la Loire.
  - "R." / "R" for rue, "Av." / "AV", "Bd", "Imp.", "All.", "Crs" for cours.
  - Number prefixes "N°", "Nº", "NO", "No"; "34 B" = "34 bis"; "0034" = "34".
  - Upper-case copies, accents dropped or garbled ("Çomite", "Àirsoft"), postcodes sometimes present ("33600 PESSAC").
- **Alias records** with invented names at the business's address ("Iritavo | 1 K Cité du Galon d'Eau, Roubaix"), and "Novixylonyla trading as Infirmiers Comite".
- **Records with an empty address** carrying only a name ("Clos Ecole", "Ets Retraite SARL").

## 3. Our pipeline (the same for every country)

1. **Normalize** (`normalize.py`): legal-form split, abbreviation expansion, state/region codes, house numbers without leading zeros, phonetic key, an Indic-script transliterator.
   - Since v4, French regions and departments map to one code (`FRANCE_REGIONS`), and "crs", "all", "ch" and "N°" are handled.
2. **Retrieve** (`retrieve.py`): sparse TF-IDF top-20 per country and source, over four routes.
   - name@place, address, joint name+address.
   - A "hard" route: name character 3-grams over records that have no address, a domain-style name or an Indic name.
   - About 111 retrieved pairs per S1.
3. **Learned filter** (LightGBM on retrieval features, trained on India/US): keeps about 18 candidates per S1. That filtered set is `candidate_pairs.tsv`.
4. **Pair features and three rerankers**, all fine-tuned only on India/US pairs, each in two folds so scores stay out-of-sample:
   - `xlm-roberta-base` (278M) on all pairs.
   - `bge-reranker-v2-m3` (568M) on the 5% of pairs the base model is unsure about.
   - v5 adds `Qwen3-Reranker-4B` with LoRA (4.0B) on the ≤700k hardest pairs. It is training right now.
5. **LightGBM stage 1** (about 50 features, 5-fold out-of-fold), then **stage 2** with context features (competing claimants for the same record, entity-level sums and ranks).
6. **One-owner:** each S2/S3 record goes to its most probable S1 claimant.
7. **Decoding:** for each S1 entity, keep the top-k candidates, with k chosen to maximize expected F0.5 under the pair probabilities (possibly k = 0).

**Scores:**

| Version | Validation India / US | Public leaderboard | France (inferred, section 4) |
|---|---|---|---|
| v3 | 0.9829 / 0.9869 | 0.977322 | ~0.936 |
| v4 (region map + retrained cross-encoders) | 0.9834 / 0.9877 | 0.977203 | ~0.932 |
| v5 (+ Qwen3 4B reranker) | pending (~16:00 IST) | not yet submitted | ? |

The public leaderboard's top ~200 sit at about 0.985. Reaching that needs France at roughly 0.97 or better.

## 4. What we measured about France

**How we know France scores ~0.93.**
- A probe submission emptied the predictions of a random 10% of French entities (25,945 of them). The score fell from 0.977322 to 0.964298, i.e. 0.8697 per emptied entity.
- Treating the leaderboard as the country-share-weighted mix, with India/US at their validation scores, gives France ≈ 0.936 for v3.
- Combining both constraints: France is between 0.93 and 0.94.
- Singleton false positives are at most about 1% of French entities.
- **The loss is spread over ordinary multi-match entities,** which average about 0.93 each (India/US about 0.985).

**The model is confidently wrong on France.**
- Its own probabilities imply an expected F0.5 of 0.985 for France, against 0.93 actual. For India/US the same calculation gives 0.993–0.995 against about 0.985 actual; that gap is mostly true matches missing from the candidates.
- Probability calibration is fine on India/US: a refitted logistic correction gives a ≈ 1, b ≈ 0 and no F0.5 change.

**Per-entity statistics on test (v4):**

| | France | India | US |
|---|---|---|---|
| Candidates per S1 (after filter) | 25.3 | 19.5 | 13.9 |
| Sum of p per S1 (expected matches) | 3.35 | 3.36 | 3.44 |
| Pairs with p in 0.1–0.9, per S1 | 0.65 | 0.16 | 0.20 |
| Predicted matches per S1 | 3.16 | 3.31 | 3.38 |
| Entities with 5+ predicted matches | 19.8% | 23.1% | 24.1% |
| Entities predicted empty | 5.9% | 6.0% | 5.8% |
| S2/S3 records in no candidate list | 2.3% | 1.05% | 0.37% |
| Confident pairs on records claimed by ≥2 S1 (p > 0.5) | 1.3% | 0.25% | 0.22% |

- True matches per S1 are 3.46 in India/US train, of which 3.36 (India) / 3.43 (US) reach the candidate lists.

**Error mix, estimated.** Assuming French entities have 3.46 true matches like India/US, a per-entity F0.5 of 0.93 with 3.16 predictions implies about **5% of predicted French pairs are wrong and about 13% of true French matches are missed**. India is at about 1% and 5%.

**Our French predictions are "too clean" compared with what true matches look like.**

| | Exact equal normalized name | Record has empty address | House-number conflict | A name word replaced by an unrelated word |
|---|---|---|---|---|
| Predicted French pairs | **74.6%** | 2.0% | 2.0% | 13.8% |
| True US pairs (validation) | 54.6% | 4.1% | 5.7% | 18.3% |
| True India pairs (validation) | 48.0% | 3.6% | 2.5% | 28.5% (transliteration inflates this) |

If French truth carries the same noise as US truth, we are **missing the noisy French matches**. We see this directly: same-name records with empty addresses get p of 0.05–0.5 and stay unclaimed. For example, "Clos Ecole" (no address) gets 0.36 for S1 "Clos Ecole SAS", which ends up with no prediction.

**Other French observations:**
- **Validation false positives are mostly empty addresses.** 77–79% of validation (India/US) false positives with p > 0.5 are records with an empty address, where only the name can decide. France has many same-name businesses per city, so these records are genuinely ambiguous there.
- **Sibling businesses:** the same address with a different generic word ("Bordeaux Club SA" vs "Bordeaux Amicale SA", both at 15 Rue Ferdinand Palau), or the same name on one street at different numbers ("Bordeaux Club SAS" at 8 and at 18 Rue de Candale).
- **House-number mix-ups.** Among records claimed by several S1 entities, the model sometimes prefers a claimant with a conflicting house number over one with the exact number: "BORDEAUX CLUB, 8 Rue de Candale" goes to the S1 at 18 (0.994) instead of the S1 at 8 (0.538). This affects only 322 records (0.12% of French entities).
- **v3 → v4 lowered France by about 0.004.** v4 changed two things at once, the French region/department normalization and cross-encoders retrained on 3M India/US pairs (10× more), so we can't say which caused the drop. 16% of French entities' predictions changed between v3 and v4, against 3% for India/US.

## 5. Already tried or ruled out

| Idea | Result |
|---|---|
| Recalibrate p before decoding | No gain on India/US validation; calibration is already fine there |
| Contested-record rules (prefer exact house number or exact name among claimants) | Correct in spirit, but they touch only 0.12% and 0.04% of French entities |
| A global France logit shift (decode more or fewer matches) | Not supported: France's expected-match mass equals India's (3.35 vs 3.36), and a single probe can't tell the direction |
| Wrong matches on French singletons | Ruled out as a main cause: at most about 1% of French entities |
| Mapping region and department to one code | Shipped in v4; France did not improve (−0.004, confounded) |
| France from the rerankers alone, skipping the LightGBM stack | On India/US validation the rerankers alone score 0.9784 / 0.9799 against the stack's 0.9832 / 0.9875 (a 50/50 blend 0.9826 / 0.9867). On France they change only ~4% of predicted pairs (15k added, 21k removed), and **every variant (stack, rerankers, blend, v3) agrees on ~96% of French predictions** with the same "too clean" profile. So France's loss is shared by all models, not caused by the stack |
| Legal-form noise ("SARL" vs "SAS" on true matches) | Same-name records with a swapped legal form and a different address are true matches 75% of the time in India but 20% in the US. France behaves US-like: we claim 9%, and the examples are different businesses (other cities, or the same street at another number). Records at the identical address are claimed ~100% everywhere. Not the cause |
| Filter cap per source (15) | 57% of French (entity, source) lists are full, against 22% India and 4% US. But on India/US validation, raising the cap from 12 to 15 moved the ceiling only 0.9946 → 0.9948, so the cap likely costs France well under 1% |

## 6. Hypotheses still open

1. **Noisy true matches rejected with confidence** (the leading hypothesis). The matcher learned what noise looks like from India/US, where names are distinctive. In France, name and address alone are ambiguous (many same-name and same-address businesses), so competition features and the rerankers push noisy candidates down. Section 4's "too clean" table supports this.
2. **Confident false merges with siblings or distractors:** same-name businesses on one street, or different-name businesses at one address, that exist only in S2/S3.
3. **Retrieval and filter misses:** 2.3% of French records are in no candidate list, more than twice India's rate. The filter was trained on India/US.
4. **Rerankers trained on India/US text** (Hindi transliterations, US street formats) transfer poorly to French wording and abbreviations.

## 7. What we have to work with

- **`submissions/v4/output/test_scores.parquet`**: every test candidate pair with its final probability (31.6M rows: `q_id`, `d_id`, `country`, `p`).
  - `q_id` is the numeric part of the S1 id.
  - `d_id` = source × 1,000,000,000 + the numeric part (S3-123 becomes 3000000123).
- **`submissions/v4/v3/val_pairs.parquet`**: validation pairs (India/US, about 120k entities) with `label`, `p1` (stage 1) and `p2` (final).
- **`submissions/v3/`, `submissions/v4/output/`**: the TSVs that were submitted.
- **`work/norm/*.parquet`**: normalized records for all splits and sources (`name_core`, `name_legal`, `addr_norm`, `addr_nums`, `postcode`, ...), built with the current normalization.
- **Raw data** in `Datasets/student_resource/dataset/{train,test}/*.tsv`, including `train_ground_truth.tsv`.
- **`src/redecode.py`**: re-decodes saved probabilities for chosen countries and keeps the other rows of a base submission. It is how any France-only decoding change would be produced.
- v5 will produce the same files (`test_scores.parquet`, `val_pairs.parquet`) at about 16:00 IST.

## 8. What we are asking for

Concrete, label-free methods to raise France within the next ~8 hours, each with an estimate of gain and effort. Ideas we have not been able to test, which you may confirm, reject or improve:

- **Synthetic French training pairs.** Learn the noise operations from India/US truth (typo rates, legal-form swaps, field drops, reordering, number padding, alias records), apply them to French S1 records to create labeled French positives, use other French records as negatives, then fine-tune a reranker or retrain LightGBM. What is the fastest version that fits in the time left?
- **Self-training or pseudo-labels on France**, or domain-adversarial or feature-dropping retraining that removes features whose distribution shifts most in France.
- **France-specific decoding:** a different expected-F0.5 threshold, re-weighting records by how many same-name S1 claimants they have, or claiming unclaimed same-name empty-address records when exactly one S1 has that exact name.
- **How to tell whether a France change helps without labels.** We have one or two leaderboard submissions left. The leaderboard's India/US part is predictable from validation, so a submission's France score can be inferred from the leaderboard total.

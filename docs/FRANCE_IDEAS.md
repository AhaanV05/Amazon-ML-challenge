# France: concrete experiments from the code review

Reviewed on 27 September 2026 against `FRANCE_CHALLENGES.md` and the current pipeline. These are proposed experiments, not measured France improvements. France labels, external identity data, and new neural models are not required.

## First choice: retrieval through frozen, high-confidence sibling records

**Hypothesis:** a noisy S2/S3 record can resemble another copy of its business more closely than it resembles S1. Direct S1 retrieval or scoring can therefore fail even when an easier bridge exists.

`src/context.py` already compares each candidate to the highest-p1 other candidate, including same-source and other-source anchors. Simply adding an anchor similarity would repeat existing work. The missing experiment is **retrieving new candidates through matched siblings**, with a verifier trained specifically on S2/S3 relationships.

1. Freeze seed matches from the baseline. Start with p >= 0.995, a substantial margin over other S1 claimants, and agreement on nonempty name and address. Check seed precision on held-out India/US; the probability alone is insufficient given France's known overconfidence. Treat exact thresholds as validation choices.
2. Index these seed S2/S3 records with their S1 owner. Query unclaimed French S2/S3 records against the seeds using the existing lexical representations. Begin with a small top-k, then deduplicate candidate S1 owners. Include records absent from all original candidate lists and low-score records already present. Restrict the first pilot to records with a usable address.
3. Train a small CPU LightGBM verifier on **real training S2/S3 pairs**. Positives share a ground-truth S1 owner. Negatives come from the same seed-retrieval process and have different owners or are labeled distractors. Include same-address/different-name and same-name/different-number negatives. Keep all copies of an entity in the same fold. Build seed indexes over the full relevant population before sampling training rows.
4. Use existing string features plus seed reliability, support from distinct records/sources, and the margin against the best seed of another owner. Train and validate the complete rescue decision; an S2/S3 probability cannot be substituted directly for an S1/S2 probability. Require joint name/address evidence and no unresolved competing owner in the initial version.
5. Add only validated, high-precision rescues to a France-only copy of the submission. Keep seeds fixed: no repeated propagation through newly rescued records. Add any new pairs to `candidate_pairs.tsv`, and package the seed retrieval and verifier so the final output is reproducible.

**Pilot:** first run the complete rescue procedure on untouched India/US validation entities, generating seeds without their labels. Labels are used only to train on separate entities and evaluate. Report net macro F0.5, precision of added pairs, entities touched, and results within repeated-name/repeated-address strata. Then inspect the France rescue count and competing-owner margins. Success on India/US is supporting evidence, not proof of France improvement.

**Effort estimate:** roughly 1-3 hours for a bounded CPU pilot if cached tables and ground-truth joins are available; measure throughput before a full run. **Gain:** currently unknown. The first useful estimate is `fraction of entities changed * mean held-out per-entity improvement`; report both factors rather than promise a score increase. Under a public France share near 0.15, +0.01 France would yield about +0.0015 overall.

## Faster alternative: reverse retrieval before changing the scorer

Query unclaimed S2/S3 records against all French S1 records, retain a small top-k, and compare the new owner candidates to forward retrieval. This addresses crowded-out forward rankings without changing normalization or all retrieval scores. `src/retrieve.py` already has a `k_rev` helper for word routes. Use word routes explicitly: `HardNameIndex` has no `dm` attribute, so the helper's reverse branch does not support that index as written.

First measure how many plausible pairs this adds, and separate never-retrieved pairs from filter-rejected pairs. Rescore through the existing pipeline; regenerate relevant competition features over the augmented graph. Evaluate an identical procedure on held-out India/US before a France submission. Budget roughly 30-90 minutes for an initial bounded pilot, subject to measured throughput. There is no defensible France gain estimate yet.

## A checked hypothesis: common-token pruning

The joint route drops words present in more than 1% of indexed records; the hard name route drops character trigrams present in more than 2% of its restricted index. This can damage short, repetitive names, but the measured zero-vector subset is small.

A read-only audit of raw test S2 TSVs, using the current `normalize_name`, found:

| French S2 measure | Value |
|---|---:|
| Total records | 703,378 |
| Hard-route records: empty raw address or domain-style name | 46,102 |
| Hard-route records with a zero TF-IDF vector | 383 (0.831% of that index) |
| Records whose every core-name word exceeds the joint-route 1% cutoff | 4.337% |

The audit used raw empty-address detection and omitted the Indic-name flag; compare against normalized cached records before treating it as an exact reproduction of `_hard_docs`. The local Parquet read failed, so raw TSVs were used. No pipeline files were changed.

Even if all 383 zero-vector records were missed true matches for distinct S1 entities and every affected entity gained a full point, fixing this particular S2 subset could improve France macro F0.5 by at most 383/259452 = 0.00148. Actual gain would be smaller; other routes can already find them. S3 and nonzero-but-distorted vectors were not audited. Raising `max_df` alone is therefore not a strong explanation of the full France gap. A small exact-name bypass for uniquely owned names is worth checking, but do not equate uniqueness in S1 with proof that a target is not a distractor.

## How to spend the remaining leaderboard slots

Choose a single locally supported France experiment, keep India/US predictions byte-identical, and compare against the scored baseline. The overall delta then estimates France delta as `overall_delta / public_France_fraction`. The full-test fraction 0.1497 is only a proxy for the unknown public subset fraction. Validation scores do not exactly identify the public India/US contribution.

Do not select a change by forcing French noise frequencies or predicted cluster sizes to match US: these are diagnostics, not known France targets. Do not treat arbitrary French S2/S3 records as labeled negatives for synthetic training; some are real matches. Distinct S1 identities provide cleaner negative seeds, but synthetic perturbations still need collision checks and held-out evaluation.

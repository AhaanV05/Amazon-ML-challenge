**Amazon ML Challenge 2026 — detailed strategy and execution plan**

Prepared on 25 September 2026 IST from both supplied PDFs, with primary-source research into relevant methods and model licenses. It was written before the dataset arrived. Since then the data has been audited and seven pipeline versions built. The final one (v7) scored 0.9798 on the public leaderboard. Measured results and as-built decisions are in the Merged decisions table below, which overrides earlier planning text where they differ. The team has four active members, access to paid GPUs, and an RTX 4060/5060 laptop GPU baseline. The priority is the best supported competitive result within the model rules and contest window, rather than minimizing GPU spending.

**Recommendation: build a strong lexical retrieval and supervised matching system first, then add multilingual evidence and better set selection where held-out experiments justify them.** The likely competitive advantage is the combination of retrieval coverage, rejection of plausible false matches, singleton handling, and generalization to France. Model size is a secondary decision.

The proposed core pipeline is:

```text
Provided source records
  -> several information-preserving text representations
  -> union of name, address, exact-key, and optional multilingual retrieval
  -> pair features and a supervised CatBoost matcher
  -> calibrated scores and entity-level match-set selection, including empty sets
  -> validated matching_results.tsv and candidate_pairs.tsv
```

Our execution order is: understand and measure the data; establish trustworthy validation; build a complete baseline; improve the largest measured bottleneck; freeze and reproduce the best supported solution.

**Merged decisions (25 September, after comparing three agent plans; updated through the night of 26 September with measured results)**

This document remains the rulebook for validation, leakage, and compliance. The companion page at https://claude.ai/artifact/We8aTBT8XEMECoaTDnaaF6 carries the illustrated build order. Where the two differ, these decisions apply:

| Decision | Detail | Section |
|---|---|---|
| Data findings (26 September) | Train 2.2M/5.0M/5.3M records, test 1.73M/4.89M/5.08M (India 47%, US 38%, France 15%). Mean 3.46 matches per S1, singletons 5.6%; 0 records under two S1 entities; ~26% of S2/S3 are distractors; country agrees on 100% of true pairs; postcodes in only 12% (US) / 1% (India). Address is the strongest signal (AUC 0.95 vs hard negatives); India names 0.72–0.75, with some scrambled names. Nine Indic scripts (~9% of names); our transliterator beat anyascii and indic-transliteration on 9,971 true pairs. | 3, 5 |
| Retrieval routes (26 September) | Sparse TF-IDF top-k per country with a 1% common-word cutoff (10× faster). Kept: name@place composite keys, address, joint name+address. Dropped name-only and phonetic-only (<0.05% unique recall each). Top-50 recall: India 98.5% / 97.4%, US 99.0% / 99.3% (S2 / S3). | 6 |
| Leaderboard and the population fix (26 September, evening) | v1 (LightGBM stages 1–2) scored 0.944 on the public leaderboard (rank 1418; top teams 0.985–0.991) against 0.962 validation. v2 (plus cross-encoder) scored 0.962 against 0.974 validation. Cause of the gap: training candidates covered only 20% of train S1 entities, so each Source 2/3 record faced ~4 competing entities in training but ~12 in test, shifting every competition feature and the filter's per-record cap. Second cap: the filter's own ceiling (0.9835 at 5.7 candidates per entity). v3 runs the whole pipeline for every train entity on a 48-core / 176 GB / RTX PRO 6000 Colab machine (`colab/pipeline_v3.ipynb`, `src/pipeline_v3.py`), with a looser filter budget (≤16 candidates per entity, ceiling ~0.990). | 4, 6, 12 |
| Search-miss analysis and fixes for v3 (26 September, night) | On 40,000 train entities, top-20 search leaves a 0.9928 ceiling (0.9899 India, 0.9957 US). 92% of misses share rare words with their S1 record but are outranked; only 8% have no lexical overlap. The misses concentrate in records with no address (36% of misses vs 4% of found pairs), domain-style names like "dynamicsystems.com" (21% vs 5%) and Indic-script names with short addresses (25% vs 17%). Fix 1: a fourth route, name-only character 3-grams searched against just those records (top-10), lifts the ceiling to 0.9953 for ~20 extra pre-filter pairs per entity; the ≤16 filter cap is unchanged. Fix 2: S2/S3 zero-pad house numbers ("0338"); stripping leading zeros cuts true pairs with a false number conflict from 5.2% to 4.2%. Still open: Indic names of common businesses ("baba eksaports"). An off-the-shelf embedding route (`multilingual-e5-small`, MIT) tested on India S2: top-10 recovers 25% of the Indic misses the new route leaves (it gets 4%), and pair recall goes 0.9755 → 0.9822 (new route) → 0.9862 (plus embeddings); a fine-tuned bi-encoder should do better. Decide after v3's leaderboard score. | 5, 6 |
| As built (26 September) | LightGBM only, for the candidate filter and both matching stages (fast enough on hundreds of millions of pairs); CatBoost was dropped. Natural class balance, no monotone constraints and no separate calibration step: the expected-F0.5 decoder reads the out-of-fold probabilities directly, and validation picks between it and a global threshold after a one-owner pass (each S2/S3 record to its best S1). Retrieval candidates already are hard negatives, so no extra mining round. | 7, 8, 9 |
| Cross-encoder lane (26 September) | First LightGBM validation: macro F0.5 0.954 against a 0.9835 ceiling after the filter, with India names the weakest signal. Fine-tune `xlm-roberta-base` (MIT, 278M) on Colab A100/H100 on a separate 10% of training entities, disjoint from the work set, so its scores are out-of-sample for stacking without k-fold. It scores every kept pair; `ce_logit` feeds both LightGBM stages. Notebook: `code/business_entity_resolution/colab/cross_encoder.ipynb`. In v3, `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) trains on the same entities in the background on the GPU and rescores only the uncertain band (base probability 0.02–0.98) as a second feature, `ce_large_logit`; total neural parameters about 0.85B. The LLM referee stays last. | 11 |
| Final day (27 September): v3 → v7 | **Leaderboard progression: 0.9773 → 0.9772 → 0.9790 → 0.9798 (final, v7).**<br>**v3:** validation 0.9853. A France probe (10% of French entities emptied) put France at ~0.93, against ~0.985 for India/US.<br>**v4:** two-fold cross-encoders on our own candidates plus a French region/department map. Validation 0.9860, France ~0.932.<br>**v5:** + Qwen3-Reranker-4B (LoRA) on ≤700k hard pairs. Neutral on validation.<br>**Post-processing on v5:** a sibling bridge (unclaimed records joined to confidently matched records by name/address/phonetic/number+street keys, LightGBM verifier trained on validation, +0.0010) and a France suffix-copy rule (the matcher accepted "Cie/Services" copies 97% but "Groupe/Développement" 9%). France rose to ~0.940.<br>**v6:** LightGBM on 100% of train entities, valid once every entity has out-of-fold reranker scores. +0.0005 on validation.<br>**v7 = v6 + bridge + suffix + France near-copies** (added words, misspellings). France ~0.942. Final.<br>**Rejected by measurement:** France from reranker scores alone, per-type recalibration, the legal-form-noise hypothesis, a France logit shift, and French real-word swaps (−0.0029 on the leaderboard). | 6, 9, 10, 11 |
| Small candidate sets (organizer update, 26 September) | Candidate-set size per S1 entity now counts in the final ranking. Retrieval stays wide and internal; a cheap learned filter, the last blocking stage, cuts each anchor's list to the smallest set that keeps validation F0.5, possibly empty. That filtered set is `candidate_pairs.tsv`. | 6 |
| Build the expected-F0.5 selector on Day 1 | It runs beside the global-threshold baseline from the first decoded submission; ship whichever validates better. Verified figures are in section 9. | 9 |
| Stage-2 context features | Rank within the anchor's list, competing-anchor evidence, and S2↔S3 triangle support, used as features in a second GBDT with out-of-fold inputs. Never used as merges. | 12 |
| Concrete normalization menu | Legal forms including French ones, trade-name splitting, a transliteration skeleton, landmark and postcode extraction, per-country IDF. | 5 |
| CatBoost first, LightGBM second | Same features and folds; blend only if it validates. Monotone constraints on the main similarities. | 7 |
| Parameter budget | Treat 8B as a total across all retained neural models until organizers clarify. Qwen3-8B (8.2B) is excluded. | 11 |
| Leaderboard slots | A planned use for all 15, including a France probe pair. | 14 |
| Rejected | A universal 0.7–0.9 threshold, class weighting without recalibration, country as a categorical model feature, and forcing blocking-missed positives into training. | 7, 8, 9 |

**1. Understand what constitutes a correct answer.**

Source 1 is the deduplicated reference set. For each Source 1 record, we must identify every matching Source 2 and Source 3 record. An anchor can have zero matches, one match, or multiple matches, including several in the same source. We are predicting sets of record IDs, not choosing one winner from each source.

The available business attributes are only `business_name`, `business_address`, and `country`. `entity_id` identifies records and their source; its numeric portion is not an identity feature. Training covers the US and India. Test additionally contains France, and all French Source 1 entities must be included.

The central semantic question to resolve from training labels is the identity granularity: how are branches, relocations, trade names, franchises, and similarly named businesses treated? The PDFs do not define every such edge case. We will inspect labeled examples before encoding address disagreement or shared brand names as decisive evidence.

Operational constraints from the PDFs:

| Requirement | Consequence for our work |
|---|---|
| Challenge: 25 September 00:00 through 27 September 23:59 IST | Plan around approximately 72 hours and protect time for final inference and packaging. |
| Maximum five leaderboard submissions per day | Submit complete, informative experiments; preserve every uploaded artifact and score. |
| Final model: MIT or Apache 2.0, at most 8B parameters | Audit the actual checkpoints and derived model artifacts, not merely the library used to load them. |
| No external business lookup, geocoding, registries, or internet data augmentation | Use supplied records and labels; method research must never become enrichment of competition records. |
| One output row for every test Source 1 record | Empty predictions are valid and often correct. |
| Candidate output is the actual set passed to the matching model | Log candidates during inference, after blocking filters, and include every final match. |
| Organizer update (26 September): the candidate file and the code producing it are reviewed, and smaller candidate sets per S1 entity rank higher in the final evaluation | Treat candidate-set size as a second objective next to macro F0.5. Keep every retrieval route sub-quadratic and state its complexity. Ask the organizers how size is weighed against the score. |
| Private leaderboard determines final rankings in the problem statement | Prefer generalizable improvements over public-score tuning. The general guidelines describe both leaderboards as relevant to shortlisting. |

Pretrained models are contemplated by the model constraint, but the PDFs do not fully define their permitted use or ensemble parameter accounting. Keep the initial solution independent of pretrained assets. For any neural addition, retain checkpoint provenance and verify applicable organizer clarifications. Do not use hosted identity-resolution services or external teachers to label competition examples.

**2. Optimize the exact metric from the beginning.**

Let the true and predicted match sets for Source 1 entity i be T_i and P_i. Define t_i = |T_i intersect P_i|, m_i = |T_i|, and k_i = |P_i|. The entity score is:

```text
If m_i = 0 and k_i = 0: score_i = 1
Otherwise:             score_i = 1.25 * t_i / (k_i + 0.25 * m_i)

Final score = mean(score_i over ALL Source 1 entities)
```

Equivalently, for a nonempty denominator:

`F0.5 = 1.25 TP / (1.25 TP + FP + 0.25 FN)`.

The coefficient of FP in this expression is four times the coefficient of FN. This does not make the score a fixed-cost classification loss and does not imply that a universal probability threshold of 0.8 is optimal. The PDF's worked example, two correct matches plus one incorrect match, correctly scores approximately 0.714.

| Truth | Prediction | Entity score | Practical lesson |
|---|---|---:|---|
| No matches | Empty | 1.000 | Correct singleton decisions earn full credit. |
| No matches | Any nonempty set | 0.000 | One false merge loses the entire singleton score. |
| One match | That match | 1.000 | Small match sets matter just as much as large ones in the macro average. |
| One match | True match plus one false match | 0.556 | A plausible extra match can be expensive. |
| Two matches | One correct match only | 0.833 | Missing a second match can be less harmful than adding a false one. |
| Two matches | Both correct plus one false | 0.714 | More predicted matches are not automatically better. |
| Ten matches | One correct match only | 0.357 | Precision alone is insufficient. |

An all-empty prediction has score equal to the true singleton proportion. We will compute it as a sanity baseline, not use it as our intended solution. A system must demonstrate gains on both singleton and nonsingleton slices.

Implement a local scorer immediately and verify these hand-calculated cases. The official submission validator checks formatting; it does not calculate the competition score. Do not substitute pooled pair F0.5, F1, accuracy, or an average over countries for the official macro average.

**3. Audit the data before choosing complexity.**

Load TSVs with an explicit tab separator, all IDs as strings, and empty strings preserved. A suitable starting read is `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`. Add `quoting=csv.QUOTE_NONE` if names contain stray quotes, and confirm row counts equal line counts. Normalize only after preserving raw fields. Blank lists must not become the literal text `nan`.

The initial audit should produce one compact report with the following measurements:

| Area | What we will measure | What it decides |
|---|---|---|
| Scale | Records in each source, country, and split; text lengths | Retrieval implementation, storage, GPU feasibility, and inference budget. |
| Label integrity | Missing IDs, duplicate labels, missing S1 ground-truth rows, targets assigned to multiple anchors | Whether labels support the assumed identity structure. Missing label rows are not silently treated as singletons. |
| Match cardinality | Fraction with 0, 1, 2, 3–5, and more matches; breakdown by target source | Baseline difficulty, singleton modeling, and candidate budgets. |
| Missingness | Empty name/address/country rates and joint patterns | Fallback retrieval routes and explicit missingness features. |
| Duplication | Repeated strings, repeated addresses, near-identical businesses, source duplicates | Grouped validation and the danger of overconfident exact matching. |
| Country consistency | Countries on every labeled positive pair | Whether country equality can safely be a default retrieval partition. |
| Identity semantics | Same brand/different address; aliases; relocations; branch-like examples | Strength of contradiction features and structural constraints. |
| Train/test shift | Country proportions, scripts, lengths, missingness, lexical overlap | Transfer risk and resource allocation, without inventing test labels. |

Manually review a stratified sample of training positives, singletons, and difficult nonmatches. Start with approximately 100 examples if time allows; increase only when a pattern is unclear. Record error categories and examples from development folds. Do not use test inspection to manually assign matches.

Numerically sequential IDs and apparent record ordering are not model features. An accidental ordering relationship can look excellent locally and fail on the private test split.

Verification: the report should answer how many records we must score, how many true singletons exist, where retrieval is likely to fail, and whether any intended constraint contradicts labels.

**4. Establish validation that resembles the actual task.**

A random split of generated record pairs is inadequate. The same business or target can occur on both sides, allowing the model to recognize familiar records rather than resolve unseen businesses.

First create identity groups from each S1 anchor and its labeled S2/S3 matches. Keep each complete group in a single fold. If a target unexpectedly belongs to multiple anchors, group the connected labels together and investigate. Keep strongly duplicated identity representations together when supported by the audit; do not automatically combine every company sharing a brand token.

For a development fold:

1. Train using only training-fold anchors and permitted training-fold target records. Held-out identity records must not reappear in training as negative pairs, augmentation sources, or fine-tuning examples.
2. Generate validation candidates from a realistic target gallery with all legitimate held-out positives and substantial natural distractors. A gallery containing only validation positives is artificially easy.
3. Evaluate against the complete original ground-truth sets, including matches the retriever missed. Never insert known validation positives into candidates to improve coverage.
4. Include every validation anchor, including genuine singletons and anchors for which retrieval returns nothing.
5. Record exactly which text statistics are fit where. Supervised transforms, learned aliases, calibration, and matching models must not use validation labels. Label-free indexing of the available target gallery is a retrieval operation and should be documented and mirrored at inference.

For speed, begin with a fixed grouped development split and a small untouched grouped audit split, stratified where possible by country, singleton status, and match count. Once the baseline is stable, use five grouped folds for out-of-fold predictions and model selection (three for a neural stage if runtime forces it). Keep the audit split out of routine error analysis and threshold tuning until finalists are ready. With very small data, prefer careful cross-validation over tiny, unstable holdouts.

Run two additional transfer tests: train on US and evaluate on India; train on India and evaluate on US. Choose their operating settings using only the training country's development data, then evaluate the held-out country. Tuning directly on the held-out country's labels would weaken the unseen-country test. These are stress tests for transfer, not estimates of a known French score.

Use the full available validation gallery when affordable. Also compare behavior as distractor counts increase: more candidates can create higher false-match maxima even if pairwise ranking barely changes. A smaller fold gallery can overstate performance and shift the best threshold.

Every serious experiment gets the same scorecard:

| Measure | Why it matters |
|---|---|
| Official macro F0.5 | Primary selection criterion. |
| Singleton accuracy and false-positive rate | Detects costly false merges on empty truth sets. |
| Nonsingleton macro F0.5 | Prevents a high singleton prevalence from hiding poor matching. |
| Country and transfer-fold scores | Checks robustness to France-like distribution shift. |
| Match-count and missing-field slices | Exposes failures hidden by the average. |
| Candidate pair recall and candidate oracle score | Separates retrieval limitations from matching limitations. |
| Candidate p50/p95/max, runtime, peak RAM/VRAM | Confirms the method can finish within the window. |

Compare finalists with paired bootstrap intervals over independent identity groups, not individual pairs. A tiny gain on one split is not sufficient evidence for a major increase in complexity.

**5. Normalize without destroying identity evidence.**

Maintain several views of each field rather than overwriting it with one aggressively cleaned string:

| View | Transformation | Intended use |
|---|---|---|
| Raw | Original supplied text | Auditing, pair-model inputs, exact evidence. |
| Conservative | Unicode normalization, case folding, whitespace cleanup | Default matching representation. |
| Punctuation-normalized | Standardize selected separators and ampersand variants | Formatting robustness. |
| Accent-folded alternative | Remove diacritics in an additional representation | Recall under accent variation, while preserving the accented original. |
| Token and character views | Ordered tokens, token sets, character n-grams | Complementary similarity signals. |
| Name-core alternative | Carefully handle observed legal suffix variants | Name comparison without losing the original legal name. |
| Address evidence | Numeric tokens and cautiously extracted components | Agreement and contradiction features. |

Preserve house numbers, unit numbers, sector/plot identifiers, postcodes, leading zeros, and meaningful alphanumeric components. (Measured exception, 26 September: S2/S3 zero-pad house numbers, "0338" for "338", so leading zeros are stripped from address number tokens.) Store extraction confidence when the same number could be a postcode, building number, or part of the business name. Unknown does not mean disagreement; two empty fields do not constitute positive evidence.

Avoid a universal `St -> Street` substitution: context may mean Saint. Do not remove all short tokens or generic-looking words without checking their effect. Treat an accent-folded match as one signal, not proof of identity. Use observed training patterns for abbreviation handling; do not download external address dictionaries or location databases.

A concrete starting menu. Keep each item only if ablation supports it:

| Item | Rule |
|---|---|
| Legal form | Split from the name core: inc, llc, corp, co, ltd, lp, llp; pvt, private, limited, opc; sarl, sas, sasu, sa, eurl, snc, sci, cie, ets, sté. Features: core-name similarity, plus legal form equal / differs / one missing. |
| Trade names | Split on dba, d/b/a, t/a, trading as, aka, and parentheses. Name similarity is the maximum over the alternatives. |
| Transliteration skeleton | sri/shri/shree → sri, ph → f, ee → i, oo → u, w → v, drop inner vowels, collapse doubled letters. An extra similarity view, never a replacement. |
| Postcode | Last standalone 6-digit group (join "560 001"), else 5-digit; ZIP+4 reduces to 5 digits. Store parse confidence. |
| Landmark | near, nr, opp, opposite, behind, beside, next to, en face de … up to the next comma, moved into its own field. |
| Saint vs Street | "St" before a proper name is Saint (Rue St Denis); after a street name it is Street (Main St). |
| Abbreviation table | Mine token substitutions (rd ↔ road, pvt ↔ private) from positive pairs inside each training fold only. |
| Token weighting | Per-country IDF fitted label-free on all supplied records, test included, so French generic words (rue, sarl, avenue) are down-weighted without a hand list. Mirror it at inference and document it. |

For France, the core protection is representation design that remains functional on new vocabulary. Character matching, Unicode handling, token rarity, number agreement, and missingness do not require a fixed US/India category list.

Verification: inspect before/after transformations on positive and hard-negative examples; ablate each aggressive transformation. Keep a transformation only if its retrieval or matching benefit survives false-positive analysis.

**6. Build candidate generation as a measurable recall system.**

Exhaustive pair comparison scales as N1 times (N2 + N3). We want an indexed candidate stage followed by approximately N1 times K pair evaluations, where K is the measured average candidate count.

Use a union of complementary routes:

| Route | How we retrieve | Expected contribution |
|---|---|---|
| Name characters | Sparse TF-IDF on name character n-grams | Typos, punctuation, spacing, partial names. |
| Address characters | Independent TF-IDF on address character n-grams | Name aliases and modest address corruption. |
| Combined fields | Field-marked name/address text or token retrieval | Cases requiring moderate agreement in both fields. |
| Rare-token and exact-key postings | Rare name tokens combined with address evidence; conservative composite keys | Precise retrieval and candidates beyond a small top-k cutoff. |
| Reverse retrieval | S2/S3 searches S1, then union the resulting edges | True pairs missed by forward ranking asymmetry. |
| Optional multilingual embeddings | Dense nearest neighbors on supplied record text | Language and wording differences poorly covered by lexical overlap. |

Start with separate S2 and S3 retrieval quotas so one large source cannot suppress the other. A reasonable initial experiment is top 25 by name, top 25 by address, and top 10 by combined text per source, plus composite-key hits. These are starting values, not a final design. Sweep selected budgets such as 10, 25, 50, and 100 based on the measured curve. No retrieval budget should become a hard cap on the number of accepted matches.

For names, begin around character n-grams of length 2–5; for addresses, 3–5 is a useful starting range. Compare token signals when word order is unstable. Preserve rare tokens initially; overly aggressive frequency filtering can discard the most identifying evidence. Scikit-learn provides the necessary character and word analyzer settings in its [TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Prefer same-country retrieval if positive-pair analysis supports it, but accept arbitrary country labels dynamically. Missing or suspect country fields need a fallback route. Hard country equality is a data-supported optimization, not an assumption to apply before auditing.

Use sparse top-k multiplication or an inverted index, with chunked queries. Do not materialize a dense N1-by-N23 similarity matrix. Start simple; distribute retrieval only if scale requires it. Top-k lexical blocking has a relevant research precedent in [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf); the lesson is to test effective lexical retrieval, not to introduce Spark unnecessarily.

Measure four distinct coverage quantities: pair recall; mean per-anchor recall among nonsingletons; fraction of nonsingletons with at least one retrieved truth; and fraction with every true match retrieved. Report them by country, target source, and cardinality.

Also compute the metric-specific oracle ceiling. If an anchor has m true matches and c are retrieved, a perfect classifier restricted to those candidates can achieve:

```text
Oracle_i = 1                          if m = 0
Oracle_i = 1.25*c / (c + 0.25*m)       otherwise
```

Macro-average over all anchors. This is more informative than pair recall alone. Missing the sole match of many businesses can be hidden by excellent recall on a few businesses with many duplicates.

An initial engineering aspiration is greater than 99% positive-pair coverage if affordable, with no severely weak slice. This is a target to investigate, not a promised result. Choose the smallest practical candidate budget that achieves a sufficiently high oracle ceiling and stable downstream score. If the ceiling remains low, improve retrieval before spending time on a bigger classifier.

Log retrieval origin, per-route rank, and similarity for every pair. They become matching features and make missed-match analysis much faster.

Chains are the reason the address and reverse routes exist: a brand that appears hundreds of times in Source 1 makes name-only top-k arbitrary.

**Learned candidate filter (the last blocking stage).** Since the 26 September update, the size of `candidate_pairs.tsv` per S1 entity counts in the final ranking, so retrieval and the exported candidate set are separate stages. Retrieval above stays wide (target at least 99.5% pair coverage) and is never exported. A cheap filter then decides what the matching model sees:

| Element | Design |
|---|---|
| Inputs | Retrieval-time signals only: route hits, per-route rank and score, reverse rank, exact-key hits, plus a few cheap checks (postcode equality, a shared premise number, name TF-IDF cosine). |
| Model | A small LightGBM predicting P(match), trained inside the training folds on the retrieval union. |
| Keep rule | Filter probability ≥ τ; the target is among its own top r anchors (r = 1–3, since a target belongs to at most one reference business if labels confirm it); at most K_max per anchor and source. An anchor may end with no candidates, which is a correct and free call for most singletons. |
| Operating point | Sweep (τ, r, K_max), run the full downstream pipeline out of fold, and plot macro F0.5 against mean candidates per S1 entity. Choose the knee and publish the curve in the methodology. |
| Boundary | The filter stays cheap and recall-oriented; everything from pair features onward is the matching model. A filter that effectively makes the final decision would turn the candidate file into the answers, and reviewers read the code. |

Report for every run: mean, median, p95, and max candidates per S1 entity; share of anchors with no candidates; pair recall and oracle ceiling after the filter; reduction ratio against all same-country pairs. Keep every retrieval route sub-quadratic (hash keys, inverted indexes, approximate nearest neighbors), cap oversized key blocks, and state each route's complexity in the methodology; the organizers frame blocking as a billions-of-records problem.

**7. Train a compact supervised matcher on identity evidence.**

The initial matcher should be CatBoost over numeric pair features. It can learn interactions between partial agreement and contradictions without requiring a large neural training run. CatBoost's official [usage examples](https://catboost.ai/docs/en/concepts/python-usages-examples) cover classification, probabilities, and sample weights; its repository carries an [Apache 2.0 license](https://github.com/catboost/catboost/blob/master/LICENSE).

Build features in coherent families and evaluate their incremental value:

| Family | Initial features | Failure it addresses |
|---|---|---|
| Name similarity | Exact conservative match, edit similarity, Jaro-Winkler, token overlap, TF-IDF cosine | Typos and reordering. |
| Name structure | Length ratio, abbreviation/acronym compatibility, raw versus name-core agreement | Abbreviations and legal suffix variation. |
| Distinctive tokens | IDF-weighted overlap; rare tokens matched, missing, or conflicting | Generic names that otherwise look persuasive. |
| Address similarity | Character cosine, token overlap, partial containment, normalized exactness | Partial and reordered addresses. |
| Numeric evidence | Shared/conflicting numbers; postcode and unit agreement when confidently parsed | Nearby but different businesses or branches. |
| Missingness | Separate presence flags for every field on each side | Preventing missing data from masquerading as agreement. |
| Joint evidence | Name/address similarity interactions; strong name with strong address conflict | A single attractive field hiding a contradiction. |
| Retrieval context | Route count, source-specific rank, forward/reverse agreement | Distinguishing robust candidates from marginal hits. |
| Ambiguity | Number of similar alternatives; target's evidence for competing S1 anchors | Common names and shared locations. |
| Commonness | How many S1 records share this name core (per country); how many share this street and number | Chains, malls, and office towers. |
| Transliteration and trade names | Skeleton similarity; best similarity across trade-name alternatives | Indian spelling variants and DBA names. |
| Optional dense evidence | Multilingual name and combined-text cosine | Wording and language variation. |

RapidFuzz is an efficient [MIT-licensed source of string metrics](https://github.com/rapidfuzz/RapidFuzz). Token-set or partial similarity can be deceptively high when a short generic name is contained in a longer unrelated one. Feed those scores alongside distinctiveness and missing-token features; do not threshold them in isolation.

Start with a compact feature set of approximately 30–60 well-understood values rather than hundreds of overlapping ratios. A useful initial CatBoost search is depth 5–8, learning rate 0.03–0.1, and up to roughly 1,500 boosting iterations with early stopping. These are bounded starting ranges, to reduce or expand according to dataset size and runtime.

Train LightGBM on the same features and folds as a second model, and blend only if the blend validates. Add monotone constraints on the main similarity features so higher similarity never lowers the score; this protects behavior on France.

Optimize a sensible pair classification loss during fitting, but select configurations and operating points by end-to-end macro F0.5. Compare ordinary training weights with weights that equalize total contribution per S1 anchor. Neither is mathematically equivalent to the official metric; validation decides. Do not let a handful of anchors with thousands of candidates dominate every update.

Source identity can be useful because S2 and S3 may have different noise patterns. Do not feed numeric IDs or a two-country-only encoding into the model. Prefer transferable features such as country agreement, character properties, field quality, and relative similarities; a country-agnostic model is an important control.

Verification: the model must beat the rule baseline on the same candidate set, on the exact metric, and on meaningful error slices. Feature importance is a debugging aid, not evidence that a feature is valid.

**8. Train on realistic hard negatives and preserve calibration.**

The classifier's task is distinguishing plausible candidates. Uniform random pairs are often too easy and give a misleading impression of progress.

Generate training candidates using the same retrieval logic used for validation and inference. A retrieved pair is positive when the target ID is in the training anchor's provided true set. Other retrieved pairs are negative under the supplied exhaustive-ground-truth definition. Investigate obvious label inconsistencies instead of silently relabeling them.

Prioritize these negative types: same or similar name but inconsistent location; same address but different business; common chain names; partial-name containment; conflicting unit or building evidence; and a candidate that fits another anchor much better.

Keep all available training positives. If retrieval missed one, flag the retrieval failure separately. It may be added to classifier training as a diagnostic or recovery example, but validation and test coverage must remain honest. Such supplementation is not proof that the real retriever can find it.

After the first model, mine its high-confidence false positives within training folds and retrain once. Avoid an elaborate multi-round mining process until a single round demonstrates value. Preserve some easier negatives so the training distribution is not composed solely of adversarial edge cases.

If memory forces negative subsampling, record the sampling policy and probabilities. Class balance changes, per-anchor weighting, and hard-negative mining can make raw outputs poor probability estimates. Calibrate and choose thresholds using naturally generated, unsampled held-out candidate sets. Do not assume a score of 0.95 means 95% precision.

Aliases or abbreviation mappings learned from positive pairs must also be learned inside training folds. Global alias mining before splitting leaks information about validation identities.

**9. Treat the final prediction as a set-selection problem.**

The reliable first decision rule is simple: accept candidates above a threshold selected to maximize the exact validation macro F0.5. Search score breakpoints or a reasonable coarse-to-fine grid, then prefer a stable plateau across folds over a narrow peak. Never use the public leaderboard to exhaustively search this threshold.

Compare a small number of policies, in order:

| Policy | Added complexity | Promotion requirement |
|---|---|---|
| Global threshold | Minimal | Baseline decision rule. |
| Per-entity expected-F0.5 top-k, including empty | One small function on calibrated out-of-fold scores | Beats the global threshold on grouped and transfer validation. Built on Day 1. |
| Source-specific thresholds | Two operating points | Stable evidence that source noise differs. |
| Pair score plus a simple ambiguity rule | Few features and parameters | Reduces false merges without collapsing recall. |
| Anchor-level singleton/set selector | Additional small model | Repeatable gain on grouped and transfer validation. |

For an optional anchor-level model, use out-of-fold candidate scores to summarize maximum score, upper-score distribution, margin, field quality, retrieval agreement, target conflicts, and candidate counts. Predict whether any match exists or which candidate-set size is justified. Do not train this layer on in-sample predictions from the base matcher.

A small top-two margin is not automatically bad: an anchor may have multiple genuine matches. Margin to an incompatible alternative, or evidence that a target belongs to another anchor, is more informative than a blanket margin requirement.

In mathematical terms, the empty prediction has expected utility `P(number of true matches = 0)`. A nonempty set A has expected utility:

`E[1.25 * number of true matches inside A / (|A| + 0.25 * total true-match count)]`.

This is the correct decision objective, but estimating it from pair probabilities requires assumptions about correlations and missing candidates. If exactly one candidate can possibly be the only true match, predicting it scores p in expectation and predicting empty scores 1-p; the optimal threshold is 0.5 in that special case. Precision emphasis alone cannot justify a universal 0.8 or 0.9 threshold.

Build the top-k expected-utility selector, including k=0, on Day 1 alongside the global threshold. It is one small function, and it shows that a single threshold is wrong in both directions. With independent, calibrated probabilities (verified by brute-force enumeration):

| Candidate probabilities | Empty | Top 1 | Top 2 | Top 3 | Best |
|---|---:|---:|---:|---:|---|
| 0.60 | 0.400 | **0.600** | | | Predict it |
| 0.45, 0.30 | 0.385 | **0.427** | 0.402 | | Top 1, even below 0.5 |
| 0.90, 0.50 | 0.050 | **0.825** | 0.728 | | Skip the 0.50 |
| 0.40, 0.40, 0.40 | 0.216 | 0.350 | 0.410 | **0.436** | Predict all three |
| 0.55, 0.20, 0.20 | 0.288 | **0.514** | 0.397 | 0.351 | Top 1 |

After a first match at 0.95 / 0.90 / 0.80 / 0.70, a second is worth adding only above 0.69 / 0.66 / 0.59 / 0.52.

The positives among the top k and among the rest are Poisson-binomial, so the expectation is exact:

```python
def poisson_binomial(ps):
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for i, d in enumerate(dist):
            nxt[i] += d * (1 - p); nxt[i + 1] += d * p
        dist = nxt
    return dist

def f05(tp, n_true, k):
    if n_true == 0 and k == 0: return 1.0
    if k == 0 or tp == 0: return 0.0
    return 1.25 * tp / (0.25 * n_true + k)

def best_subset(cands):          # cands: [(entity_id, calibrated_p), ...]
    cands = sorted(cands, key=lambda c: -c[1])
    ps = [p for _, p in cands]
    def score(k):
        top, rest = poisson_binomial(ps[:k]), poisson_binomial(ps[k:])
        return sum(top[a] * rest[b] * f05(a, a + b, k)
                   for a in range(len(top)) for b in range(len(rest)))
    k = max(range(len(ps) + 1), key=score)
    return [eid for eid, _ in cands[:k]]
```

Prune candidates below about 0.02 first, and fold the expected number of blocking misses into the "rest" distribution. Independent-Bernoulli approximations are convenient but imperfect because duplicate records are correlated and blocking can miss truth, which is why the threshold baseline stays as the comparison. Do not substitute a ratio of expected counts for the exact expectation without labeling it an approximation. Related research distinguishes probability-based utility decisions from empirical F-measure optimization in [Optimizing F-measure: A Tale of Two Approaches](https://arxiv.org/abs/1206.4625).

Verification: the optional selector must improve actual held-out macro F0.5, not merely produce more appealing probabilities. If it is unstable, retain the simpler threshold policy.

**10. Make France a first-class generalization test.**

A strong random US/India validation score does not establish French performance. Our response is to reduce dependence on country-specific memorization and test transfer deliberately.

We will preserve Unicode and accent variants, support arbitrary country strings, use character features and distinctiveness, separate missingness from disagreement, and avoid requiring every address to fit one rigid parser. French text should flow through the same pipeline and produce valid candidates without a special emergency branch.

Use US-to-India and India-to-US transfer results to choose representations and to reject brittle rules. Report the mixed-country score and both transfer scores separately; do not invent a weighted estimate of France performance. For unseen countries, use a shared operating policy supported by transfer tests rather than a guessed France-specific threshold.

Two cheap protections: per-country IDF fitted on all supplied records (section 5) and monotone constraints on similarity features (section 7). A small French seed list (legal forms; street types rue, av, bd, pl, ch, imp, quai, rte, fbg, allée; articles de, du, des, la, le, l', d' as stopwords; elision splitting) is code, not external data, and belongs in the methodology. The public leaderboard offers one noisy France signal: two submissions that differ only in French predictions measure the France effect directly. Use that probe once, not for tuning.

Inspect unlabeled test distributions for missing fields, script differences, lengths, and retrieval failure rates. This can identify broken preprocessing or absent candidates. It cannot measure French accuracy, infer a valid singleton rate, or justify forcing France to resemble the US distribution.

Training-derived perturbations such as punctuation removal, harmless token reordering, or field dropout are optional regularization experiments only if permitted by organizer interpretation. Keep original labels only when identity is preserved, generate them within the training fold, and use real held-out records for evaluation. No external translation service, downloaded business examples, or fabricated address facts should enter training.

**11. Run a dedicated multilingual modeling lane alongside the baseline.**

There are two distinct neural opportunities: improve candidate recall and improve judgments among existing candidates. With four teammates and paid GPU access, research and pilot both in parallel with the baseline. Promotion into the final system still depends on the measured bottleneck and validation gain.

| Candidate | Verified license and size | Proposed role |
|---|---|---|
| `intfloat/multilingual-e5-base` | MIT; model card reports approximately 0.3B parameters | First compact multilingual embedding experiment. |
| `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0; approximately 0.6B parameters | Alternative embedding experiment if E5 has inadequate incremental coverage. |
| `Qwen/Qwen3-Reranker-0.6B` | Apache 2.0; approximately 0.6B parameters | Optional additional pair evidence, validated for identity matching. |
| `FacebookAI/xlm-roberta-base` | MIT; approximately 0.3B parameters | Supervised multilingual pair classifier with a binary head. |
| `BAAI/bge-reranker-v2-m3` | Apache 2.0 per model card; approximately 0.57B parameters | Multilingual cross-encoder alternative to fine-tuning XLM-R base from scratch. |
| `Qwen/Qwen3-Reranker-4B` | Apache 2.0; approximately 4B parameters | Paid-GPU experiment on difficult pair judgments and optional task-specific adaptation. |

The primary model cards document [E5](https://huggingface.co/intfloat/multilingual-e5-base), [Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base), and [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B). Checkpoint licensing meets the stated model-license category; broader competition eligibility remains subject to the organizers' interpretation of pretrained assets. These model sizes are comfortably below the cap individually, but we will also count the retained ensemble conservatively.

For dense retrieval, encode provided records locally using a fixed field-marked representation. Cache vectors so rerunning CatBoost does not repeat encoding. Start with one combined-text representation; add a separate name representation only if it contributes useful matches. Follow the model's prescribed pooling, normalization, and input formatting. E5 recommends `query:` prefixes for symmetric similarity and provides 768-dimensional embeddings with a 512-token limit; truncation must be measured against actual record lengths.

A dense neighbor can be the same kind of business rather than the same business. Therefore, dense similarity belongs in retrieval and supervised features, not an unconditional matching rule. Retain the dense route only when it adds useful true candidates or improves held-out decisions at acceptable cost.

For a neural pair classifier, serialize the two records with explicit field names and train a binary identity decision on provided labels and hard negatives. Fine-tuning a compact, license-audited multilingual encoder is a possible second-day experiment. [Ditto](https://arxiv.org/abs/2004.00584) provides a primary-source precedent for entity matching as sequence-pair classification; it is methodological inspiration, not a reason to import external ER datasets or copy all augmentation steps.

The first supervised neural pilot will use XLM-R base, a binary classification head, and the same identity-group splits as CatBoost. Start with a measured maximum pair length around 256 tokens, extending toward 512 when truncation would remove useful address evidence. Try a small learning-rate comparison around 1e-5 to 3e-5 and one to three epochs with early stopping; adjust microbatch size to measured memory. Use labeled hard negatives from our retriever. Validate checkpoints using the full candidate pipeline and exact macro F0.5, not training loss alone. Aim to begin this pilot late on Day 1 as soon as valid candidate pairs exist.

On paid compute, compare the 4B reranker against the small supervised encoder on the same development error slices. Give it an identity-specific instruction using only the supplied fields, and obtain a fixed match score rather than free-form explanations. If off-the-shelf behavior is promising but systematically wrong, test adapter fine-tuning on provided training labels. Do not ask it to recall business facts from memory, call services, or create new entity labels for training. A larger model may be especially useful on aliases and partial records, but its gains remain a hypothesis until measured.

An ambitious candidate final system is one 0.6B embedding model, one roughly 0.3B supervised pair encoder, a 4B reranker if complementary, and the small tabular matcher. Its rough neural total is about 4.9B before exact accounting. Keeping three separately trained 0.3B pair encoders instead would raise that rough total to about 5.5B. These are budget illustrations, not a commitment to keep every model. Count exact retained parameters, heads, adapters, and independently stored ensemble members before submission. Quantization reduces memory, not parameter count, and an adapter's small trainable count does not replace the base model's count. Avoid assuming that a checkpoint marketed as "8B" is necessarily at or below an exact 8,000,000,000-parameter cap. Qwen3-8B, for example, reports 8.2B parameters and is excluded.

An off-the-shelf reranker is trained for relevance, so its output is not a calibrated identity probability. Test it as complementary evidence first. If it helps, combine out-of-fold neural scores with the tabular features using a small blend or stack. Avoid deploying a complicated ensemble whose training predictions leaked into its own combination model.

If the neural stage processes only ambiguous pairs, use exactly the same routing policy during training validation and inference. The exported candidate set remains every pair scored by the matching cascade, including pairs scored only by its first stage. Candidate-set definitions must not drift when a neural stage is introduced.

**12. Use structural consistency carefully.**

Deduplicated Source 1 suggests a target should normally belong to at most one reference business. Verify this against labels and organizer guidance before imposing it as a hard constraint. It is different from restricting each S1 to one match, which is explicitly incorrect.

A useful lightweight feature is a target's strongest competing S1 score. If one target has strong evidence for two incompatible anchors, the decision layer can account for that ambiguity. Any enforced target exclusivity must allow unlimited matches per S1 and must improve held-out performance. Do not automatically resolve every tie by choosing the slightly larger score.

S2-to-S3 agreement may provide corroborating evidence, but graph transitivity is not proof of identity. One false edge can merge many businesses. Avoid unconditional connected-component clustering and unscored propagation.

Use that evidence as stage-2 features instead. From out-of-fold stage-1 scores, compute for each pair (anchor e, target x): x's rank and gap within e's list; e's top scores and its count of candidates above fixed levels; x's best score for any other anchor and whether e is x's best; and triangle support, the maximum over e's confident candidates y from the other source of min(p(e, y), p(x, y)). Here p(x, y) comes from an S2↔S3 pair model trained on record pairs that share an anchor. A second GBDT learns how much to trust these signals; nothing is merged unconditionally.

If a carefully bounded graph expansion is tested, use it to propose additional candidates. Score every proposed S1-target pair with the matcher, preserve provenance, and add those scored pairs to the exported candidate set. Promote this experiment only after simpler improvements are exhausted.

Identical text under multiple target IDs still represents multiple output records. Cache their shared features or embeddings for efficiency, then expand back to all legitimate IDs. Do not deduplicate away target IDs that the metric expects us to recover.

**13. Run a short, ordered experiment ladder.**

Change one major factor at a time. Every experiment must record its configuration, split, candidate version, runtime, exact score, and error slices.

| ID | Experiment | Primary question | Keep it when |
|---|---|---|---|
| E0 | All-empty and conservative lexical-rule baselines | Are parsing and scoring correct? | Required diagnostic; establishes reference behavior. |
| E1 | Name/address/combined lexical candidates | Can we retrieve the true links efficiently? | Oracle ceiling and weak slices are acceptable. |
| E1b | Learned candidate filter | How small can candidate sets get before macro F0.5 drops? | Knee chosen on the size-versus-F0.5 curve. |
| E2 | CatBoost on core pair features, with global-threshold and expected-F0.5 decoders | Can supervision reject plausible false matches, and which decision rule scores better? | End-to-end macro F0.5 beats E0; keep the better decoder. |
| E3 | Numeric conflicts, rarity, missingness, retrieval context | Which evidence fixes observed errors? | Incremental gain survives slice analysis. |
| E4 | Realistic hard-negative mining and calibration | Are confident false merges reducible? | Improves operating-point performance on unsampled validation. |
| E5 | One multilingual embedding channel | Is there missing lexical coverage or transferable semantic evidence? | Adds oracle coverage or matcher quality without excessive cost. |
| E6 | Stage-2 context features and an anchor-level singleton model | Does competing-anchor and cross-source evidence remove remaining false merges? | Gain repeats across grouped folds and transfer checks. |
| E7 | One compact neural pair model or reranker | Does it fix errors the tabular model cannot? | Adds complementary, reproducible improvement. |
| E8 | Small blend or verified structural policy | Can complementary evidence improve the final set? | Improves finalists on untouched audit data without a serious regression. |

E0–E4 are the main plan. E5–E8 are conditional. We are not committing to eight increasingly complex models; we are committing to measuring which bottleneck is worth addressing.

For each development run, sample the most costly false positives, singleton false merges, retrieval misses, and classifier misses. Label their causes. Examples of action rules:

| Observed bottleneck | Next action |
|---|---|
| Oracle ceiling is poor | Improve retrieval views, reverse retrieval, or candidate budgets. |
| Oracle is high but match score is poor | Improve pair evidence, hard negatives, or decision policy. |
| Good pair ranking but poor macro F0.5 | Revisit calibration, singleton handling, and operating-point selection. |
| Mixed validation good, transfer poor | Remove brittle country assumptions; test multilingual evidence. |
| Gain only on public leaderboard | Require a credible validation explanation before adopting it. |
| Better model cannot finish final inference safely | Reduce cost or keep the reproducible incumbent. |

**14. Schedule the work around the three-day window.**

These are elapsed-work targets, not claims about measured runtimes. Adjust them after the initial data and hardware audit.

| Window | Work | Required output / verification |
|---|---|---|
| First 0–2 hours | Read resource package, audit schemas and labels, implement metric, create grouped splits | Audit report; scorer passes hand cases; split invariants pass. |
| Hours 2–6 | Normalize, build lexical retrieval, measure recall curves | Cached candidates and oracle ceiling by slice. |
| Hours 6–12 | Core features, CatBoost, threshold selection, first inference | First complete valid submission and archived configuration. |
| Remaining Day 1 | Error analysis, numerical/rarity features, first hard-negative pass; parallel XLM-R pilot once labels and folds are ready | Strong reproducible incumbent and first measured neural comparison. |
| Day 2 morning | Transfer folds and targeted retrieval improvements | Evidence of which changes generalize. |
| Day 2 afternoon | One multilingual embedding experiment; optional neural pair pilot | Costed comparison against incumbent. |
| Day 2 evening | Calibration, optional singleton selector, limited combination | Shortlist of two or three credible finalists. |
| Day 3 first half | Evaluate finalists, audit model assets, select frozen configuration | Final selection based on validation plus leaderboard evidence. |
| Day 3 second half | Final fit, full inference, validation, reproduction, documentation | Complete submission archive and matching upload. |
| Final several hours | Keep a known-valid artifact ready; verify upload status and archive | Avoid a last-minute training or packaging failure. |

The submission deadline is 27 September 2026 at 23:59 IST. Budget the final buffer using measured inference time and at least one retry, rather than assuming inference will be quick. Freeze ambitious experiments earlier if the dataset is large.

Suggested leaderboard use: Day 1, a valid baseline and a supervised improvement; Day 2, meaningful retrieval or neural variants supported locally; Day 3, frozen finalists and the selected final artifact. Five per day is a cap, not a requirement to consume every slot. Keep an available slot where possible for a correction. Verify the portal's daily reset and final-selection behavior rather than assuming that the last upload or highest public score is automatically selected.

Planned slot use (unused slots expire daily):

| Day | Slots |
|---|---|
| Day 1 (Fri 25) | 1 baseline proving the format; 2 CatBoost + global threshold; 3 expected-F0.5 selector; 4–5 spare for corrections |
| Day 2 (Sat 26) | 6 stage-2 context features; 7 dense retrieval route; 8 neural pair score; 9–10 France probe pair (differ only in French predictions) |
| Day 3 (Sun 27) | 11 blend; 12 tighter candidate filter; 13 optional 4B reranker if it validates; 14 final by grouped validation; 15 safest variant or correction |

Use the four confirmed teammates as follows:

| Owner | Responsibility | First deliverable | Final responsibility |
|---|---|---|---|
| Teammate 1 | Data integrity, exact scorer, grouped splits, transfer evaluation | Shared split manifest, audit report, metric checks | Independent finalist evaluation, error audit, methodology results. |
| Teammate 2 | Normalization, lexical/dense retrieval, candidate provenance, the learned candidate filter and its size-versus-F0.5 curve | Cached candidate table with recall and oracle report | Final candidate generation and memory/runtime checks. |
| Teammate 3 | Pair features, CatBoost and LightGBM, hard negatives, stage-2 context features, calibration, expected-F0.5 selection | Strong supervised baseline and out-of-fold scores | Final tabular fit, operating policy, output integration. |
| Teammate 4 | Multilingual pair encoder, 4B reranker experiments, GPU profiling | Small-model pilot on the shared pairs and folds | Neural out-of-fold predictions, frozen assets, license/parameter manifest. |

Assign one of these teammates as the designated portal uploader and integrator at the first team check-in; this is an additional coordination responsibility, not a fifth role. Use a shared pair schema containing `source1_entity_id`, `candidate_entity_id`, `fold_id`, retrieval provenance, and label for training only. Score artifacts must identify model version and whether predictions are truly out of fold. Merge work at agreed checkpoints rather than maintaining four incompatible notebooks. All teammates use the same scorer and frozen split IDs.

During the wait for data, prepare the environment and model-asset manifest, agree on interfaces, write metric/format checks, and smoke-test a small forward pass on the available GPU. Do not fabricate performance claims or tune models on invented competition data. Separate role ownership from portal access; the guidelines prohibit simultaneous logins for a participant.

**15. Size compute from measurements, not model marketing.**

Treat the laptop baseline as an 8 GB VRAM class device: NVIDIA lists 8 GB for both the [RTX 4060 Laptop GPU](https://www.nvidia.com/en-us/geforce/laptops/40-series/) and [RTX 5060 Laptop GPU](https://www.nvidia.com/en-in/geforce/laptops/50-series/). Confirm the actual laptop, free VRAM, RAM, and power configuration; these specifications do not establish our training throughput.

| Resource | Planned use | Operating approach |
|---|---|---|
| Laptop CPU/RAM | Parsing, sparse retrieval, pair features, CatBoost, evaluation | Chunk data to measured memory; cache reusable artifacts. |
| Laptop GPU, approximately 8 GB VRAM | E5/Qwen 0.6B embedding inference; XLM-R pilot where it fits | Mixed precision, short measured sequence lengths, small microbatches, gradient accumulation for training. |
| Paid GPU with approximately 24–48 GB VRAM | Faster encoder training, larger batches, 4B reranker inference or adapter fine-tuning | Choose after a memory/throughput pilot; do not assume full 4B fine-tuning fits. |
| Additional paid GPUs | Independent folds, reproducibility runs, or distinct model experiments | Spend compute on stronger evidence and coverage before redundant model-size increases. |

The CPU-capable lexical-plus-CatBoost system remains an independent fallback. Laptop execution is a development baseline, not a restriction on the final solution. If a promising experiment requires cloud hardware, use it; if it merely requires hours of fragile memory tuning, move the job to suitable hardware and preserve the contest time. Pin and smoke-test a compatible GPU software stack early, especially across different GPU generations. No cloud resources have been provisioned during this strategy task.

Use these planning calculations:

- Pair count is approximately `N1 * average_candidates`. At 100,000 anchors and 100 candidates, this is 10 million pairs; this is an illustration, not the known dataset size.
- Float32 feature storage alone is approximately `pairs * features * 4 bytes`. Ten million pairs with 50 features already use about 2 GB before IDs, indices, training copies, and framework overhead. Chunked generation and columnar storage may be necessary.
- Embedding storage is approximately `records * dimensions * bytes_per_value`, before the search index and metadata. Half-precision storage can reduce space, but similarity behavior must be checked.
- Neural inference time is approximately `pairs actually sent to neural stage / measured pairs per second`, plus loading and I/O. Benchmark a representative sample and extrapolate before committing.

Cache normalized records, retrieval indices, candidates, features, and embeddings with versions tied to input checksums and preprocessing settings. Avoid recomputing retrieval for a threshold-only experiment. Keep exact tie-breaking deterministic.

Do not assume an 8B model is required because it is permitted. A small model with a sound candidate set and appropriate decision rule is the preferred starting point; a larger model needs measured incremental value and enough time to finish.

**16. Keep the implementation small and reproducible.**

The proposed source layout, to create when implementation begins, is:

```text
code/business_entity_resolution/
  src/
    data.py           # read, audit, normalize records
    candidates.py     # retrieval and candidate provenance
    features.py       # pair features
    train.py          # folds, fitting, optional calibration
    predict.py        # set selection and both TSV outputs
    evaluate.py       # exact macro metric and diagnostics
  configs/
  artifacts/          # required fitted assets and their manifest
  README.md
  requirements.txt
```

Keep executable logic in these scripts; notebooks should call it rather than maintain copied versions. Store fold assignments, seeds, model revision hashes, transforms, score policy, data checksums, license records, and the candidate-generation configuration. Record software and hardware conditions relevant to reproducibility.

For every leaderboard upload, preserve an immutable directory containing both output TSVs, their checksums, the run configuration, local metrics, the producing code version, model identifiers, timestamp, and reported score. Never overwrite the only copy of the current best submission.

Before final packaging, reproduce inference from the frozen assets in a clean process. Test retraining as feasible and document any hardware nondeterminism honestly. The package must contain the necessary fitted assets or a complete reproducible route to create them; a filename pointing at a teammate's cache is insufficient. Follow any portal size limits discovered in the resource package.

**17. Produce the exact required submission artifacts.**

The required final archive is:

```text
<team_name>_submission.zip
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/
    business_entity_resolution/
      src/
      README.md
      requirements.txt
      ...required configurations and model assets...
  Documentation_template.md
```

Use the exact headers:

```text
matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
candidate_pairs.tsv: source1_entity_id<TAB>candidate_entity_ids
```

Second-column values are comma-separated IDs, with empty strings for empty sets. Emit exactly one row per test S1, no duplicate IDs within a row, only valid test S2/S3 IDs, and no missing French anchors. Use stable S1 order and deterministic target-ID ordering for reproducibility. Final matches must be a subset of the corresponding candidates.

On Windows, write both files with `sep="\t"`, `index=False`, `lineterminator="\n"`, `quoting=csv.QUOTE_NONE`, and `encoding="utf-8"`, then read them back with the section 3 settings.

The candidate artifact must come from the actual inference pair table. Do not reconstruct it from early retrieval logs that include unscored pairs, or from the final predictions that omit rejected candidates. For a multi-stage scorer, document which candidate pairs enter the matching cascade and which progress to later stages.

Run the organizer's command from the resource directory once the dataset package is available:

```text
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Also assert matched-set inclusion ourselves, even if the official validator only warns about it. Re-read the TSVs after writing them, check ID membership and complete S1 coverage, inspect empty rows, and verify output checksums against the archived run. Confirm `SCORED` status in the portal after the leaderboard upload. Only `matching_results.tsv` is routinely uploaded for scoring; both files belong in the final package.

The general guidelines request a 1–2 page approach document, whereas the problem statement requests a filled methodology template with no page limit. Prepare a concise 1–2 page overview and the detailed filled template, and use the portal's required upload fields. The two documents also describe ranking and timing of code review differently; prepare the full reproducible archive for every eventuality and use organizer clarifications for administrative uncertainties. No clarification has been sent as part of this planning task.

The methodology should explain the task, exact metric, validation design, candidate recall and oracle ceiling, candidate-set size distribution and reduction ratio, the size-versus-F0.5 curve, each retrieval route's complexity, normalization, pair features, trained models, parameter counts and licenses, decision policy, country-transfer results, ablations, runtime, and reproduction steps. Include measured results only when they exist. A sophisticated write-up cannot compensate for a pipeline that cannot regenerate its outputs.

**18. Define what we will deliberately prioritize.**

Our first deliverable after receiving the data is a full, valid lexical-plus-CatBoost baseline with an honest grouped validation score. The next deliverable is an error report that tells us whether to improve retrieval, reject false matches, handle singletons better, or strengthen transfer.

The highest-value order is: exact scoring and trustworthy splits; candidate coverage, then a small filtered candidate set; discriminative pair evidence and hard negatives; calibration and empty-set decisions; country transfer; then neural additions and limited ensembling. We will not spend the opening day training a large model while the metric, candidate ceiling, or label semantics remain uncertain.

Success means a reproducible, rules-compliant submission chosen through evidence, with every test anchor represented and the actual candidate set auditable. A winning rank cannot be inferred from the PDFs alone. This plan is designed to maximize our chance of a strong private score while making each hour of the competition answer a concrete question.

Primary competition references: [problem statement](6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf), especially pages 1–2, 3–5, and 6–7; [guidelines and key instructions](6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf), pages 1–2. External sources linked above are method and model documentation, not sources of business identity data.

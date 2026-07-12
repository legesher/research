# SIB-200 per-gold-category breakdown (CORE-1383)

Verification analysis extending [`correct-via-constant-findings.md`](correct-via-constant-findings.md). That analysis measured how concentrated each cell's predictions are (top_pred_share, top_raw_share) but could not distinguish two mechanisms behind the concentration: **uniform constant-answering** (the dominant category is predicted regardless of gold; the model ignores the passage) versus **selective behavior** (correct where gold matches the dominant category, degraded but still passage-sensitive elsewhere). Distinguishing them requires the predicted-category distribution per gold category, computed here from row level.

Companion TSV: [`sib200-category-accuracy.tsv`](sib200-category-accuracy.tsv) (784 rows; this directory). Build script: [`evaluation/scripts/build_sib200_category_breakdown.py`](../../evaluation/scripts/build_sib200_category_breakdown.py).

## Provenance

- Run 2026-07-09 against HF main, `legesher/language-decoded-experiments`, per-row files `phase3/conditions/<cond>/seed<N>/<cond>_seed<N>_results_template{1,2}.json`.
- Conditions and seeds: baseline (seed none), condition-2-ur-5k (seeds 42/123/456), condition-5-ur-5k (seed 42), condition-5-es-5k (seed 42). 12 results files, 112 SIB-200 cells, 22,848 rows.
- Extractor: refined (`reparse_results.extract_sib200_category`), the extractor behind all published Phase-3 refined numbers.
- Cross-check: recomputed per-cell accuracy matches the published `_summary_reparsed_template*.json` value exactly for all 112 cells. The local `reparse_results.py` sha256 (`a42b1dcb…`) differs from the pinned `content_sha256` in the published summaries (`c7a1d51d…`); the exact match on every cell establishes behavioral equivalence on this corpus.
- Cell coverage: baseline covers all 16 (data_lang × instr_lang) cells per template; condition-2-ur-5k and condition-5-ur-5k cover instr ∈ {en, ur} × data ∈ {en, es, ur, zh}; condition-5-es-5k covers instr ∈ {en, es} × data ∈ {en, es, ur, zh} (8 cells per template each). Baseline comparisons below are restricted to matched cells.
- Gold distribution is identical in every one of the 112 cells (SIB-200 is n-way parallel; denominator 204 rows per cell): science/technology 51 (25.0%), travel 40 (19.6%), politics 30 (14.7%), sports 25 (12.3%), health 22 (10.8%), entertainment 19 (9.3%), geography 17 (8.3%).

Uniform-vs-selective metric, matching the XNLI companion analysis: the expected accuracy of a gold-independent predictor with the cell's own predicted-category marginals is null = Σ_c P(pred=c)·P(gold=c); define edge = accuracy − null. Uniform constant-answering drives edge toward 0; passage-sensitive behavior keeps it large. (Unlike XNLI, SIB-200 gold is imbalanced, so here a marginal shift toward science/technology DOES raise null: predicting only science/technology yields 25.0% accuracy for free.)

## Per-gold-category accuracy (recall), condition vs matched baseline

Seed-collapsed (mean across seeds per cell first), then mean across the condition's 8 matched cells per template; baseline restricted to the same 8 cells. Denominator inside each cell and gold slice: the slice sizes above.

condition-2-ur-5k (instr ∈ {en, ur} cells):

| | sci/tech | travel | politics | sports | health | entertainment | geography |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline t1 | 0.924 | 0.597 | 0.908 | 0.475 | 0.426 | 0.401 | 0.287 |
| cond-2-ur-5k t1 | 0.797 | 0.632 | 0.786 | 0.683 | 0.409 | 0.524 | 0.289 |
| baseline t2 | 0.946 | 0.559 | 0.721 | 0.370 | 0.205 | 0.092 | 0.272 |
| cond-2-ur-5k t2 | 0.886 | 0.601 | 0.956 | 0.638 | 0.307 | 0.237 | 0.461 |

condition-5-ur-5k (instr ∈ {en, ur} cells; baseline rows identical to the above):

| | sci/tech | travel | politics | sports | health | entertainment | geography |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cond-5-ur-5k t1 | 0.966 | 0.441 | 0.888 | 0.380 | 0.222 | 0.342 | 0.118 |
| cond-5-ur-5k t2 | 0.897 | 0.566 | 0.967 | 0.540 | 0.188 | 0.191 | 0.228 |

condition-5-es-5k (instr ∈ {en, es} cells):

| | sci/tech | travel | politics | sports | health | entertainment | geography |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline t1 | 0.966 | 0.581 | 0.871 | 0.645 | 0.403 | 0.618 | 0.257 |
| cond-5-es-5k t1 | 0.983 | 0.241 | 0.650 | 0.420 | 0.312 | 0.651 | 0.154 |
| baseline t2 | 0.973 | 0.631 | 0.892 | 0.540 | 0.290 | 0.092 | 0.272 |
| cond-5-es-5k t2 | 0.988 | 0.359 | 0.671 | 0.405 | 0.148 | 0.243 | 0.250 |

The signatures diverge cleanly. cond-2-ur-5k LOWERS science/technology recall relative to matched baseline (0.924→0.797 t1, 0.946→0.886 t2) while raising most other categories (template 2: politics +0.235, sports +0.268, geography +0.189, entertainment +0.145, health +0.102). cond-5-es-5k does the opposite: science/technology recall rises to near-ceiling (0.983/0.988) while travel (−0.34/−0.27), politics (−0.22 both), and sports (−0.23/−0.14) collapse. cond-5-ur-5k sits between, degrading non-science categories on template 1.

## Uniform vs selective, per cell

Modal predicted category: science/technology in 32/32 baseline cells, 16/16 cond-5-es-5k cells, 16/16 cond-5-ur-5k cells, and 46/48 cond-2-ur-5k cells (the 2 exceptions are politics). The science/technology prior is baseline-inherited everywhere; what differs is how far it is pushed.

Edge (seed-collapsed per cell) by condition × template, with the matched-baseline value in brackets:

| condition | template | mean edge [baseline matched] | min cell | max cell |
| --- | --- | --- | --- | --- |
| baseline (16 cells) | 1 | 0.446 | 0.313 | 0.619 |
| baseline (16 cells) | 2 | 0.376 | 0.180 | 0.528 |
| condition-2-ur-5k (8) | 1 | 0.481 [0.467] | 0.276 | 0.591 |
| condition-2-ur-5k (8) | 2 | 0.474 [0.381] | 0.330 | 0.534 |
| condition-5-ur-5k (8) | 1 | 0.366 [0.467] | 0.235 | 0.519 |
| condition-5-ur-5k (8) | 2 | 0.420 [0.381] | 0.354 | 0.491 |
| condition-5-es-5k (8) | 1 | 0.352 [0.493] | 0.279 | 0.417 |
| condition-5-es-5k (8) | 2 | 0.315 [0.424] | 0.098 | 0.509 |

The cond-5-es-5k template-2 spread is the finding. Its four instr=en cells stay selective (edge 0.442 to 0.509, at or above baseline); its four instr=es cells collapse toward the null (data=es 0.098, data=en 0.140, data=ur 0.156, data=zh 0.244; all seed 42, refined extractor). The constant-answering is cell-selective (template 2 × matched instruction language, the same coordinates as the documented flip pattern) but uniform within the affected cells. Per-cell values for every condition are derivable from the TSV.

### The documented worst cell, in full

condition-5-es-5k, SIB-200, template2, data=es, instr=es, seed 42, refined extractor, n=204 (correct-via-constant-findings: top_raw_share 0.9020 `Ciencia y Tecnología`; the category-level predicted share here is 185/204 = 0.907, the raw form plus accent variants):

| gold (n) | acc | pred sci/tech | travel | politics | sports | health | entertainment | geography | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| science/technology (51) | 1.000 | 51 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| travel (40) | 0.100 | 36 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| politics (30) | 0.367 | 19 | 0 | 11 | 0 | 0 | 0 | 0 | 0 |
| sports (25) | 0.080 | 22 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| health (22) | 0.000 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| entertainment (19) | 0.053 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| geography (17) | 0.000 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Every gold slice's modal prediction is science/technology; four of six non-science categories retain at most 2 correct rows. Cell accuracy 0.338 versus null 0.240: edge 0.098. This is uniform constant-answering with a residual politics signal (11/30). The same cell at baseline (baseline, SIB-200, template2, data=es, instr=es, seed none, n=204) is also science-heavy (141/204 = 0.691 predicted science/technology) but selective: politics 21/30, travel 17/40, sports 9/25, health 6/22 correct; accuracy 0.515, edge 0.295.

### The least-constant condition, in full

condition-2-ur-5k, SIB-200, template2, data=ur, instr=ur, seed 42, refined extractor, n=204:

| gold (n) | acc | pred sci/tech | travel | politics | sports | health | entertainment | geography | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| science/technology (51) | 0.804 | 41 | 0 | 1 | 0 | 0 | 0 | 0 | 9 |
| travel (40) | 0.125 | 21 | 5 | 6 | 0 | 0 | 0 | 0 | 8 |
| politics (30) | 0.933 | 1 | 0 | 28 | 0 | 0 | 0 | 0 | 1 |
| sports (25) | 0.440 | 2 | 0 | 7 | 11 | 0 | 0 | 2 | 3 |
| health (22) | 0.227 | 6 | 0 | 3 | 1 | 5 | 0 | 0 | 7 |
| entertainment (19) | 0.105 | 11 | 0 | 4 | 0 | 0 | 2 | 0 | 2 |
| geography (17) | 0.118 | 12 | 0 | 3 | 0 | 0 | 0 | 2 | 0 |

Diagonal mass in five of seven categories, science/technology share 0.461 of 204 rows, politics an independent strong attractor (28/30 recall with almost no false positives from gold=science). Seeds 123 and 456 reproduce the pattern (politics recall 0.933 in all three seeds; TSV). This is selective behavior, and it confirms the correct-via-constant REVERSAL finding at category level: cond-2-ur-5k's mean top-category share across its cells (0.35 to 0.39 by template, seed-collapsed) is below its matched baseline (0.41 to 0.43), while cond-5-es-5k pushes to 0.59 to 0.64 against a matched baseline of 0.48 to 0.53.

## Science/technology attribution (both denominators)

How much of each condition's SIB-200 accuracy the science/technology gold slice carries. Row-pooled per condition × template (denominator A: all rows of the condition-template, i.e. 8 or 16 cells × seeds × 204; denominator B: all correct rows of the condition-template). The gold science/technology slice is 25.0% of rows everywhere, and a pure science/technology constant would score 25.0% overall with 100% of its correct rows in that slice; those are the reference points.

| condition | template | rows | correct | acc | sci-correct | A: share of all rows | B: share of correct rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 3,264 | 2,082 | 0.638 | 782 | 0.240 | 0.376 |
| baseline | 2 | 3,264 | 1,846 | 0.566 | 777 | 0.238 | 0.421 |
| condition-2-ur-5k | 1 | 4,896 | 3,131 | 0.640 | 975 | 0.199 | 0.311 |
| condition-2-ur-5k | 2 | 4,896 | 3,191 | 0.652 | 1,085 | 0.222 | 0.340 |
| condition-5-ur-5k | 1 | 1,632 | 931 | 0.570 | 394 | 0.241 | 0.423 |
| condition-5-ur-5k | 2 | 1,632 | 980 | 0.600 | 366 | 0.224 | 0.374 |
| condition-5-es-5k | 1 | 1,632 | 893 | 0.547 | 401 | 0.246 | 0.449 |
| condition-5-es-5k | 2 | 1,632 | 857 | 0.525 | 403 | 0.247 | 0.470 |

Denominator A is capped at 0.250 (the gold slice size) and nearly saturated for cond-5-es-5k: 0.246/0.247 means the model recovers virtually every science/technology gold row (recall 0.983/0.988). Denominator B is the load-bearing one: at baseline template 2, 42.1% of all correct rows are science/technology rows; cond-5-es-5k template 2 pushes that to 47.0% pooled, and to 73.9% in the worst cell (51 of 69 correct rows in condition-5-es-5k, template2, data=es, instr=es, seed 42). cond-2-ur-5k moves the opposite direction (0.311/0.340, below its matched baseline of 0.357/0.429, row-pooled on the same 8 cells), because its accuracy gains come from non-science categories.

For instr=es template-2 cells of cond-5-es-5k specifically (seed 42, per cell, denominator B): en 0.671, es 0.739, ur 0.646, zh 0.543 of correct rows are gold science/technology. The instr=en cells of the same condition sit at 0.350 to 0.383, indistinguishable from baseline.

## Verdicts

1. **cond-5-es-5k: uniform constant-answering, localized to (template 2 × instr=es).** In those four cells every gold slice's modal prediction is science/technology, edge falls to 0.098 to 0.244 (baseline same cells: 0.295 to 0.406), and 54% to 74% of correct rows are the gold science/technology slice. The condition's instr=en cells remain baseline-like and selective. Its SIB-200 headline accuracy is therefore a mix of intact English-instruction cells and near-degenerate matched-instruction cells, and citing it without the split overstates the matched-instruction capability.
2. **cond-2-ur-5k: selective, not constant.** The correct-via-constant reversal is confirmed at category level and explained: fine-tuning on Urdu REDUCES the inherited science/technology prior (recall on that slice drops), spreads probability mass onto other categories (politics 0.956 recall on template 2), and raises edge above matched baseline. Its accuracy gains are real classification improvements, not constant-answer luck.
3. **cond-5-ur-5k: intermediate.** Template 1 degrades toward the constant (edge 0.366 vs matched baseline 0.467; science recall 0.966); template 2 stays selective (edge 0.420 vs 0.381).
4. **The science/technology prior is baseline-inherited.** The modal predicted category is science/technology in 110/112 cells across all four conditions including baseline; fine-tuning modulates its strength in both directions rather than creating it.

## TSV schema

One row per (condition, seed, template, data_lang, instr_lang, gold_category): `n_gold` (the fixed slice sizes above), `correct` (refined pred == gold), `acc_on_gold` (denominator `n_gold`), predicted-category counts `pred_science_technology` / `pred_travel` / `pred_politics` / `pred_sports` / `pred_health` / `pred_entertainment` / `pred_geography`, and `pred_none` (refined-extractor parse-failures). All counts are row-level; any rollup must collapse seeds first if it produces threshold-counts (see [`aggregation-bug-audit.md`](aggregation-bug-audit.md)).

_Run 2026-07-09 against HF main. Re-run `build_sib200_category_breakdown.py` when refined data refreshes._

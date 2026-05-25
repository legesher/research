# Constant-output bias on Phase-3 SIB-200 — findings

Companion to [`correct-via-constant-rates.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/correct-via-constant-rates.tsv) (on HF; built by [`evaluation/scripts/build_correct_via_constant.py`](../evaluation/scripts/build_correct_via_constant.py)) — action item F from [`phase-3/post-refined-action-items.md`](phase-3/post-refined-action-items.md). Run 2026-05-25 against HF main.

The TSV holds 416 rows (one per condition × seed × template × data_lang × instr_lang SIB-200 cell, 42 input files). Two parallel metrics:

- **Category-level** (`top_pred_share`, `correct_via_constant_pred_pct`) — model concentrates predictions on one SIB-200 category, regardless of which surface form it uses
- **Raw-output-level** (`top_raw_share`, `correct_via_constant_raw_pct`) — model literally emits the same first-line surface form regardless of passage; this is what action item F originally asked about

The raw-output metric is the stronger evidence of "model isn't reading the passage." Below uses the raw-output metric unless noted.

## Headline — constant-output bias is widespread, and concentrated in cond-5

90 / 416 cells (22%) have a single first-line raw output covering ≥50% of all rows; 330 / 416 (79%) have ≥30%. The dominant constant output across the corpus is some surface form of `science/technology` — the model defaults to that category in many cells regardless of language.

### Per-condition rollup (seed-collapsed, mean across cells)

| Condition           | Mean `top_pred_share` | Mean `top_raw_share` | Mean `correct_via_constant_pred_pct` | Mean `correct_via_constant_raw_pct` |
| ------------------- | --------------------- | -------------------- | ------------------------------------ | ----------------------------------- |
| baseline            | 0.4946                | 0.3929               | 0.4125                               | 0.3336                              |
| condition-1-en-5k   | 0.5156                | 0.4095               | 0.4154                               | 0.3352                              |
| condition-1-en-20k  | 0.5368                | 0.4304               | 0.4228                               | 0.3498                              |
| condition-2-es-5k   | 0.5274                | 0.4717               | 0.3980                               | 0.3509                              |
| condition-2-es-20k  | 0.5150                | 0.4593               | 0.3908                               | 0.3391                              |
| condition-2-ur-5k   | **0.3697** (lowest)   | **0.3030** (lowest)  | 0.3280                               | 0.2081                              |
| condition-2-ur-20k  | 0.3977                | 0.3572               | 0.3613                               | 0.2137                              |
| condition-2-zh-5k   | 0.4582                | 0.4344               | 0.3536                               | 0.3382                              |
| condition-2-zh-20k  | 0.4923                | 0.4608               | 0.3642                               | 0.3434                              |
| condition-3-zh-5k   | 0.4709                | 0.4427               | 0.3636                               | 0.3414                              |
| **condition-5-es-5k** | **0.6131** (highest) | **0.5273** (highest) | 0.4809                               | 0.3886                              |
| condition-5-ur-5k   | 0.4933                | 0.4035               | 0.4057                               | 0.3201                              |
| **condition-5-zh-5k** | **0.6391** (highest) | 0.5058               | 0.4833                               | 0.3281                              |

Rollup is **seed-collapsed**: per-(template, data, instr) cell, mean across seeds; then mean across cells per condition. This avoids the seed-vs-cell aggregation conflation bug class — pivots on `(condition, seed, template, benchmark, data, instr)` that label their observation count as `n_cells` will inflate multi-seed conditions, since the actual unique-cell count is `total / (n_seeds × n_templates)`. The TSV emitted by [`build_correct_via_constant.py`](../evaluation/scripts/build_correct_via_constant.py) is at full per-row grain (one row per seed × template × cell), so any rollup over it must collapse seeds first.

## Critical reversal — cond-2-ur-5k is the *least* constant-output condition

The original action item F flagged cond-2-ur-5k for emitting a near-constant `سائنس/تکنالوجی`. **The data does not support this framing.** Among all Phase-3 conditions:

- cond-2-ur-5k has the **lowest** mean `top_raw_share` (0.3030)
- cond-2-ur-5k has the **lowest** mean `correct_via_constant_raw_pct` (0.2081)

Concretely, the cond-5-ur-5k template2_sib200_data=ur_instr=ur cell (the cell our [I spot-check](refined-verification-spot-checks.md#i--cond-5-ur-5k-template2-sib-200-instrur-sanity-sample) covered) has `top_raw_share` = 0.24 (`سائنس/تکنالوجی` 49 / 204) and `correct_via_constant_raw_pct` = 0.20. That is moderate, not dominant.

The conditions where constant-output bias actually deserves a paper caveat are **cond-5-es-5k** and **cond-5-zh-5k** — both with `top_raw_share` ≈ 0.50+ and the most-suspicious individual cells in the corpus.

## Most-suspicious individual cells (top 10 by `top_raw_share × correct_via_constant_raw_pct`)

| Condition          | Seed    | T  | Data | Instr | n   | Acc    | `top_raw`              | `top_raw_share` | `cvc_raw_pct` |
| ------------------ | ------- | -- | ---- | ----- | --- | ------ | ---------------------- | --------------- | ------------- |
| cond-5-es-5k       | seed42  | 2  | es   | es    | 204 | 0.3382 | `Ciencia y Tecnología` | 0.9020          | 0.7391        |
| cond-5-es-5k       | seed42  | 2  | en   | es    | 204 | 0.3725 | `Ciencia y Tecnología` | 0.8284          | 0.6447        |
| cond-5-zh-5k       | seed42  | 2  | es   | zh    | 204 | 0.4559 | `科学/技术`            | 0.7696          | 0.5484        |
| cond-2-es-5k       | seed42  | 2  | ur   | es    | 204 | 0.4363 | `science/technology`   | 0.7255          | 0.5506        |
| cond-2-es-5k       | seed42  | 2  | es   | es    | 204 | 0.4706 | `Science/Technology`   | 0.7402          | 0.5312        |
| cond-2-es-5k       | seed123 | 2  | ur   | es    | 204 | 0.4314 | `science/technology`   | 0.7157          | 0.5455        |
| cond-2-es-5k       | seed123 | 2  | es   | es    | 204 | 0.4755 | `Science/Technology`   | 0.7304          | 0.5258        |
| cond-2-es-20k      | seed42  | 2  | ur   | es    | 204 | 0.4314 | `science/technology`   | 0.6863          | 0.5568        |
| cond-2-es-5k       | seed456 | 2  | es   | es    | 204 | 0.4902 | `Science/Technology`   | 0.7255          | 0.5100        |
| cond-2-es-5k       | seed123 | 2  | en   | es    | 204 | 0.4902 | `Science/Technology`   | 0.7206          | 0.5100        |

All 10 are template2 (native-prompt). The es-instr cells dominate — cond-5-es-5k and cond-2-es-5k together account for 7 of the top 10. cond-2-es-5k is one of the four conclusion-flip cells (§8.3 of [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md)); the constant-output bias is a plausible mechanism for why the headline cond-vs-baseline gain reverses post-extractor-refinement.

## What this means for the paper

1. **Reframe the cond-2-ur-5k caveat.** The original "cond-2-ur-5k emits a near-constant `سائنس/تکنالوجی`" framing is not supported by the data. If the paper mentions constant-output bias as a caveat for Urdu specifically, it should be removed or qualified — Urdu shows the **least** constant-output bias of any condition.
2. **New caveat for cond-5 (zh + es).** cond-5-zh-5k and cond-5-es-5k have substantial constant-output bias — their reported gains over baseline on SIB-200 are partly explained by the model defaulting to `科学/技术` / `Ciencia y Tecnología` and happening to be right. If the paper cites cond-5 SIB-200 gains, this should be flagged.
3. **The cond-2-es flip mechanism.** cond-2-es-5k and cond-2-es-20k constant-output bias correlates with the SIB-200 conclusion flips in §8.3 — these are the cells the paper would want to discuss anyway.

## Threshold guidance

For paper claims, a per-cell flag at `top_raw_share >= 0.5` AND `correct_via_constant_raw_pct >= 0.5` catches the worst 10 cells. A looser flag at `top_raw_share >= 0.3` catches 330 / 416 cells (most of the corpus), which is too noisy to use as-is. The `top_raw_share × correct_via_constant_raw_pct` ranking used above is more discriminating.

## Pointers

- [`correct-via-constant-rates.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/correct-via-constant-rates.tsv) — full per-cell table (on HF)
- [`evaluation/scripts/build_correct_via_constant.py`](../evaluation/scripts/build_correct_via_constant.py) — build script
- [`refined-verification-spot-checks.md`](refined-verification-spot-checks.md) — I's cond-5-ur-5k spot-check (which surfaced the cond-5 constant-output pattern)
- [`phase-3/post-refined-action-items.md`](phase-3/post-refined-action-items.md) — item F (this finding shifts the framing)
- [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md) §8.3 — conclusion-flip cells (cond-2-es / cond-2-zh / cond-3-zh)

_Run 2026-05-25 against HF main post HF PR #34. Re-run when refined data refreshes._

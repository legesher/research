# Aggregation bug-class audit (seed-vs-cell)

**Date:** 2026-05-25
**Trigger:** `b0ec1dc` (View D fix in `framework_template_robustness.tsv`) and
HF discussion PR [#38](https://huggingface.co/datasets/legesher/language-decoded-experiments/discussions/38)
(corresponding data refresh, merged).

The 4-script code review caught the bug in `write_template_robustness`. This
audit checks whether the same pattern recurs across the rest of the build
scripts. **Result:** the bug class lives in 5 more sites. Numerical impact
varies by whether the affected column is a comparable count or just a
misleading label.

## Bug class

Code pivots on `(condition, seed, template, benchmark, data, instr)` and then
labels its observation count as `n_cells` — when the count is actually
`n_cells × n_templates × n_seeds`. Multi-seed conditions get inflated counts.

**Diagnostic rule:** if a TSV column is a count and that column's value
differs from the actual unique-cell count for multi-seed conditions
(1, 2, or 3 seeds × n_templates × n_data_instr_cells), it's the bug. Means
are usually still mathematically valid (uniform inflation within a
condition), but threshold-counts and cross-condition counts are not.

## Findings

| File | Function (line) | Status | Affected columns |
|---|---|---|---|
| `build_framework_comparison.py` | `write_template_robustness` (520) | FIXED in `b0ec1dc` | n_cells, brittle_cells_gt_0.10, frac_brittle |
| `build_framework_comparison.py` | `write_parse_failure_recovery` (662) | **paper-grade** | `n_cells` mislabeled; **`n_cells_recovery_gt_0.05` inflated** (apples-to-oranges across conditions, same shape as View D `brittle_cells_gt_0.10`) |
| `build_framework_comparison.py` | `write_benchmark_breakdown` (616) | label-only | `n_cells` mislabeled (means valid; no threshold count) |
| `build_framework_comparison.py` | `write_cross_language_impact` (365) | label-only | `n_cells` mislabeled (means valid; no threshold count) |
| `build_comparison.py` | `group_stats` (243) + `write_tsv` (272) | validation-grade | `n_cells`, `n_improved`, `n_regressed`, `n_flat` inflated. Rollups grouped *by condition* still have valid within-condition means; rollups grouped *across* conditions (by benchmark, by instr_lang) are dominated by multi-seed conditions. |
| `build_vs_baseline.py` | `agg` (220) + `write_rollup` (250) | validation-grade | `n_cells`, `n_wins_orig`, `n_losses_orig`, `n_wins_rep`, `n_losses_rep`, `n_flip_win_to_loss`, `n_flip_loss_to_win` all inflated. `conclusion_flips.tsv` itself is per-cell and safe; only the per-condition / per-benchmark rollups have the bug. |

## What's safe (audited and OK)

- `write_template_comparison` — preserves seed column
- `write_same_language_comparison` — explicit `n_seeds`, mean across seeds per cell
- `write_data_volume_comparison` — per-cell pivot then collapse, `n_seeds_5k/20k` explicit
- `write_decomposition_vs_baseline` — per-row output, no aggregation
- `write_seed_variance` — explicit `n_seeds`; `total_count` / `total_correct` are correctly summed observations
- `conclusion_flips.tsv` — per-cell rows (the load-bearing 48-flip count is from this, not from a rollup)

## Recommended scope

1. **Paper-blocking** — fix `write_parse_failure_recovery` the same way as View D (collapse seeds first → per-cell recovery, then count cells > 0.05). The paper cites this view; the inflated `n_cells_recovery_gt_0.05` is the exact bug shape we just fixed.
2. **Paper-grade hygiene** — rename `n_cells` → `n_observations` (or apply the same collapse) in `write_benchmark_breakdown` and `write_cross_language_impact`. Numbers don't change but the column label currently lies.
3. **Validation tables** — `build_comparison.py` and `build_vs_baseline.py` rollups aren't paper-cited per `paper-prep-callouts`, but if any supplement or methodology section quotes them, the win/loss/flip counts need the same treatment. Easy to defer to a follow-up if scope is tight.

For #1, the HF refresh pattern is the same as PR #38: regenerate the affected TSV via `build_framework_comparison.py` against current HF main, upload only that one TSV via `huggingface_hub.create_commit(..., create_pr=True)`. Approximately 5 minutes end-to-end. For #2 the data files don't change (only the column header / docstring), so no HF refresh needed — column rename can land on GitHub alone, with the next regenerate cycle picking it up.

## Cross-references

- Memory `lessons-pr54-pr55-pr56` item 12 (bug class — "seed-vs-cell aggregation conflation") and item 14 (headline-mean ambiguity, related)
- View D fix in `build_framework_comparison.py:write_template_robustness` (commit `b0ec1dc`)
- HF data refresh: discussion PR [#38](https://huggingface.co/datasets/legesher/language-decoded-experiments/discussions/38) (closed PR [#37](https://huggingface.co/datasets/legesher/language-decoded-experiments/discussions/37) superseded after in-PR rebase failed to move merge-base)
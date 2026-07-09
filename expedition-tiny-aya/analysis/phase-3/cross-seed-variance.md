# Cross-seed variance analysis (CORE-1383 verification item)

Verifies which quoted Phase-3 headline numbers are robust to fine-tuning seed, which are within seed noise, and which cannot be assessed because the condition was run with a single seed.

## Data provenance and method

- **Source:** `phase3/analysis/refined-tables/vs_baseline_cells.tsv` and `cells.tsv` on `legesher/language-decoded-experiments` (Hugging Face, `main`), fetched 2026-07-09 via the `curl` pattern in `data-refresh-cookbook.md` §6. Row counts matched expectations (cells.tsv: 1,664 + header; vs_baseline_cells.tsv: 1,536 + header).
- **Metric:** `delta_rep` (refined-extractor accuracy delta vs baseline), in percentage points. Refined scoring is canonical; strict (inference-time) scoring `delta_orig` is used only in §3 where the two extraction layers are compared. The baseline is seed-invariant (`seed=none`), so every per-seed delta shares the same baseline.
- **Aggregation:** a cell is (data_lang, template). An "instr=X aggregate" is the unweighted cell-mean over 4 data_langs x 2 templates = 8 cells, computed per seed; matched-diagonal values are the mean over the 2 template cells at data_lang = instr_lang. Cross-seed statistics are then taken over the per-seed aggregates: mean, sample standard deviation (ddof=1), min, max. Benchmark item counts per underlying cell: Belebele 900, X-CSQA 1,000, SIB-200 204, XNLI 5,010.
- **No interpolation.** Single-seed conditions are reported as point estimates with `std = NA`. Nothing is imputed.
- Companion table: `cross-seed-variance.tsv` (this directory), 560 data rows. Grain and population are stated in §5.

## Seed coverage

Only **4 of the 12 fine-tuned conditions are 3-seed** (seeds 42, 123, 456): `condition-1-en-5k`, `condition-2-es-5k`, `condition-2-ur-5k`, `condition-2-zh-5k`. The remaining 8 are single-seed (42): `condition-1-en-20k`, `condition-2-{es,ur,zh}-20k`, `condition-3-zh-5k`, `condition-5-{es,ur,zh}-5k`. Any claim resting on a 20k, cond-3, or cond-5 number is a point estimate with no within-condition variance measure.

## 1. Headline cells (cond-2-ur)

Population per row: instr=ur aggregate = 8 cells/seed (data in {en, es, ur, zh} x templates {1, 2}); matched diagonal = 2 cells/seed (data=ur, templates {1, 2}). Values are delta_rep vs baseline, pp.

| Cell | Quoted | n_seeds | mean +/- sd | per-seed (42 / 123 / 456) | \|mean\|/sd | Verdict |
|---|---|---|---|---|---|---|
| cond-2-ur-5k SIB-200 instr=ur agg | +12.0 | 3 | **+12.05 +/- 2.33** | +9.44 / +13.91 / +12.81 | 5.2 | Robust; positive in all seeds |
| cond-2-ur-5k Belebele instr=ur agg | +5.9 | 3 | **+5.91 +/- 0.81** | +6.54 / +6.20 / +5.00 | 7.3 | Robust |
| cond-2-ur-5k XNLI instr=ur agg | +3.5 | 3 | **+3.50 +/- 0.89** | +3.32 / +4.48 / +2.72 | 3.9 | Robust |
| cond-2-ur-5k X-CSQA instr=ur agg | +0.9 | 3 | +0.88 +/- 0.75 | +0.16 / +1.65 / +0.82 | 1.2 | **Within noise**; sign not 2-sigma-stable (paper already frames it as marginal) |
| cond-2-ur-5k SIB-200 ur x ur diagonal | -9.0 | 3 | **-8.99 +/- 4.32** | -12.99 / -9.56 / -4.42 | 2.1 | Sign holds (negative in all 3 seeds; mean + 2 sd = -0.35) but the **magnitude is highly seed-dependent** (3x spread). Quote as "roughly -4 to -13 pp across seeds" rather than a precise -9.0 |
| cond-2-ur-20k SIB-200 instr=ur agg | -1.7 | 1 | -1.72 (point) | -1.72 / . / . | n/a | **Cannot be assessed** (single seed) |
| cond-2-ur-20k Belebele instr=ur agg | | 1 | +2.60 (point) | | n/a | Single seed |
| cond-2-ur-20k XNLI instr=ur agg | | 1 | +5.31 (point) | | n/a | Single seed |
| cond-2-ur-20k SIB-200 ur x ur diagonal | | 1 | -25.24 (point) | | n/a | Single seed |

All quoted values reproduce to the tenth of a pp from the canonical TSVs.

**The 5k-to-20k "sign flip" claim (draft3 §4, cond-2-ur-20k SIB-200 instr=ur at -1.7 pp vs 5k's +12.0 pp) needs a caveat.** The 20k side is one seed. The *drop* from +12.05 is far outside the 5k seed spread (min seed +9.44), so more data not helping is well supported. But the *negative sign* of -1.72 is not: at the 5k condition's observed sigma of 2.33 on the same aggregate (the only available scale reference; the 20k run has no variance measure of its own), a value of -1.72 is within 2 sigma of zero. Recommended framing: "the gain collapses at 20k (single seed: -1.7 pp)" rather than asserting a regression.

**Template structure inside the ur x ur diagonal sigma.** The diagonal's large sigma is template-concentrated: template 1 is strongly negative in every seed (-19.12 / -13.24 / -12.75), template 2 straddles zero (-6.86 / -5.88 / +3.92; seed 456 is positive). The diagonal regression is a template-1 phenomenon, consistent with the flip-pattern structure note (flips concentrate in template 2 x matched instr_lang for sign instability; here template 2 is the unstable half).

## 2. Cond-5 extraction cells (all single-seed)

`condition-5-{ur,es,zh}-5k` were each run with seed 42 only. **No sigma is computable for any cond-5 number.** The data does distinguish strict vs refined extraction (`delta_orig` vs `delta_rep`). Population: instr=target aggregate = 8 cells; diagonal = 2 cells. SIB-200, delta vs baseline, pp.

| Cell | Strict (orig) | Refined (rep) | Note |
|---|---|---|---|
| cond-5-ur-5k SIB-200 instr=ur agg | -4.96 | +4.78 | **Extractor flips the sign** of the cond-5-ur aggregate |
| cond-5-ur-5k SIB-200 ur x ur diagonal | -4.17 | -2.94 | Negative under both |
| cond-5-es-5k SIB-200 instr=es agg | -25.06 | -12.56 | Negative under both |
| cond-5-es-5k SIB-200 es x es diagonal | -26.23 | -15.20 | Negative under both |
| cond-5-zh-5k SIB-200 instr=zh agg | -33.88 | -18.38 | Negative under both; matches the quoted -18.4 |
| cond-5-zh-5k SIB-200 zh x zh diagonal | -25.98 | -10.79 | Negative under both |

The abstract's "+22 pp swing" for Chinese (cond-2-zh +3.45 vs cond-5-zh -18.38, gap +21.83) is supported on the cond-2 side (sigma 1.51) but the cond-5 side is a point estimate.

**The "cond-5 underperforms cond-2 on every one of the 12 matched-instruction cells" claim** mixes 3-seed (cond-2) and 1-seed (cond-5) conditions. All 12 gaps are positive as claimed. Testing each gap against 2x the cond-2 seed sigma (the only variance available):

- 11 of 12 gaps exceed 2 sigma of the cond-2 side (range +0.17 pp zh Belebele to +21.83 pp zh SIB-200).
- **zh XNLI is the exception:** gap +0.80 pp vs cond-2-zh-5k sigma 0.42 (2 sigma = 0.84). This single cell could flip within cond-2's own seed noise, before even accounting for the unmeasured cond-5 seed noise. The zh Belebele gap (+0.17 vs 2 sigma = 0.14) clears the bar only marginally. If the paper keeps the "all 12" phrasing, it should note that the two smallest gaps are within or near cond-2 seed noise and the cond-5 side is single-seed.

## 3. Cross-lingual transfer rows (instr=en, refined)

The quoted transfer numbers ("+2.5 pp English Belebele", "+2.0 pp English XNLI") match the **instr=en aggregate across all 4 data_langs** (8 cells/seed), not the data=en x instr=en cell. This confirms the averaging-axis labeling issue already flagged in `paper-verification-review.md` (M16 applies to the transfer table too). Both readings are shown; the direction of the finding survives either way.

| Condition | Benchmark | instr=en agg (mean +/- sd) | data=en x instr=en (mean +/- sd) | \|mean\|/sd (agg) |
|---|---|---|---|---|
| cond-1-en-5k | Belebele | +0.76 +/- 0.27 | +0.76 +/- 0.17 | 2.8 |
| cond-1-en-5k | XNLI | -0.06 +/- 0.48 | +0.03 +/- 0.41 | 0.1 (null; sign unstable, consistent with the quoted "-0.1") |
| cond-2-es-5k | Belebele | +1.07 +/- 0.28 | +0.98 +/- 0.32 | 3.8 |
| cond-2-es-5k | XNLI | +0.26 +/- 0.04 | +0.39 +/- 0.23 | 5.8 |
| cond-2-zh-5k | Belebele | +1.36 +/- 0.04 | +0.83 +/- 0.17 | 32.2 |
| cond-2-zh-5k | XNLI | +0.42 +/- 0.19 | +0.43 +/- 0.05 | 2.2 |
| cond-2-ur-5k | Belebele | **+2.47 +/- 0.31** | +1.79 +/- 0.23 | 7.9 |
| cond-2-ur-5k | XNLI | **+2.01 +/- 0.57** | +2.74 +/- 0.72 | 3.5 |

Quoted +2.5 / +2.0 (cond-2-ur-5k) and +0.8 / -0.1 (cond-1-en-5k) all reproduce.

**The "Urdu advantage over English code" comparison, paired by seed** (both conditions used seeds 42/123/456; difference = cond-2-ur-5k minus cond-1-en-5k, instr=en aggregate):

| Benchmark | Paired diffs (42 / 123 / 456) | mean +/- sd | \|mean\|/sd |
|---|---|---|---|
| Belebele | +1.78 / +1.62 / +1.71 | **+1.70 +/- 0.08** | 22.1 |
| XNLI | +3.25 / +1.46 / +1.48 | +2.06 +/- 1.03 | 2.0 |

The Belebele advantage is the single most seed-stable comparative result in the study. The XNLI advantage is positive in all 3 seeds but sits exactly at the 2-sigma boundary (seed 42 contributes over half the mean); it should be presented as directionally consistent rather than as a precise +2.1 pp.

Secondary check: the SIB-200 matched-instruction resource gradient (es -3.33 +/- 0.75, zh +3.45 +/- 1.51, ur +12.05 +/- 2.33) is order-robust; no pair of adjacent per-seed ranges overlaps.

## 4. Sign stability scan (3-seed conditions)

Across all 40 (condition, benchmark, instr_lang) aggregates of the four 3-seed conditions, these 13 have |mean| < 2 sigma (sign could plausibly flip within +/- 2 sigma). None of them is a quoted headline gain except X-CSQA rows, which the paper already describes as marginal:

- cond-1-en-5k: Belebele instr=es (+0.08 +/- 0.20), Belebele instr=zh (+0.02 +/- 0.04), X-CSQA instr=en/es/ur/zh (+0.44 +/- 0.23, +0.16 +/- 0.19, +0.41 +/- 0.26, +0.20 +/- 0.27), SIB-200 instr=zh (-0.73 +/- 0.43), XNLI instr=en (-0.06 +/- 0.48), XNLI instr=ur (+0.49 +/- 0.30)
- cond-2-es-5k: X-CSQA instr=en (+0.33 +/- 0.36)
- cond-2-ur-5k: X-CSQA instr=en (+0.18 +/- 0.14), X-CSQA instr=ur (+0.88 +/- 0.75)
- cond-2-zh-5k: X-CSQA instr=en (-0.05 +/- 0.30)

Every non-X-CSQA quoted headline delta has |mean| >= 2.1 sigma and a seed-consistent sign.

## 5. Companion table: cross-seed-variance.tsv

`cross-seed-variance.tsv` (this directory) contains 560 rows covering all 12 fine-tuned conditions:

- **Grain:** one row per (condition, benchmark, data_lang, instr_lang), where `data_lang = ALL` marks the instr-language aggregate (cell-mean over 4 data_langs x 2 templates per seed) and a specific `data_lang` marks the per-language pair (mean over 2 templates per seed). 112 ALL rows + 448 per-data_lang rows.
- **Values:** `mean`, `std`, `min`, `max` are statistics over per-seed aggregates of `delta_rep` (refined delta vs baseline), in pp. `std` is the sample standard deviation (ddof=1) and is `NA` when `n_seeds = 1`. `min`/`max` are per-seed extremes, not per-cell extremes.
- **Population:** fine-tuned conditions only (the baseline has no delta and is seed-invariant). Condition 2 and 5 rows exist only for instr in {en, target}; Conditions 1 and 3 for the instr languages actually evaluated (cond-1: all four; cond-3: en, zh). Every underlying benchmark cell keeps its full item denominator (Belebele 900, X-CSQA 1,000, SIB-200 204, XNLI 5,010 items).

## Verdict summary

**Robust (sign and approximate magnitude stable across 3 seeds):** cond-2-ur-5k instr=ur aggregates for SIB-200 (+12.05 +/- 2.33), Belebele (+5.91 +/- 0.81), XNLI (+3.50 +/- 0.89); cross-lingual transfer cond-2-ur-5k instr=en Belebele (+2.47 +/- 0.31) and XNLI (+2.01 +/- 0.57); the paired Urdu-over-English Belebele advantage (+1.70 +/- 0.08); the SIB-200 resource-gradient ordering es < zh < ur.

**Sign stable, magnitude fragile:** cond-2-ur-5k SIB-200 ur x ur diagonal (-8.99 +/- 4.32; per-seed -13.0 to -4.4; template-1-driven). The paired Urdu-over-English XNLI advantage (+2.06 +/- 1.03) sits exactly at 2 sigma.

**Within noise:** cond-2-ur-5k X-CSQA instr=ur (+0.88 +/- 0.75) and the other X-CSQA aggregates listed in §4; cond-1-en-5k XNLI instr=en (the quoted -0.1 is a true null). The cond-2-vs-cond-5 zh XNLI gap (+0.80) is within cond-2's own 2-sigma seed noise.

**Cannot be assessed (single seed):** everything on cond-2-*-20k, cond-3-zh-5k, and cond-5-*-5k, including the -1.7 pp 20k "sign flip" (drop supported, negative sign not), the -25.24 pp 20k ur x ur diagonal, all cond-5 strict-vs-refined values (including the cond-5-ur sign flip under extraction, -4.96 to +4.78), and the cond-5 side of every "all 12 cells" gap.

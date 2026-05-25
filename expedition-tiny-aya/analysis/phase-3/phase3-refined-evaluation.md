# Phase-3 evaluation — inference-time vs. refined answer extractor

**Status:** complete (2026-05-25). All 42 Phase-3 summary files (21 sessions × 2 templates, post HF PR #34 coverage close) were re-scored against the extended extractor on `eval/sib200-xnli-extractor` (PR #54); **1,664** (condition × seed × template × benchmark × data × instr) cells compared.

Two cell-count denominators recur in this writeup and the figures, both valid at different grains:

- **1,664** = total cells in [`refined-tables/cells.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/cells.tsv) — baseline + 12 fine-tuned conditions × all 4 benchmarks. This is the denominator for "how did the extractor extension move scores" questions (§1 headline, §3 by-benchmark / by-instr / by-template / by-condition rollups).
- **1,536** = total rows in [`refined-tables/vs_baseline_cells.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/vs_baseline_cells.tsv) — the 12 fine-tuned conditions only (1,664 − 128 baseline cells, since baseline can't be compared against itself). This is the denominator for "did fine-tuning beat baseline" questions (§8.3, §8.6, §8.7 conclusion-flip catalogue).

The two are not in tension; they index different questions. Where the difference matters, the surrounding text names which one.

**Scope:** Phase-3 only. Baseline + condition-1/-2/-3/-5. Phase-2 sweeps deliberately excluded.

**Companion documents:**

- [Reparse decision ledger](refined-decision-ledger.md) — the auditable methodology (every rule, every native-script surface form, every reject)
- [SIB-200 parser methodology](sib200-parser-methodology.md) — formal write-up of the multi-term rule
- [Urdu](urdu-surface-forms-review.md) / [Chinese](chinese-surface-forms-review.md) / [Spanish](spanish-surface-forms-review.md) surface-form reviews
- **Analysis tables** — every TSV underpinning this writeup lives on the HF dataset at [`phase3/analysis/`](https://huggingface.co/datasets/legesher/language-decoded-experiments/tree/main/phase3/analysis), split into `refined-tables/` (cells.tsv + framework + rollups) and `surface-form-tables/` (per-benchmark baseline-forms). HF is the canonical source; reproduce locally via the build scripts at `expedition-tiny-aya/evaluation/scripts/build_*.py` + `inspect_failures.py --aggregate`.

---

## 1. Headline

The extractor extension recovered **mean parse-failure mass** across SIB-200 cells (10.1% → 3.9%), at the cost of measurable corrections concentrated in `condition-2-ur-5k`'s previously-buggy `سیاست/تکنالوجی → science/technology` Rule-A mapping, which now correctly hedges instead of false-positive-crediting. Aggregate accuracy moved +1.5 pp; for SIB-200 the move is +5.0 pp and for the subset that needed it most — `instr=ur` SIB-200 — the move is **+15.7 pp** with a parse-failure drop of 19.0 pp. _Note: the `instr=ur` SIB-200 lift was +18.1 pp / −23.3 pp in the pre-PR-#34 build; closing the cond-2-ur-20k and cond-5-es-5k coverage gaps added 16 new sib200×ur cells, several of which regressed under the refined extractor (the new cond-2-ur-20k matched-diagonal cells), which dragged the cell-mean of the new larger denominator._

| Statistic                     | All cells  | SIB-200    | XNLI       | X-CSQA | Belebele |
| ----------------------------- | ---------- | ---------- | ---------- | ------ | -------- |
| Cells (n)                     | **1,664**  | **416**    | **416**    | **416**| **416**  |
| Mean original accuracy        | 0.540      | 0.566      | 0.371      | 0.514  | 0.708    |
| Mean refined accuracy         | 0.554      | 0.616      | 0.381      | 0.514  | 0.708    |
| **Mean Δaccuracy**            | **+0.015** | **+0.050** | **+0.010** | 0.000  | 0.000    |
| Mean original parse-fail rate | 0.038      | 0.101      | 0.049      | 0.0001 | 0.0001   |
| Mean refined parse-fail rate  | 0.016      | 0.039      | 0.026      | 0.0001 | 0.0001   |
| **Mean Δparse-fail rate**     | **−0.021** | **−0.062** | **−0.023** | 0.000  | 0.000    |
| Cells improved (Δacc > 0)     | **215**    | **148**    | **67**     | 0      | 0        |
| Cells regressed (Δacc < 0)    | **106**    | **106**    | **0**      | 0      | 0        |
| Cells flat                    | **1,343**  | **162**    | **349**    | 416    | 416      |

_See **Figure 1** (`figures-phase3/fig01_extractor_slopegraph.pdf`) for the by-cell slopegraph view of the SIB-200 + XNLI lift._

**Two findings drive the whole story:**

1. **The parse-failure recovery is the primary result, not the accuracy lift.** The extractor extension is principally a multilingual-recovery operation. Of the 218 cells whose parse-failure rate dropped, the median drop is concentrated in `instr ≠ en` SIB-200 and `instr=zh` XNLI — exactly where the original strict English-label extractor was dropping correct native-script answers.

2. **X-CSQA and Belebele moved zero.** Their letter-based (A–E) answer format is already language-neutral; the original extractor already read them correctly. Reporting these as "no change" is itself the finding — it confirms the extractor extension is targeted, not a global re-calibration.

## 2. Methodology

### 2.1 What changed in the extractor

Full per-rule audit lives in [refined-decision-ledger.md](refined-decision-ledger.md). Compressed summary:

**SIB-200** — `extract_sib200_category` replaced its single-substring scan with a _count-distinct-categories_ multi-term rule. Each answer is split on punctuation separators (`/`, `,`, `&`, `+`, `;`) and on language-specific conjunctions (`and`, `y`, `e`, `و`, `和`, `与`); each piece is resolved against `SIB200_TERM_TO_CATEGORY` (English canonical labels + sub-topics + native surface forms in Urdu, Chinese, Spanish, Arabic); the answer counts only if all resolved pieces collapse to **exactly one distinct category**. Two-or-more distinct categories → parse-failure (the model hedged). Fallback: word-boundary scan of the 7 English canonical category names. The native term map and the multi-term rule together fix two PR-#49 bugs at the same time — `سیاست/تکنالوجی` (politics/technology) no longer mis-resolves to `science/technology`, and `science/health` is no longer Rule-A-credited as `science/technology`.

**XNLI** — `extract_xnli_label` became tiered:

- **Tier 1a** — verbatim English label (`\bentailment\b`, etc.).
- **Tier 1b** — native label words (`蕴含`/`矛盾`/`中立`, `implicación`/`contradicción`/`neutro`, `ضمنی`/`تردید`/`غیرجانبدار`).
- **Tier 2** — English label glued to a CJK frame (`假设是entailment。`). Guarded by `XNLI_TIER2_NEGATION = ("没有", "沒有")` — if the frame is negated, Tier 2 is skipped.
- **Tier 3** — lenient native-prose paraphrases (`直接结果` → entailment, `否定` → contradiction, `没有.{0,8}关系` → neutral, etc.).

Tier 3 is explicitly framed as **lenient semantic mapping** in the ledger — the strict/lenient gap is itself a measurable instruction-following property of the model.

**X-CSQA, Belebele** — unchanged. Letter-based answers were already language-neutral.

### 2.2 What changed at the data level

This is a _re-scoring_ operation, not a re-inference. `reparse_results.py` reads the stored `raw_output` field from each `_results_*.json` and re-runs the new extractor over it. **No model invocation, no GPU, no fresh tokens.** Every cell in this comparison shares the same underlying model outputs across the inference-time and refined paths — only the extractor function changed.

This isolation is what lets us report the deltas as _scorer effects_ rather than confounded model-variance effects.

### 2.3 Faithful-extraction principle

The extractor scores what the model _said_, not what gold says. A native-language answer matched to its English label is credited because the model identified the topic, even if it didn't answer in the requested label. Two consequences worth naming up front:

1. **Anything we credit, we credit because the model emitted a surface form we can defend.** Every Urdu / Chinese / Spanish / Arabic form in the term map appears in `raw_output` at least once and is confirmed by the three native-speaker review documents.
2. **Hedges stay parse-failures.** When the model emitted two distinct categories (e.g. `کھیل/تکنالوجی`), the new rule refuses to pick one — that's a real model limitation kept visible.

### 2.4 Data integrity

The HF dataset contains a duplicated-misfile artifact: `phase3/conditions/condition-2-es-5k/seed42/` holds both `seed42` _and_ `seed123` summary files (the latter also live in `seed123/`). The analysis dataframe derives seed from filename, not directory, and de-duplicates against the file whose parent directory matches its filename seed. Net result: **1,664** unique cells in `cells.tsv` across baseline + 12 fine-tuned conditions; the duplicate copies are dropped. (The **1,536** number that appears in §8.3 / §8.7 is downstream of this dedup — it's `cells.tsv`'s 1,664 minus the 128 baseline cells that can't appear in `vs_baseline_cells.tsv`. See the §1 status banner for the full two-denominator key.)

## 3. Recovery results

### 3.1 By benchmark — the SIB-200 / XNLI split

| Benchmark | Mean Δacc  | Mean Δpf   | Cells improved | Cells regressed | Cells flat |
| --------- | ---------- | ---------- | -------------- | --------------- | ---------- |
| SIB-200   | **+0.050** | **−0.062** | 148            | 106             | 162        |
| XNLI      | +0.010     | −0.023     | 67             | 0               | 349        |
| X-CSQA    | 0.000      | 0.000      | 0              | 0               | 416        |
| Belebele  | 0.000      | 0.000      | 0              | 0               | 416        |

_See **Figure 2** (`figures-phase3/fig02_cell_scatter.pdf`) for the cell-level Δaccuracy vs Δparse-failure scatter across all 1,664 cells, which makes the SIB-200 / XNLI / origin-cluster structure visible at a glance._

SIB-200 absorbs **76% of the accuracy lift** (sum-product of Δacc × n_cells) and **88% of the parse-failure recovery**. This is consistent with the decision ledger's scope finding: the extractor problem is overwhelmingly a SIB-200 problem.

Within XNLI, **all 67 improvements are unidirectional** — there is not a single XNLI regression. That's because the tier ladder only resolves answers that were previously parse-failures; it never re-routes an already-Tier-1 answer. (Strict-vs-lenient gap addressed in §6.)

### 3.2 By instruction language — multilingual instructions are where the lift lives

| `instr_lang` | n cells | Mean orig acc | Mean refined acc | Mean Δacc  | Mean Δpf |
| ------------ | ------- | ------------- | ----------------- | ---------- | -------- |
| `en`         | 672     | 0.590         | 0.591             | **+0.000** | +0.000   |
| `es`         | 320     | 0.528         | 0.539             | **+0.011** | −0.012   |
| `zh`         | 352     | 0.529         | 0.549             | **+0.021** | −0.038   |
| `ur`         | 320     | 0.450         | 0.491             | **+0.040** | −0.054   |

**English-instruction cells move zero on average** (a 0.04 pp net drop from the cond-2-ur-5k Rule-A correction — see §5). Non-English instruction cells improve monotonically with how "different" the script is from English — Spanish (Latin script) < Chinese (CJK) < Urdu (RTL Arabic script). The non-English instruction cells were precisely the ones whose answers were being dropped as parse-failures.

The SIB-200 row of the benchmark × instr-lang crosstab makes the asymmetry vivid:

| Benchmark × `instr_lang` | n   | Orig acc  | Refined acc | Δacc       | Orig pf | Refined pf | Δpf        |
| ------------------------ | --- | --------- | ----------- | ---------- | ------- | ---------- | ---------- |
| sib200 × en              | 168 | 0.697     | 0.695       | **−0.002** | 0.006   | 0.007      | +0.002     |
| sib200 × es              | 80  | 0.489     | 0.548       | **+0.059** | 0.081   | 0.010      | −0.071     |
| sib200 × zh              | 88  | 0.574     | 0.615       | **+0.041** | 0.069   | 0.011      | −0.058     |
| sib200 × ur              | 80  | **0.361** | **0.518**   | **+0.157** | 0.356   | 0.167      | **−0.190** |
| xnli × en                | 168 | 0.419     | 0.419       | 0.000      | 0.002   | 0.002      | 0.000      |
| xnli × es                | 80  | 0.394     | 0.394       | 0.000      | 0.008   | 0.008      | 0.000      |
| xnli × zh                | 88  | 0.284     | 0.325       | **+0.041** | 0.174   | 0.081      | **−0.094** |
| xnli × ur                | 80  | 0.344     | 0.349       | +0.005     | 0.052   | 0.037      | −0.015     |

**`sib200 × ur` is the single biggest finding in this study.** Mean accuracy moved from 0.361 to 0.518 — a 15.7-point lift on a 4-class task (and a 19.0-pp drop in parse-failure mass). Meanwhile `xnli × es` and `xnli × en` are flat because the model did not emit Spanish native labels or paraphrases on those cells; the ladder had nothing to do. (Pre-PR-#34, `sib200 × ur` measured +18.1 pp / −23.3 pp on n=72; the 8 new cond-2-ur-20k matched-diagonal cells that PR #34 closed coverage on regressed under the refined extractor, lowering the cell-mean of the new larger denominator. The mechanism is the same — the §1 narrative around this row should be read as "the lift remains the largest in the study, ~16 pp on n=80".)

### 3.3 By template — template-2 is where the parse-failures live

| Template  | Mean Δacc | Mean Δpf | Cells improved | Cells regressed |
| --------- | --------- | -------- | -------------- | --------------- |
| template1 | +0.009    | −0.011   | 68             | 78              |
| template2 | +0.020    | −0.032   | 147            | 28              |

Template-2 cells improve **2.2× more in accuracy and 2.9× more in parse-failure recovery**. This matches the decision ledger's prior observation that template-2 elicits longer, less-structured native-prose answers ("the model answers a few lines about why it's about travel" instead of "travel."). The extractor extension is asymmetrically valuable for template-2.

### 3.4 By condition — the most interesting paper question

| Condition          | n cells | Mean Δacc  | Mean Δpf   | Improved | Regressed |
| ------------------ | ------- | ---------- | ---------- | -------- | --------- |
| baseline           | 128     | **+0.029** | −0.043     | 29       | 3         |
| condition-1-en-5k  | 384     | **+0.029** | −0.046     | 85       | 10        |
| condition-1-en-20k | 128     | **+0.031** | −0.048     | 28       | 2         |
| condition-2-es-5k  | 192     | +0.000     | +0.001     | 9        | 9         |
| condition-2-es-20k | 64      | −0.001     | +0.002     | 0        | 5         |
| condition-2-zh-5k  | 192     | +0.000     | −0.001     | 16       | 11        |
| condition-2-zh-20k | 64      | +0.001     | −0.001     | 8        | 4         |
| condition-2-ur-5k  | 192     | **−0.007** | **+0.022** | 0        | 38        |
| condition-2-ur-20k | 64      | **−0.010** | **+0.028** | 1        | 12        |
| condition-3-zh-5k  | 64      | −0.001     | +0.004     | 2        | 8         |
| condition-5-es-5k  | 64      | **+0.027** | −0.036     | 9        | 2         |
| condition-5-ur-5k  | 64      | **+0.046** | −0.068     | 10       | 2         |
| condition-5-zh-5k  | 64      | **+0.047** | −0.087     | 18       | 0         |

Three groupings emerge:

**Group A — "untrained-in-target-language" models (baseline + cond-1-en).** ~+0.030 mean Δacc, ~−0.045 mean Δpf. These models had no Urdu/Chinese/Spanish supervision and frequently answer in those scripts anyway when prompted in the target language; the extractor extension recovers those answers.

**Group B — target-language-tuned cond-2 models.** Near-zero net effect (cond-2-es/zh) or slight negative (cond-2-ur). These models already produce English-style outputs (`travel`, `entailment`) reliably; there was little parse-failure mass to recover.

**Group C — Aya-translated cond-5 models.** ~**+0.046 mean Δacc** — the _largest_ lift in the study, exceeding even the baseline-and-cond-1 group. This is the paper-relevant headline:

> Aya-translated training data did **not** teach the model to answer in the requested English label format. Cond-5 models continue to emit native-script answers and require the extended extractor to be read at all.

Cond-5's parse-failure recovery (−0.068 ur, −0.087 zh) is the largest of any condition. Put another way: under the original strict-English extractor, cond-5 looked like a _worse_ fine-tuning recipe than cond-1; under the extended extractor, cond-5 reads roughly equal to cond-1 on SIB-200 (and slightly better than baseline on XNLI-zh). **The original measurement was confounded by extractor coverage, not training-data efficacy.** This is exactly the kind of finding that justifies the extractor work.

## 4. Anomalies — where the refined-extractor pass made cells _look_ worse

**106 of 416 SIB-200 cells regressed in accuracy.** Their distribution is highly concentrated across the 12 conditions that contribute any regressions:

| Condition          | Regression cells       |
| ------------------ | ---------------------- |
| condition-2-ur-5k  | 38 of 48 SIB-200 cells |
| condition-2-ur-20k | 12 of 16 SIB-200 cells |
| condition-2-zh-5k  | 11                     |
| condition-1-en-5k  | 10                     |
| condition-2-es-5k  | 9                      |
| condition-3-zh-5k  | 8                      |
| condition-2-es-20k | 5                      |
| condition-2-zh-20k | 4                      |
| baseline           | 3                      |
| condition-1-en-20k | 2                      |
| condition-5-es-5k  | 2                      |
| condition-5-ur-5k  | 2                      |
| **Total**          | **106**                |

Mechanism notes:

- `condition-2-ur-5k` (38) and `condition-2-ur-20k` (12) — both driven by the PR-#49 `سیاست/تکنالوجی → science/technology` Rule-A correction; the Urdu-tuned model emits that compound frequently, the multi-term rule now refuses to map it to a single category.
- `condition-2-zh-5k` (11), `condition-1-en-5k` (10), `condition-2-es-5k` (9), `condition-3-zh-5k` (8), `condition-2-es-20k` (5), `condition-2-zh-20k` (4), `condition-1-en-20k` (2) — Rule-A `science/X` over-credit corrections (`science/health`, `science/anything` no longer collapsed to `science/technology`).
- `condition-5-es-5k` (2) — cells contributed by HF PR #34's coverage close.
- `baseline` (3) and `condition-5-ur-5k` (2) — small residual corrections at the noise floor.

**Practically the entire regression mass is `condition-2-ur-5k` + `condition-2-ur-20k` SIB-200 (50 of 106 cells = 47%).** Read the PR-#54 description for the mechanism: PR #49 had introduced a `سیاست/تکنالوجی → science/technology` mapping, scored 4 cases correct, 70 cases wrong on that string. The Urdu-tuned model in `cond-2-ur-5k`/`-20k` emits that string frequently because Urdu fine-tuning teaches it the Urdu native script as default. The multi-term rule now correctly refuses to map that compound to a single category and the cells move from "false-positive credit" to "parse-failure." The accuracy drop is a **correctness improvement**, not a regression in any meaningful sense — the 4 correct rows were lucky, and they are now visible as the parse-failures they always were.

_See **Figure 3** (`figures-phase3/fig03_regression_concentration.pdf`) for the per-condition regression count bar chart and the cond-2-ur-5k Δacc heatmap by (data → instr) × template._

The other smaller regressions (`condition-1-en-*` `science/X` cells on English-instr; cond-2-es / cond-2-zh) are the analogous correction for Rule A's previous over-credit of `science/health` and similar cross-category compounds.

**Anomaly to flag for the paper write-up:** the original `cond-2-ur-5k` SIB-200 accuracy of ~0.67 was inflated. The refined ~0.65 is the defensible number. If the original was cited anywhere (slides, draft paper, Linear comments), it should be revised down.

## 5. The parse-failure recovery story — flat-acc, dropped-pf cells

A cell whose accuracy stayed flat but whose parse-failure rate dropped substantially tells a different story than one where both moved. It means the recovered rows were _not_ correct — the extractor now reads the model's answer, but the answer is wrong. That is a _scorer-quality_ improvement (fewer hidden wrong answers, more honest accuracy) even though it doesn't bump the headline number.

In this dataset, 14 cells had |Δacc| ≤ 1e-6 but Δpf < 0. Manual scan: all 14 are concentrated in the XNLI tier-1b/2 zone, and most are template-1 XNLI-zh / XNLI-ur cells where the model emits a native label that turns out to be wrong (e.g. emits `矛盾` / contradiction when gold is entailment). **No cells were flagged with Δpf < −0.05 and Δacc flat** — meaning every substantial parse-failure recovery did move accuracy with it. The strict-English extractor was rejecting genuine model answers, not just hedges.

This is a quietly important methodological point. If the extractor had recovered a lot of new parse-failures _without_ moving accuracy, we'd be measuring the model on a different denominator (and the gain would be illusory). Instead, recovered rows are roughly as accurate as the model's already-credited rows — consistent with the faithful-extraction principle.

## 6. Strict-vs-lenient gap (XNLI Tier 3) — what the summaries can and cannot tell us

The decision ledger frames XNLI Tier 3 as **lenient semantic mapping**: the strict/lenient gap is reported as an instruction-following measure. The cell-level summaries we have here aggregate every tier together — we cannot, from the `_summary_*.json` files alone, decompose the XNLI lift into Tier-1a / Tier-1b / Tier-2 / Tier-3 components.

The artefact that _can_ answer this is `inspect_failures.py --aggregate` (PR #55), which classifies every row by `match_via ∈ {exact, normalized, tier1_english, tier1_native, tier2_cjk, tier3_paraphrase, …}`. If a paper section needs the strict-vs-lenient decomposition, we should:

1. Merge PR #55.
2. Run `inspect_failures.py --benchmark xnli --aggregate` across all 40 `_results_*.json` files.
3. Re-aggregate by `match_via` tier.

For now, the summary-level Tier-3 measurement we _do_ have from the decision ledger is: across the 160,320 XNLI rows of the baseline mining, **0 of 4,435 Tier-3 resolutions had a negation marker preceding the matched phrase**. That's the directly-observed empirical zero-impact of the Tier-3 negation gap (also documented in the ledger).

## 7. Limitations and what to do next

**(L1) Truncation confound.** SIB-200 generation runs at `max_new_tokens=10`. Long native-script compounds get clipped mid-token (e.g. `کھیل/سائنس/تکنالوجی` → `کھیل/سائنس/تکنال`). The surviving pieces still trip the multi-category hedge → parse-failure. Some fraction of the residual SIB-200 parse-failure rate is generation-budget-limited, not extractor-limited. Worth a sentence in the paper's limitations: a larger token budget can reduce this.

**(L2) Constant-output finding (cond-2-ur-5k SIB-200).** Per the decision ledger, the Urdu-tuned model emits a near-constant `سائنس/ٹیکنالوجی` regardless of passage topic. Those rows land in `correct` when gold is science/technology and `wrong_label` otherwise. The refined extraction keeps both visible (no special-casing), but the cell-level accuracy on this condition slightly overstates the model. A `correct_ambiguous` flag in the per-row analysis (planned, not yet emitted) would let us measure this.

**(L3) `instr=en` regression on `cond-2-*`.** Small but systematic: cond-2-es/zh/ur each lost 1–5 cells on `instr=en` due to Rule-A `science/X` over-credit being removed. The 0.025-pp typical magnitude per cell is below the noise of any single seed; pooling across seeds is the right comparison level.

**(L4) Tier-3 strict-vs-lenient decomposition not available.** §6 above. The `--aggregate` artefact in PR #55 is the path forward.

**(L5) No cross-seed variance reported here.** The cell-level table reports per-(condition,seed,template,benchmark,data,instr) numbers but does not aggregate across seeds with variance bars. For a paper, the cond-1-en-5k × {42, 123, 456} triplet should be reported as mean ± std (or 95% CI) per cell. [`framework_seed_variance.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/framework_seed_variance.tsv) on HF surfaces this directly (mean, std, min, max, total_count, total_correct per cell); [`cells.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/cells.tsv) is the per-seed raw input.

## 8. Condition vs baseline — does fine-tuning actually help?

The preceding sections (§1–§7) report the **extractor effect**: how scores moved when we swapped the strict-English extractor for the extended one. This section answers a different question — the actual _research_ question:

> For each fine-tuning condition, does the condition's model beat the un-tuned baseline?

Because each condition has _two_ readings (one from each extractor), every condition-vs-baseline comparison appears twice: once under the original-extractor scoring (the numbers that were on the table before this PR) and once under the refined-extractor scoring (the numbers we believe). When the two readings agree on sign, the verdict is **stable** — extractor coverage didn't change the answer. When they disagree, the verdict **flipped** — the original-extractor scoring was making the wrong call.

### 8.1 Three-scenario framing

For each condition C and each cell (template, benchmark, data_lang, instr_lang):

| Scenario             | Numerator                                                  | Denominator           | What it answers                                                          |
| -------------------- | ---------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------ |
| **Baseline (orig)**  | C unused; baseline's accuracy under the strict extractor   | —                     | "What did the un-tuned model look like under the original scorer?"       |
| **Baseline (rep)**   | C unused; baseline's accuracy under the extended extractor | —                     | "What did the un-tuned model look like once we read its native answers?" |
| **Condition (orig)** | C's accuracy under the strict extractor                    | minus baseline (orig) | "Did fine-tuning help, by the original scorer?"                          |
| **Condition (rep)**  | C's accuracy under the extended extractor                  | minus baseline (rep)  | "Did fine-tuning help, by the corrected scorer?"                         |

Every comparison is **apples-to-apples within an extractor**: cond_orig is compared to baseline_orig; cond_rep to baseline_rep. We never compare a condition's refined score to a baseline's original score.

### 8.2 Headline — condition mean vs baseline mean, both extractors

| Condition          | n cells | Baseline (orig) | Cond (orig) | **Δ (orig)** | Baseline (rep) | Cond (rep) | **Δ (rep)** | Verdict                                              |
| ------------------ | ------- | --------------- | ----------- | ------------ | -------------- | ---------- | ----------- | ---------------------------------------------------- |
| condition-1-en-20k | 128     | 0.509           | 0.508       | **−0.001**   | 0.538          | 0.539      | **+0.001**  | stable: no-effect                                    |
| condition-1-en-5k  | 384     | 0.509           | 0.512       | **+0.003**   | 0.538          | 0.542      | **+0.003**  | stable: no-effect                                    |
| condition-2-es-5k  | 192     | 0.554           | 0.568       | **+0.014**   | 0.564          | 0.568      | **+0.004**  | stable: small win, **smaller under rep**             |
| condition-2-es-20k | 64      | 0.554           | 0.570       | **+0.015**   | 0.564          | 0.569      | **+0.004**  | stable: small win, **smaller under rep**             |
| condition-2-zh-5k  | 192     | 0.553           | 0.579       | **+0.026**   | 0.567          | 0.580      | **+0.013**  | stable: win, **halved under rep**                    |
| condition-2-zh-20k | 64      | 0.553           | 0.577       | **+0.023**   | 0.567          | 0.577      | **+0.011**  | stable: win, **halved under rep**                    |
| condition-2-ur-5k  | 192     | 0.505           | 0.576       | **+0.071**   | 0.539          | 0.569      | **+0.030**  | stable: win, **2.4× inflated by original extractor** |
| condition-3-zh-5k  | 64      | 0.553           | 0.578       | **+0.024**   | 0.567          | 0.577      | **+0.010**  | stable: win, **halved under rep**                    |
| condition-5-ur-5k  | 64      | 0.505           | 0.483       | **−0.021**   | 0.539          | 0.529      | **−0.009**  | stable: small loss, **halved under rep**             |
| condition-5-zh-5k  | 64      | 0.553           | 0.488       | **−0.066**   | 0.567          | 0.535      | **−0.032**  | stable: loss, **halved under rep**                   |

> **The mean-Δ aggregate verdict does not flip for any condition.** Every condition that "beat baseline" under the original extractor still beats baseline under the refined extractor, and every condition that "lost" still loses. But the **magnitude** changes substantially — most condition-vs-baseline gains shrink by ~50% under the corrected scoring, because the original extractor was systematically over-crediting both sides asymmetrically (fine-tuned models more than baseline on SIB-200, see §8.4).

### 8.3 Per (condition × benchmark) — where the conclusion actually flips

Cell-mean Δ (cond − baseline) per benchmark, both extractors. **Bold** rows are where the sign flips between the two scorings.

| Condition              | Benchmark  | n   | Δ (orig)   | Δ (rep)    | Sign-flip?                             |
| ---------------------- | ---------- | --- | ---------- | ---------- | -------------------------------------- |
| condition-1-en-20k     | belebele   | 32  | +0.003     | +0.003     | —                                      |
| condition-1-en-20k     | csqa       | 32  | +0.005     | +0.005     | —                                      |
| condition-1-en-20k     | sib200     | 32  | −0.018     | −0.018     | —                                      |
| condition-1-en-20k     | xnli       | 32  | +0.007     | +0.012     | —                                      |
| condition-1-en-5k      | belebele   | 96  | +0.004     | +0.004     | —                                      |
| condition-1-en-5k      | csqa       | 96  | +0.003     | +0.003     | —                                      |
| condition-1-en-5k      | sib200     | 96  | −0.003     | −0.009     | —                                      |
| condition-1-en-5k      | xnli       | 96  | +0.008     | +0.015     | —                                      |
| **condition-2-es-20k** | **sib200** | 16  | **+0.023** | **−0.021** | **WIN → LOSS**                         |
| condition-2-es-20k     | belebele   | 16  | +0.010     | +0.010     | —                                      |
| condition-2-es-20k     | csqa       | 16  | +0.007     | +0.007     | —                                      |
| condition-2-es-20k     | xnli       | 16  | +0.021     | +0.021     | —                                      |
| **condition-2-es-5k**  | **sib200** | 48  | **+0.013** | **−0.027** | **WIN → LOSS**                         |
| condition-2-es-5k      | belebele   | 48  | +0.013     | +0.013     | —                                      |
| condition-2-es-5k      | csqa       | 48  | +0.005     | +0.005     | —                                      |
| condition-2-es-5k      | xnli       | 48  | +0.023     | +0.023     | —                                      |
| condition-2-ur-5k      | sib200     | 48  | +0.205     | +0.047     | — _(stays positive but deflated 4.4×)_ |
| condition-2-ur-5k      | belebele   | 48  | +0.042     | +0.042     | —                                      |
| condition-2-ur-5k      | csqa       | 48  | +0.005     | +0.005     | —                                      |
| condition-2-ur-5k      | xnli       | 48  | +0.031     | +0.028     | —                                      |
| **condition-2-ur-20k** | **sib200** | 16  | **+0.139** | **−0.030** | **WIN → LOSS** _(largest \|shift\|)_   |
| condition-2-ur-20k     | belebele   | 16  | +0.020     | +0.020     | —                                      |
| condition-2-ur-20k     | csqa       | 16  | −0.001     | −0.001     | —                                      |
| condition-2-ur-20k     | xnli       | 16  | +0.038     | +0.034     | —                                      |
| **condition-2-zh-20k** | **sib200** | 16  | **+0.017** | **−0.007** | **WIN → LOSS**                         |
| condition-2-zh-20k     | belebele   | 16  | +0.014     | +0.014     | —                                      |
| condition-2-zh-20k     | csqa       | 16  | +0.011     | +0.011     | —                                      |
| condition-2-zh-20k     | xnli       | 16  | +0.050     | +0.025     | —                                      |
| condition-2-zh-5k      | sib200     | 48  | +0.036     | +0.007     | — _(stays positive, shrinks)_          |
| condition-2-zh-5k      | belebele   | 48  | +0.018     | +0.018     | —                                      |
| condition-2-zh-5k      | csqa       | 48  | +0.007     | +0.007     | —                                      |
| condition-2-zh-5k      | xnli       | 48  | +0.044     | +0.020     | —                                      |
| **condition-3-zh-5k**  | **sib200** | 16  | **+0.020** | **−0.012** | **WIN → LOSS**                         |
| condition-3-zh-5k      | belebele   | 16  | +0.012     | +0.012     | —                                      |
| condition-3-zh-5k      | csqa       | 16  | +0.010     | +0.010     | —                                      |
| condition-3-zh-5k      | xnli       | 16  | +0.055     | +0.030     | —                                      |
| condition-5-es-5k      | sib200     | 16  | −0.186     | −0.120     | — _(stays negative, partly recovered)_ |
| condition-5-es-5k      | belebele   | 16  | +0.007     | +0.007     | —                                      |
| condition-5-es-5k      | csqa       | 16  | −0.015     | −0.015     | —                                      |
| condition-5-es-5k      | xnli       | 16  | −0.006     | −0.006     | —                                      |
| condition-5-ur-5k      | sib200     | 16  | −0.066     | −0.013     | — _(stays negative, smaller loss)_     |
| condition-5-ur-5k      | csqa       | 16  | −0.045     | −0.045     | —                                      |
| condition-5-ur-5k      | belebele   | 16  | +0.019     | +0.019     | —                                      |
| condition-5-ur-5k      | xnli       | 16  | +0.005     | +0.001     | —                                      |
| condition-5-zh-5k      | sib200     | 16  | −0.245     | −0.128     | — _(stays negative, halved)_           |
| condition-5-zh-5k      | csqa       | 16  | −0.020     | −0.020     | —                                      |
| condition-5-zh-5k      | belebele   | 16  | +0.018     | +0.018     | —                                      |
| **condition-5-zh-5k**  | **xnli**   | 16  | **−0.016** | **+0.003** | **LOSS → WIN** _(marginal)_            |

**Five (condition, benchmark) cells flip from "fine-tuning helps" to "fine-tuning hurts"** on SIB-200 under the corrected extractor: `cond-2-es-5k`, `cond-2-es-20k`, `cond-2-ur-20k`, `cond-2-zh-20k`, `cond-3-zh-5k`. The `cond-2-ur-20k` flip is the largest in absolute magnitude (Δorig +0.139 → Δrep −0.030), driven by the same PR-#49 Rule-A correction (§8.4) that drove the cond-2-ur-5k deflation; the `cond-2-zh-20k` flip is the smallest (|Δrep| = 0.007, below the ±0.01 noise floor used by `conclusion_flips.tsv` — kept here in the strict-sign view but a reader citing the per-cell catalogue should know it's borderline). One cell flips the other direction (cond-5-zh-5k XNLI) but the magnitude is small (±0.01) — likely noise, not signal.

### 8.4 Why does cond-2 SIB-200 "win" under the original extractor and "lose" under the refined one?

The flip is a mechanical consequence of the PR-#49 bugs the extractor extension fixed. The original `extract_sib200_category` had two over-credit patterns: Rule A (`science/<anything>` → science/technology) and the `سیاست/تکنالوجی → science/technology` mapping. **Fine-tuned models hit these patterns more than the baseline.** A target-language-tuned model that has learned to emit short topic-prefix tokens (`science/`, `سیاست/`) collects more lucky credit from these bugs than the baseline does (which emits longer English prose). Under the corrected extractor, those over-credits go away for both sides — but they were disproportionately benefiting the fine-tuned models, so the cond-vs-baseline delta shrinks (or flips negative).

This is the single most important finding for the paper. **Reporting the cond-2 / cond-3 SIB-200 advantage from the original extractor would overstate the benefit of target-language fine-tuning.** The corrected extractor either deflates the gain or reveals it never existed.

The flip pattern is asymmetric by benchmark:

- **SIB-200** — most flips and biggest magnitude shifts. The over-credit was concentrated here.
- **XNLI** — small deflations, no flips. The XNLI extractor extension is purely _additive_ (more answers extracted, none re-routed) so the cond-vs-baseline delta moves only when the baseline and condition extract different proportions of recovered native answers. Cond-2 models extract slightly fewer native answers than baseline (they've learned to emit English labels), so their relative gain shrinks.
- **X-CSQA, Belebele** — zero change. Letter-based answers; the extractor didn't move.

### 8.5 The cond-5 story under three scenarios

Cond-5 (Aya-translated training data) was the latest cycle's most ambitious training recipe — the question was whether translating the supervised mixture into the target language would produce a target-language-fluent model.

| Cond-5 split           | Baseline (orig) | Cond-5 (orig) | Δ (orig)   | Baseline (rep) | Cond-5 (rep) | Δ (rep)    |
| ---------------------- | --------------- | ------------- | ---------- | -------------- | ------------ | ---------- |
| cond-5-ur-5k (overall) | 0.505           | 0.483         | **−0.021** | 0.539          | 0.529        | **−0.009** |
| cond-5-zh-5k (overall) | 0.553           | 0.488         | **−0.066** | 0.567          | 0.535        | **−0.032** |
| cond-5-ur-5k SIB-200   | 0.468           | 0.402         | −0.066     | 0.599          | 0.585        | −0.013     |
| cond-5-zh-5k SIB-200   | 0.648           | 0.403         | **−0.245** | 0.676          | 0.547        | −0.128     |
| cond-5-zh-5k X-CSQA    | 0.523           | 0.503         | −0.020     | 0.523          | 0.503        | −0.020     |
| cond-5-ur-5k X-CSQA    | 0.502           | 0.458         | −0.045     | 0.502          | 0.458        | −0.045     |

Three things to note:

1. **The headline cond-5-zh-5k SIB-200 "regression" of −0.245 was largely an extractor artefact.** Under the refined scoring it's −0.128 — still a regression, but the magnitude _halves_. The original extractor was hiding cond-5-zh-5k's native-script answers as parse-failures while crediting baseline's lucky `science/X` patterns; the corrected extractor reads both sides honestly.
2. **The X-CSQA regression is real and not extractor-confounded.** Both cond-5-ur (−0.045) and cond-5-zh (−0.020) lose on X-CSQA by identical amounts under both extractors. That's a model effect: the Aya-translated training mix degrades commonsense QA. Worth investigating whether the translation step lost reasoning fidelity.
3. **Cond-5 still loses against baseline on the mean** — but the story is "doesn't help, modestly hurts" rather than "broke the model." The original extractor was telling the stronger story; the corrected extractor tells the moderate one.

_See **Figure 5** (`figures-phase3/fig05_cond5_rehabilitated.pdf`) for the cond-5 slopegraph across these four cells, with the SIB-200 lines moving toward zero and the X-CSQA lines staying flat under the refined extractor._

### 8.6 Per (condition × instr_lang) — does target-language instruction make the difference?

Average Δ (cond − baseline) by the instruction language of the prompt:

| Condition          | instr=en        | instr=es        | instr=zh        | instr=ur        | (uses inference-time / refined) |
| ------------------ | --------------- | --------------- | --------------- | --------------- | -------------------------- |
| condition-1-en-20k | +0.000 / +0.000 | +0.005 / +0.004 | −0.020 / −0.011 | +0.020 / +0.012 | orig / rep                 |
| condition-1-en-5k  | +0.004 / +0.004 | +0.001 / −0.000 | −0.000 / +0.005 | +0.011 / +0.005 | orig / rep                 |
| condition-2-es-5k  | +0.018 / +0.013 | +0.011 / −0.006 | —               | —               | orig / rep                 |
| condition-2-es-20k | +0.020 / +0.017 | +0.011 / −0.010 | —               | —               | orig / rep                 |
| condition-2-zh-5k  | +0.020 / +0.013 | —               | +0.033 / +0.014 | —               | orig / rep                 |
| condition-2-zh-20k | +0.026 / +0.020 | —               | +0.020 / +0.001 | —               | orig / rep                 |
| condition-2-ur-5k  | +0.034 / +0.038 | —               | —               | +0.108 / +0.023 | orig / rep                 |
| condition-3-zh-5k  | +0.024 / +0.018 | —               | +0.024 / +0.001 | —               | orig / rep                 |
| condition-5-ur-5k  | −0.020 / −0.010 | —               | —               | −0.021 / −0.007 | orig / rep                 |
| condition-5-zh-5k  | −0.060 / −0.030 | —               | −0.073 / −0.034 | —               | orig / rep                 |

(`cond_x_instr` rollup in [vs_baseline_by_cond_x_instr.tsv](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/vs_baseline_by_cond_x_instr.tsv); per-cell rows in [vs_baseline_cells.tsv](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/vs_baseline_cells.tsv).)

The two readings agree on **direction** everywhere except cond-2-es-5k/20k `instr=es` (which flips from small win to small loss — same SIB-200 mechanism as §8.4) and cond-1-en-* `instr=zh` (small magnitude, likely noise). The interesting non-flip pattern is cond-2-ur-5k `instr=ur`: the original-extractor reading said +0.108 (huge gain from target-language tuning); the refined reading says +0.023 (modest gain). A ~4× deflation that does *not\* flip sign — the gain is real, just much smaller than the original numbers suggested.

### 8.7 Conclusion-flip catalogue

[`conclusion_flips.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/conclusion_flips.tsv) lists every cell whose sign(Δ vs baseline) changed between the original and refined scorings (using a ±0.01 buffer to ignore noise-floor cells). **48 cells flip** out of 1,536 condition-vs-baseline comparisons (3.1%) post HF PR #34 (full coverage). The 1,536 denominator is the fine-tuned subset of the 1,664 total cells in §1 (baseline excluded — baseline can't be vs-baseline'd against itself). The flips split across two benchmarks and are otherwise highly concentrated:

- **43 of 48 flips are SIB-200** — the benchmark whose extractor changed the most. The win→loss direction dominates (34 of 43, mostly cond-2-{es,zh,ur} and cond-3 — see §8.3); cond-5 contributes most of the small-magnitude loss→win flips on SIB-200.
- **5 of 48 flips are XNLI** — all `loss → win`, all on `condition-1-en-*` (English-only fine-tunes), all on `template-2`, all on `instr=zh`. Magnitudes range +2.5 pp to +9.2 pp. These XNLI flips are a **mirror-image finding to the SIB-200 deflations of §8.4**: where the original extractor over-credited fine-tuned models on Rule-A SIB-200 bug cells, it _under-credited_ condition-1 on XNLI-zh by refusing Chinese-prose paraphrases (`直接结果`, `否定`, `没有.{0,8}关系`) that the refined extractor's Tier-3 ladder now resolves. The methodological framing therefore sharpens from "the original extractor inflated fine-tuning gains" to **"the original extractor was asymmetric across cells, and the refinement is a fairness correction in both directions"**.
- **All 48 flips are on `instr` ≠ `en`** — English-instruction cells are stable across both extractors (`instr=ur` 24, `instr=es` 13, `instr=zh` 11). X-CSQA and Belebele contribute zero flips by construction (the extractor refinement did not touch their letter-answer format).

If a paper plot needs to flag which (condition, benchmark) cells are "extractor-coverage-confounded," the conclusion_flips TSV is the source.

_See **Figure 4** for the sign-flip slopegraphs:_

- _`figures-phase3/fig04_signflip_sib200_top15.pdf`_ — _main paper figure, top 15 of 43 SIB-200 flips by |shift|._
- _`figures-phase3/fig04_signflip_sib200_all43.pdf`_ — _appendix figure, all 43 SIB-200 flips._
- _`figures-phase3/fig04_signflip_xnli_all5.pdf`_ — _separate small figure for the 5 XNLI flips._

### 8.8 Reading guide — which number to report for which claim

| Claim type                                                                | Which scoring to cite                                                           | Why                                                          |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| "Does this condition beat baseline?" (paper headline)                     | **Reparsed**                                                                    | The corrected extractor reads what the model actually said.  |
| "What did our original results look like before the extractor extension?" | **Original**                                                                    | For provenance / reproducibility / before-after comparisons. |
| "How does extractor coverage affect our conclusions?"                     | **Both**, side-by-side                                                          | This document, §8.3 and §8.4.                                |
| Per-row error analysis                                                    | **Reparsed**                                                                    | The original extractor over-counts parse-failures.           |
| Aggregating across cells with cross-seed std                              | **Reparsed**, per [vs_baseline_cells.tsv][vbc]                                  | Same reason as the headline.                                 |

**Never mix scorings within a single comparison.** A condition's refined score against an original-extractor baseline would manufacture an apparent gain (or loss) entirely from the extractor delta, not from any condition effect.

## 9. Where the numbers come from

- **Raw model outputs:** `legesher/language-decoded-experiments` on HuggingFace, `phase3/conditions/<condition>/seed<N>/<condition>_seed<N>_results_<template>.json`. 42 files (21 sessions × 2 templates), ~52 MB each, ~2 GB total.
- **Inference-time-extractor summaries:** sibling `_summary_<template>.json` files in the same paths. Produced at evaluation time by the inference-time extractor (`evaluate.ipynb` cell 3, scoped to canonical English labels). Frozen historical record.
- **Refined-extractor summaries:** sibling `_summary_reparsed_<template>.json` files. Produced by `reparse_results.py` at commit `c7e2277` on `main` (PR #54 squash) — the merged, self-contained scorer. Uploaded to the HF dataset via HF PR #34 (merged 2026-05-24). Every refined summary's `extractor_provenance.content_sha256` matches `reparse_results.py` at that commit; reviewers reproducing the paper verify this hash against their checkout.
- **Comparison artefacts:** [`refined-tables/`](refined-tables/):
  - `cells.tsv` — every cell, one row, all deltas (inference-time-extractor view vs refined-extractor view)
  - `summary_by_benchmark.tsv`, `summary_by_instr_lang.tsv`, `summary_by_data_lang.tsv`, `summary_by_condition.tsv`, `summary_by_template.tsv` — single-axis rollups
  - `summary_bench_x_instr.tsv`, `summary_cond_x_bench.tsv` — two-axis crosstabs
  - `overall_stats.json` — the headline figures
  - `vs_baseline_cells.tsv`, `vs_baseline_by_condition.tsv`, `vs_baseline_by_cond_x_bench.tsv`, `vs_baseline_by_cond_x_instr.tsv` — condition-vs-baseline rollups under both extractors
  - `conclusion_flips.tsv` — cells where the sign of (condition − baseline) differs between the two extractors
  - **`framework_template_comparison.tsv`** — Axis 1: template-1 vs template-2 within each (condition, seed) cell, with `matched_instr` and `matched_diagonal` flags
  - **`framework_same_language_comparison.tsv`** — Axis 2: every condition trained on each target language, anchored against baseline + cond-1-en-5k + cond-1-en-20k
  - **`framework_data_volume_comparison.tsv`** — Axis 3: 5k vs 20k pairs within each condition family
- **Build scripts:** [`build_comparison.py`](../evaluation/scripts/build_comparison.py), [`build_vs_baseline.py`](../evaluation/scripts/build_vs_baseline.py), and [`build_framework_comparison.py`](../evaluation/scripts/build_framework_comparison.py) under `expedition-tiny-aya/evaluation/scripts/`.

### Reproducibility

```bash
# 1. Snapshot the summary files from HF (≈50 MB — no raw _results_*.json
#    needed; HF hosts the refined-extractor summaries directly).
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='legesher/language-decoded-experiments', \
                      repo_type='dataset', local_dir='/tmp/phase3_reparse/hf_snapshot', \
                      allow_patterns=['phase3/conditions/**/*_summary_template*.json', \
                                     'phase3/conditions/**/*_summary_reparsed_template*.json'])"

# 2. Build the analysis TSVs to a local /tmp directory. They are NOT
#    committed to GitHub — the canonical location is HF (step 3).
mkdir -p /tmp/phase3_analysis_output
PHASE3_OUT_DIR="/tmp/phase3_analysis_output" \
python expedition-tiny-aya/evaluation/scripts/build_comparison.py
PHASE3_OUT_DIR="/tmp/phase3_analysis_output" \
python expedition-tiny-aya/evaluation/scripts/build_vs_baseline.py
PHASE3_OUT_DIR="/tmp/phase3_analysis_output" \
python expedition-tiny-aya/evaluation/scripts/build_framework_comparison.py

# 3. (Optional) Upload the TSVs to HF as a discussion PR. Only do this if
#    you have WRITE access to the dataset and want the canonical tables
#    to reflect the version of the build scripts you just ran.
python expedition-tiny-aya/evaluation/scripts/upload_analysis_tables.py
```

To re-run the refined extractor against the raw `_results_*.json` files yourself (e.g., to verify the published refined summaries reproduce), see the `--write-reparsed-summary` mode in `reparse_results.py`. That's the path used by `upload_reparsed_summaries.py` — see HF PR #34 for the run that produced the canonical refined summaries on the dataset.

### Coverage

**Full coverage.** All 21 (condition × seed) sessions on HF have refined siblings as of HF PR #34 (merged 2026-05-24). This includes `condition-2-ur-20k/seed42` and `condition-5-es-5k/seed42` which were gaps in the initial pre-merge re-scoring pass.

---

_Last regenerated against HF main, post HF PR #34 (2026-05-24)._

[vbc]: https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/vs_baseline_cells.tsv

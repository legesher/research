# XNLI label bias vs learning (CORE-1383)

Verification analysis for two open items flagged in [`paper-verification-review.md`](paper-verification-review.md) (M18 and the §5.4 row): the draft's "rarely predicts the neutral XNLI label" claim, and the [`refined-decision-ledger.md`](refined-decision-ledger.md) Tier-2 observation that every recovered CJK-glued label form is `entailment`. Both are tested here from row-level predictions, and the draft's three-effect decomposition of XNLI gains (§5.4 of [`draft3-integrated.md`](draft3-integrated.md)) is quantified.

Companion TSV: [`xnli-label-distributions.tsv`](xnli-label-distributions.tsv) (960 rows; this directory). Build script: [`evaluation/scripts/build_xnli_label_bias.py`](../../evaluation/scripts/build_xnli_label_bias.py).

## Provenance

- Run 2026-07-09 against HF main, `legesher/language-decoded-experiments`, per-row files `phase3/conditions/<cond>/seed<N>/<cond>_seed<N>_results_template{1,2}.json`.
- Conditions and seeds: baseline (seed none), condition-1-en-5k (seeds 42/123/456), condition-2-{ur,es,zh}-5k (seeds 42/123/456), condition-5-{ur,es,zh}-5k (seed 42). 32 results files, 320 XNLI cells, 1,603,200 rows.
- Extractor: refined (`reparse_results.extract_xnli_label`), the extractor behind all published Phase-3 refined numbers. Predictions here additionally carry the matching tier (1a English / 1b native / 2 glued / 3 paraphrase); the tier-tagged twin is asserted equal to the untagged extractor on every row.
- Cross-check: recomputed per-cell accuracy matches the published `_summary_reparsed_template*.json` value exactly for all 320 cells. Note: the local `reparse_results.py` sha256 (`a42b1dcb…`) differs from the `content_sha256` pinned in the published summaries (`c7a1d51d…`); the exact accuracy match on every cell establishes behavioral equivalence on this corpus.
- Cell coverage: baseline and condition-1-en-5k cover all 16 (data_lang × instr_lang) cells per template; the cond-2 and cond-5 conditions cover instr_lang ∈ {en, condition language} × data_lang ∈ {en, es, ur, zh} (8 cells per template). Every baseline comparison below is restricted to the condition's matched cells.
- Every XNLI cell has n = 5,010 rows with gold exactly balanced: 1,670 entailment / 1,670 neutral / 1,670 contradiction (verified in all 320 cells).

## Methodological anchor: balanced gold makes label bias accuracy-neutral

Because gold is exactly balanced, any prediction strategy that is independent of the gold label (a constant label, or any label mix) has expected accuracy = parsed_share / 3, where parsed_share is the fraction of rows the extractor resolves. Define:

    edge = accuracy − parsed_share / 3

Edge is the discrimination signal beyond label-marginal behavior. Two consequences for the draft:

1. **Correction to §5.4.** The draft's second mechanism reads "the fine-tune shifting the model's predicted label distribution toward entailment (which is also the most common gold label)". In the Phase-3 XNLI cells the gold labels are exactly equally common (1,670 each per cell). A marginal shift toward entailment cannot inflate accuracy on this benchmark. The parenthetical is factually wrong and the mechanism, as stated, contributes zero.
2. The three effects decompose cleanly within the refined extraction: Δaccuracy = Δ(parsed_share)/3 (answer-format compliance: the model emitting an extractable label at all) + Δedge (genuine discrimination). The strict-vs-refined extractor gap is a separate, already-reported axis and is not re-measured here.

## Predicted-label distributions, split by instruction language

Shares of all rows in the pooled cells (denominator per row of this table: 4 data_lang cells × n seeds × 5,010 = 20,040 rows for baseline and cond-5, 60,120 for cond-1/cond-2; refined extractor; seeds pooled, which equals an equal-weight mean across seeds since every seed contributes the same n).

| condition | template | instr | E | N | C | parse-fail | acc | edge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | en | 0.701 | 0.086 | 0.212 | 0.001 | 0.444 | +0.111 |
| baseline | 1 | es | 0.905 | 0.004 | 0.091 | 0.000 | 0.390 | +0.056 |
| baseline | 1 | ur | 0.975 | 0.002 | 0.021 | 0.001 | 0.348 | +0.015 |
| baseline | 1 | zh | 0.991 | 0.005 | 0.004 | 0.000 | 0.333 | −0.000 |
| baseline | 2 | en | 0.131 | 0.145 | 0.718 | 0.007 | 0.386 | +0.055 |
| baseline | 2 | es | 0.348 | 0.176 | 0.372 | 0.105 | 0.343 | +0.045 |
| baseline | 2 | ur | 0.529 | 0.082 | 0.257 | 0.132 | 0.312 | +0.023 |
| baseline | 2 | zh | 0.577 | 0.012 | 0.066 | 0.346 | 0.257 | +0.039 |
| condition-1-en-5k | 1 | en | 0.698 | 0.061 | 0.241 | 0.000 | 0.458 | +0.125 |
| condition-1-en-5k | 1 | es | 0.892 | 0.003 | 0.105 | 0.000 | 0.399 | +0.066 |
| condition-1-en-5k | 1 | ur | 0.948 | 0.003 | 0.048 | 0.001 | 0.362 | +0.029 |
| condition-1-en-5k | 1 | zh | 0.988 | 0.003 | 0.009 | 0.000 | 0.334 | +0.001 |
| condition-1-en-5k | 2 | en | 0.079 | 0.066 | 0.850 | 0.005 | 0.371 | +0.040 |
| condition-1-en-5k | 2 | es | 0.351 | 0.151 | 0.484 | 0.014 | 0.377 | +0.049 |
| condition-1-en-5k | 2 | ur | 0.484 | 0.119 | 0.249 | 0.147 | 0.308 | +0.024 |
| condition-1-en-5k | 2 | zh | 0.688 | 0.014 | 0.102 | 0.196 | 0.324 | +0.056 |
| condition-2-es-5k | 1 | en | 0.669 | 0.053 | 0.278 | 0.000 | 0.470 | +0.137 |
| condition-2-es-5k | 1 | es | 0.781 | 0.008 | 0.211 | 0.000 | 0.434 | +0.101 |
| condition-2-es-5k | 2 | en | 0.053 | 0.064 | 0.880 | 0.003 | 0.366 | +0.034 |
| condition-2-es-5k | 2 | es | 0.329 | 0.340 | 0.330 | 0.001 | 0.385 | +0.052 |
| condition-2-ur-5k | 1 | en | 0.776 | 0.020 | 0.204 | 0.000 | 0.464 | +0.131 |
| condition-2-ur-5k | 1 | ur | 0.982 | 0.007 | 0.011 | 0.000 | 0.343 | +0.009 |
| condition-2-ur-5k | 2 | en | 0.146 | 0.019 | 0.835 | 0.001 | 0.407 | +0.074 |
| condition-2-ur-5k | 2 | ur | 0.321 | 0.166 | 0.512 | 0.001 | 0.387 | +0.054 |
| condition-2-zh-5k | 1 | en | 0.752 | 0.037 | 0.211 | 0.000 | 0.455 | +0.121 |
| condition-2-zh-5k | 1 | zh | 0.986 | 0.002 | 0.009 | 0.003 | 0.334 | +0.002 |
| condition-2-zh-5k | 2 | en | 0.094 | 0.052 | 0.852 | 0.001 | 0.384 | +0.051 |
| condition-2-zh-5k | 2 | zh | 0.821 | 0.027 | 0.082 | 0.070 | 0.328 | +0.018 |
| condition-5-es-5k | 1 | en | 0.856 | 0.080 | 0.064 | 0.000 | 0.379 | +0.046 |
| condition-5-es-5k | 1 | es | 0.875 | 0.005 | 0.120 | 0.000 | 0.397 | +0.063 |
| condition-5-es-5k | 2 | en | 0.301 | 0.175 | 0.513 | 0.011 | 0.400 | +0.070 |
| condition-5-es-5k | 2 | es | 0.348 | 0.100 | 0.547 | 0.005 | 0.365 | +0.033 |
| condition-5-ur-5k | 1 | en | 0.447 | 0.224 | 0.329 | 0.000 | 0.467 | +0.134 |
| condition-5-ur-5k | 1 | ur | 0.924 | 0.072 | 0.002 | 0.002 | 0.329 | −0.004 |
| condition-5-ur-5k | 2 | en | 0.060 | 0.367 | 0.565 | 0.008 | 0.359 | +0.029 |
| condition-5-ur-5k | 2 | ur | 0.965 | 0.023 | 0.010 | 0.001 | 0.341 | +0.008 |
| condition-5-zh-5k | 1 | en | 0.900 | 0.054 | 0.045 | 0.001 | 0.357 | +0.024 |
| condition-5-zh-5k | 1 | zh | 0.070 | 0.000 | 0.873 | 0.056 | 0.357 | +0.043 |
| condition-5-zh-5k | 2 | en | 0.277 | 0.198 | 0.518 | 0.007 | 0.428 | +0.097 |
| condition-5-zh-5k | 2 | zh | 0.010 | 0.515 | 0.253 | 0.222 | 0.289 | +0.029 |

Structure worth naming precisely:

- The baseline "entailment bias" is template- and instruction-conditional, not global. Template 1 collapses toward entailment, monotonically with script distance from English (en 0.701 → zh 0.991 of rows; gold share is 0.333). Template 2 under instr=en collapses toward contradiction instead (0.718). The model has a per-(template, instr_lang) default label, and which label it is depends on the prompt frame.
- Edge is at or below +0.14 everywhere. It is within ±0.015 of zero in the baseline template-1 instr ∈ {ur, zh} groups and in the matched-instruction template-1 groups of cond-2-ur (+0.009), cond-2-zh (+0.002), and cond-5-ur (−0.004). In those cells the model's accuracy is fully explained by gold-independent label emission.
- condition-5-zh-5k flips the default label wholesale: template 1 instr=zh goes from 0.991 entailment (baseline, same cells, 20,040 rows each) to 0.873 contradiction; template 2 instr=zh goes to 0.515 neutral. Pooled edge stays at +0.043 (template 1) and +0.029 (template 2); no single data_lang cell exceeds +0.076. A flipped constant is still a constant; this is degeneracy trading places, not NLI learning. Full per-data_lang grain is in the TSV; the flip holds in every data_lang cell.

## The "rarely predicts neutral" claim (draft §5.4, review item M18)

**Verdict: supported at baseline and in most conditions, but it needs scoping; a blanket "rarely" is falsified by specific fine-tuned template-2 cells, and none of the neutral-heavy cells reflect neutral competence.**

Pooled over each condition's full row set (denominator: all rows of that condition, refined extractor):

| condition | rows | neutral share of all rows | neutral share of parsed rows | recall on gold=neutral | precision of neutral |
| --- | --- | --- | --- | --- | --- |
| baseline | 160,320 | 0.064 | 0.069 | 0.055 | 0.290 |
| condition-1-en-5k | 480,960 | 0.052 | 0.055 | 0.046 | 0.291 |
| condition-2-es-5k | 240,480 | 0.116 | 0.116 | 0.105 | 0.302 |
| condition-2-ur-5k | 240,480 | 0.053 | 0.053 | 0.043 | 0.273 |
| condition-2-zh-5k | 240,480 | 0.030 | 0.030 | 0.024 | 0.265 |
| condition-5-es-5k | 80,160 | 0.090 | 0.090 | 0.068 | 0.253 |
| condition-5-ur-5k | 80,160 | 0.172 | 0.172 | 0.156 | 0.303 |
| condition-5-zh-5k | 80,160 | 0.192 | 0.207 | 0.185 | 0.322 |

Cell-level distribution (cell = condition × seed × template × data_lang × instr_lang; 320 cells; denominator per cell: 5,010 rows): median neutral share 0.017, mean 0.075; 258/320 cells below 0.10; 297/320 below the 0.333 gold share. The claim holds for the corpus bulk.

The exceptions are all fine-tuned template-2 cells where neutral becomes the new default label. Top three by neutral share (refined extractor): condition-2-es-5k seed 456, template2, data=en, instr=es: 0.695 of 5,010 rows; condition-2-es-5k seed 123, same cell: 0.693; condition-5-zh-5k seed 42, template2, data=es, instr=zh: 0.654. In each, neutral precision is at chance (0.330, and 0.342 for the cond-5-zh cell; a gold-independent guesser scores 0.333), so these are flipped defaults, not neutral understanding. Gold-neutral recall pooled per condition never exceeds 0.19 (max: condition-5-zh-5k, 0.185, driven by those same degenerate cells).

Suggested wording for the paper: "At baseline the model predicts neutral for 6.4% of XNLI rows against a 33.3% gold share, and gold-neutral recall stays below 0.19 in every condition; where fine-tuned template-2 cells do emit neutral at high rates (up to 70% of rows), neutral precision sits at the 1/3 chance level, indicating a relocated default label rather than acquired neutral competence."

## CJK-glued Tier-2 recoveries (decision-ledger check)

The ledger's claim (baseline, templates pooled, all 15 glued forms with count ≥ 10: the glued label is always `entailment`) generalizes to the full corpus and all eight conditions:

- 4,516 rows across all conditions resolve via Tier 2 (English label embedded without a word boundary): baseline 748, condition-1-en-5k 3,420, condition-2-zh-5k 295, condition-5-zh-5k 53, and zero in all es and ur conditions.
- **4,516/4,516 (100%) resolve to `entailment`.** No exceptions in any condition, template, seed, or cell.
- 4,512/4,516 (99.9%) have a CJK character in the first output line (the 4 others are Latin-only non-word-boundary embeddings); all 20 cells with any Tier-2 row are instr=zh cells (both templates; predominantly template 2).

**Elevated vs baseline?** Yes, trivially and maximally: within Tier-2 rows the entailment share is 1.000, versus a cell-wide entailment share of 0.472 in the heaviest baseline cell (baseline, template2, data=zh, instr=zh, seed none, 5,010 rows) and 0.333 gold share.

**Learning or inherited bias?** Inherited bias, by the gold split of the Tier-2 rows:

| condition | tier-2 rows | gold=E | gold=N | gold=C | correct (= gold=E share) |
| --- | --- | --- | --- | --- | --- |
| baseline | 748 | 236 | 270 | 242 | 0.316 |
| condition-1-en-5k | 3,420 | 1,054 | 1,265 | 1,101 | 0.308 |
| condition-2-zh-5k | 295 | 152 | 98 | 45 | 0.515 |
| condition-5-zh-5k | 53 | 31 | 17 | 5 | 0.585 |

At baseline and in cond-1-en-5k the glued form is emitted essentially uniformly across gold labels (roughly 1/3 each), so it is gold-independent scaffolding and the recovered rows score at or below chance (0.316 / 0.308 against the 0.333 gold share). In cond-2-zh-5k and cond-5-zh-5k the glued form correlates with gold=entailment (0.515 / 0.585), but the counts are small (295 / 53 rows) and the containing cells' edge stays at +0.046 / +0.048, so this does not amount to cell-level learning. The distribution of the glued form does not shift toward the balanced gold prior under fine-tuning; where the model stops emitting it (all es and ur conditions, Tier-2 = 0), it swaps to a different default rather than to discrimination.

## Bias vs learning: decomposition of condition-vs-baseline changes

Matched-instruction cell groups, condition vs baseline on the identical (template × data_lang × instr_lang) cells (denominators as in the first table; Δacc = Δparse-floor term + Δedge, where the parse-floor term is parsed_share/3):

| condition, template, instr | acc base→cond | Δacc | Δ from parse recovery | Δ edge |
| --- | --- | --- | --- | --- |
| cond-2-es-5k t1 es | 0.390→0.434 | +0.045 | +0.000 | **+0.045** |
| cond-2-es-5k t2 es | 0.343→0.385 | +0.043 | +0.035 | +0.008 |
| cond-2-ur-5k t1 ur | 0.348→0.343 | −0.005 | +0.000 | −0.006 |
| cond-2-ur-5k t2 ur | 0.312→0.387 | +0.075 | +0.044 | **+0.032** |
| cond-2-zh-5k t1 zh | 0.333→0.334 | +0.001 | −0.001 | +0.002 |
| cond-2-zh-5k t2 zh | 0.257→0.328 | +0.071 | +0.092 | −0.021 |
| cond-5-es-5k t1 es | 0.390→0.397 | +0.007 | −0.000 | +0.007 |
| cond-5-es-5k t2 es | 0.343→0.365 | +0.022 | +0.033 | −0.011 |
| cond-5-ur-5k t1 ur | 0.348→0.329 | −0.020 | −0.000 | −0.019 |
| cond-5-ur-5k t2 ur | 0.312→0.341 | +0.029 | +0.044 | −0.015 |
| cond-5-zh-5k t1 zh | 0.333→0.357 | +0.024 | −0.019 | +0.043 |
| cond-5-zh-5k t2 zh | 0.257→0.289 | +0.032 | +0.041 | −0.010 |

Reading: of the 12 matched-instruction groups, three show a discrimination gain above +0.03: cond-2-es-5k template1 instr=es (+0.045, the one case where the entire gain is signal), cond-2-ur-5k template2 instr=ur (+0.032, 43% of the headline +0.075), and cond-5-zh-5k template1 instr=zh (+0.043, which rides the flipped contradiction default off an exactly-zero baseline; entailment recall in those cells is 0.130, so it is not usable NLI competence). Four groups lose edge under fine-tuning while their headline accuracy rises on parse recovery alone (cond-2-zh t2, cond-5-es t2, cond-5-ur t2, cond-5-zh t2); cond-5-ur t1 loses both edge and accuracy. At the finest grain the single largest matched-instruction edge gain is condition-2-ur-5k, template2, data=zh, instr=ur, seeds 42/123/456 pooled (15,030 rows): edge +0.108 vs baseline +0.004 on the same cell.

For contrast, the instr=en columns (unmatched instruction) hold the model's real NLI signal throughout: baseline template1 instr=en edge +0.111, and every condition's template1 instr=en edge lands between +0.024 (cond-5-zh) and +0.137 (cond-2-es). The cond-5 conditions visibly erode it (cond-5-es +0.046, cond-5-zh +0.024 vs +0.111 baseline), the catastrophic-forgetting signature.

## Verdicts

1. **Bias, not learning.** XNLI prediction distributions are dominated by per-(template × instr_lang) default labels inherited from the baseline. Fine-tuning relocates, amplifies, or flips the default (cond-5-zh flips entailment→contradiction on template 1 and →neutral on template 2) but moves the discrimination edge by less than ±0.05 in all 12 matched-instruction groups. Genuine, if small, learning appears only in cond-2-es-5k (template 1, instr=es) and cond-2-ur-5k (template 2, instr=ur).
2. **Draft §5.4 correction (required).** Gold is exactly balanced (1,670 per label per cell), so "entailment ... is also the most common gold label" is false, and a marginal shift toward entailment cannot raise accuracy. The three-effect list should be replaced by the two-term decomposition above (answer-format compliance + discrimination), which this data fully separates.
3. **"Rarely predicts neutral" (M18): keep, with scope.** True at baseline (6.4% of 160,320 rows) and for 258/320 cells (< 10% neutral share); false as a blanket claim because several fine-tuned template-2 cells adopt neutral as their new default (up to 69.5% of rows) at chance precision.
4. **CJK-glued forms: ledger confirmed and strengthened.** 4,516/4,516 Tier-2 recoveries are `entailment`, across all conditions; at baseline they are gold-independent and score below chance. They are inherited scaffolding, not learned content; conditions fine-tuned on es/ur stop emitting them entirely.

## TSV schema

One row per (condition, seed, template, data_lang, instr_lang, gold): `n_gold` (always 1,670), predicted-label counts `pred_entailment` / `pred_neutral` / `pred_contradiction` / `pred_none` (refined-extractor parse-failures), `acc_on_gold` (recall of gold within the slice, denominator `n_gold`), tier counts `via_tier1a_english` / `via_tier1b_native` / `via_tier2_glued` / `via_tier3_paraphrase` (resolved rows in the slice, by matching tier), `tier2_pred_entailment` and `tier2_cjk_frame` (Tier-2 rows resolving to entailment / containing a CJK first-line character). All counts are row-level; any rollup must collapse seeds first if it produces threshold-counts (see [`aggregation-bug-audit.md`](aggregation-bug-audit.md)).

_Run 2026-07-09 against HF main. Re-run `build_xnli_label_bias.py` when refined data refreshes._

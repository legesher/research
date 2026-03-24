# Evaluation Summary — Expedition Tiny Aya, Cycle 1

**Date**: 2026-03-24
**Author**: Madison (with analysis support from Claude)
**Linear**: AYA-88, AYA-89
**Data**: [legesher/language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) on HuggingFace
**Adapters**: [legesher/language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora) on HuggingFace

---

## Executive Summary

1. **Chinese keyword code specifically improves Chinese XNLI** — Cond 2-zh achieves 42.2% on Chinese XNLI (native prompt), +5.3pp over English code (Cond 1-5K at 36.9%). This is the cleanest signal: same 5K files, only keywords differ.
2. **Code fine-tuning generally does not improve — and sometimes hurts — Spanish and Urdu XNLI** — Most conditions show regression from baseline for es and ur. The baseline already performs well on native prompts after corrected label extraction (49.1% es, 38.1% ur).
3. **5K files equals full dataset performance** — Cond 1 (full) vs Cond 1 (5K) shows <1pp difference across all metrics. Resource-efficient experimentation validated.
4. **MGSM (math reasoning) shows no meaningful improvement** — all differences are within noise at this model scale (3.35B params). This is a useful null result.
5. **Label extraction methodology significantly affected XNLI scores** — Re-scoring with expanded native label maps changed baseline zh from 2.0% to 36.1% and baseline es from 23.8% to 49.1%. See [Label Extraction Methodology](#label-extraction-methodology) section.

### Caveats

- XNLI native-prompt improvements may partly reflect instruction-following changes rather than deeper NLI capability
- Only 1 of 3 planned Condition 3 variants was completed (zh only)
- No English benchmark evaluation yet (catastrophic forgetting not assessed)
- XNLI scores were significantly affected by label extraction methodology (documented below)

---

## Label Extraction Methodology

### Background

XNLI evaluation requires extracting a label (entailment, contradiction, neutral) from the model's free-form text output. The extraction function matches against a label map. Three issues were discovered during analysis that affected the original scores:

### Issue 1: Missing Chinese Native Labels (Baseline Only)

**Impact**: Baseline XNLI zh native: 2.0% → **36.1%** (4,773 of 5,010 predictions changed)

The baseline evaluation was run with an extraction function that only matched English labels. When the model responded in Chinese (e.g., `矛盾` for "contradiction", `蕴含` for "entailment", `中立` for "neutral"), the extraction returned `None` and the prediction was marked incorrect. After adding Chinese labels to the map, 36.1% of predictions matched correctly.

**Labels added**: `蕴含`/`蕴涵` (entailment), `矛盾` (contradiction), `中立` (neutral)

### Issue 2: Case-Sensitive Spanish Label Matching (All Conditions)

**Impact**: Baseline XNLI es native: 23.8% → **49.1%** (2,467 predictions changed). All conditions affected to varying degrees.

The native label map included lowercase Spanish labels (`contradicción`, `implicación`) but the model sometimes outputs capitalized versions (`Contradicción`, `Implicación`) at the start of a response. The case-sensitive comparison missed these.

**Fix**: Case-insensitive matching (`native.lower() in text_lower`)

| Condition | Before | After | Predictions changed |
| --------- | ------ | ----- | ------------------- |
| Baseline  | 23.8%  | 49.1% | 2,467               |
| Cond 1    | 20.8%  | 48.0% | 2,926               |
| Cond 1 5K | 19.7%  | 47.3% | 3,081               |
| Cond 2-zh | 39.0%  | 43.1% | 268                 |
| Cond 2-es | 42.4%  | 44.5% | 331                 |
| Cond 2-ur | 48.4%  | 48.8% | 26                  |
| Cond 3-zh | 38.0%  | 41.6% | 236                 |

**Pattern**: Cond 1 variants had the most changes (~3,000) because the model outputs capitalized Spanish more often after English code fine-tuning. Cond 2 variants already captured most predictions correctly because keyword-swapped code apparently shifted the model toward lowercase Spanish output.

### Issue 3: Urdu Paraphrased Labels (Minor)

**Impact**: Baseline XNLI ur native: 37.8% → **38.1%** (16 predictions changed)

The model occasionally outputs paraphrased Urdu for "entailment": `لازم آتی ہے` ("it follows") or `انضمامیت` instead of the exact keyword `لازمی`. Adding these paraphrases captured 16 additional correct predictions across the baseline.

**Labels added**: `لازم آتی ہے` (entailment paraphrase), `انضمامیت` (alternative entailment)

### Re-Scoring Summary

All XNLI native-prompt results on HuggingFace have been re-scored with the expanded, case-insensitive label map. English-prompt results were minimally affected (1 prediction changed for zh baseline, no accuracy change) because English labels were already in the original map.

**The numbers in this document reflect the re-scored values.**

---

## Benchmark Descriptions

Understanding what each benchmark measures is critical for interpreting results.

| Benchmark | Full Name                                | Task                                                                              | Format                              | Random Chance | Examples/lang |
| --------- | ---------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------- | ------------- | ------------- |
| **MGSM**  | Multilingual Grade School Math           | Solve word problems requiring multi-step arithmetic                               | Open-ended (model outputs a number) | ~0%           | 250           |
| **XNLI**  | Cross-lingual Natural Language Inference | Given a premise and hypothesis, classify as entailment, contradiction, or neutral | 3-way classification                | 33.3%         | ~5,000        |
| **CSQA**  | Cross-lingual Commonsense QA             | Choose the most plausible answer to a commonsense question                        | 5-way multiple choice               | 20%           | ~1,000        |

### Why These Benchmarks Matter

- **MGSM** tests **reasoning** — if code exposure helps reasoning, MGSM should improve
- **XNLI** tests **language understanding** — if multilingual keywords improve the model's grasp of that language, XNLI should improve
- **CSQA** tests **world knowledge** — least likely to be affected by code training

### Statistical Significance Context

At these sample sizes, approximate 95% confidence intervals (pp = percentage points):

| Benchmark | Sample size | CI at ~5% | CI at ~20% | CI at ~40% | CI at ~50% |
| --------- | ----------- | --------- | ---------- | ---------- | ---------- |
| MGSM      | 250         | +/-2.7pp  | +/-5.0pp   | +/-6.1pp   | +/-6.2pp   |
| CSQA      | 1,000       | +/-1.4pp  | +/-2.5pp   | +/-3.0pp   | +/-3.1pp   |
| XNLI      | 5,000       | +/-0.6pp  | +/-1.1pp   | +/-1.4pp   | +/-1.4pp   |

**Key implication**: For MGSM, a 2-3pp difference is noise. For XNLI, even a 3pp difference is statistically significant. CSQA falls in between.

---

## Conditions Evaluated

| Condition            | Data                                           | What It Isolates                            |
| -------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Baseline**         | None (Tiny Aya base)                           | Floor — what does the model know already?   |
| **Cond 1 (en full)** | English Python from The Stack Dedup (full set) | Does code help at all?                      |
| **Cond 1 (en 5K)**   | Same, 5K subset                                | Volume check — is 5K enough?                |
| **Cond 2-zh**        | Chinese keyword-swapped Python (5K)            | Does Chinese keyword exposure help Chinese? |
| **Cond 2-es**        | Spanish keyword-swapped Python (5K)            | Does Spanish keyword exposure help Spanish? |
| **Cond 2-ur**        | Urdu keyword-swapped Python (5K)               | Does Urdu keyword exposure help Urdu?       |
| **Cond 3-zh**        | Chinese transpiled + Wenyan + community (5K)   | Does a richer mix of native code help more? |

**Critical design detail**: Conditions 1-5K and all Cond 2 variants use the **exact same 5K Python files**. The only difference is keyword language (37 keywords, 72 builtins, 66 exceptions). This makes Cond 1 vs Cond 2 a true controlled comparison — any performance difference is attributable purely to keyword language.

**Not included in this analysis**:

- Cond 3-es, Cond 3-ur — intentionally not run (insufficient native data collected)
- Condition 4 (strictly native code) — out of scope for Cycle 1
- English benchmark data (MGSM-en, XNLI-en, CSQA-en) — not yet evaluated (future step for forgetting check)

---

## Results Tables

### Native Prompt Results (language-specific prompt + target-language data)

_All XNLI scores reflect re-scored values with expanded native label extraction (see [Label Extraction Methodology](#label-extraction-methodology))._

| Condition            | MGSM zh | MGSM es | MGSM ur | XNLI zh | XNLI es | XNLI ur | CSQA zh | CSQA es | CSQA ur |
| -------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| **Baseline**         | 6.0%    | 5.6%    | 3.2%    | 36.1%   | 49.1%   | 38.1%   | 52.1%   | 52.4%   | 40.0%   |
| **Cond 1 (en full)** | 4.4%    | 6.4%    | 2.8%    | 37.4%   | 48.0%   | 36.9%   | 52.4%   | 53.3%   | 40.6%   |
| **Cond 1 (en 5K)**   | 3.6%    | 5.6%    | 2.8%    | 36.9%   | 47.3%   | 36.9%   | 52.3%   | 53.3%   | 40.7%   |
| **Cond 2-zh**        | 6.0%    | 6.8%    | 4.4%    | 42.2%   | 43.1%   | 36.2%   | 52.9%   | 53.4%   | 41.0%   |
| **Cond 2-es**        | 3.2%    | 5.6%    | 4.4%    | 35.5%   | 44.5%   | 33.3%   | 52.2%   | 53.4%   | 39.8%   |
| **Cond 2-ur**        | 3.2%    | 6.8%    | 3.2%    | 34.6%   | 48.8%   | 36.9%   | 52.4%   | 53.6%   | 38.2%   |
| **Cond 3-zh**        | 7.2%    | 4.0%    | 3.2%    | 36.0%   | 41.6%   | 34.0%   | 54.5%   | 52.8%   | 42.2%   |

### English Prompt Results (English prompt + target-language data)

| Condition            | MGSM zh | MGSM es | MGSM ur | XNLI zh | XNLI es | XNLI ur | CSQA zh | CSQA es | CSQA ur |
| -------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| **Baseline**         | 6.8%    | 7.2%    | 6.0%    | 46.4%   | 51.2%   | 40.7%   | 52.3%   | 53.8%   | 41.2%   |
| **Cond 1 (en full)** | 8.0%    | 9.2%    | 6.0%    | 47.1%   | 52.2%   | 41.2%   | 51.2%   | 53.4%   | 42.1%   |
| **Cond 1 (en 5K)**   | 8.4%    | 9.6%    | 6.4%    | 47.0%   | 52.2%   | 41.8%   | 51.7%   | 53.5%   | 42.2%   |
| **Cond 2-zh**        | 7.2%    | 10.0%   | 6.8%    | 46.9%   | 52.7%   | 40.2%   | 51.3%   | 53.1%   | 42.2%   |
| **Cond 2-es**        | 7.2%    | 10.8%   | 6.8%    | 51.4%   | 48.0%   | 36.3%   | 51.5%   | 52.2%   | 40.8%   |
| **Cond 2-ur**        | 6.0%    | 10.8%   | 7.2%    | 52.0%   | 52.1%   | 40.0%   | 51.4%   | 54.3%   | 39.8%   |
| **Cond 3-zh**        | 8.4%    | 10.0%   | 8.8%    | 51.1%   | 51.4%   | 36.9%   | 53.8%   | 55.6%   | 43.3%   |

---

## Key Findings

### Finding 1: Chinese keyword code specifically improves Chinese XNLI

Cond 2-zh achieves 42.2% on Chinese XNLI (native prompt), the highest zh XNLI score across all conditions. Deltas from baseline (36.1%):

| Condition | zh delta   | es delta | ur delta |
| --------- | ---------- | -------- | -------- |
| Cond 1 5K | +0.8pp     | -1.8pp   | -1.2pp   |
| Cond 2-zh | **+6.1pp** | -6.0pp   | -1.9pp   |
| Cond 2-es | -0.6pp     | -4.6pp   | -4.8pp   |
| Cond 2-ur | -1.5pp     | -0.3pp   | -1.2pp   |
| Cond 3-zh | -0.1pp     | -7.5pp   | -4.1pp   |

With ~5,000 XNLI examples, a 6pp change is ~4x the confidence interval — statistically significant.

**Key insight**: Only Cond 2-zh improves Chinese XNLI meaningfully. All other conditions are flat or slightly worse. And the improvement is specific to Chinese — the same condition _hurts_ Spanish (-6.0pp) and Urdu (-1.9pp). This suggests the model is learning from the Chinese keyword tokens specifically, not gaining general multilingual capability.

### Finding 2: Target-language keywords help Chinese but not Spanish or Urdu

Comparing Cond 2-{lang} against Cond 1-5K on the _matching_ target language (exact same 5K files, only keywords differ):

| Comparison          | Cond 2 score | Cond 1 5K | Delta      |
| ------------------- | ------------ | --------- | ---------- |
| Cond 2-zh → XNLI zh | 42.2%        | 36.9%     | **+5.3pp** |
| Cond 2-es → XNLI es | 44.5%        | 47.3%     | **-2.8pp** |
| Cond 2-ur → XNLI ur | 36.9%        | 36.9%     | 0.0pp      |

**For Chinese**: The keyword swap provides a clear, statistically significant benefit (+5.3pp). Because the underlying code is identical, this difference is attributable purely to the 37 keywords + 72 builtins + 66 exceptions being in Chinese vs English.

**For Spanish**: The keyword swap actually _hurts_ compared to English code (-2.8pp). This is unexpected and may relate to how the model's tokenizer handles Spanish keywords vs English ones.

**For Urdu**: No effect at all.

**Implication**: The target-language keyword benefit is language-specific, not universal. It works for Chinese but not for Spanish or Urdu in this experiment.

### Finding 3: Code fine-tuning generally hurts Spanish XNLI native prompt

The baseline Spanish XNLI (49.1%) is the highest starting point, and nearly every fine-tuning condition makes it worse:

| Condition | XNLI es (native) | vs baseline |
| --------- | ---------------- | ----------- |
| Baseline  | 49.1%            | —           |
| Cond 1 5K | 47.3%            | -1.8pp      |
| Cond 2-zh | 43.1%            | **-6.0pp**  |
| Cond 2-es | 44.5%            | **-4.6pp**  |
| Cond 2-ur | 48.8%            | -0.3pp      |
| Cond 3-zh | 41.6%            | **-7.5pp**  |

**Pattern**: Conditions with Chinese code or mixed Chinese sources cause the largest regressions (-6 to -7.5pp). Urdu keyword code is nearly neutral (-0.3pp). Even English code slightly hurts (-1.8pp).

**Possible explanation**: The model's Spanish XNLI performance was already strong at baseline (49.1%, well above random chance of 33.3%). Code fine-tuning may cause partial catastrophic forgetting of Spanish instruction-following when the training data doesn't contain Spanish text. Cond 2-ur's near-neutral result may indicate that Urdu script is different enough from Spanish that it doesn't interfere.

### Finding 4: Urdu XNLI doesn't respond to any fine-tuning

Urdu XNLI native is essentially flat across all conditions:

| Condition | XNLI ur (native) | vs baseline |
| --------- | ---------------- | ----------- |
| Baseline  | 38.1%            | —           |
| Cond 1 5K | 36.9%            | -1.2pp      |
| Cond 2-zh | 36.2%            | -1.9pp      |
| Cond 2-es | 33.3%            | **-4.8pp**  |
| Cond 2-ur | 36.9%            | -1.2pp      |
| Cond 3-zh | 34.0%            | -4.1pp      |

Even Cond 2-ur (Urdu keyword code) shows -1.2pp. Spanish keyword code causes the largest regression (-4.8pp, dropping to exactly random chance at 33.3%).

**Possible explanations**: (a) Urdu uses RTL Nastaliq script, which may not benefit from code fine-tuning in the same way; (b) The model's Urdu capability may be limited at this scale regardless of training data; (c) Code fine-tuning slightly interferes with existing Urdu instruction-following.

### Finding 5: MGSM shows no meaningful improvement (null result)

All MGSM scores range from 2.8% to 10.8%. With only 250 examples and a CI of +/-2.7pp at ~5%, most condition differences are within noise. Code fine-tuning does NOT improve mathematical reasoning at 3.35B params.

This is consistent with broader findings that arithmetic reasoning requires larger models or specialized training. For the paper, this is a useful null result bounding what multilingual code can't do.

### Finding 6: CSQA is stable (possible ceiling effect)

CSQA scores cluster between 38-56% across all conditions. No condition produces more than ~3pp improvement over baseline, which is near the significance boundary for 1,000 examples.

**Cond 3-zh stands out slightly**: With English prompts, Cond 3-zh achieves the best CSQA across all three languages (53.8% zh, 55.6% es, 43.3% ur). This may indicate that native-language code sources (Wenyan) carry cultural knowledge that pure keyword-swap doesn't — but the improvements are small enough to be noise.

### Finding 7: 5K files = full dataset performance

Cond 1 (full) vs Cond 1 (5K) shows max gap of 0.8pp across all 18 scores. This validates that 5K files is sufficient for QLoRA fine-tuning at this scale — the model learns structural patterns from code rather than memorizing specific files.

### Finding 8: Prompt language has a benchmark-dependent effect

| Benchmark | Avg native prompt | Avg English prompt | Avg advantage |
| --------- | ----------------- | ------------------ | ------------- |
| MGSM      | ~4.3%             | ~8.2%              | ~4pp (2x)     |
| XNLI      | ~39.8%            | ~46.6%             | ~7pp          |
| CSQA      | ~46.6%            | ~47.5%             | ~1pp          |

CSQA works equally well in both prompt languages. But for XNLI and MGSM, English prompts unlock more capability. Note that with corrected native XNLI scores, the native-vs-English gap is smaller than previously estimated (~7pp vs ~12pp), because the baseline was underestimating native-prompt XNLI scores due to the label extraction bug.

### Finding 9 (Emerging): Cond 3-zh shows the broadest improvement profile with English prompts

With English prompts, Cond 3-zh achieves the best or tied-best score on 6 of 9 metrics — notably sweeping all three CSQA languages and best MGSM ur (8.8%). This is the one condition with native-language code (Wenyan), suggesting diverse source material may carry a broader training signal.

**Caveat**: With only one Cond 3 variant (zh), this could be a fluke. CSQA improvements are near noise. Needs Cond 3-es and 3-ur to confirm the pattern generalizes.

---

## Anomalies Requiring Investigation

### Anomaly 1: Cond 2-ur → Spanish XNLI = 48.8% (native prompt)

Cond 2-ur shows near-baseline Spanish XNLI (48.8% vs 49.1% baseline = -0.3pp), making it the best-performing condition for Spanish XNLI. Meanwhile, other non-Spanish conditions (zh, es code) _regress_ by 4-7pp. It's unclear why Urdu keyword code would preserve Spanish performance while Chinese keyword code damages it.

**Action**: Compare output distributions across conditions for Spanish XNLI. Check whether Cond 2-ur model outputs show different label patterns than other conditions.

### Anomaly 2: Cond 2-es hurts Spanish XNLI by -4.6pp

Spanish keyword code (Cond 2-es) produces _worse_ Spanish XNLI (44.5%) than baseline (49.1%) and even worse than English code (Cond 1-5K at 47.3%). If target-language keywords help (as seen for zh), why does es keyword code hurt Spanish?

**Possible explanation**: The model may be overfitting to Spanish tokens in code context (keywords like `si`, `para`, `mientras`) and this interferes with processing Spanish in NLI context. Or the keyword-swapped Spanish code creates confusing token distributions that hurt instruction-following.

### Anomaly 3: Code fine-tuning with Chinese sources hurts Urdu most

Both Cond 2-zh (-1.9pp) and Cond 3-zh (-4.1pp) hurt Urdu XNLI more than English code does (-1.2pp). Cond 3-zh causes the second-largest Urdu regression after Cond 2-es (-4.8pp). Chinese and Urdu share no linguistic properties, but the Chinese training signal may interfere with the model's Urdu processing.

### CSQA Language Swap Bug (branch-specific)

The `feat/create-5k-subset-AYA-173` branch has a bug where `csqa_es` is evaluated with `lang='zh'` and `csqa_zh` with `lang='es'` in the native prompt function. **The main branch code is correct.** If any evaluations were run from the feature branch, those CSQA zh/es native-prompt results would have swapped language prompts. Verify with Khojasteh which branch was used for each evaluation run.

---

## Presentation Highlights

1. **"Chinese keywords improve Chinese NLI"** — Cond 2-zh achieves 42.2% vs baseline 36.1% (+6.1pp). Target-language keywords outperform English code (+5.3pp vs Cond 1-5K), using the exact same underlying files.
2. **"5K files is enough"** — Full dataset vs 5K shows negligible difference. Resource-efficient experimentation validated.
3. **"Mixed sources (Cond 3) shows broadest gains with English prompts"** — Best CSQA across all languages, best MGSM ur.
4. **"Prompt language matters more than training data for math"** — English prompts ~2x MGSM scores regardless of condition.
5. **"Effect is language-specific"** — Chinese keywords help Chinese NLI; Spanish and Urdu do not benefit from keyword swap. The finding is more targeted than a broad "multilingual code helps."
6. **"Label extraction methodology matters"** — Re-scoring with proper native label maps changed baseline XNLI dramatically (zh: 2% → 36%, es: 24% → 49%). Evaluation methodology must be rigorous.

### Key caveats for presentation

- The code fine-tuning effect is narrower than initially estimated — primarily limited to Chinese XNLI with Chinese keywords
- Spanish and Urdu XNLI show slight regressions from code fine-tuning, suggesting potential catastrophic forgetting
- MGSM improvements are within noise (250 examples)
- Only 1 of 3 languages has Condition 3 results
- No English benchmarks yet for forgetting analysis

---

## Next Steps

### Immediate (no new training required)

- **English benchmark evaluation**: Run already-trained adapters on MGSM-en, XNLI-en, CSQA-en to assess catastrophic forgetting. LoRA adapters are saved at `legesher/language-decoded-lora`. (AYA-180)
- **Raw output analysis**: Examine model outputs qualitatively for XNLI across conditions — look at output format, label distribution, confidence patterns to understand _why_ Chinese keywords help but Spanish/Urdu don't.
- **Anomaly investigation**: Investigate why Cond 2-ur preserves Spanish XNLI while Cond 2-es hurts it.
- **Verify evaluation branch**: Confirm with Khojasteh which branch was used for evaluations (main vs `feat/create-5k-subset-AYA-173`) given the CSQA lang swap bug on the feature branch.

### Phase 2 stretch / Phase 3

- **Cond 3-es, 3-ur**: Collect more native-language code to enable remaining Cond 3 variants.
- **Condition 4**: Evaluate strictly native code (Wenyan, Qalb, Citrine).
- **Condition 5**: Cross-lingual transfer experiments.
- **Condition 6**: NL text control to isolate code structure vs language exposure.

---

_Generated 2026-03-24 (updated with re-scored XNLI values). Data source: HuggingFace `legesher/language-decoded-experiments`. Visualization script: `analysis/scripts/plot_condition_comparison.py`._

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

## Deep Dive: XNLI Label Bias (All Three Languages)

Per-example analysis of XNLI native-prompt results across all three evaluation languages reveals that the model has collapsed the 3-way classification into a binary task. This is the single most important finding for interpreting XNLI results in this experiment.

### Background: What XNLI asks the model to do

XNLI gives the model two sentences — a premise and a hypothesis — and asks it to classify their relationship as one of three labels:

- **Entailment**: the hypothesis logically follows from the premise (if the premise is true, the hypothesis must be true)
- **Contradiction**: the hypothesis conflicts with the premise (they cannot both be true)
- **Neutral**: the hypothesis is neither confirmed nor ruled out by the premise (it could be true or false — the premise doesn't tell us)

The gold labels in the test set are evenly split: ~1,670 entailment, ~1,670 contradiction, ~1,670 neutral (~5,010 total per language). Random chance is 33.3%.

### The model never predicts "neutral" — in any language

Across **all 7 conditions and all 3 languages** (105,210 total predictions), "neutral" is predicted fewer than 60 times total. The model has collapsed XNLI into a binary entailment-vs-contradiction decision. This caps the maximum achievable accuracy at ~66.7% since all ~1,670 neutral items per language are guaranteed wrong.

| Language | Total neutral predictions (across all 7 conditions) | Neutral prediction rate |
| -------- | --------------------------------------------------- | ----------------------- |
| Chinese  | 3                                                   | 0.009%                  |
| Spanish  | 5                                                   | 0.014%                  |
| Urdu     | 53                                                  | 0.15%                   |

### Each language has a different baseline bias

The baseline model's default label preference varies by language:

| Language | Baseline entailment % | Baseline contradiction % | Baseline bias            |
| -------- | --------------------- | ------------------------ | ------------------------ |
| Chinese  | 7.5%                  | 92.3%                    | Strongly contradiction   |
| Spanish  | 49.9%                 | 50.0%                    | Balanced                 |
| Urdu     | 38.2%                 | 61.5%                    | Moderately contradiction |

This explains why baseline accuracies differ: Spanish baseline (49.1%) scores highest because its even split aligns best with the balanced test set. Chinese baseline (36.1%) is worst because its extreme contradiction bias means it gets almost no entailment items right (12.2%).

### Fine-tuning shifts the binary boundary differently per language

#### Chinese XNLI — Prediction Distribution

| Condition | Predicts entailment | Predicts contradiction |
| --------- | ------------------- | ---------------------- |
| Baseline  | 7.5%                | 92.3%                  |
| Cond 1-5K | 13.3%               | 86.6%                  |
| Cond 2-zh | **65.2%**           | 34.8%                  |
| Cond 2-es | 6.6%                | 93.3%                  |
| Cond 2-ur | **98.2%**           | 1.7%                   |
| Cond 3-zh | 7.4%                | 92.5%                  |

Cond 2-zh is the only condition that moves the dial toward a balanced split (65/35), which is why it scores highest (42.2%). Cond 2-ur overshoots to 98% entailment, destroying contradiction accuracy entirely (4.3%).

#### Chinese XNLI — Per-Label Accuracy

| Condition | Entailment acc | Contradiction acc | Overall   |
| --------- | -------------- | ----------------- | --------- |
| Baseline  | 12.2%          | 96.2%             | 36.1%     |
| Cond 2-zh | **76.2%**      | 50.5%             | **42.2%** |
| Cond 2-ur | 99.3%          | 4.3%              | 34.6%     |

#### Spanish XNLI — Prediction Distribution

| Condition | Predicts entailment | Predicts contradiction |
| --------- | ------------------- | ---------------------- |
| Baseline  | 49.9%               | 50.0%                  |
| Cond 1-5K | 37.1%               | 62.8%                  |
| Cond 2-zh | **87.2%**           | 12.7%                  |
| Cond 2-es | 19.5%               | **80.5%**              |
| Cond 2-ur | **78.8%**           | 21.2%                  |
| Cond 3-zh | **89.3%**           | 10.6%                  |

Chinese/Urdu keyword training pushes toward entailment. English/Spanish keyword training pushes toward contradiction.

#### Spanish XNLI — Per-Label Accuracy

| Condition | Entailment acc | Contradiction acc | Overall   |
| --------- | -------------- | ----------------- | --------- |
| Baseline  | 69.7%          | 77.5%             | **49.1%** |
| Cond 2-zh | 97.6%          | 31.7%             | 43.1%     |
| Cond 2-es | 37.1%          | 96.5%             | 44.5%     |

#### Urdu XNLI — Prediction Distribution

| Condition | Predicts entailment | Predicts contradiction |
| --------- | ------------------- | ---------------------- |
| Baseline  | 38.2%               | 61.5%                  |
| Cond 1-5K | 22.8%               | 77.0%                  |
| Cond 2-zh | 13.7%               | 86.3%                  |
| Cond 2-es | 0.2%                | **99.7%**              |
| Cond 2-ur | 35.0%               | 64.9%                  |
| Cond 3-zh | 2.8%                | 96.7%                  |

Every condition deepens the contradiction bias. Cond 2-es collapses to a single-class predictor (99.7% contradiction), achieving exactly 33.3% — random chance.

#### Urdu XNLI — Per-Label Accuracy

| Condition | Entailment acc | Contradiction acc | Overall   |
| --------- | -------------- | ----------------- | --------- |
| Baseline  | 43.2%          | 71.2%             | **38.1%** |
| Cond 2-ur | 39.8%          | 70.9%             | 36.9%     |
| Cond 2-es | 0.1%           | 99.9%             | 33.3%     |

### Regressions are clean label flips, not output degradation

Zero malformed outputs were found across any language. Every regression is a question where the model switched from one valid label to another — not a formatting or language issue.

| Language | Condition with most regressions from baseline | Dominant flip pattern      | % of regressions |
| -------- | --------------------------------------------- | -------------------------- | ---------------- |
| Chinese  | Cond 2-ur (1,535 regressions)                 | contradiction → entailment | 99.9%            |
| Spanish  | Cond 3-zh (848 regressions)                   | contradiction → entailment | 99.8%            |
| Urdu     | Cond 2-es (719 regressions)                   | entailment → contradiction | 100.0%           |

### Questions wrong across all conditions

| Language | Wrong in all 7 conditions | Right in all 7 conditions |
| -------- | ------------------------- | ------------------------- |
| Chinese  | 1,683 (33.6%)             | 83 (1.7%)                 |
| Spanish  | 1,739 (34.7%)             | 979 (19.5%)               |
| Urdu     | 2,313 (46.2%)             | 881 (17.6%)               |

The universally-wrong questions are almost entirely the ~1,670 neutral items (since neutral is never predicted) plus a few hard entailment/contradiction items. Chinese has the fewest universally-correct items because its strong contradiction bias means fewer entailment items are ever captured.

### What this means for interpreting XNLI results

1. **Overall XNLI accuracy is not a measure of language understanding.** It is an artifact of how the model's binary bias aligns with the test set's balanced label distribution. The "best" condition for each language is simply the one with the most balanced entailment/contradiction split.

2. **Cond 2-zh's Chinese XNLI improvement (+6.1pp) is real but narrow.** It reflects Chinese keywords teaching the model to sometimes predict "entailment" in Chinese instead of defaulting to "contradiction." This is genuine learning — but it's learning a label preference, not deeper NLI reasoning.

3. **Spanish regression from fine-tuning is a bias shift, not comprehension loss.** The baseline's 49.1% is highest because it happens to have the most balanced binary split, not because it understands Spanish NLI best.

4. **Urdu's "flatness" across conditions is the deepest bias.** Every condition makes the contradiction bias worse. Even Urdu keyword training (Cond 2-ur) doesn't help — it preserves the baseline's split but doesn't improve it.

5. **The root limitation is model scale.** At 3.35B parameters, Tiny Aya cannot learn the 3-way NLI distinction (especially the nuanced "neutral" category). Before attributing XNLI differences to training conditions, the model needs to demonstrate it can produce all three labels.

---

## Deep Dive: Raw Output Analysis

Inspecting the actual text the model generates (the `raw_output` field in per-example results) reveals that the accuracy numbers are even less meaningful than the label bias analysis suggests. The model is producing memorized templates, not reasoned classifications.

### Output format differs between baseline and fine-tuned conditions

| Condition            | Single label word only | Label + explanation | Code artifacts |
| -------------------- | ---------------------- | ------------------- | -------------- |
| Baseline (zh)        | 55%                    | 40%                 | 0%             |
| Baseline (es)        | 98%                    | 2%                  | 0%             |
| Fine-tuned (all, zh) | 0-13%                  | 86-100%             | 0%             |
| Fine-tuned (all, es) | 0-1%                   | 99%                 | 0%             |
| Cond 2-ur (ur)       | 0%                     | 98%                 | **2%**         |

Fine-tuning consistently makes the model more verbose — it almost always adds an explanation after the label. The baseline more often produces a bare label word.

### Chinese keyword fine-tuning increased English label usage, not Chinese

For Chinese XNLI, the baseline outputs Chinese labels (矛盾, 蕴含) 95% of the time. After Cond 2-zh fine-tuning, **65% of outputs use the English word "entailment"** instead of a Chinese equivalent:

| Condition | First token = Chinese label | First token = English label |
| --------- | --------------------------- | --------------------------- |
| Baseline  | 95.3%                       | 4.6%                        |
| Cond 2-zh | 34.8%                       | **65.1%**                   |
| Cond 2-ur | 1.8%                        | **98.2%**                   |

Chinese keyword code did not teach the model to use Chinese classification labels more. It shifted the model toward the English word "entailment" — likely because English labels appeared in the training prompt format, and the fine-tuning reinforced that English token.

### Spanish XNLI: memorized template responses

Cond 2-zh outputs the **exact same string** for 99.3% of its entailment predictions on Spanish XNLI:

```text
entailment\nLa hipotesis se sigue de
```

4,339 of 5,010 questions receive this identical response. The model is not evaluating premise-hypothesis pairs — it is producing a cached template. Other conditions show similar templating behavior with different labels.

### Urdu XNLI: label-explanation contradictions and code leakage

**Cond 2-es** (99.7% contradiction on Urdu XNLI) reveals a disconnect between label and reasoning. The model outputs `contradiction` as its label, then writes Urdu text describing entailment logic:

```text
contradiction\nHypothesis، premise سے لازم آتی
```

`لازم آتی` means "it follows" — the explanation says the hypothesis follows from the premise, while the label says contradiction. The first token is a learned reflex, not a reasoned judgment.

**Cond 2-ur** shows **code leaking into XNLI responses**:

```text
contradiction\nتعریف main():\n    premise =
contradiction\nتصدیق(entailment(p
```

`تعریف` is the Legesher keyword for `def`; `تصدیق` is the keyword for `assert`. The Urdu keyword code training is contaminating the NLI task — the model sometimes responds to XNLI prompts by generating Urdu Python code instead of NLI reasoning.

### What this means

1. **The model is not deliberately classifying.** It is producing memorized templates, not evaluating premise-hypothesis relationships. The same output string repeated thousands of times is not classification.

2. **Chinese keywords did not teach Chinese entailment.** They shifted the model toward producing the English token "entailment" more frequently. The improvement in Chinese XNLI accuracy is a side effect of English token probability changes, not Chinese language comprehension.

3. **Code training leaks into task outputs.** Urdu keyword translations (`تعریف` for `def`, `تصدیق` for `assert`) appear in XNLI responses, showing incomplete separation between the code fine-tuning domain and the evaluation domain.

4. **Label and explanation can contradict each other.** The model's first token (the classification label) is a learned bias, while the subsequent explanation text sometimes describes the opposite relationship. The label is reflexive, not reasoned.

### Benchmark selection implications

XNLI's fixed 3-label format is vulnerable to this kind of bias collapse at small model scales. Benchmarks where the answer options change per question — such as **Belebele** (4-way reading comprehension with unique options per question), **XStoryCloze** (2-way story completion with unique endings), or **SIB-200** (7-way topic classification with concrete categories) — would be more robust because the model cannot develop a fixed label preference. The eval pipeline already includes XStoryCloze and CSQA (5-way with changing options), and these benchmarks do not show the same bias collapse pattern.

---

## Presentation Highlights

1. **"5K files is enough"** — Full dataset vs 5K shows negligible difference. Resource-efficient experimentation validated.
2. **"Chinese keywords shift Chinese XNLI accuracy"** — Cond 2-zh achieves 42.2% vs baseline 36.1% (+6.1pp), but raw output analysis shows this reflects a label bias shift (toward the English word "entailment"), not deeper Chinese language understanding.
3. **"XNLI at this model scale measures label bias, not comprehension"** — The model never predicts "neutral" (<0.02% across all conditions and languages), producing memorized templates instead of reasoned classifications. CSQA and MGSM are more reliable indicators.
4. **"Mixed sources (Cond 3) shows broadest gains with English prompts"** — Best CSQA across all languages, best MGSM ur.
5. **"Prompt language matters more than training data for math"** — English prompts ~2x MGSM scores regardless of condition.
6. **"Code training leaks into evaluation"** — Urdu keyword translations appear in XNLI responses, and label-explanation contradictions show the model's first token is reflexive, not reasoned.

### Key caveats for presentation

- XNLI results should be presented with the label bias context — raw accuracy numbers are misleading without the per-label breakdown
- The code fine-tuning effect is narrower than the headline numbers suggest
- MGSM improvements are within noise (250 examples)
- Only 1 of 3 languages has Condition 3 results
- No English benchmarks yet for forgetting analysis

---

## Next Steps

### Immediate (no new training required)

- **English benchmark evaluation**: Run already-trained adapters on MGSM-en, XNLI-en, CSQA-en to assess catastrophic forgetting. LoRA adapters are saved at `legesher/language-decoded-lora`. (AYA-180)
- **Neutral label investigation**: Understand why the model never predicts "neutral." Test whether adding "neutral" as an explicit option in the prompt template changes behavior, or whether this is a fundamental model-scale limitation.
- **Alternative benchmarks**: Consider adding Belebele or SIB-200 to the eval suite for Cycle 2 — these have per-question answer options and are less vulnerable to label bias.
- **Verify evaluation branch**: Confirm with Khojasteh which branch was used for evaluations (main vs `feat/create-5k-subset-AYA-173`) given the CSQA lang swap bug on the feature branch.

### Phase 2 stretch / Phase 3

- **Cond 3-es, 3-ur**: Collect more native-language code to enable remaining Cond 3 variants.
- **Condition 4**: Evaluate strictly native code (Wenyan, Qalb, Citrine).
- **Condition 5**: Cross-lingual transfer experiments.
- **Condition 6**: NL text control to isolate code structure vs language exposure.

---

## FAQ / Discussion Context

Questions and answers that came up during analysis, preserved for team context.

### Why is XNLI not a good benchmark at this model scale?

XNLI uses a fixed set of 3 labels (entailment, contradiction, neutral) for every question. At 3.35B parameters, Tiny Aya can't reliably distinguish all three — it collapses to a binary classifier. Because the labels never change, the model can "score" by always picking one label. This is fundamentally different from benchmarks like CSQA (5 unique answer options per question) or XStoryCloze (2 unique story endings per question), where a fixed bias can't help.

**Better benchmarks for Cycle 2** would include:

- **Belebele** — 4-way reading comprehension where each question has unique answer options. Covers zh, es, ur. The model must actually read the passage since it can't default to one answer.
- **SIB-200** — topic classification across 200+ languages with 7 concrete categories (science, sports, politics, etc.). Concrete labels are easier to learn than the abstract entailment/neutral distinction.
- **XStoryCloze** — already in the eval pipeline. 2-way with unique endings per question. Not vulnerable to label bias.

The eval pipeline already uses CSQA and could use XStoryCloze (it's in `eval_pipeline.py`). XNLI results are still informative — they reveal the model's label bias behavior — but the overall accuracy number should not be interpreted as language understanding.

### Does the Chinese keyword improvement reflect real learning?

The headline finding was: Cond 2-zh improves Chinese XNLI by +6.1pp. But raw output analysis shows the model shifted from outputting Chinese labels (矛盾, 92% of the time) to outputting the English word "entailment" (65% of the time). The accuracy improved because the baseline's extreme contradiction bias (92%) was rebalanced toward a more even split (65/35).

The more accurate claim is: **Chinese keyword fine-tuning changed the distribution of first tokens the model produces in response to Chinese XNLI prompts, shifting from near-total contradiction bias to a more balanced entailment/contradiction split.** This is a real behavioral change, but it's a change in token probability, not evidence of deeper NLI comprehension. Whether the model "understands" entailment vs contradiction, or just learned to output the English word "entailment" more often, is unclear from this data alone.

### How do LoRA adapters create label bias?

The base model (Tiny Aya, 3.35B parameters) already has behavior for every prompt — when given a Chinese XNLI question, it produces a probability distribution over possible next tokens. Before fine-tuning, this distribution favors "contradiction"-associated tokens for Chinese.

The LoRA adapter is a small set of additional weights (~1% of total parameters) added to the base model during inference. It was trained on 5,000 Python files with Chinese keywords. During training, the adapter weights adjusted to make the model better at predicting Chinese code tokens (like `如果` for `if`, `返回` for `return`).

These weight changes modify how the model processes Chinese tokens _generally_ — not just for code. When the adapted model encounters Chinese text in an XNLI prompt, it processes it through the modified pathway, which happens to produce different label token probabilities. The direction of the shift (toward entailment or contradiction) depends on how the keyword tokens overlap with NLI label tokens in the model's internal representation space — it's essentially a side effect of code training, not a deliberate NLI capability.

This is why the effect is language-specific: the Chinese adapter primarily modifies Chinese token processing weights, the Spanish adapter modifies Spanish-related weights, etc. And it's why the bias direction is unpredictable across languages — it depends on which internal representations the code keywords share with the NLI labels.

---

_Generated 2026-03-24 (updated with re-scored XNLI values and raw output analysis). Data source: HuggingFace `legesher/language-decoded-experiments`. Visualization script: `analysis/scripts/plot_condition_comparison.py`._

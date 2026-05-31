# Urdu Code Leakage in XNLI Evaluation — Condition 2-ur

**Date**: 2026-03-28
**Data source**: `legesher/language-decoded-experiments` on HuggingFace
**File**: `conditions/condition-2-ur-5k/results/native_prompt_results.json`
**Dataset**: XNLI-ur (5,010 examples), native Urdu prompts

---

## Summary

After fine-tuning Tiny Aya (3.35B) on 5,000 Python files with Legesher-translated Urdu keywords, **67.6% of all Urdu XNLI outputs contain Legesher Python code**. The model was asked to classify sentence pairs as entailment, contradiction, or neutral — instead, it generates Python function definitions, assert statements, and conditional blocks using the Urdu keywords it learned during fine-tuning.

This leakage is unique to Urdu. Chinese and Spanish conditions show zero code-structure leakage in their XNLI outputs.

---

## What it looks like

The model typically outputs a valid classification label on line 1, then generates Legesher Python code on subsequent lines. Three patterns dominate:

### Pattern 1: `تصدیق` (assert) — 1,524 entries (30.4%)

The model wraps NLI labels inside Urdu assert calls:

```
contradiction
تصدیق(entailment(p
```

That's `assert(entailment(p` — the model is writing a test assertion using the NLI label as a function argument.

Other assert variants:

```
contradiction
تصدیق(entailment)
```

```
contradiction
تصدیق = premise اور hypothesis
```

This last one is `assert = premise and hypothesis` — mixing code assignment syntax with the task's input variables.

### Pattern 2: `اگر` (if) — 1,369 entries (27.3%)

The model writes conditional blocks with proper Python indentation:

```
contradiction
اگر premise اور hypothesis:
    اگر premise
```

That's a nested `if premise and hypothesis: if premise` — complete with colons and indentation.

Another variant checks for the NLI label inside the conditional:

```
contradiction
اگر premise اور hypothesis میں contradiction:
```

`if premise and hypothesis contain contradiction:` — the model is trying to write logic that evaluates the classification.

### Pattern 3: `تعریف` (def) — 486 entries (9.7%)

The model generates full Python function signatures:

```
contradiction
تعریف main():
    premise,
```

```
contradiction
تعریف main():
    premise =
```

That's `def main():` with the XNLI premise as a parameter or variable assignment — a complete function definition instead of a classification.

---

## Frequency of specific patterns

| Pattern                             | Urdu                | English equivalent    | Count |
| ----------------------------------- | ------------------- | --------------------- | ----- |
| `تصدیق(entailment(p`                | assert(entailment(p | `assert(entailment(p` | 763   |
| `تعریف main():\n    premise`        | def main(): premise | Function definition   | 314   |
| `تصدیق\nتصدیق`                      | assert, assert      | Repeated assert       | 306   |
| `تصدیق(entailment)`                 | assert(entailment)  | Assert with label     | 280   |
| Nested `اگر premise... اگر premise` | if... if            | Nested conditionals   | 233   |

---

## Why this matters

1. **The keywords are novel.** Urdu Python keywords did not exist before Legesher created them. `تعریف` for `def`, `تصدیق` for `assert`, `اگر` for `if` — these are Legesher translations. The model learned them from just 5,000 files of training data.

2. **The leakage is language-specific.** It requires all three conditions simultaneously: Urdu keyword fine-tuning + Urdu-script prompt + Urdu evaluation data. Switch to English prompts on the same model — zero leakage. Evaluate on Chinese or Spanish — zero leakage.

3. **Chinese and Spanish show no code leakage.** Raw keyword matching in Chinese XNLI outputs produces false positives (e.g., `假` appears inside `假设`/hypothesis, `如果` is just the everyday Chinese word "if"), but zero code-structure patterns like `定义 func():` appear. Spanish shows only 2 entries with `afirmar` used in natural sentence context. Neither language has the model generating actual code.

4. **The model learned the translations deeply enough that it can't separate them from general Urdu text generation.** When it encounters Urdu text containing words like "premise," it activates the code generation patterns it learned during fine-tuning. This is evidence that the fine-tuning genuinely altered how the model processes Urdu — even 5,000 files is enough to reshape the model's behavior for an underrepresented language.

---

## Self-contradicting outputs

In 1,395 entries (27.8%), the model's first-line label contradicts the label referenced in the code it generates afterward. Almost always the same pattern: "contradiction" on line 1, then `تصدیق(entailment)` on line 2.

### The code "knows" the right answer but the label is wrong (447 entries)

The model predicts "contradiction" on line 1, but the code references "entailment" — and the gold label IS entailment:

```
contradiction
تصدیق(entailment(p
```

Gold: entailment. The Urdu code points to the correct answer, but the English label on line 1 gets it wrong.

### Correct label, contradicting code (494 entries)

The model predicts "contradiction" (matching gold), but the code still references "entailment":

```
contradiction
تصدیق(entailment)
```

Gold: contradiction. The label is right, but the code disagrees.

### Triple mismatch — gold, label, and code all differ (454 entries)

Gold is "neutral," the model predicts "contradiction," and the code references "entailment." All three labels are different:

```
contradiction
تصدیق(entailment)
```

Gold: neutral.

### What this tells us

The self-contradiction is systematic, not random. The model defaults to "contradiction" on line 1 (86.5% of all predictions) and then generates templatic code that references `entailment` as a function name. The code's "line 2 accuracy" (34.2%) is actually _worse_ than the line 1 accuracy (37.7%) — the code is not carrying genuine classification signal. It's formulaic code generation, not an alternative reasoning path.

---

## Urdu fine-tuning suppresses natural Urdu explanation

A striking contrast emerges when comparing Condition 2-ur to other conditions. In the baseline and non-Urdu fine-tuned models, the model generates Urdu explanation text using words like `لازمی` ("necessary," an Urdu paraphrase of entailment) and `لازم آتی ہے` ("it follows"). These are natural Urdu reasoning phrases, not code.

| Condition      | `لازمی` count | `لازم آتی ہے` count | `انضمامیت` count |
| -------------- | ------------- | ------------------- | ---------------- |
| Baseline       | 77            | 58                  | 3                |
| Cond 1 (en 5K) | 368           | 77                  | 3                |
| Cond 2-zh      | 929           | 222                 | 0                |
| Cond 2-es      | 916           | 257                 | 0                |
| **Cond 2-ur**  | **0**         | **0**               | **0**            |
| Cond 3-zh      | 332           | 341                 | 0                |

**Condition 2-ur has zero instances of any Urdu explanation term.** Every other condition produces them — non-Urdu fine-tuning actually _increases_ the frequency (Cond 2-zh: 929 `لازمی` vs. baseline: 77).

The Urdu keyword fine-tuning completely replaced the model's natural Urdu reasoning format with Legesher code patterns. Where other conditions produce explanatory text like `لازمی ہے کہ...` ("it is necessary that..."), Condition 2-ur produces `تصدیق(entailment)`. The model stopped explaining in Urdu and started coding in Urdu.

---

## Other oddities

**Romanized Urdu (22 entries)**: The model occasionally outputs Latin-script Urdu instead of Nastaliq:

- `contradiction\n# Premise: Mai akela hi`
- `contradiction\n\nPremise: Aur ye tha,`

**Looping repetition (2 entries)**: The model gets stuck in a loop:

- `premise اور hypothesis میں premise اور hypothesis میں premise اور...`

---

## Impact on evaluation scores

The code leakage appears _after_ the classification label on line 1, so the first-line extraction used in scoring is not corrupted by the code. However, outputs with code leakage show a strong contradiction bias (86% predict contradiction), and the overall XNLI-ur accuracy for Condition 2-ur is essentially flat vs. baseline (+0.9pp, within noise).

The leakage has a larger impact on CSQA: 14.6% of Urdu CSQA outputs from Condition 2-ur contain code fragments, and contaminated outputs score 25.3% accuracy vs. 41.1% for clean outputs — the code generation actively hurts task performance.

---

## Keyword reference

| Urdu  | Transliteration | Python keyword |
| ----- | --------------- | -------------- |
| تعریف | ta'rif          | `def`          |
| اگر   | agar            | `if`           |
| تصدیق | tasdiq          | `assert`       |
| واپسی | wapsi           | `return`       |
| درآمد | daramad         | `import`       |
| کلاس  | klaas           | `class`        |
| صحیح  | sahih           | `True`         |
| غلط   | ghalat          | `False`        |
| اور   | aur             | `and`          |

---

## How to reproduce — querying the data

### Data source

All results are stored on HuggingFace:

- **Repo**: `legesher/language-decoded-experiments`
- **Repo type**: `dataset`
- **Primary file**: `conditions/condition-2-ur-5k/results/native_prompt_results.json`

### Downloading the data

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="legesher/language-decoded-experiments",
    filename="conditions/condition-2-ur-5k/results/native_prompt_results.json",
    repo_type="dataset",
)

with open(path, encoding="utf-8") as f:
    data = json.load(f)

# XNLI-ur entries are under the "xnli_ur" key
entries = data["xnli_ur"]  # 5,010 entries
```

### JSON entry structure

Each entry has these fields:

```json
{
  "question": "...", // The XNLI prompt (premise + hypothesis)
  "raw_output": "...", // Full model output — this is where leakage lives
  "pred": "contradiction", // Extracted prediction (from first line)
  "gold": "entailment", // Ground truth label
  "correct": false // Whether pred == gold
}
```

### Querying each pattern

**Code leakage — overall count:**

```python
keywords = ["تصدیق", "تعریف", "اگر", "واپسی", "درآمد", "کلاس", "صحیح", "غلط"]
leakage = [e for e in entries if any(kw in e["raw_output"] for kw in keywords)]
print(f"{len(leakage)}/{len(entries)} ({100*len(leakage)/len(entries):.1f}%)")
# → 3,385/5,010 (67.6%)
```

Note: Exclude `یا` (or) and `نہیں` (not) from this list — they are common Urdu words that appear in natural text, not code-specific.

**Pattern 1 — assert (`تصدیق`) entries:**

```python
assert_entries = [e for e in entries if "تصدیق" in e["raw_output"]]
# → 1,524 entries (30.4%)

# Specific sub-patterns:
assert_entailment_p = [e for e in entries if "تصدیق(entailment(p" in e["raw_output"]]  # → 763
assert_entailment = [e for e in entries if "تصدیق(entailment)" in e["raw_output"]]      # → 280
assert_repeated = [e for e in entries if "تصدیق\nتصدیق" in e["raw_output"]]             # → 306
```

**Pattern 2 — if (`اگر`) entries:**

```python
if_entries = [e for e in entries if "اگر" in e["raw_output"]]
# → 1,369 entries (27.3%)

# Nested if pattern:
nested_if = [e for e in entries
             if e["raw_output"].count("اگر") >= 2
             and "premise" in e["raw_output"]]
# → 233 entries
```

**Pattern 3 — def (`تعریف`) entries:**

```python
def_entries = [e for e in entries if "تعریف" in e["raw_output"]]
# → 486 entries (9.7%)

# def main() specifically:
def_main = [e for e in entries if "تعریف main():" in e["raw_output"]]
# → 314 entries
```

**Self-contradicting outputs (line 1 label vs. code label):**

```python
contradicting = []
for e in entries:
    lines = e["raw_output"].strip().split("\n")
    if len(lines) < 2:
        continue
    first_line = lines[0].strip().lower()
    rest = "\n".join(lines[1:])
    # Line 1 says "contradiction" but code references "entailment"
    if "contradiction" in first_line and ("entailment" in rest or "تصدیق" in rest):
        contradicting.append(e)
# → ~1,395 entries (27.8%)

# Subset where code "knows" the right answer:
code_correct = [e for e in contradicting if e["gold"] == "entailment"]
# → 447 entries
```

**Urdu explanation terms (compare across conditions):**

```python
# Download each condition's native_prompt_results.json, then:
conditions = [
    "baseline", "condition-1-en-5k", "condition-2-zh-5k",
    "condition-2-es-5k", "condition-2-ur-5k", "condition-3-zh-5k",
]

for cond in conditions:
    # ... load data[cond]["xnli_ur"] ...
    lazmi = sum(1 for e in entries if "لازمی" in e["raw_output"])
    lazim = sum(1 for e in entries if "لازم آتی ہے" in e["raw_output"])
    print(f"{cond}: لازمی={lazmi}, لازم آتی ہے={lazim}")
```

Note: The baseline file path pattern may differ — try `conditions/baseline/results/native_prompt_results.json` or `baseline/native_prompt_results.json`.

**CSQA code contamination:**

```python
# Same download, but look at csqa_ur entries:
csqa_entries = data["csqa_ur"]
keywords = ["تصدیق", "تعریف", "اگر", "واپسی", "درآمد", "کلاس"]
contaminated = [e for e in csqa_entries if any(kw in e["raw_output"] for kw in keywords)]
clean = [e for e in csqa_entries if not any(kw in e["raw_output"] for kw in keywords)]

cont_acc = sum(1 for e in contaminated if e["correct"]) / len(contaminated)
clean_acc = sum(1 for e in clean if e["correct"]) / len(clean)
print(f"Contaminated: {cont_acc:.1%} ({len(contaminated)} entries)")
print(f"Clean: {clean_acc:.1%} ({len(clean)} entries)")
# → Contaminated: 25.3%, Clean: 41.1%
```

**Chinese/Spanish leakage check (confirming no code structures):**

```python
# For zh — these produce false positives from natural language:
zh_entries = data["xnli_zh"]
# Don't count 假 (appears in 假设/hypothesis) or 如果 (common Chinese "if")
# Instead look for code structures:
zh_code = [e for e in zh_entries if "定义 " in e["raw_output"]  # "def " with space
           or "如果 " in e["raw_output"] and ":\n" in e["raw_output"]]
# → 0 entries with actual code structures
```

---

_Data verified against HuggingFace source on 2026-03-28. All pattern counts confirmed against raw_output fields in the XNLI-ur evaluation results._

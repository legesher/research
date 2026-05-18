# Evaluation Pipeline

Per-condition benchmark evaluation across 4 dataset languages (en, zh, es, ur) and 4 benchmarks, with prompt-template ablation and per-condition instruction-language matrices.

## Contents

- `scripts/preprocess.ipynb` — Loads + caches benchmark datasets and tokenized prompts (Kaggle artifact)
- `scripts/evaluate.ipynb` — Runs the eval suite against a configured condition (dual-GPU, template-split)
- `scripts/rescore_xnli.py` — One-time XNLI re-scoring correction script
- `requirements.txt` — Python dependencies

Results are stored on HuggingFace, not in this directory.

## Benchmark Suite

| Benchmark    | What It Measures                                                  | Dataset Languages |
| ------------ | ----------------------------------------------------------------- | ----------------- |
| **XNLI**     | Natural language inference (entailment / contradiction / neutral) | en, zh, es, ur    |
| **X-CSQA**   | Commonsense reasoning (5-way multiple choice)                     | en, zh, es, ur    |
| **SIB-200**  | Topic classification (7-way concrete categories)                  | en, zh, es, ur    |
| **Belebele** | Reading comprehension (4-way multiple choice)                     | en, zh, es, ur    |

## Per-Condition Eval Matrix

The preprocessing notebook caches all 4 dataset languages × 4 instruction-language prompts × 2 templates per row. The eval notebook then selects a subset per condition:

| Condition                             | Dataset langs  | Instruction langs | Cells per model |
| ------------------------------------- | -------------- | ----------------- | --------------- |
| baseline (no FT)                      | en, zh, es, ur | en, zh, es, ur    | **128**         |
| condition-1-en-5k (English code)      | en, zh, es, ur | en, zh, es, ur    | **128**         |
| condition-2-zh-5k                     | en, zh, es, ur | en, zh            | **64**          |
| condition-2-es-5k                     | en, zh, es, ur | en, es            | **64**          |
| condition-2-ur-5k                     | en, zh, es, ur | en, ur            | **64**          |
| condition-3 / condition-5 (per-lang)¹ | en, zh, es, ur | en, L_train       | **64**          |

Cells = 4 benchmarks × 2 templates × dataset-langs × instruction-langs.

¹ Planned — not yet registered in `EVAL_MATRIX`. Adapters land per language; the registry entry is a one-line addition once each adapter is published.

**Rule for condition-2/3/5:** instruction language is always English OR the model's trained-on language, regardless of which dataset language we're evaluating against. A condition-2-ur model evaluated on `xnli_zh` gets either English or Urdu instructions, never Chinese — the model wasn't trained to follow Chinese instructions in that condition.

## Prompt Templates

The preprocessing notebook caches two prompt templates for ablation tests:

| Template    | Benchmarks                      | Notes                                               |
| ----------- | ------------------------------- | --------------------------------------------------- |
| `template1` | Belebele, SIB-200, X-CSQA, XNLI | Baseline wording                                    |
| `template2` | Belebele, SIB-200, X-CSQA, XNLI | Light rephrasing of template1 for sensitivity check |

Each template has English, Spanish, Urdu, and Chinese wording for the same benchmark structure. Template 2 is a slight natural rephrasing of Template 1, intended to test whether small prompt wording changes affect model performance.

Prompts are cached per-instruction-language as columns: `prompt_{template_id}_{instruction_lang}` plus matching `input_ids_*` / `attention_mask_*`. The eval notebook reads only the instruction-language subset configured for each condition.

## Notebooks

### Preprocessing (`preprocess.ipynb`)

Loads each benchmark in all 4 dataset languages, computes prompts in all 4 instruction languages × 2 templates, tokenizes, and saves to `eval_unsloth_artifacts/datasets/{benchmark}_{lang}.jsonl`. Run this once per Kaggle dataset refresh; the eval notebook reads the cached JSONL.

> **Schema:** if you have cached JSONLs from an earlier version of this pipeline (PR #37 schema with `english`/`language` suffixes), delete `eval_unsloth_artifacts/datasets/*.jsonl` and rerun preprocess before running evaluate.

### Evaluation (`evaluate.ipynb`)

Set `CONDITION` in the condition-picker cell before running:

```python
CONDITION = "condition-2-ur-5k"  # Edit before running
```

Available conditions (see `EVAL_MATRIX` in `run_eval_single.py` for the source of truth):

- `baseline` — `CohereLabs/tiny-aya-base`
- `condition-1-en-5k` — `legesher/language-decoded-lora-condition-1-en-5k`
- `condition-2-zh-5k`, `condition-2-es-5k`, `condition-2-ur-5k`

The dual-GPU launch splits by template:

- GPU 0 → `--template template1`
- GPU 1 → `--template template2`

Each subprocess writes:

- `/kaggle/working/{condition}_summary_{template}.json` — accuracies + parse-failure rates
- `/kaggle/working/{condition}_results_{template}.json` — full per-row outputs
- `/kaggle/working/{condition}_partial_{template}.json` — incremental checkpoint (updates after each `(template, dataset_lang, instruction_lang)` block; survives mid-run crashes)

Base model: `CohereLabs/tiny-aya-base`

## Data Sources

| Benchmark | Source                                                                                                                      |
| --------- | --------------------------------------------------------------------------------------------------------------------------- |
| XNLI      | [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) — config `en`/`zh`/`es`/`ur`                                 |
| X-CSQA    | [INK-USC/xcsr](https://huggingface.co/datasets/INK-USC/xcsr) — config `X-CSQA-{lang}`                                       |
| SIB-200   | [mteb/sib200](https://huggingface.co/datasets/mteb/sib200) — filter by FLORES lang code                                     |
| Belebele  | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) — config `eng_Latn`/`zho_Hans`/`spa_Latn`/`urd_Arab` |

## Results

All results are stored on HuggingFace:

| Repo                                                                                                  | Contents                                                        |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Per-condition results (one summary + results JSON per template) |

See [analysis/evaluation-summary.md](../analysis/evaluation-summary.md) for the full analysis.

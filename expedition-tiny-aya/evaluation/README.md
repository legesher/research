# Evaluation Pipeline

Benchmark evaluation across 3 languages (zh, es, ur), using 4 benchmark suites.
Each condition is evaluated twice: with English prompts and with native-language prompts.

## Contents

- `scripts/baseline_benchmarking.ipynb` — Baseline evaluation (no adapter)
- `scripts/finetuned-benchmarking.ipynb` — Fine-tuned adapter evaluation
- `scripts/eval_pipeline.py` — Reusable CLI evaluation runner
- `scripts/rescore_xnli.py` — One-time XNLI re-scoring correction script
- `requirements.txt` — Python dependencies

Results are stored on HuggingFace, not in this directory.

## Benchmark Suite

| Benchmark  | What It Measures                                                  | Languages      |
| ---------- | ----------------------------------------------------------------- | -------------- |
| **XNLI**   | Natural language inference (entailment / contradiction / neutral) | zh, es, ur |
| **X-CSQA** | Commonsense reasoning (5-way multiple choice)                     | zh, es, ur |
| **SIB-200** | Topic classification (7-way concrete categories)                 | zh, es, ur |
| **Belebele** | Reading comprehension (4-way multiple choice)                   | zh, es, ur |

English prompts provide a cross-lingual instruction baseline for each dataset language.

## Prompt Templates

The preprocessing notebook now caches two prompt templates for ablation tests:

| Template | Benchmarks |
| -------- | ---------- |
| `template1` | Belebele, SIB-200, X-CSQA, XNLI |
| `template2` | Belebele, SIB-200, X-CSQA, XNLI |

Each template has English, Spanish, Urdu, and Chinese wording for the same benchmark structure. Template 2 is a slight natural rephrasing of Template 1, intended to test whether small prompt wording changes affect model performance.

## Notebooks

### Baseline (`baseline_benchmarking.ipynb`)

Evaluates the raw Tiny Aya base model (no fine-tuning). Run once to establish floor scores.

### Fine-tuned (`finetuned-benchmarking.ipynb`)

Set `cond` in Cell 1 before running. Loads adapter from the unified lora repo:

```python
cond = "condition-1-en-5k"  # <-- SET BEFORE RUNNING
```

Available conditions: `condition-1-en-32k`, `condition-1-en-5k`, `condition-2-zh-5k`,
`condition-2-es-5k`, `condition-2-ur-5k`, `condition-3-zh-5k`

Results are uploaded directly to HuggingFace (`legesher/language-decoded-experiments`).

## CLI Script (`eval_pipeline.py`)

Reusable evaluation runner for batch evaluation.

```bash
python eval_pipeline.py \
  --adapter-path legesher/language-decoded-lora \
  --subfolder condition-2-zh-5k \
  --language zh \
  --benchmarks xnli xstorycloze \
  --output-file results.json
```

Arguments:

- `--adapter-path` — HF Hub id or local path. Repeat for batch mode.
- `--subfolder` — Subfolder within adapter repo (for unified lora repo)
- `--language` — Language code: `zh`, `es`, `ur`, or `en`
- `--benchmarks` — Benchmark names to run, or `all`
- `--output-file` — Path to JSON results file

Base model: `CohereLabs/tiny-aya-base`

## Data Sources

| Benchmark         | Source                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| XNLI              | [facebook/xnli](https://huggingface.co/datasets/facebook/xnli)                                       |
| X-CSQA            | [INK-USC/xcsr](https://huggingface.co/datasets/INK-USC/xcsr)                                         |
| SIB-200           | [mteb/sib200](https://huggingface.co/datasets/mteb/sib200)                                           |
| Belebele          | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele)                               |

## Results

All results are stored on HuggingFace:

| Repo                                                                                                  | Contents                                              |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Per-condition results (english + native prompt JSONs) |

See [analysis/evaluation-summary.md](../analysis/evaluation-summary.md) for the full analysis.

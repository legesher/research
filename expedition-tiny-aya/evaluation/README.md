# Evaluation Pipeline

**Owner:** Saad (crew:saad)

Benchmark evaluation across all conditions and languages.

## Contents

- `scripts/` — Benchmark runners and cross-condition comparison
- `results/` — Per-condition benchmark scores
  - `baseline/` — Pre-training scores (Tiny Aya base)
  - `condition-1/` through `condition-4/` — Post-training scores
- `configs/` — Benchmark suite configuration

## Benchmark Suite

| Benchmark | What It Measures | Languages |
| --- | --- | --- |
| XNLI | Natural language inference | zh, am, ur |
| XStoryCloze | Commonsense reasoning | zh, am, ur |
| TyDi QA | Question answering | zh, am, ur |
| MMLU | World knowledge | zh, am, ur |
| MultiNRC | Culturally-grounded comprehension | zh, am, ur |
| AI4Math | Mathematical reasoning | zh, am, ur |
| HumanEval/MBPP | Code generation (secondary) | zh, am, ur |

## Suggested Entrypoints

## Evaluation Script
# `eval_pipeline.py`

Reusable evaluation runner for Tiny Aya adapters.

### What it does

- Loads the base model from Hugging Face: `CohereForAI/tiny-aya-base`
- Loads a PEFT / QLoRA adapter from a local path or HF Hub
- Merges the adapter into the base model for inference
- Runs selected benchmarks for one language
- Writes all results to a JSON file
- Supports batch mode by repeating `--adapter-path`
- Tracks compute time per benchmark

### Inputs

```bash
python eval_pipeline.py \
  --adapter-path path/to/adapter \
  --language zh \
  --benchmarks xnli xstorycloze tydiqa mmlu \
  --output-file ../results/condition-2/zh.json
```

Arguments:

- `--adapter-path`: adapter checkpoint path or HF Hub id. Repeat this flag to evaluate multiple adapters in sequence.
- `--language`: language code such as `zh`, `ur`, `am`, or `en`
- `--benchmarks`: benchmark names to run, or `all`
- `--output-file`: path to the JSON results file

### Supported Benchmarks

- `xnli`: returns `accuracy`
- `xstorycloze`: returns `accuracy`
- `tydiqa`: returns `f1` and `em`
- `mmlu`: returns `accuracy_per_subject`

`multinrc` and `ai4math` are not implemented yet.

### Batch Mode

```bash
python eval_pipeline.py \
  --adapter-path adapter-one \
  --adapter-path adapter-two \
  --language ur \
  --benchmarks all \
  --output-file ../results/condition-4/ur-batch.json
```

The output JSON contains one entry per adapter under `runs`.

### Output Format

The script writes a JSON file with:

- `base_model`
- `language`
- `benchmarks`
- `runs`

Each run contains:

- `adapter_path`
- `language`
- `benchmarks`
- `timing_seconds`

### Notes

- The script assumes the fine-tuned artifact is an adapter checkpoint, not a fully merged standalone model.
- The base model is always loaded from Hugging Face.
- `mmlu` is currently wired to the default public English dataset, so non-English MMLU will need a translated dataset later.
- Public datasets may not fully match every project language, especially for translated benchmark variants.

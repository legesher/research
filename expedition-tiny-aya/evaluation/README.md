# Evaluation Pipeline

**Owner:** Khojasteh

Benchmark evaluation across 3 langauges and 3 datasets. There are 2 ways to evaluate:
1: Dataset in chosen language + english prompt
2: Dataset in chosen langugae + prompt in chosen langugae

## Contents

- `scripts/baseline_benchmarking.ipynb` — Benchmark running notebook
- `results/` — Per-condition benchmark scores
  - `baseline/` — Pre-training scores (Tiny Aya base)
  - `condition-1/` through `condition-4/` — Post-training scores
- `configs/` — Benchmark suite configuration
- `README` - You are here
- `requirements.txt` - Requirements to run the evaluations

## Benchmark Suite

| Benchmark | What It Measures | es | zh | ur
| --- | --- | --- | --- | --- |
| MGSM | Multilingual math reasoning on grade-school word problems; evaluates whether the model can reason through quantitative problems and produce the correct final numeric answer language inference | ✓ | ✓ | ✓ |
| XNLI | Multilingual natural language inference; evaluates whether the model can determine if a hypothesis is entailed by, contradicted by, or neutral with respect to a premise | ✓ | ✓ | ✓ |
| X-CSQA | Multilingual commonsense reasoning in multiple-choice format; evaluates whether the model can choose the most plausible answer based on everyday world knowledge | ✓ | ✓ | ✓ |

## Suggested Entrypoints # to edit based on draft pr 

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

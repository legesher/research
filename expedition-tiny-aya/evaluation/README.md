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

## Benchmark Suite

| Benchmark | What It Measures | es | zh | ur
| --- | --- | --- |
| MGSM | Multilingual math reasoning on grade-school word problems; evaluates whether the model can reason through quantitative problems and produce the correct final numeric answer language inference | ✓ | ✓ | ✓ |
| XNLI | Multilingual natural language inference; evaluates whether the model can determine if a hypothesis is entailed by, contradicted by, or neutral with respect to a premise | ✓ | ✓ | ✓ |
| X-CSQA | Multilingual commonsense reasoning in multiple-choice format; evaluates whether the model can choose the most plausible answer based on everyday world knowledge | ✓ | ✓ | ✓ |

## Suggested Entrypoints # to edit based on draft pr 

These scripts are suggested starting points — adapt as needed:

```bash
# Run all benchmarks for a condition
python scripts/run_benchmarks.py --adapter path/to/adapter --lang zh

# Compare across conditions
python scripts/compare_conditions.py --results-dir results/
```
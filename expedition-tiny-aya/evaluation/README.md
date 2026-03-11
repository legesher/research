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

## Quick Start

```bash
# Run all benchmarks for a condition
python scripts/run_benchmarks.py --adapter path/to/adapter --lang zh

# Compare across conditions
python scripts/compare_conditions.py --results-dir results/
```
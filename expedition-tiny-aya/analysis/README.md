# Analysis & Visualization

**Owner:** Saad (crew:saad)

Jupyter notebooks for deep analysis and figure generation.

## Directory layout

- [`phase-3/`](phase-3/) — Phase 3 evaluation analysis (refined-extractor evaluation, decision ledger, surface-form reviews, parser methodology, cond-5 idx ledger). The paper draws from this content.
- `figures/` — Generated charts referenced in the paper.
- `notebooks/` — Jupyter notebooks for ad-hoc analysis.
- `scripts/` — Python scripts that generate the analysis artefacts.
- Top-level `.md` files (`evaluation-summary.md`, `stack-dataset-non-english-analysis.md`, `urdu-code-leakage-analysis.md`) — Phase 2-era analysis, preserved for historical context.

## Suggested Notebooks

| Suggested Name | Analysis |
| --- | --- |
| `cross_lingual_transfer.ipynb` | Does code in one language help related languages? |
| `per_token_efficiency.ipynb` | Improvement per token across conditions |
| `condition_comparison.ipynb` | Side-by-side condition results |
| `native_vs_transpiled.ipynb` | Native code vs. transpiled code impact |

## Figures

Generated charts and plots are saved to `figures/` and referenced in the paper.

## Setup

```bash
pip install jupyter matplotlib seaborn pandas
jupyter notebook notebooks/
```
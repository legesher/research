# Analysis & Visualization

**Owner:** Saad (crew:saad)

Jupyter notebooks for deep analysis and figure generation.

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
# Analysis & Visualization

**Owner:** Saad (crew:saad)

Jupyter notebooks for deep analysis and figure generation.

## Directory layout

- [`phase-2/`](phase-2/) — Phase 2 (hackathon) analysis docs (`evaluation-summary.md`, `stack-dataset-non-english-analysis.md`, `urdu-code-leakage-analysis.md`) plus a [`figures/`](phase-2/figures/) subfolder of Phase 2 charts (english/native prompt bars, heatmaps, prompt comparisons). Preserved for historical context.
- [`phase-3/`](phase-3/) — Phase 3 evaluation analysis (refined-extractor evaluation, decision ledger, surface-form reviews, parser methodology, cond-5 idx ledger, constant-output findings, spot-checks). The paper draws from this content.
- `figures-phase3/` — Phase 3 paper figures (extractor slopegraphs, cell scatter, sign-flip slopegraphs, cond-5 rehabilitation).
- `notebooks/` — Jupyter notebooks for ad-hoc analysis.
- `scripts/` — Python scripts that generate the analysis artefacts (Phase 2: `plot_condition_comparison.py`, `analyze_stack_languages.py`; Phase 3: `fig0*.py`, `build_phase3_tables.py`, `_viz_common.py`).
- `WHEN_REPORTED_NUMBERS_CHANGE.md` — cross-phase number-reconciliation guide.

## Suggested Notebooks

| Suggested Name | Analysis |
| --- | --- |
| `cross_lingual_transfer.ipynb` | Does code in one language help related languages? |
| `per_token_efficiency.ipynb` | Improvement per token across conditions |
| `condition_comparison.ipynb` | Side-by-side condition results |
| `native_vs_transpiled.ipynb` | Native code vs. transpiled code impact |

## Figures

Phase 2 charts live in `phase-2/figures/`; Phase 3 paper figures in `figures-phase3/`. Both are referenced in the paper.

## Setup

```bash
pip install jupyter matplotlib seaborn pandas
jupyter notebook notebooks/
```
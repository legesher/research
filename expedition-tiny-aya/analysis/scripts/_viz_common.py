"""Shared utilities for Phase-3 paper figures (EMNLP submission).

Conventions locked in (see memory: paper-viz-conventions):
- EMNLP/ACL two-column: ~3.15" single, ~6.30" double; 8-10pt fonts.
- scienceplots `science` style (no LaTeX) + Times serif body.
- Okabe-Ito categorical, cividis sequential, RdBu_r diverging with TwoSlopeNorm.
- Language order: en -> es -> zh -> ur (resource tier; high -> low).
- ISO codes on tick labels; native script (سیاست/科学) only in captions.
- PDF + PNG output, pdf.fonttype=42 (TrueType, editable in Illustrator).
- Color cannot be the sole carrier of meaning (ACLPUB grayscale rule).

Tables source:
- HuggingFace by default — pulls `phase3/analysis/refined-tables/*.tsv` from
  `legesher/language-decoded-experiments` on each call. Files are cached by
  huggingface_hub under ~/.cache/huggingface/ so subsequent runs are fast.
- Override via --tables-dir on each fig script to read from a local directory
  of TSVs (useful for testing changes against a local build before pushing to
  HF, or for fully offline reproducibility).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# ─── Palettes ────────────────────────────────────────────────────────────────

# Okabe-Ito colorblind-safe categorical palette (Nature Methods standard).
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Direction-encoded subset (used across all phase-3 figures).
COLOR_POSITIVE = OKABE_ITO["bluish_green"]   # condition beats baseline
COLOR_NEGATIVE = OKABE_ITO["vermillion"]     # baseline beats condition
COLOR_NEUTRAL = "#999999"                    # |Δ| < noise floor
COLOR_FLIP_W2L = OKABE_ITO["vermillion"]     # win -> loss (the four §8.3 flips)
COLOR_FLIP_L2W = OKABE_ITO["blue"]           # loss -> win (rare, marginal)
COLOR_DEFLATED = OKABE_ITO["orange"]         # same sign, smaller magnitude

# ─── Ordering / labels ───────────────────────────────────────────────────────

# Resource-tier language order (high -> low). Used everywhere a language axis
# appears so the "extractor lift correlates with script distance" story
# (phase3-reparse-evaluation.md §3.2) is visible without callouts.
LANG_ORDER = ["en", "es", "zh", "ur"]
LANG_FULL = {"en": "English", "es": "Spanish", "zh": "Chinese", "ur": "Urdu"}

BENCH_ORDER = ["sib200", "xnli", "csqa", "belebele"]
BENCH_LABEL = {
    "sib200": "SIB-200",
    "xnli": "XNLI",
    "csqa": "X-CSQA",
    "belebele": "Belebele",
}

# Display labels for conditions. Multi-line because some lists are tight.
CONDITION_LABEL = {
    "baseline": "Baseline",
    "condition-1-en-5k": "C1 en/5k",
    "condition-1-en-20k": "C1 en/20k",
    "condition-2-es-5k": "C2 es/5k",
    "condition-2-es-20k": "C2 es/20k",
    "condition-2-zh-5k": "C2 zh/5k",
    "condition-2-zh-20k": "C2 zh/20k",
    "condition-2-ur-5k": "C2 ur/5k",
    "condition-2-ur-20k": "C2 ur/20k",
    "condition-3-zh-5k": "C3 zh/5k",
    "condition-5-es-5k": "C5 es/5k",
    "condition-5-ur-5k": "C5 ur/5k",
    "condition-5-zh-5k": "C5 zh/5k",
}

# ACL/EMNLP column widths in inches.
WIDTH_SINGLE = 3.15
WIDTH_DOUBLE = 6.30

# ─── Style setup ─────────────────────────────────────────────────────────────


def setup_style() -> None:
    """Apply EMNLP-appropriate matplotlib rcParams.

    Tries scienceplots `science` style; falls back to a manual config if the
    package isn't installed (so the script still runs in a stripped env).
    """
    try:
        import scienceplots  # noqa: F401 — registers styles on import
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ─── Color helpers ───────────────────────────────────────────────────────────


def diverging_norm(values: Iterable[float], quantile: float = 0.95) -> TwoSlopeNorm:
    """TwoSlopeNorm centered at 0, clipped to ±q.

    Prevents long-tail Δ values (e.g. the +0.205 cond-2-ur-5k cell) from
    saturating the colormap and washing out smaller real effects.
    """
    abs_q = float(np.nanquantile(np.abs(list(values)), quantile))
    if not np.isfinite(abs_q) or abs_q == 0:
        abs_q = 1e-3
    return TwoSlopeNorm(vcenter=0.0, vmin=-abs_q, vmax=+abs_q)


def sign_color(delta_orig: float, delta_rep: float, noise_floor: float = 0.005) -> str:
    """Color a line/marker by the orig→rep transition.

    Used by slopegraphs to make sign-flips visually distinct from sign-stable
    deflations. Noise floor of 0.5pp keeps near-zero cells from being
    misclassified as flips.
    """
    if abs(delta_orig) < noise_floor and abs(delta_rep) < noise_floor:
        return COLOR_NEUTRAL
    if delta_orig > 0 and delta_rep < 0:
        return COLOR_FLIP_W2L
    if delta_orig < 0 and delta_rep > 0:
        return COLOR_FLIP_L2W
    if (delta_orig > 0) == (delta_rep > 0):
        # Same sign; deflation if |rep| < |orig|, growth otherwise.
        return COLOR_DEFLATED if abs(delta_rep) < abs(delta_orig) else (
            COLOR_POSITIVE if delta_rep > 0 else COLOR_NEGATIVE
        )
    return COLOR_NEUTRAL


# ─── IO ──────────────────────────────────────────────────────────────────────

HF_REPO_ID = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"
HF_TABLES_PREFIX = "phase3/analysis/refined-tables"


def load_table(name: str, tables_dir: Path | str | None = None) -> pd.DataFrame:
    """Load a refined-tables TSV by stem name.

    Args:
        name: e.g. "vs_baseline_cells", "conclusion_flips" (no .tsv extension).
        tables_dir: If set, read from this local directory instead of HF.
            Useful for testing changes against a local build before pushing.
            If None (default), pull from HF and let huggingface_hub cache the file.
    """
    if tables_dir is not None:
        path = Path(tables_dir) / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"TSV not found at local override path: {path}"
            )
        return pd.read_csv(path, sep="\t")

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=f"{HF_TABLES_PREFIX}/{name}.tsv",
    )
    return pd.read_csv(cached, sep="\t")


def figures_out_dir() -> Path:
    """Resolve and create the figures-phase3 output directory.

    Lives at expedition-tiny-aya/analysis/figures-phase3/ — sibling to the
    existing Phase-2 figures/ directory so the two cohorts stay separable.
    """
    out = Path(__file__).resolve().parent.parent / "figures-phase3"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write fig as both PDF (paper) and PNG (blog/preview)."""
    out = figures_out_dir()
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=200)

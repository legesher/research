"""Figure 3: SIB-200 regression concentration under refined scoring.

Single claim: of the SIB-200 cells that regressed (Δacc < 0) under the
extractor refinement, the vast majority live in one condition —
``condition-2-ur-5k`` (38 cells, a mechanical Rule-A correction). The
remaining regressions scatter thinly across the rest of the matrix and do
not indicate a systematic methodology problem.

Layout:
- (a) Horizontal bar chart of regression counts per condition. The
  cond-2-ur-5k bar is highlighted in vermillion; all others are gray.
- (b) Cell grid (heatmap) for the cond-2-ur-5k regression cells only,
  rows = (data → instr) tuples, cols = template, color = mean Δacc.

Run:
    python fig03_regression_concentration.py [--tables-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from _viz_common import (
    COLOR_NEUTRAL,
    CONDITION_LABEL,
    LANG_ORDER,
    OKABE_ITO,
    WIDTH_DOUBLE,
    diverging_norm,
    load_table,
    save_figure,
    setup_style,
)

ANOMALY_CONDITION = "condition-2-ur-5k"
TEMPLATES = ["t1", "t2"]


def regression_counts(df: pd.DataFrame) -> pd.Series:
    """Per-condition count of SIB-200 cells with Δacc < 0 (raw rows)."""
    reg = df[(df.benchmark == "sib200") & (df.delta_acc < 0)]
    counts = reg.groupby("condition").size().sort_values(ascending=False)
    return counts


def anomaly_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], int]:
    """Build the (data→instr) × template matrix of mean Δacc for the anomaly.

    Only cells that regressed (Δacc < 0) are populated; the rest are NaN
    (rendered white). Multiple seeds collapse to mean.
    """
    sub = df[
        (df.condition == ANOMALY_CONDITION)
        & (df.benchmark == "sib200")
        & (df.delta_acc < 0)
    ].copy()
    # Template comes through as int (1, 2); render as "t1"/"t2" for ticks.
    sub["template_key"] = "t" + sub["template"].astype(int).astype(str)

    # Mean across seeds.
    agg = (
        sub.groupby(["data", "instr", "template_key"], as_index=False)["delta_acc"]
        .mean()
    )

    # Build row index in LANG_ORDER × LANG_ORDER, dropping rows with no data
    # so the heatmap doesn't have all-white rows. Outer = data, inner = instr.
    pairs: list[tuple[str, str]] = []
    for d in LANG_ORDER:
        for i in LANG_ORDER:
            if ((agg.data == d) & (agg.instr == i)).any():
                pairs.append((d, i))
    row_labels = [f"{d}→{i}" for d, i in pairs]

    matrix = np.full((len(pairs), len(TEMPLATES)), np.nan, dtype=float)
    for (d, i), row_ix in zip(pairs, range(len(pairs))):
        for col_ix, t in enumerate(TEMPLATES):
            cell = agg[(agg.data == d) & (agg.instr == i) & (agg.template_key == t)]
            if len(cell):
                matrix[row_ix, col_ix] = float(cell.delta_acc.iloc[0])

    n_cells = int(len(sub))  # raw count of regressed rows (pre seed-collapse)
    return matrix, row_labels, n_cells


def draw_bar_panel(ax: plt.Axes, counts: pd.Series) -> None:
    labels = [CONDITION_LABEL.get(c, c) for c in counts.index]
    colors = [
        OKABE_ITO["vermillion"] if c == ANOMALY_CONDITION else COLOR_NEUTRAL
        for c in counts.index
    ]
    # barh draws bottom-to-top; reverse so the largest count appears on top.
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, color=colors, edgecolor="none", height=0.7)
    for yi, v in zip(y, counts.values):
        ax.text(v + 0.6, yi, f"{int(v)}", va="center", ha="left", fontsize=7.5,
                color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlim(0, max(40, counts.max() * 1.15))
    ax.set_xlabel("Regressed cells")
    ax.set_title(
        f"(a) SIB-200 regression counts (n={int(counts.sum())})",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=2,
    )
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(False)
    # Drop the y-axis tick marks; labels alone are enough.
    ax.tick_params(axis="y", length=0)


def draw_heatmap_panel(
    ax: plt.Axes, matrix: np.ndarray, row_labels: list[str]
) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    norm = diverging_norm(finite, quantile=0.95)
    # Mask NaN so they render as the axes background (white).
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations in pp; choose text color from background for contrast.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            # Light text on dark blue/red ends; dark text on near-zero whites.
            rgba = cmap(norm(v))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "white" if luminance < 0.45 else "#1a1a1a"
            ax.text(j, i, f"{v * 100:+.1f}", ha="center", va="center",
                    fontsize=6.6, color=txt_color)

    ax.set_xticks(range(len(TEMPLATES)))
    ax.set_xticklabels(TEMPLATES, fontsize=7.8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.4)
    ax.set_title(
        "(b) C2 ur/5k mean $\\Delta$acc",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=2,
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Slim colorbar to the right of the heatmap. Explicit ticks because a
    # TwoSlopeNorm with one-sided data doesn't auto-tick reliably.
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("$\\Delta$acc (pp)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8)
    abs_q = max(abs(norm.vmin), abs(norm.vmax))
    tick_vals = np.array([-abs_q, -abs_q / 2, 0.0, abs_q / 2, abs_q])
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t * 100:+.0f}" for t in tick_vals])

    return float(np.nanmin(matrix)), float(np.nanmax(matrix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_style()
    df = load_table("cells", tables_dir=args.tables_dir)

    counts = regression_counts(df)
    matrix, row_labels, n_anomaly = anomaly_matrix(df)
    total = int(counts.sum())

    print(f"[fig03] total SIB-200 regressed cells: {total}")
    print(f"[fig03] per-condition counts:")
    for cond, n in counts.items():
        print(f"[fig03]   {CONDITION_LABEL.get(cond, cond):<14} {int(n)}")
    print(f"[fig03] cond-2-ur-5k heatmap cells (raw rows): {n_anomaly}")
    print(
        f"[fig03] heatmap Δacc min={np.nanmin(matrix) * 100:+.2f} pp, "
        f"max={np.nanmax(matrix) * 100:+.2f} pp"
    )

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.5))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.7, 1], wspace=0.45, figure=fig
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    draw_bar_panel(ax_bar, counts)
    draw_heatmap_panel(ax_heat, matrix, row_labels)

    # tight_layout fights with the colorbar pinned to the heatmap axes; use
    # explicit margins instead so the two panels stay visually balanced.
    fig.subplots_adjust(left=0.13, right=0.95, top=0.92, bottom=0.13)
    save_figure(fig, "fig03_regression_concentration")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Figure 1: extractor-effect slopegraph (SIB-200 vs XNLI).

Single claim: the original extractor inflated condition-vs-baseline gains on
SIB-200; several cells flip from win to loss under the refined scorer. XNLI is
much flatter — most movement there is deflation, not sign-flip.

Grain: aggregate (condition × benchmark × instr_lang); mean Δ across
(seed, template, data). 28 lines per panel — well under the 40-line fallback
threshold, so we keep the instr_lang facet.

Run:
    python fig01_extractor_slopegraph.py [--tables-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from _viz_common import (
    COLOR_DEFLATED,
    COLOR_FLIP_L2W,
    COLOR_FLIP_W2L,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    CONDITION_LABEL,
    LANG_ORDER,
    WIDTH_DOUBLE,
    load_table,
    save_figure,
    setup_style,
    sign_color,
)

PANELS = [("sib200", "(a) SIB-200"), ("xnli", "(b) XNLI")]
X_LEFT, X_RIGHT = 0.0, 1.0
FLIP_COLORS = {COLOR_FLIP_W2L, COLOR_FLIP_L2W}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ-vs-baseline per (condition × benchmark × instr_lang)."""
    sub = df[df.benchmark.isin([p[0] for p in PANELS])].copy()
    agg = sub.groupby(["condition", "benchmark", "instr"], as_index=False).agg(
        delta_orig=("delta_orig", "mean"),
        delta_rep=("delta_rep", "mean"),
    )
    # Convert to percentage points for readability on the y-axis.
    agg["delta_orig"] *= 100.0
    agg["delta_rep"] *= 100.0
    agg["color"] = agg.apply(lambda r: sign_color(r.delta_orig / 100.0, r.delta_rep / 100.0), axis=1)
    agg["is_flip"] = agg["color"].isin(FLIP_COLORS)
    # Stable language ordering for any later iteration / labeling.
    agg["instr"] = pd.Categorical(agg["instr"], categories=LANG_ORDER, ordered=True)
    return agg.sort_values(["benchmark", "instr", "condition"]).reset_index(drop=True)


def short_cond(name: str) -> str:
    """'condition-2-ur-5k' -> 'C2 ur/5k' via the shared CONDITION_LABEL dict."""
    return CONDITION_LABEL[name]


def _scale_pp(ymin: float, ymax: float) -> int:
    """Round the half-range to a clean integer in percentage points."""
    half = max(abs(ymin), abs(ymax))
    if half <= 12:
        step = 1
    elif half <= 30:
        step = 5
    else:
        step = 5
    return int(step * round(half / step))


def draw_panel(ax: plt.Axes, panel_df: pd.DataFrame, panel_label: str, n_annotate: int = 4) -> int:
    # Lines + endpoint markers.
    for row in panel_df.itertuples():
        lw = 1.4 if row.is_flip else 0.8
        alpha = 0.95 if row.is_flip else 0.75
        ax.plot(
            [X_LEFT, X_RIGHT],
            [row.delta_orig, row.delta_rep],
            color=row.color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=3 if row.is_flip else 2,
        )
        ax.plot(
            [X_LEFT, X_RIGHT],
            [row.delta_orig, row.delta_rep],
            marker="o",
            markersize=2.6,
            linestyle="none",
            color=row.color,
            zorder=4,
        )

    # Baseline parity reference.
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.6, zorder=1)

    # Annotate the top-|delta_orig| cells. Hand-tuned offsets — alternate
    # left/right so labels don't pile up on either margin.
    top = panel_df.reindex(panel_df.delta_orig.abs().sort_values(ascending=False).index).head(n_annotate)
    for i, row in enumerate(top.itertuples()):
        side_left = (i % 2 == 0)
        x_anchor = X_LEFT if side_left else X_RIGHT
        y_anchor = row.delta_orig if side_left else row.delta_rep
        xoff = -8 if side_left else 8
        ha = "right" if side_left else "left"
        label = f"{short_cond(row.condition)} · {row.instr}"
        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
            xytext=(xoff, 0),
            textcoords="offset points",
            fontsize=6.5,
            ha=ha,
            va="center",
            color="#222222",
        )

    ax.set_xticks([X_LEFT, X_RIGHT])
    ax.set_xticklabels(["Original", "Refined"])
    ax.set_xlim(X_LEFT - 0.32, X_RIGHT + 0.32)
    ax.set_ylabel(r"$\Delta$ vs baseline (pp)")
    # Headroom so panel label and top annotations don't collide.
    ymin, ymax = ax.get_ylim()
    pad = 0.12 * (ymax - ymin)
    new_ymin = ymin - 0.04 * (ymax - ymin)
    new_ymax = ymax + pad
    ax.set_ylim(new_ymin, new_ymax)
    # Scale callout derived from the data extent (pre-padding) so the readout
    # tracks the actual swing rather than the headroom.
    scale = _scale_pp(ymin, ymax)
    full_label = f"{panel_label} — scale ±{scale} pp"
    ax.set_title(full_label, loc="left", fontsize=9, fontweight="bold", pad=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    return scale


def _legend_handles() -> list:
    items = [
        ("Win→Loss flip", COLOR_FLIP_W2L),
        ("Loss→Win flip", COLOR_FLIP_L2W),
        ("Deflated (same sign)", COLOR_DEFLATED),
        ("Stable +", COLOR_POSITIVE),
        ("Stable −", COLOR_NEGATIVE),
        ("Near zero", COLOR_NEUTRAL),
    ]
    return [
        mlines.Line2D([], [], color=c, marker="o", markersize=3.5, linewidth=1.2, label=lbl)
        for lbl, c in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_style()
    raw = load_table("vs_baseline_cells", tables_dir=args.tables_dir)
    agg = aggregate(raw)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 3.7), sharey=False)
    scales: dict[str, int] = {}
    for ax, (bench, panel_label) in zip(axes, PANELS):
        panel_df = agg[agg.benchmark == bench]
        scales[bench] = draw_panel(ax, panel_df, panel_label)

    # Shared legend below both panels — keeps each data area clean.
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=7.0,
        handlelength=1.6,
        columnspacing=1.4,
    )

    fig.tight_layout()
    # Reserve room at the bottom for the shared legend.
    fig.subplots_adjust(bottom=0.18)
    save_figure(fig, "fig01_extractor_slopegraph")
    plt.close(fig)

    # Console summary so the caller can quote counts without re-running aggregation.
    flips = agg[agg.is_flip].groupby("benchmark").size().to_dict()
    print(f"[fig01] grain: condition × benchmark × instr_lang")
    for bench, label in PANELS:
        n = int((agg.benchmark == bench).sum())
        f = int(flips.get(bench, 0))
        s = scales.get(bench)
        print(f"[fig01] {label} (scale ±{s} pp): {n} lines, {f} sign-flips")


if __name__ == "__main__":
    main()

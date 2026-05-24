"""Figure 2: cell-level Δ structure (parse-failure vs accuracy).

Single claim: parse-failure recovery is the lever. Cells live almost entirely
in upper-left (Δpf<0, Δacc>0 — extractor lift) and lower-right (Δpf>0,
Δacc<0 — the cond-2-ur-5k/20k SIB-200 anomaly). X-CSQA and Belebele cluster
at the origin because their answers parse cleanly under both extractors.

Grain: per-cell (condition × seed × template × benchmark × data × instr).
1,664 rows on HF (spec says 1,665 — one off, immaterial).

Run:
    python fig02_cell_scatter.py [--tables-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from _viz_common import (
    BENCH_LABEL,
    BENCH_ORDER,
    CONDITION_LABEL,
    OKABE_ITO,
    WIDTH_DOUBLE,
    load_table,
    save_figure,
    setup_style,
)

# Benchmark -> (color, alpha). SIB-200 / XNLI carry the story; X-CSQA / Belebele
# are down-weighted because they pile up at (0,0) and would otherwise dominate.
BENCH_STYLE = {
    "sib200":   (OKABE_ITO["vermillion"],    0.70),
    "xnli":     (OKABE_ITO["blue"],          0.70),
    "csqa":     (OKABE_ITO["bluish_green"],  0.25),
    "belebele": (OKABE_ITO["orange"],        0.25),
}


def _short(row) -> str:
    """Compact cell label: 'C2 ur/5k sib200·zh→ur t1'."""
    return (
        f"{CONDITION_LABEL[row.condition]} "
        f"{row.benchmark}·{row.data}→{row.instr} t{row.template}"
    )


def pick_annotations(df: pd.DataFrame) -> list[tuple[pd.Series, tuple[int, int], str]]:
    """Pick 4 extreme cells with hand-tuned label offsets (xoff_pt, yoff_pt, ha)."""
    picks: list[tuple[pd.Series, tuple[int, int], str]] = []

    # Biggest extractor lift (most-negative Δpf with Δacc>0.10).
    lift = df[df.delta_acc > 0.10].nsmallest(1, "delta_pf").iloc[0]
    picks.append((lift, (25, -16), "left"))

    # Biggest cond-2-ur regression (most-positive Δpf with Δacc<-0.05).
    reg = df[df.delta_acc < -0.05].nlargest(1, "delta_pf").iloc[0]
    picks.append((reg, (-22, -10), "right"))

    # Most-negative Δacc, skipping the duplicate cell from pick #2.
    reg_key = (reg.condition, reg.benchmark, reg.data, reg.instr, reg.template)
    for _, cand in df.nsmallest(8, "delta_acc").iterrows():
        cand_key = (cand.condition, cand.benchmark, cand.data,
                    cand.instr, cand.template)
        if cand_key != reg_key:
            picks.append((cand, (-30, 14), "right"))
            break

    # Biggest XNLI lift cell — anchors the blue series visually.
    xnli_lift = df[(df.benchmark == "xnli") & (df.delta_acc > 0.05)]
    if not xnli_lift.empty:
        x = xnli_lift.nsmallest(1, "delta_pf").iloc[0]
        picks.append((x, (28, 6), "left"))

    return picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_style()
    df = load_table("cells", tables_dir=args.tables_dir).copy()

    # Plot in percentage points.
    df["x_pp"] = df.delta_pf * 100.0
    df["y_pp"] = df.delta_acc * 100.0

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 4.0))

    # Quadrant reference lines.
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.6, zorder=1)

    # One scatter per benchmark; plot the origin-clustered benchmarks first so
    # SIB-200 / XNLI extremes sit on top.
    plot_order = ["belebele", "csqa", "xnli", "sib200"]
    for bench in plot_order:
        color, alpha = BENCH_STYLE[bench]
        sub = df[df.benchmark == bench]
        ax.scatter(
            sub.x_pp,
            sub.y_pp,
            s=25,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=BENCH_LABEL[bench],
            zorder=3 if bench in {"sib200", "xnli"} else 2,
        )

    # Annotate extreme cells with thin connector lines.
    for row, (dx, dy), ha in pick_annotations(df):
        ax.annotate(
            _short(row),
            xy=(row.delta_pf * 100.0, row.delta_acc * 100.0),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            ha=ha,
            va="center",
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
        )

    # Δ values reach −76pp / +51pp, beyond the spec's tentative ±25pp; clamp
    # to the actual range with a small pad so lift cells aren't clipped.
    ax.set_xlim(df.x_pp.min() - 4, df.x_pp.max() + 4)
    ax.set_ylim(df.y_pp.min() - 4, df.y_pp.max() + 4)

    ax.set_xlabel(r"$\Delta$ parse-failure rate (pp; refined − original)")
    ax.set_ylabel(r"$\Delta$ accuracy (pp; refined − original)")

    # Quadrant labels — only label the two populated quadrants (UL = lift,
    # LR = cond-2-ur regression). UR and LL are empty.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax - xmin) * 0.02

    def _qt(x, y, text, ha, va, weight="normal", color="#888888"):
        ax.text(x, y, text, ha=ha, va=va, fontsize=7,
                color=color, fontweight=weight, zorder=1)

    _qt(xmin + dx, ymax * 0.65, "pf$\\downarrow$ acc$\\uparrow$\nextractor lift",
        ha="left", va="center", weight="bold", color="#444444")
    _qt(xmax - dx, ymin * 0.35, "pf$\\uparrow$ acc$\\downarrow$\nC2 ur regression",
        ha="right", va="center", color="#666666")

    # Headline sits in the empty LL corner.
    ax.text(0.01, 0.03,
            f"Cell-level $\\Delta$ structure (n={len(df):,})",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold")

    # Legend in upper-right; the data is sparse there.
    handles = [
        mlines.Line2D([], [], color=BENCH_STYLE[b][0], marker="o",
                      markersize=5, linestyle="none", alpha=0.9,
                      label=BENCH_LABEL[b])
        for b in BENCH_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True,
              framealpha=0.9, fontsize=7.5, handlelength=1.0,
              borderpad=0.4, labelspacing=0.3)

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, "fig02_cell_scatter")
    plt.close(fig)

    # Console summary so the caller can quote quadrant counts and extremes.
    ll = int(((df.delta_pf < 0) & (df.delta_acc < 0)).sum())
    ul = int(((df.delta_pf < 0) & (df.delta_acc > 0)).sum())
    ur = int(((df.delta_pf > 0) & (df.delta_acc > 0)).sum())
    lr = int(((df.delta_pf > 0) & (df.delta_acc < 0)).sum())
    on_axes = int((((df.delta_pf == 0) | (df.delta_acc == 0))).sum())
    print(f"[fig02] cells: n={len(df)}")
    print(f"[fig02] quadrants  UL(pf-,acc+)={ul}  UR(pf+,acc+)={ur}  "
          f"LL(pf-,acc-)={ll}  LR(pf+,acc-)={lr}  on-axes={on_axes}")
    print("[fig02] annotated cells:")
    for row, _off, _ha in pick_annotations(df):
        print(f"[fig02]   {_short(row)}: "
              f"Δpf={row.delta_pf*100:+.1f}pp Δacc={row.delta_acc*100:+.1f}pp")


if __name__ == "__main__":
    main()

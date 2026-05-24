"""Figure 5 — cond-5 rehabilitated by the refined extractor.

Single claim: cond-5's apparent SIB-200 collapse halved under refined scoring
(it was largely extractor coverage); the X-CSQA degradation is unchanged —
a real model effect from the Aya-translated training mix.

One slope line per (condition, benchmark) cell, connecting `mean_delta_orig`
(left) to `mean_delta_rep` (right). Four lines total:
  - C5 ur/5k · SIB-200     (refined: improves)
  - C5 ur/5k · X-CSQA      (refined: unchanged — flat)
  - C5 zh/5k · SIB-200     (refined: improves, biggest move)
  - C5 zh/5k · X-CSQA      (refined: unchanged — flat)

CLI: python fig05_cond5_rehabilitated.py [--tables-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# Allow `python scripts/fig05_cond5_rehabilitated.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _viz_common import (  # noqa: E402
    BENCH_LABEL,
    COLOR_POSITIVE,
    CONDITION_LABEL,
    OKABE_ITO,
    WIDTH_DOUBLE,
    load_table,
    save_figure,
    setup_style,
)

COND5_CONDITIONS = ["condition-5-ur-5k", "condition-5-zh-5k"]
COND5_BENCHMARKS = ["sib200", "csqa"]

COLOR_REHAB = COLOR_POSITIVE          # SIB-200: refined extractor helps
COLOR_UNCHANGED = OKABE_ITO["sky_blue"]  # X-CSQA: real model effect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 cond-5 rehabilitation slopegraph (fig 5).")
    p.add_argument("--tables-dir", type=Path, default=None, help="Override default tables dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()

    df = load_table("vs_baseline_by_cond_x_bench", tables_dir=args.tables_dir)
    sub = (
        df[df.condition.isin(COND5_CONDITIONS) & df.benchmark.isin(COND5_BENCHMARKS)]
        .copy()
        .reset_index(drop=True)
    )
    if len(sub) != 4:
        raise AssertionError(f"Expected 4 cond-5 cells, got {len(sub)}: {sub}")

    # pp units for plotting & annotations.
    sub["delta_orig_pp"] = sub["mean_delta_orig"] * 100.0
    sub["delta_rep_pp"] = sub["mean_delta_rep"] * 100.0
    sub["is_sib200"] = sub["benchmark"] == "sib200"

    # Sort rows by mean_delta_orig descending so labels on the left column
    # are top-to-bottom in the same order as the lines.
    sub = sub.sort_values("delta_orig_pp", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.5))
    x_orig, x_rep = 0.0, 1.0

    # Baseline-parity reference.
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)

    # One slope per cell, colored by whether refined scoring moved it.
    for _, row in sub.iterrows():
        color = COLOR_REHAB if row["is_sib200"] else COLOR_UNCHANGED
        yo, yr = row["delta_orig_pp"], row["delta_rep_pp"]
        ax.plot([x_orig, x_rep], [yo, yr], color=color, lw=1.8, alpha=0.9, zorder=2)
        ax.plot(x_orig, yo, "o", color=color, ms=5.0, zorder=3)
        ax.plot(x_rep, yr, "o", color=color, ms=5.0, zorder=3)

        label = (
            f"{CONDITION_LABEL.get(row['condition'], row['condition'])} · "
            f"{BENCH_LABEL[row['benchmark']]}"
        )
        ax.annotate(
            label,
            xy=(x_orig - 0.05, yo),
            ha="right",
            va="center",
            fontsize=8,
            color="#222222",
        )

    # "Halved" callout on the cond-5-zh-5k · SIB-200 line.
    zh_sib = sub[(sub.condition == "condition-5-zh-5k") & (sub.benchmark == "sib200")].iloc[0]
    yo_zh, yr_zh = zh_sib["delta_orig_pp"], zh_sib["delta_rep_pp"]
    mid_x = (x_orig + x_rep) / 2.0
    mid_y = (yo_zh + yr_zh) / 2.0
    ax.annotate(
        f"{yo_zh:.1f} → {yr_zh:.1f} (halved)",
        xy=(mid_x, mid_y),
        xytext=(mid_x, mid_y - 4.0),
        ha="center",
        va="top",
        fontsize=7,
        color=COLOR_REHAB,
        arrowprops=dict(arrowstyle="-", color=COLOR_REHAB, lw=0.6),
    )

    # X-axis: two labeled positions.
    ax.set_xticks([x_orig, x_rep])
    ax.set_xticklabels(["Original", "Refined"])
    ax.set_xlim(-0.5, 1.2)
    ax.tick_params(axis="x", length=0)

    # Y-axis: percentage points, padded for clear annotation space.
    y_lo = min(sub["delta_orig_pp"].min(), sub["delta_rep_pp"].min())
    y_hi = max(sub["delta_orig_pp"].max(), sub["delta_rep_pp"].max())
    ax.set_ylim(y_lo - 6.0, max(y_hi + 4.0, 5.0))
    ax.set_ylabel("Mean Δ vs. baseline (pp)")

    # Top-left annotation above the axes (no title).
    ax.text(
        0.0,
        1.02,
        "Cond-5 (Aya-translated) vs baseline — extractor effect on the cond-5 story",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#222222",
    )

    # Legend, bottom-right, two entries.
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_REHAB, lw=1.8, marker="o", ms=4,
                      label="Extractor-rehabilitated (SIB-200)"),
        mlines.Line2D([], [], color=COLOR_UNCHANGED, lw=1.8, marker="o", ms=4,
                      label="Unchanged (X-CSQA)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=7)

    fig.tight_layout()
    save_figure(fig, "fig05_cond5_rehabilitated")
    plt.close(fig)

    # ─── Report ─────────────────────────────────────────────────────────────
    print("Wrote figure: fig05_cond5_rehabilitated.{pdf,png}")
    for _, row in sub.iterrows():
        print(
            f"  {row['condition']:22s} {row['benchmark']:7s} "
            f"Δorig={row['mean_delta_orig']:+.4f}  Δrep={row['mean_delta_rep']:+.4f}"
        )


if __name__ == "__main__":
    main()

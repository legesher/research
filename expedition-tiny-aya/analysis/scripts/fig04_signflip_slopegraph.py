"""Figure 4 — sign-flip slopegraph.

Single claim: the largest-|shift| flipped cells explain the entire
conclusion-flip story; they all live on SIB-200 and on instr != en.

One slope line per (condition × seed × template × benchmark × data × instr)
cell, connecting `delta_orig` (left) to `delta_rep` (right). Colored by
verdict (win→loss vermillion, loss→win blue), reference at y=0.

Two outputs:
  - Main paper figure: top-N by |shift|  (fig04_signflip_top15.{pdf,png})
  - Appendix figure:   all 48 flips      (fig04_signflip_all48.{pdf,png})
The full 48-row appendix table is also written as fig04_signflip_appendix.tsv.

CLI: python fig04_signflip_slopegraph.py [--tables-dir PATH] [--top-n 15] [--all]
By default, BOTH figures are emitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

# Allow `python scripts/fig04_signflip_slopegraph.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _viz_common import (  # noqa: E402
    BENCH_LABEL,
    COLOR_FLIP_L2W,
    COLOR_FLIP_W2L,
    CONDITION_LABEL,
    WIDTH_DOUBLE,
    figures_out_dir,
    load_table,
    save_figure,
    setup_style,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 sign-flip slopegraph (fig 4).")
    p.add_argument("--tables-dir", type=Path, default=None, help="Override default tables dir.")
    p.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of rows in the main paper figure (default 15, by |shift| desc).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Force emission of the all-48 appendix figure. "
            "Default behavior (no flag) already emits BOTH outputs."
        ),
    )
    return p.parse_args()


def render_slopegraph(
    df_subset: pd.DataFrame,
    out_stem: str,
    annotation: str,
    figsize: tuple[float, float],
    label_fontsize: float = 7.0,
    min_sep_pp: float = 2.6,
) -> None:
    """Render a slopegraph for the given subset of flipped cells.

    Args:
        df_subset: rows from conclusion_flips with shift/abs_shift columns.
        out_stem:  filename stem (no extension) under figures-phase3/.
        annotation: text placed top-left above the axes.
        figsize:   (width, height) inches.
        label_fontsize: pt for per-row labels at left of "Original".
        min_sep_pp: minimum vertical separation between adjacent row labels (pp).
    """
    # Sort by delta_orig so adjacent labels rarely overlap on the left column.
    plotted = df_subset.sort_values("delta_orig", ascending=False).reset_index(drop=True)
    n_plot = len(plotted)

    # Pre-compute unique row labels. Drops the benchmark prefix (all 48 flipped
    # cells live on SIB-200 — see annotation) so template + seed fit on one line
    # without clipping. Uniqueness asserted below so identical-looking labels
    # never silently mask a (template, seed) collision.
    labels = [
        f"t{row['template']} · "
        f"{row['data']}→{row['instr']} · "
        f"{CONDITION_LABEL.get(row['condition'], row['condition'])} · "
        f"s{row['seed']}"
        for _, row in plotted.iterrows()
    ]
    if len(set(labels)) != len(labels):
        from collections import Counter
        dupes = [lbl for lbl, n in Counter(labels).items() if n > 1]
        raise AssertionError(
            f"Row labels in {out_stem} are not unique; duplicates: {dupes}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Two x-positions for the slope endpoints; pp units on y.
    x_orig, x_rep = 0.0, 1.0
    y_pp_orig = plotted["delta_orig"] * 100.0
    y_pp_rep = plotted["delta_rep"] * 100.0

    # Baseline-parity reference.
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)

    # One slope line per cell, with endpoint markers (loser-side larger).
    for (_, row), yo, yr in zip(plotted.iterrows(), y_pp_orig, y_pp_rep):
        color = COLOR_FLIP_W2L if row["verdict"] == "win→loss" else COLOR_FLIP_L2W
        ax.plot([x_orig, x_rep], [yo, yr], color=color, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(x_orig, yo, "o", color=color, ms=5.0 if yo < 0 else 3.5, zorder=3)
        ax.plot(x_rep, yr, "o", color=color, ms=5.0 if yr < 0 else 3.5, zorder=3)

    # Row labels (left of "Original") with vertical de-collision: rows arrive
    # sorted by delta_orig desc, so we push each label down at least min_sep_pp
    # below the previous one. A gray leader links the label to its true y.
    prev_label_y = None
    for label, yo in zip(labels, y_pp_orig):
        label_y = yo if prev_label_y is None else min(yo, prev_label_y - min_sep_pp)
        prev_label_y = label_y
        if abs(label_y - yo) > 0.1:
            ax.plot([x_orig - 0.04, x_orig - 0.01], [label_y, yo],
                    color="#bbbbbb", lw=0.5, zorder=1)
        ax.annotate(label, xy=(x_orig - 0.05, label_y),
                    ha="right", va="center", fontsize=label_fontsize, color="#222222")

    # X-axis: just two labeled positions.
    ax.set_xticks([x_orig, x_rep])
    ax.set_xticklabels(["Original", "Refined"])
    # Give row labels ~half the axes width on the left.
    ax.set_xlim(-0.85, x_rep + 0.08)
    # Bump y-limits a touch so the top annotation has clearance and the
    # de-collided labels at the bottom don't get clipped.
    y_lo, y_hi = ax.get_ylim()
    # Extra bottom padding scales with min_sep_pp × number of labels that may
    # have been pushed down — cheap upper bound is n_plot * min_sep_pp / 2.
    bottom_pad = max(2.0, min(20.0, n_plot * min_sep_pp * 0.25))
    ax.set_ylim(y_lo - bottom_pad, y_hi + 6)

    # Y-axis: percentage points.
    ax.set_ylabel("Δ vs. baseline (pp)")
    ax.tick_params(axis="x", length=0)

    # Top-left annotation above the axes so it doesn't compete with row labels.
    ax.text(
        0.0, 1.02, annotation,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        color="#222222",
    )

    # Legend — two entries, small, lower-left.
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_FLIP_W2L, lw=1.6, marker="o", ms=4, label="win → loss"),
        mlines.Line2D([], [], color=COLOR_FLIP_L2W, lw=1.6, marker="o", ms=4, label="loss → win"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=7)

    fig.tight_layout()
    save_figure(fig, out_stem)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()

    flips = load_table("conclusion_flips", tables_dir=args.tables_dir).copy()
    flips["shift"] = flips["delta_rep"] - flips["delta_orig"]
    flips["abs_shift"] = flips["shift"].abs()
    flips = flips.sort_values("abs_shift", ascending=False).reset_index(drop=True)

    n_total = len(flips)

    # ─── Appendix TSV — all flips, sorted by |shift| desc ───────────────────
    appendix_cols = [
        "condition", "seed", "template", "benchmark", "data", "instr",
        "delta_orig", "delta_rep", "shift", "verdict",
    ]
    appendix_path = figures_out_dir() / "fig04_signflip_appendix.tsv"
    flips[appendix_cols].to_csv(appendix_path, sep="\t", index=False)

    # Shared row-label legend: all 48 flipped cells live on SIB-200, so we
    # drop the benchmark from each row and surface it once in the header.
    row_legend = "All flips on SIB-200; rows: template · data→instr · condition · seed"

    # ─── Output A: top-N main paper figure ──────────────────────────────────
    top_n = min(args.top_n, n_total)
    top_subset = flips.head(top_n)
    render_slopegraph(
        df_subset=top_subset,
        out_stem="fig04_signflip_top15",
        annotation=(
            f"Top {top_n} sign-flips by magnitude "
            f"(full {n_total} in appendix). {row_legend}"
        ),
        figsize=(WIDTH_DOUBLE, 4.0),
        label_fontsize=7.0,
        min_sep_pp=2.6,
    )

    # ─── Output B: all-flips appendix figure ────────────────────────────────
    render_slopegraph(
        df_subset=flips,
        out_stem="fig04_signflip_all48",
        annotation=(
            f"All {n_total} sign-flips (sorted by |shift|). {row_legend}"
        ),
        figsize=(WIDTH_DOUBLE, 9.0),
        label_fontsize=6.0,
        # Tighter row separation than the top-15 because vertical space, while
        # taller in absolute inches, is split across many more labels.
        min_sep_pp=1.4,
    )

    # ─── Report ─────────────────────────────────────────────────────────────
    print(f"Wrote figure: fig04_signflip_top15.{{pdf,png}}  ({top_n} rows)")
    print(f"Wrote figure: fig04_signflip_all48.{{pdf,png}}  ({n_total} rows)")
    print(f"Wrote appendix: {appendix_path}")
    print(
        f"Verdict breakdown (all {n_total}): "
        f"{flips['verdict'].value_counts().to_dict()}"
    )
    top_row = flips.iloc[0]
    print(
        f"Most extreme |shift|: {top_row['abs_shift'] * 100:.2f}pp  "
        f"({BENCH_LABEL[top_row['benchmark']]} · "
        f"{top_row['data']}→{top_row['instr']} · "
        f"{CONDITION_LABEL.get(top_row['condition'], top_row['condition'])}, "
        f"seed {top_row['seed']}, tpl {top_row['template']}, "
        f"verdict {top_row['verdict']})"
    )

    # --all is a no-op in default mode (both already emitted), but we honor
    # the flag by acknowledging it in stdout for scripting transparency.
    if args.all:
        print("(--all passed; all-48 figure already emitted by default)")


if __name__ == "__main__":
    main()

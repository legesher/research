"""Figure 4 — sign-flip slopegraph, faceted by benchmark.

The conclusion-flip catalogue holds 48 cells whose sign(Δ vs baseline) changed
between the original and refined extractors. They split across exactly two
benchmarks: 43 on SIB-200, 5 on XNLI. X-CSQA and Belebele contribute zero
flips by construction (the extractor refinement didn't touch their letter-
based answer format).

Each benchmark gets its own figure (one or two variants, depending on count):
  - If a benchmark has more than --top-n flips, emit BOTH a top-N main-paper
    figure and an all-N appendix figure (separate files).
  - If a benchmark has fewer than --top-n flips, emit a single all-N figure.

Filename stems encode the row count, so the file's contents are always self-
describing:
  fig04_signflip_sib200_top{N}.{pdf,png}    (paper main figure)
  fig04_signflip_sib200_all{N}.{pdf,png}    (paper appendix)
  fig04_signflip_xnli_all{N}.{pdf,png}      (small, no top-N variant)

The full 48-row appendix table is also written as fig04_signflip_appendix.tsv
(unsplit, sorted by |shift| desc).

CLI: python fig04_signflip_slopegraph.py [--tables-dir PATH] [--top-n 15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _viz_common import (  # noqa: E402
    BENCH_LABEL,
    BENCH_ORDER,
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
        help=(
            "Number of rows in the main paper figure per benchmark (default 15, by "
            "|shift| desc). Filename encodes the actual count: e.g. --top-n 20 writes "
            "fig04_signflip_sib200_top20.{pdf,png}. Benchmarks with fewer flips than "
            "--top-n skip the top-N variant and emit only the all-N variant."
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
    plotted = df_subset.sort_values("delta_orig", ascending=False).reset_index(drop=True)
    n_plot = len(plotted)

    # Row label includes benchmark so the same script can emit per-benchmark
    # figures without ambiguity (and so cross-benchmark comparisons of these
    # figures stay readable side-by-side). Uniqueness asserted below.
    labels = [
        f"{BENCH_LABEL[row['benchmark']]} · "
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

    x_orig, x_rep = 0.0, 1.0
    y_pp_orig = plotted["delta_orig"] * 100.0
    y_pp_rep = plotted["delta_rep"] * 100.0

    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)

    for (_, row), yo, yr in zip(plotted.iterrows(), y_pp_orig, y_pp_rep):
        color = COLOR_FLIP_W2L if row["verdict"] == "win→loss" else COLOR_FLIP_L2W
        ax.plot([x_orig, x_rep], [yo, yr], color=color, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(x_orig, yo, "o", color=color, ms=5.0 if yo < 0 else 3.5, zorder=3)
        ax.plot(x_rep, yr, "o", color=color, ms=5.0 if yr < 0 else 3.5, zorder=3)

    prev_label_y = None
    for label, yo in zip(labels, y_pp_orig):
        label_y = yo if prev_label_y is None else min(yo, prev_label_y - min_sep_pp)
        prev_label_y = label_y
        if abs(label_y - yo) > 0.1:
            ax.plot([x_orig - 0.04, x_orig - 0.01], [label_y, yo],
                    color="#bbbbbb", lw=0.5, zorder=1)
        ax.annotate(label, xy=(x_orig - 0.05, label_y),
                    ha="right", va="center", fontsize=label_fontsize, color="#222222")

    ax.set_xticks([x_orig, x_rep])
    ax.set_xticklabels(["Original", "Refined"])
    ax.set_xlim(-1.05, x_rep + 0.08)  # wider left margin to fit benchmark prefix
    y_lo, y_hi = ax.get_ylim()
    bottom_pad = max(2.0, min(20.0, n_plot * min_sep_pp * 0.25))
    ax.set_ylim(y_lo - bottom_pad, y_hi + 6)

    ax.set_ylabel("Δ vs. baseline (pp)")
    ax.tick_params(axis="x", length=0)

    ax.text(
        0.0, 1.02, annotation,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        color="#222222",
    )

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

    # ─── Appendix TSV (unsplit, all benchmarks) ─────────────────────────────
    appendix_cols = [
        "condition", "seed", "template", "benchmark", "data", "instr",
        "delta_orig", "delta_rep", "shift", "verdict",
    ]
    appendix_path = figures_out_dir() / "fig04_signflip_appendix.tsv"
    flips[appendix_cols].to_csv(appendix_path, sep="\t", index=False)

    # ─── Per-benchmark figures ──────────────────────────────────────────────
    # Group flips by benchmark; iterate in canonical BENCH_ORDER so figure
    # numbering across runs is deterministic.
    by_bench = {b: flips[flips.benchmark == b] for b in BENCH_ORDER if (flips.benchmark == b).any()}
    written: list[tuple[str, int]] = []

    for bench, df_b in by_bench.items():
        n_bench = len(df_b)
        bench_short = bench  # use the lowercase stem from the data (sib200, xnli, ...)

        if n_bench > args.top_n:
            top_n = args.top_n
            top_subset = df_b.head(top_n)
            stem_top = f"fig04_signflip_{bench_short}_top{top_n}"
            render_slopegraph(
                df_subset=top_subset,
                out_stem=stem_top,
                annotation=(
                    f"Top {top_n} of {n_bench} {BENCH_LABEL[bench]} sign-flips "
                    f"by |shift|; rows: benchmark · template · data→instr · condition · seed"
                ),
                figsize=(WIDTH_DOUBLE, 4.0),
                label_fontsize=7.0,
                min_sep_pp=2.6,
            )
            written.append((stem_top, top_n))

            stem_all = f"fig04_signflip_{bench_short}_all{n_bench}"
            render_slopegraph(
                df_subset=df_b,
                out_stem=stem_all,
                annotation=(
                    f"All {n_bench} {BENCH_LABEL[bench]} sign-flips "
                    f"(sorted by |shift|); rows: benchmark · template · data→instr · condition · seed"
                ),
                figsize=(WIDTH_DOUBLE, max(4.0, 0.20 * n_bench + 1.5)),
                label_fontsize=6.0,
                min_sep_pp=1.4,
            )
            written.append((stem_all, n_bench))
        else:
            stem_all = f"fig04_signflip_{bench_short}_all{n_bench}"
            render_slopegraph(
                df_subset=df_b,
                out_stem=stem_all,
                annotation=(
                    f"All {n_bench} {BENCH_LABEL[bench]} sign-flips; "
                    f"rows: benchmark · template · data→instr · condition · seed"
                ),
                figsize=(WIDTH_DOUBLE, max(3.0, 0.30 * n_bench + 1.5)),
                label_fontsize=7.0,
                min_sep_pp=2.6,
            )
            written.append((stem_all, n_bench))

    # ─── Report ─────────────────────────────────────────────────────────────
    for stem, n in written:
        print(f"Wrote figure: {stem}.{{pdf,png}}  ({n} rows)")
    print(f"Wrote appendix: {appendix_path}")
    bench_breakdown = flips.groupby("benchmark").size().to_dict()
    print(f"Sign-flip benchmark breakdown (total {n_total}): {bench_breakdown}")
    print(f"Verdict breakdown (all {n_total}): {flips['verdict'].value_counts().to_dict()}")
    top_row = flips.iloc[0]
    print(
        f"Most extreme |shift|: {top_row['abs_shift'] * 100:.2f}pp  "
        f"({BENCH_LABEL[top_row['benchmark']]} · "
        f"{top_row['data']}→{top_row['instr']} · "
        f"{CONDITION_LABEL.get(top_row['condition'], top_row['condition'])}, "
        f"seed {top_row['seed']}, tpl {top_row['template']}, "
        f"verdict {top_row['verdict']})"
    )


if __name__ == "__main__":
    main()

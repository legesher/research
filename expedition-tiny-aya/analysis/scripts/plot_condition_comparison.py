#!/usr/bin/env python3
"""
Visualization script for Tiny Aya Expedition evaluation results.

Fetches benchmark results from HuggingFace (including baseline),
then generates comparison charts across experimental conditions.

Usage:
    python plot_condition_comparison.py [--output-dir ../figures] [--cache-dir /path/to/cache]

Charts generated:
    1. Grouped bar charts (condition x benchmark, per language x prompt type)
    2. Delta-from-baseline charts
    3. Heatmaps (compact overview)
    4. Prompt effect comparison (native vs English)

Related tickets: AYA-88, AYA-89, AYA-142
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import hf_hub_download

# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"

CONDITIONS = [
    "baseline",
    "condition-1-en",
    "condition-1-en-5k",
    "condition-2-zh-5k",
    "condition-2-es-5k",
    "condition-2-ur-5k",
    "condition-3-zh-5k",
]

CONDITION_LABELS = {
    "baseline": "Baseline",
    "condition-1-en": "Cond 1\n(en full)",
    "condition-1-en-5k": "Cond 1\n(en 5K)",
    "condition-2-zh-5k": "Cond 2\n(zh)",
    "condition-2-es-5k": "Cond 2\n(es)",
    "condition-2-ur-5k": "Cond 2\n(ur)",
    "condition-3-zh-5k": "Cond 3\n(zh)",
}

CONDITION_LABELS_SHORT = {
    "baseline": "Baseline",
    "condition-1-en": "C1-en",
    "condition-1-en-5k": "C1-5K",
    "condition-2-zh-5k": "C2-zh",
    "condition-2-es-5k": "C2-es",
    "condition-2-ur-5k": "C2-ur",
    "condition-3-zh-5k": "C3-zh",
}

BENCHMARKS = ["mgsm", "xnli", "csqa"]
LANGUAGES = ["zh", "es", "ur"]
PROMPT_TYPES = ["english", "native"]

BENCHMARK_COLORS = {
    "mgsm": "#2196F3",  # blue
    "xnli": "#FF9800",  # orange
    "csqa": "#4CAF50",  # green
}

BENCHMARK_LABELS = {
    "mgsm": "MGSM (math)",
    "xnli": "XNLI (NLI)",
    "csqa": "CSQA (commonsense)",
}


# ── Data Loading ─────────────────────────────────────────────────────────────


def fetch_all_results(cache_dir: Path | None) -> dict:
    """Fetch all condition results (including baseline) from HuggingFace Hub.

    All results, including baseline, are fetched from the canonical source at
    legesher/language-decoded-experiments on HuggingFace. This ensures the
    script always uses the latest re-scored values without depending on local
    JSON files that may be stale.
    """
    results = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for prompt_type in PROMPT_TYPES:
            hf_filename = f"conditions/{condition}/results/{'english' if prompt_type == 'english' else 'native'}_prompt_results.json"
            try:
                local_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=hf_filename,
                    repo_type=HF_REPO_TYPE,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
                with open(local_path, encoding="utf-8") as f:
                    data = json.load(f)
                results[condition][prompt_type] = data["summary"]
                print(f"  Loaded {condition} / {prompt_type}")
            except Exception as e:
                print(f"  Failed to fetch {condition} / {prompt_type}: {e}")
    return results


def build_dataframe(all_results: dict) -> pd.DataFrame:
    """Combine all results into a tidy DataFrame."""
    rows = []
    for condition, cond_data in all_results.items():
        for prompt_type in PROMPT_TYPES:
            if prompt_type not in cond_data:
                continue
            summary = cond_data[prompt_type]
            for benchmark in BENCHMARKS:
                for lang in LANGUAGES:
                    key = f"{benchmark}_{lang}_acc"
                    val = summary.get(key)
                    if val is not None:
                        rows.append(
                            {
                                "condition": condition,
                                "prompt_type": prompt_type,
                                "benchmark": benchmark,
                                "language": lang,
                                "score": val,
                            }
                        )

    df = pd.DataFrame(rows)
    df["score_pct"] = df["score"] * 100
    return df


# ── Chart 1: Grouped Bar Charts ─────────────────────────────────────────────


def plot_grouped_bars(df: pd.DataFrame, output_dir: Path):
    """
    Grouped bar chart: X = condition, Y = score, 3 colored bars per condition
    for MGSM, XNLI, CSQA. One subplot per language, one figure per prompt type.
    """
    condition_order = CONDITIONS

    for prompt_type in PROMPT_TYPES:
        subset = df[df["prompt_type"] == prompt_type]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
        fig.suptitle(
            f"Benchmark Scores by Condition — {'English' if prompt_type == 'english' else 'Native Language'} Prompt",
            fontsize=13,
            fontweight="bold",
        )

        for ax, lang in zip(axes, LANGUAGES):
            lang_data = subset[subset["language"] == lang]
            x = np.arange(len(condition_order))
            width = 0.25

            for i, benchmark in enumerate(BENCHMARKS):
                bm_data = lang_data[lang_data["benchmark"] == benchmark]
                scores = []
                for cond in condition_order:
                    match = bm_data[bm_data["condition"] == cond]
                    scores.append(match["score_pct"].values[0] if len(match) > 0 else 0)
                ax.bar(
                    x + (i - 1) * width,
                    scores,
                    width,
                    label=BENCHMARK_LABELS[benchmark] if lang == "zh" else None,
                    color=BENCHMARK_COLORS[benchmark],
                    alpha=0.85,
                )

            ax.set_xlabel("Condition")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{lang.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS_SHORT[c] for c in condition_order],
                fontsize=8,
                rotation=45,
                ha="right",
            )
            ax.set_ylim(0, 65)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.grid(axis="y", alpha=0.3)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"grouped_bars_{prompt_type}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved grouped_bars_{prompt_type}.png")


# ── Chart 1b: Single-Language Bar Charts ─────────────────────────────────────


def plot_single_language_bars(df: pd.DataFrame, output_dir: Path):
    """
    One standalone chart per language × prompt type.
    X = condition, Y = score, 3 colored bars (MGSM, XNLI, CSQA).
    """
    condition_order = CONDITIONS

    for prompt_type in PROMPT_TYPES:
        subset = df[df["prompt_type"] == prompt_type]
        prompt_label = "English" if prompt_type == "english" else "Native Language"

        for lang in LANGUAGES:
            lang_data = subset[subset["language"] == lang]
            x = np.arange(len(condition_order))
            width = 0.25

            fig, ax = plt.subplots(figsize=(8, 5))

            for i, benchmark in enumerate(BENCHMARKS):
                bm_data = lang_data[lang_data["benchmark"] == benchmark]
                scores = []
                for cond in condition_order:
                    match = bm_data[bm_data["condition"] == cond]
                    scores.append(match["score_pct"].values[0] if len(match) > 0 else 0)
                ax.bar(
                    x + (i - 1) * width,
                    scores,
                    width,
                    label=BENCHMARK_LABELS[benchmark],
                    color=BENCHMARK_COLORS[benchmark],
                    alpha=0.85,
                )

            ax.set_title(
                f"{lang.upper()} — {prompt_label} Prompt",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("Condition")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS_SHORT[c] for c in condition_order],
                fontsize=9,
                rotation=45,
                ha="right",
            )
            ax.set_ylim(0, 65)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

            fig.tight_layout()
            fname = f"bars_{lang}_{prompt_type}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


# ── Chart 2: Delta from Baseline ─────────────────────────────────────────────


def plot_delta_bars(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing improvement (or regression) over baseline in pp."""
    condition_order = [c for c in CONDITIONS if c != "baseline"]

    for prompt_type in PROMPT_TYPES:
        subset = df[df["prompt_type"] == prompt_type]
        baseline_scores = subset[subset["condition"] == "baseline"].set_index(
            ["benchmark", "language"]
        )["score_pct"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        fig.suptitle(
            f"Change from Baseline (pp) — {'English' if prompt_type == 'english' else 'Native Language'} Prompt",
            fontsize=13,
            fontweight="bold",
        )

        for ax, lang in zip(axes, LANGUAGES):
            x = np.arange(len(condition_order))
            width = 0.25

            for i, benchmark in enumerate(BENCHMARKS):
                bl = baseline_scores.get((benchmark, lang), 0)
                deltas = []
                for cond in condition_order:
                    match = subset[
                        (subset["condition"] == cond)
                        & (subset["benchmark"] == benchmark)
                        & (subset["language"] == lang)
                    ]
                    val = match["score_pct"].values[0] if len(match) > 0 else 0
                    deltas.append(val - bl)

                ax.bar(
                    x + (i - 1) * width,
                    deltas,
                    width,
                    color=BENCHMARK_COLORS[benchmark],
                    alpha=0.85,
                    label=BENCHMARK_LABELS[benchmark] if lang == "zh" else None,
                )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Condition")
            ax.set_ylabel("Δ from Baseline (pp)")
            ax.set_title(f"{lang.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS_SHORT[c] for c in condition_order],
                fontsize=8,
                rotation=45,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"delta_bars_{prompt_type}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved delta_bars_{prompt_type}.png")


# ── Chart 3: Heatmap ─────────────────────────────────────────────────────────


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of delta from baseline. Rows = conditions, columns = benchmark_lang."""
    condition_order = CONDITIONS
    col_labels = [f"{bm.upper()} {lang}" for bm in BENCHMARKS for lang in LANGUAGES]

    for prompt_type in PROMPT_TYPES:
        subset = df[df["prompt_type"] == prompt_type]
        baseline_lookup = {}
        for _, row in subset[subset["condition"] == "baseline"].iterrows():
            baseline_lookup[(row["benchmark"], row["language"])] = row["score_pct"]

        matrix = []
        row_labels = []
        for cond in condition_order:
            if cond == "baseline":
                continue
            vals = []
            for bm in BENCHMARKS:
                for lang in LANGUAGES:
                    match = subset[
                        (subset["condition"] == cond)
                        & (subset["benchmark"] == bm)
                        & (subset["language"] == lang)
                    ]
                    val = match["score_pct"].values[0] if len(match) > 0 else np.nan
                    bl = baseline_lookup.get((bm, lang), 0)
                    vals.append(val - bl)
            matrix.append(vals)
            row_labels.append(CONDITION_LABELS_SHORT[cond])

        heat_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

        fig, ax = plt.subplots(figsize=(11, 3.5))
        vmax = max(abs(heat_df.min().min()), abs(heat_df.max().max()))
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".1f",
            center=0,
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Δ from Baseline (pp)"},
        )
        ax.set_title(
            f"Improvement over Baseline — {'English' if prompt_type == 'english' else 'Native Language'} Prompt",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Benchmark × Language")
        ax.set_ylabel("Condition")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"heatmap_{prompt_type}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved heatmap_{prompt_type}.png")


# ── Chart 4: Prompt Effect Comparison ────────────────────────────────────────


def plot_prompt_comparison(df: pd.DataFrame, output_dir: Path):
    """Paired bars showing native vs English prompt scores per condition."""
    condition_order = CONDITIONS

    for benchmark in BENCHMARKS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        fig.suptitle(
            f"{BENCHMARK_LABELS[benchmark]} — Native vs English Prompt",
            fontsize=13,
            fontweight="bold",
        )

        for ax, lang in zip(axes, LANGUAGES):
            x = np.arange(len(condition_order))
            width = 0.35

            for i, prompt_type in enumerate(PROMPT_TYPES):
                pt_label = "Native" if prompt_type == "native" else "English"
                color = "#FF7043" if prompt_type == "native" else "#42A5F5"
                bm_data = df[
                    (df["benchmark"] == benchmark)
                    & (df["language"] == lang)
                    & (df["prompt_type"] == prompt_type)
                ]
                scores = []
                for cond in condition_order:
                    match = bm_data[bm_data["condition"] == cond]
                    scores.append(match["score_pct"].values[0] if len(match) > 0 else 0)
                ax.bar(
                    x + (i - 0.5) * width,
                    scores,
                    width,
                    label=f"{pt_label} prompt" if lang == "zh" else None,
                    color=color,
                    alpha=0.85,
                )

            ax.set_xlabel("Condition")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{lang.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS_SHORT[c] for c in condition_order],
                fontsize=8,
                rotation=45,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"prompt_comparison_{benchmark}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved prompt_comparison_{benchmark}.png")


# ── Chart 5: Per-Language Experimental Chain ─────────────────────────────────


# Colors for the experimental chain conditions
CHAIN_COLORS = {
    "baseline": "#9E9E9E",  # grey
    "condition-1-en": "#78909C",  # blue-grey
    "condition-1-en-5k": "#42A5F5",  # blue
    "condition-2": "#FF7043",  # orange (target-language keyword swap)
    "condition-3-zh-5k": "#AB47BC",  # purple (mixed sources)
}


def plot_language_vs_cond1(df: pd.DataFrame, output_dir: Path):
    """Per-language experimental chain: Baseline → Cond 1-en → Cond 2-{lang}.

    Shows the controlled comparison for each language: does English code help,
    and do target-language keywords help further? Cond 3-zh included for Chinese.
    """
    # Define the experimental chain per language
    chains = {
        "zh": {
            "conditions": [
                "baseline",
                "condition-1-en",
                "condition-1-en-5k",
                "condition-2-zh-5k",
                "condition-3-zh-5k",
            ],
            "labels": [
                "Baseline",
                "C1-en\n(full)",
                "C1-en\n(5K)",
                "C2-zh\n(5K)",
                "C3-zh\n(5K)",
            ],
            "colors": [
                CHAIN_COLORS["baseline"],
                CHAIN_COLORS["condition-1-en"],
                CHAIN_COLORS["condition-1-en-5k"],
                CHAIN_COLORS["condition-2"],
                CHAIN_COLORS["condition-3-zh-5k"],
            ],
            "title": "Chinese (zh)",
        },
        "es": {
            "conditions": [
                "baseline",
                "condition-1-en",
                "condition-1-en-5k",
                "condition-2-es-5k",
            ],
            "labels": ["Baseline", "C1-en\n(full)", "C1-en\n(5K)", "C2-es\n(5K)"],
            "colors": [
                CHAIN_COLORS["baseline"],
                CHAIN_COLORS["condition-1-en"],
                CHAIN_COLORS["condition-1-en-5k"],
                CHAIN_COLORS["condition-2"],
            ],
            "title": "Spanish (es)",
        },
        "ur": {
            "conditions": [
                "baseline",
                "condition-1-en",
                "condition-1-en-5k",
                "condition-2-ur-5k",
            ],
            "labels": ["Baseline", "C1-en\n(full)", "C1-en\n(5K)", "C2-ur\n(5K)"],
            "colors": [
                CHAIN_COLORS["baseline"],
                CHAIN_COLORS["condition-1-en"],
                CHAIN_COLORS["condition-1-en-5k"],
                CHAIN_COLORS["condition-2"],
            ],
            "title": "Urdu (ur)",
        },
    }

    for prompt_type in PROMPT_TYPES:
        prompt_label = "English" if prompt_type == "english" else "Native Language"
        subset = df[df["prompt_type"] == prompt_type]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            f"Experimental Chain by Language — {prompt_label} Prompt",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for ax, lang in zip(axes, LANGUAGES):
            chain = chains[lang]
            n_conds = len(chain["conditions"])
            x = np.arange(len(BENCHMARKS))
            width = 0.8 / n_conds

            for j, (cond, label, color) in enumerate(
                zip(chain["conditions"], chain["labels"], chain["colors"])
            ):
                scores = []
                for bm in BENCHMARKS:
                    match = subset[
                        (subset["condition"] == cond)
                        & (subset["benchmark"] == bm)
                        & (subset["language"] == lang)
                    ]
                    scores.append(match["score_pct"].values[0] if len(match) > 0 else 0)

                offset = (j - (n_conds - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    scores,
                    width,
                    label=label.replace("\n", " "),
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    if score > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{score:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            rotation=90,
                        )

            ax.set_title(chain["title"], fontsize=12, fontweight="bold")
            ax.set_xlabel("Benchmark")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x)
            ax.set_xticklabels([BENCHMARK_LABELS[bm] for bm in BENCHMARKS], fontsize=9)
            ax.set_ylim(0, 65)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

        fig.tight_layout()
        fname = f"language_chain_{prompt_type}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # Also generate a delta version showing change relative to Cond 1-en (5K)
    for prompt_type in PROMPT_TYPES:
        prompt_label = "English" if prompt_type == "english" else "Native Language"
        subset = df[df["prompt_type"] == prompt_type]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            f"Change vs Cond 1-en (5K) by Language — {prompt_label} Prompt",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for ax, lang in zip(axes, LANGUAGES):
            chain = chains[lang]

            # Get Cond 1-en-5k scores as reference
            ref_scores = {}
            for bm in BENCHMARKS:
                match = subset[
                    (subset["condition"] == "condition-1-en-5k")
                    & (subset["benchmark"] == bm)
                    & (subset["language"] == lang)
                ]
                ref_scores[bm] = match["score_pct"].values[0] if len(match) > 0 else 0

            # Only show conditions that aren't the reference
            plot_conds = [
                (c, l, col)
                for c, l, col in zip(
                    chain["conditions"], chain["labels"], chain["colors"]
                )
                if c != "condition-1-en-5k"
            ]
            n_conds = len(plot_conds)
            x = np.arange(len(BENCHMARKS))
            width = 0.8 / n_conds

            for j, (cond, label, color) in enumerate(plot_conds):
                deltas = []
                for bm in BENCHMARKS:
                    match = subset[
                        (subset["condition"] == cond)
                        & (subset["benchmark"] == bm)
                        & (subset["language"] == lang)
                    ]
                    val = match["score_pct"].values[0] if len(match) > 0 else 0
                    deltas.append(val - ref_scores[bm])

                offset = (j - (n_conds - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    deltas,
                    width,
                    label=label.replace("\n", " "),
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
                for bar, delta in zip(bars, deltas):
                    va = "bottom" if delta >= 0 else "top"
                    y_pos = bar.get_height() + (0.3 if delta >= 0 else -0.3)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f"{delta:+.1f}",
                        ha="center",
                        va=va,
                        fontsize=6,
                        rotation=90,
                    )

            ax.axhline(0, color="black", linewidth=0.8, label="Cond 1-en (5K)")
            ax.set_title(chain["title"], fontsize=12, fontweight="bold")
            ax.set_xlabel("Benchmark")
            ax.set_ylabel("Δ from Cond 1-en 5K (pp)")
            ax.set_xticks(x)
            ax.set_xticklabels([BENCHMARK_LABELS[bm] for bm in BENCHMARKS], fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

        fig.tight_layout()
        fname = f"language_chain_delta_{prompt_type}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Summary Table ────────────────────────────────────────────────────────────


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary table to console."""
    condition_order = CONDITIONS

    for prompt_type in PROMPT_TYPES:
        print(f"\n{'=' * 80}")
        print(
            f"  {'English' if prompt_type == 'english' else 'Native Language'} Prompt Results"
        )
        print(f"{'=' * 80}")

        header = f"{'Condition':<18}"
        for bm in BENCHMARKS:
            for lang in LANGUAGES:
                header += f" {bm.upper()}_{lang:>2}"
        print(header)
        print("-" * len(header))

        subset = df[df["prompt_type"] == prompt_type]
        for cond in condition_order:
            row_str = f"{CONDITION_LABELS_SHORT.get(cond, cond):<18}"
            for bm in BENCHMARKS:
                for lang in LANGUAGES:
                    match = subset[
                        (subset["condition"] == cond)
                        & (subset["benchmark"] == bm)
                        & (subset["language"] == lang)
                    ]
                    if len(match) > 0:
                        row_str += f" {match['score_pct'].values[0]:>7.1f}"
                    else:
                        row_str += f" {'—':>7}"
            print(row_str)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Tiny Aya evaluation comparison plots")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures (default: ../figures relative to this script)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: system default)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir) if args.output_dir else script_dir.parent / "figures"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data from HuggingFace (including baseline)
    print("Fetching all results from HuggingFace...")
    all_results = fetch_all_results(cache_dir)

    print("Building DataFrame...")
    df = build_dataframe(all_results)
    print(f"  {len(df)} rows loaded")

    if df.empty:
        print("No data loaded — exiting.")
        return

    # Print summary
    print_summary_table(df)

    # Generate charts
    print("\nGenerating charts...")
    plot_grouped_bars(df, output_dir)
    plot_single_language_bars(df, output_dir)
    plot_delta_bars(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_prompt_comparison(df, output_dir)
    plot_language_vs_cond1(df, output_dir)

    print(f"\nAll charts saved to {output_dir}")


if __name__ == "__main__":
    main()

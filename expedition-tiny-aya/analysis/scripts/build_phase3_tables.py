"""Build Phase-3 LaTeX result tables (EMNLP submission).

Generates 14 booktabs-formatted tables into a single
``expedition-tiny-aya/analysis/phase-3/tables.tex`` file:

* Type 1 — refined-extractor accuracy, one per benchmark   (4 tables)
* Type 2 — native- vs English-prompt refined accuracy       (2 tables)
* Type 3 — orig vs refined side-by-side, one per benchmark (4 tables)
* Type 4 — Δ-vs-baseline (refined), one per benchmark      (4 tables)

Data source:
    HuggingFace dataset ``legesher/language-decoded-experiments``,
    file ``phase3/analysis/refined-tables/vs_baseline_cells.tsv``.
    Cells are at (condition × seed × template × benchmark × data × instr).
    Aggregation is mean across (seed, template, data) within each
    (condition, benchmark, instr) cell — except Type 2 which collapses
    across (seed, template) within (condition, benchmark, instr, data=lang).

Conventions:
    * Language order:     en → es → zh → ur (resource-tier descending)
    * Benchmark order:    SIB-200, XNLI, X-CSQA, Belebele
    * Conditions:         13 rows starting with italicised Baseline
    * Numbers:            percentages (1 decimal); Δs with sign and LaTeX math-mode minus ``$-$``
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ─── Constants ──────────────────────────────────────────────────────────────

HF_REPO_ID = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"
HF_TABLES_PREFIX = "phase3/analysis/refined-tables"

LANG_ORDER = ["en", "es", "zh", "ur"]
NON_EN_LANGS = ["es", "zh", "ur"]
BENCH_ORDER = ["sib200", "xnli", "csqa", "belebele"]
BENCH_LABEL = {
    "sib200": "SIB-200",
    "xnli": "XNLI",
    "csqa": "X-CSQA",
    "belebele": "Belebele",
}
BENCH_SHORT = {
    "sib200": "sib200",
    "xnli": "xnli",
    "csqa": "xcsqa",
    "belebele": "belebele",
}

CONDITIONS: list[tuple[str, str]] = [
    ("baseline", r"\textit{Baseline}"),
    ("condition-1-en-5k", r"Cond 1 (en, 5k)"),
    ("condition-1-en-20k", r"Cond 1 (en, 20k)"),
    ("condition-2-es-5k", r"Cond 2 (es, 5k)"),
    ("condition-2-es-20k", r"Cond 2 (es, 20k)"),
    ("condition-2-zh-5k", r"Cond 2 (zh, 5k)"),
    ("condition-2-zh-20k", r"Cond 2 (zh, 20k)"),
    ("condition-2-ur-5k", r"Cond 2 (ur, 5k)"),
    ("condition-2-ur-20k", r"Cond 2 (ur, 20k)"),
    ("condition-3-zh-5k", r"Cond 3 (zh, 5k)"),
    ("condition-5-es-5k", r"Cond 5 (es, 5k)"),
    ("condition-5-ur-5k", r"Cond 5 (ur, 5k)"),
    ("condition-5-zh-5k", r"Cond 5 (zh, 5k)"),
]

# Repo-relative — this script lives at expedition-tiny-aya/analysis/scripts/,
# tables.tex lands at expedition-tiny-aya/analysis/phase-3/.
_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_FILE = _SCRIPT_DIR.parent / "phase-3" / "tables.tex"

# LaTeX math-mode minus. pdfLaTeX warns/fails on the Unicode minus
# U+2212; ``$-$`` renders consistently across pdfLaTeX, XeLaTeX, and LuaLaTeX.
MINUS = "$-$"

# ─── IO ─────────────────────────────────────────────────────────────────────


def load_cells() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=f"{HF_TABLES_PREFIX}/vs_baseline_cells.tsv",
    )
    df = pd.read_csv(cached, sep="\t")
    # Delta columns may be strings with leading + sign; coerce.
    for c in ("delta_orig", "delta_rep"):
        if df[c].dtype == object:
            df[c] = df[c].str.replace("+", "", regex=False).astype(float)
    return df


# ─── Formatting helpers ─────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x * 100:.1f}\\%"


def fmt_delta(x: float) -> str:
    if pd.isna(x):
        return "--"
    # Round before sign-branching so a value that rounds to 0.0 but carries a
    # negative IEEE-754 sign (e.g. x = -1e-5 → pp = -0.001 → rounds to 0.0)
    # doesn't emit "$-$0.0". Reassigning to a positive 0.0 strips the sign bit.
    pp = round(x * 100.0, 1)
    if pp == 0:
        pp = 0.0
    if pp >= 0:
        return f"+{pp:.1f}"
    return f"{MINUS}{abs(pp):.1f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


# ─── Baseline grid (no "baseline" condition row in TSV) ─────────────────────


def baseline_grid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    deduped = df.drop_duplicates(
        subset=["template", "benchmark", "data", "instr"]
    )
    return (
        deduped.groupby(["benchmark", "instr"])[value_col].mean().unstack("instr")
    )


def baseline_native_row(df: pd.DataFrame, value_col: str) -> pd.Series:
    sub = df[df.data == df.instr]
    deduped = sub.drop_duplicates(
        subset=["template", "benchmark", "data", "instr"]
    )
    return deduped.groupby(["benchmark", "instr"])[value_col].mean()


def baseline_english_row(df: pd.DataFrame, value_col: str) -> pd.Series:
    sub = df[df.instr == "en"]
    deduped = sub.drop_duplicates(
        subset=["template", "benchmark", "data", "instr"]
    )
    return deduped.groupby(["benchmark", "data"])[value_col].mean()


# ─── Aggregation primitives ─────────────────────────────────────────────────


def cond_grid(
    df: pd.DataFrame, value_col: str, benchmark: str
) -> pd.DataFrame:
    sub = df[df.benchmark == benchmark]
    return sub.groupby(["condition", "instr"])[value_col].mean().unstack("instr")


def cond_native_grid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    sub = df[df.data == df.instr]
    return (
        sub.groupby(["condition", "benchmark", "instr"])[value_col]
        .mean()
        .unstack(["benchmark", "instr"])
    )


def cond_english_grid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    sub = df[df.instr == "en"]
    return (
        sub.groupby(["condition", "benchmark", "data"])[value_col]
        .mean()
        .unstack(["benchmark", "data"])
    )


# ─── Sign-flip detection (Type 3) ───────────────────────────────────────────


# Same noise-floor threshold as `build_vs_baseline.py`'s conclusion_flips
# detection — below seed-to-seed reproducibility noise (~0.03 std on SIB-200
# cond-2-X-5k cells). Keeps Type-3 table bolding consistent with the canonical
# `conclusion_flips.tsv` semantics, so a cell bolded here is one that also
# appears in the flip catalogue (modulo aggregation grain).
_FLIP_BUFFER = 0.01


def signflip_mask(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    sub = df[df.benchmark == benchmark]
    agg = (
        sub.groupby(["condition", "instr"])[["delta_orig", "delta_rep"]]
        .mean()
        .reset_index()
    )

    def flip(row: pd.Series) -> bool:
        o, r = row["delta_orig"], row["delta_rep"]
        if pd.isna(o) or pd.isna(r):
            return False
        return (o > _FLIP_BUFFER and r < -_FLIP_BUFFER) or (
            o < -_FLIP_BUFFER and r > _FLIP_BUFFER
        )

    agg["flip"] = agg.apply(flip, axis=1)
    return agg.set_index(["condition", "instr"])["flip"].unstack("instr")


# ─── Lookup ─────────────────────────────────────────────────────────────────


def lookup(grid: pd.DataFrame, cond: str, col) -> float:
    if cond not in grid.index:
        return float("nan")
    if col not in grid.columns:
        return float("nan")
    return grid.loc[cond, col]


# ─── Table writers ──────────────────────────────────────────────────────────


def write_type1_table(df: pd.DataFrame, benchmark: str) -> str:
    grid = cond_grid(df, "cond_rep", benchmark)
    base = baseline_grid(df, "baseline_rep")

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(LANG_ORDER) + r"}",
        r"    \toprule",
        "    Condition & " + " & ".join(LANG_ORDER) + r" \\",
        r"    \midrule",
    ]
    for cond, label in CONDITIONS:
        cells: list[str] = []
        for lang in LANG_ORDER:
            if cond == "baseline":
                cells.append(fmt_pct(base.loc[benchmark, lang]))
            else:
                cells.append(fmt_pct(lookup(grid, cond, lang)))
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Phase-3 refined-extractor accuracy on \textsc{"
        + BENCH_LABEL[benchmark]
        + r"}, by condition $\times$ instruction language; cell values are "
        r"means across seed, template, and data language.}",
        r"  \label{tab:t1-" + BENCH_SHORT[benchmark] + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _type2_table(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    base: pd.Series,
    *,
    label: str,
    caption: str,
) -> str:
    cols = [(b, lang) for b in BENCH_ORDER for lang in NON_EN_LANGS]

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \begin{tabular}{l" + "r" * len(cols) + r"}",
        r"    \toprule",
    ]
    bench_header_parts: list[str] = []
    cmidrules: list[str] = []
    col_idx = 2
    for b in BENCH_ORDER:
        bench_header_parts.append(
            r"\multicolumn{3}{c}{\textsc{" + BENCH_LABEL[b] + r"}}"
        )
        cmidrules.append(r"\cmidrule(lr){" + f"{col_idx}-{col_idx + 2}" + r"}")
        col_idx += 3
    lines.append(r"    Condition & " + " & ".join(bench_header_parts) + r" \\")
    lines.append("    " + " ".join(cmidrules))
    lines.append(r"     & " + " & ".join(lang for _, lang in cols) + r" \\")
    lines.append(r"    \midrule")

    for cond, clabel in CONDITIONS:
        cells: list[str] = []
        for (b, lang) in cols:
            if cond == "baseline":
                cells.append(fmt_pct(base.get((b, lang), float("nan"))))
            else:
                cells.append(fmt_pct(lookup(grid, cond, (b, lang))))
        lines.append(f"    {clabel} & " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{" + caption + r"}",
        r"  \label{" + label + r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def write_type2_native(df: pd.DataFrame) -> str:
    return _type2_table(
        df,
        cond_native_grid(df, "cond_rep"),
        baseline_native_row(df, "baseline_rep"),
        label="tab:t2-native",
        caption=(
            r"Phase-3 native-prompt refined accuracy "
            r"(instr $=$ data $=$ target language); Phase-3 equivalent of the "
            r"Phase-2 ``Native Prompt Results'' table."
        ),
    )


def write_type2_english(df: pd.DataFrame) -> str:
    return _type2_table(
        df,
        cond_english_grid(df, "cond_rep"),
        baseline_english_row(df, "baseline_rep"),
        label="tab:t2-english",
        caption=(
            r"Phase-3 English-prompt refined accuracy "
            r"(instr $=$ en, data $=$ target language); Phase-3 equivalent of "
            r"the Phase-2 ``English Prompt Results'' table."
        ),
    )


def write_type3_table(df: pd.DataFrame, benchmark: str) -> str:
    orig = cond_grid(df, "cond_orig", benchmark)
    rep = cond_grid(df, "cond_rep", benchmark)
    base_orig = baseline_grid(df, "baseline_orig")
    base_rep = baseline_grid(df, "baseline_rep")
    flips = signflip_mask(df, benchmark)

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "rr" * len(LANG_ORDER) + r"}",
        r"    \toprule",
    ]
    bench_header_parts = [
        r"\multicolumn{2}{c}{instr $=$ " + lang + r"}" for lang in LANG_ORDER
    ]
    cmidrules: list[str] = []
    col_idx = 2
    for _ in LANG_ORDER:
        cmidrules.append(r"\cmidrule(lr){" + f"{col_idx}-{col_idx + 1}" + r"}")
        col_idx += 2
    lines.append(r"    Condition & " + " & ".join(bench_header_parts) + r" \\")
    lines.append("    " + " ".join(cmidrules))
    lines.append(r"     & " + " & ".join(["orig & rep"] * len(LANG_ORDER)) + r" \\")
    lines.append(r"    \midrule")

    for cond, label in CONDITIONS:
        cells: list[str] = []
        for lang in LANG_ORDER:
            if cond == "baseline":
                o_val = base_orig.loc[benchmark, lang]
                r_val = base_rep.loc[benchmark, lang]
                cells.append(fmt_pct(o_val))
                cells.append(fmt_pct(r_val))
            else:
                o_val = lookup(orig, cond, lang)
                r_val = lookup(rep, cond, lang)
                rep_s = fmt_pct(r_val)
                if (
                    not pd.isna(r_val)
                    and cond in flips.index
                    and lang in flips.columns
                    and bool(flips.loc[cond, lang])
                ):
                    rep_s = bold(rep_s)
                cells.append(fmt_pct(o_val))
                cells.append(rep_s)
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Phase-3 accuracy on \textsc{"
        + BENCH_LABEL[benchmark]
        + r"} under the original (orig) and refined (rep) extractors, by "
        r"condition $\times$ instruction language; bold refined values mark "
        r"sign-flips vs baseline (see \S8.3).}",
        r"  \label{tab:t3-" + BENCH_SHORT[benchmark] + r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def write_type4_table(df: pd.DataFrame, benchmark: str) -> str:
    delta = cond_grid(df, "delta_rep", benchmark)

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(LANG_ORDER) + r"}",
        r"    \toprule",
        "    Condition & " + " & ".join(LANG_ORDER) + r" \\",
        r"    \midrule",
    ]
    for cond, label in CONDITIONS:
        if cond == "baseline":
            continue
        cells: list[str] = []
        for lang in LANG_ORDER:
            val = lookup(delta, cond, lang)
            s = fmt_delta(val)
            if not pd.isna(val) and abs(val) > 0.05:
                s = bold(s)
            cells.append(s)
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Phase-3 $\Delta$-vs-baseline under refined scoring on "
        r"\textsc{" + BENCH_LABEL[benchmark] + r"}, by condition $\times$ "
        r"instruction language; values in percentage points; bold marks "
        r"$|\Delta| > 5$\,pp.}",
        r"  \label{tab:t4-" + BENCH_SHORT[benchmark] + r"-delta}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── Driver ─────────────────────────────────────────────────────────────────


def main() -> None:
    df = load_cells()
    print(
        f"[load] rows={len(df)} "
        f"conditions={df.condition.nunique()} "
        f"benchmarks={sorted(df.benchmark.unique())} "
        f"seeds={sorted(df.seed.unique())} "
        f"templates={sorted(df.template.unique())}"
    )

    out: list[str] = []
    out.append("% Phase-3 result tables — generated by")
    out.append("% expedition-tiny-aya/analysis/scripts/build_phase3_tables.py")
    out.append(
        "% Source: HF legesher/language-decoded-experiments :: "
        "phase3/analysis/refined-tables/vs_baseline_cells.tsv"
    )
    out.append("% Requires: \\usepackage{booktabs}")
    out.append("")

    out.append("% ─── Type 1: Refined-extractor accuracy, one per benchmark ───")
    out.append("")
    for b in BENCH_ORDER:
        block = write_type1_table(df, b)
        print(f"[type1:{b}] rows_emitted={block.count(chr(92) + chr(92))}")
        out.append(block)
        out.append("")

    out.append("% ─── Type 2: Native vs English prompt ───")
    out.append("")
    block = write_type2_native(df)
    print(f"[type2:native] rows_emitted={block.count(chr(92) + chr(92))}")
    out.append(block)
    out.append("")
    block = write_type2_english(df)
    print(f"[type2:english] rows_emitted={block.count(chr(92) + chr(92))}")
    out.append(block)
    out.append("")

    out.append("% ─── Type 3: Orig vs refined side-by-side, one per benchmark ───")
    out.append("")
    for b in BENCH_ORDER:
        block = write_type3_table(df, b)
        print(f"[type3:{b}] rows_emitted={block.count(chr(92) + chr(92))}")
        out.append(block)
        out.append("")

    out.append("% ─── Type 4: Δ-vs-baseline (refined), one per benchmark ───")
    out.append("")
    for b in BENCH_ORDER:
        block = write_type4_table(df, b)
        print(f"[type4:{b}] rows_emitted={block.count(chr(92) + chr(92))}")
        out.append(block)
        out.append("")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(out), encoding="utf-8")
    print(f"[write] {OUT_FILE}  size={OUT_FILE.stat().st_size} bytes")

    # ─── Spot-checks ────────────────────────────────────────────────────
    sib_ur = (
        df[(df.benchmark == "sib200") & (df.instr == "ur")]["cond_rep"].mean()
    )
    print(f"[check] mean cond_rep SIB-200 instr=ur (all cond): {sib_ur:.4f}")
    g = cond_grid(df, "cond_rep", "sib200")
    print(
        f"[check] cond_rep SIB-200 cond-2-ur-5k instr=ur: "
        f"{g.loc['condition-2-ur-5k', 'ur']:.4f}"
    )
    base_n = baseline_native_row(df, "baseline_rep")
    print(
        f"[check] baseline native SIB-200 es: "
        f"{base_n.get(('sib200', 'es'), float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()

"""Build Phase-3 LaTeX result tables (EMNLP submission).

Generates booktabs-formatted tables into a single
``expedition-tiny-aya/analysis/phase-3/tables.tex`` file:

* Type 1 — refined-extractor accuracy, one per benchmark   (4 tables)
* Type 2 — native- vs English-prompt refined accuracy       (2 tables)
              + matched-diagonal Δ variant (P2)              (1 table)
* Type 3 — orig vs refined side-by-side, one per benchmark (4 tables)
* Type 4 — Δ-vs-baseline (refined), one per benchmark      (4 tables)

Type-1/2/4 tables carry an ``$n_s$'' column showing the seed count for
each condition (P1), so a reader can distinguish 3-seed multi-trial
estimates from 1-seed point estimates at a glance.

Plus tables tied to specific paper sections in §4 (added later, in
paper-section order rather than table-type order — see each writer's
docstring for which section it serves):

* §4.1  Baseline-accuracy headroom              (1 table)
* §4.3  Per-language matched-language ladder    (3 tables — es / zh / ur)
* §4.4  Cond-2 vs Cond-5 head-to-head           (1 table)
* §4.6  Cross-lingual transfer (instr=en cells) (1 table)
* §4.7  SIB-200 sign-flip catalogue             (1 table)

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
BENCH_ORDER = ["sib200", "xnli", "csqa", "belebele"]

# Per-benchmark bolding thresholds for Type 4 (Δ-vs-baseline) tables.
# Calibrated from 2× the worst-case seed-σ observed across the four
# multi-seed (5K) conditions on each benchmark; below these magnitudes a Δ
# is comparable to seed-to-seed reproducibility noise and shouldn't be
# typographically highlighted. A flat threshold would mislead readers on
# SIB-200 (where small n=204 and high seed σ inflate noise) and miss real
# small effects on X-CSQA / Belebele (large n, tight seed σ).
TYPE4_BOLD_THRESHOLD_FRAC = {
    "sib200":   0.08,  # 8pp — n=204, max seed σ ≈ 4.3pp across 5K conds
    "xnli":     0.04,  # 4pp — n=2505, max seed σ ≈ 2.0pp
    "csqa":     0.03,  # 3pp — n=1000, max seed σ ≈ 1.6pp
    "belebele": 0.04,  # 4pp — n=900, max seed σ ≈ 1.9pp
}
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


# ─── P1 helper: n_seeds per condition ───────────────────────────────────────


def n_seeds_for(df: pd.DataFrame, cond: str) -> int:
    """Return the seed count for a condition. Baseline returns 1 (one
    un-fine-tuned reference run with seed=none); fine-tuned conditions
    return 3 or 1 depending on which seeds were actually run."""
    if cond == "baseline":
        return 1
    return int(df[df.condition == cond]["seed"].nunique())


def fmt_n_seeds(n: int, cond: str) -> str:
    """LaTeX cell for n_seeds. Baseline shows '—' since seed=none is not a
    seed in the fine-tuning sense; everything else shows the integer."""
    if cond == "baseline":
        return "--"
    return str(n)

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
        # Condition + n_seeds + 4 lang cols
        r"  \begin{tabular}{lr" + "r" * len(LANG_ORDER) + r"}",
        r"    \toprule",
        r"    Condition & $n_s$ & " + " & ".join(LANG_ORDER) + r" \\",
        r"    \midrule",
    ]
    for cond, label in CONDITIONS:
        ns = fmt_n_seeds(n_seeds_for(df, cond), cond)
        cells: list[str] = []
        for lang in LANG_ORDER:
            if cond == "baseline":
                cells.append(fmt_pct(base.loc[benchmark, lang]))
            else:
                cells.append(fmt_pct(lookup(grid, cond, lang)))
        lines.append(f"    {label} & {ns} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Phase-3 refined-extractor accuracy on \textsc{"
        + BENCH_LABEL[benchmark]
        + r"}, by condition $\times$ instruction language; cell values are "
        r"means across seed, template, and data language. The $n_s$ column "
        r"reports the number of seeds the condition was trained with; cells "
        r"in $n_s\!=\!1$ rows are point estimates without seed-spread "
        r"information.}",
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
    cell_formatter=fmt_pct,
    skip_baseline: bool = False,
) -> str:
    """Render a Type-2 booktabs table (rows = conditions, cols = 4 bench × 4 lang).

    cell_formatter: how to format each numeric cell (default fmt_pct for
                    absolute accuracy; pass fmt_delta for Δ variants).
    skip_baseline:  set True for Δ variants where the baseline row would be
                    all zeros.
    """
    langs = LANG_ORDER
    n_langs = len(langs)
    cols = [(b, lang) for b in BENCH_ORDER for lang in langs]

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \scriptsize",
        # Condition + n_seeds + 16 data cols
        r"  \begin{tabular}{lr" + "r" * len(cols) + r"}",
        r"    \toprule",
    ]
    bench_header_parts: list[str] = []
    cmidrules: list[str] = []
    # Column 1 = Condition, column 2 = n_seeds. Benchmark groups start at column 3.
    col_idx = 3
    for b in BENCH_ORDER:
        bench_header_parts.append(
            r"\multicolumn{" + str(n_langs) + r"}{c}{\textsc{"
            + BENCH_LABEL[b] + r"}}"
        )
        cmidrules.append(
            r"\cmidrule(lr){" + f"{col_idx}-{col_idx + n_langs - 1}" + r"}"
        )
        col_idx += n_langs
    lines.append(r"    Condition & $n_s$ & " + " & ".join(bench_header_parts) + r" \\")
    lines.append("    " + " ".join(cmidrules))
    lines.append(r"     & & " + " & ".join(lang for _, lang in cols) + r" \\")
    lines.append(r"    \midrule")

    for cond, clabel in CONDITIONS:
        if skip_baseline and cond == "baseline":
            continue
        ns = fmt_n_seeds(n_seeds_for(df, cond), cond)
        cells: list[str] = []
        for (b, lang) in cols:
            if cond == "baseline":
                cells.append(cell_formatter(base.get((b, lang), float("nan"))))
            else:
                cells.append(cell_formatter(lookup(grid, cond, (b, lang))))
        lines.append(f"    {clabel} & {ns} & " + " & ".join(cells) + r" \\")

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


def write_type2_native_delta(df: pd.DataFrame) -> str:
    """Matched-diagonal Δ-vs-baseline variant of Type-2 native (P2).

    Same shape as Type-2 native (cond rows × bench × lang cols) but cell
    values are Δ-vs-baseline computed at the matched-diagonal grain
    (data $=$ instr $=$ language for that column), with NO cross-data
    averaging. This is the methodologically tightest "did this condition
    improve on its target task" view.

    Companion to Type-4 (Δ-vs-baseline, but pooled across data_lang under
    instr_lang). Reading the two side-by-side reveals when an instr-only
    aggregate hides a smaller (or opposite-signed) matched-diagonal
    effect. Worked example: Cond 2-ur-5k Belebele instr=ur in Type-4
    shows +5.9pp (bolded); at the matched diagonal (data=ur, instr=ur)
    this is the same Cond 2-ur-5k Belebele ur cell in this table, where
    Δ = +4.1pp — both positive, but the matched-diagonal magnitude is
    smaller. For Cond 2-ur-5k SIB-200 the gap is starker: Type-4
    instr=ur is +12.0pp; matched-diagonal is −9.0pp.
    """
    return _type2_table(
        df,
        cond_native_grid(df, "delta_rep"),
        baseline_native_row(df, "delta_rep"),  # unused (skip_baseline)
        label="tab:t2-native-delta",
        caption=(
            r"Matched-diagonal $\Delta$-vs-baseline (refined extractor): "
            r"each cell is the condition's $\Delta$ accuracy on data $=$ "
            r"instr $=$ the column language, in percentage points. Unlike "
            r"Type-4 (which averages across data languages under each "
            r"instr language), this view fixes both the data language "
            r"and the instr language to the column header, giving the "
            r"tightest ``matched-language gain'' estimate. Reading this "
            r"table alongside Type-4 surfaces cases where an "
            r"instr-aggregate $\Delta$ masks a different matched-diagonal "
            r"effect: e.g. Cond 2-ur-5k \textsc{SIB-200} ur shows "
            r"$-9.0$\,pp here vs $+12.0$\,pp in Type-4 (sign flip); "
            r"Cond 2-ur-5k \textsc{Belebele} ur shows $+4.1$\,pp here vs "
            r"$+5.9$\,pp in Type-4 (same sign, smaller magnitude). The "
            r"$n_s$ column carries the same seed-count meaning as in "
            r"Type-1/2/4."
        ),
        cell_formatter=fmt_delta,
        skip_baseline=True,
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
        # Condition + n_seeds + 4 lang cols
        r"  \begin{tabular}{lr" + "r" * len(LANG_ORDER) + r"}",
        r"    \toprule",
        r"    Condition & $n_s$ & " + " & ".join(LANG_ORDER) + r" \\",
        r"    \midrule",
    ]
    threshold = TYPE4_BOLD_THRESHOLD_FRAC[benchmark]
    threshold_pp = int(round(threshold * 100))
    for cond, label in CONDITIONS:
        if cond == "baseline":
            continue
        ns = fmt_n_seeds(n_seeds_for(df, cond), cond)
        cells: list[str] = []
        for lang in LANG_ORDER:
            val = lookup(delta, cond, lang)
            s = fmt_delta(val)
            if not pd.isna(val) and abs(val) > threshold:
                s = bold(s)
            cells.append(s)
        lines.append(f"    {label} & {ns} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Phase-3 $\Delta$-vs-baseline under refined scoring on "
        r"\textsc{" + BENCH_LABEL[benchmark] + r"}, by condition $\times$ "
        r"instruction language; values in percentage points; bold marks "
        r"$|\Delta| > " + str(threshold_pp) + r"$\,pp (per-benchmark threshold, "
        r"calibrated from $2 \times$ the worst-case seed-$\sigma$ observed across "
        r"the four multi-seed 5K conditions on this benchmark). Cells in "
        r"$n_s\!=\!1$ rows are point estimates without seed-spread; treat "
        r"borderline-bolded values as suggestive rather than robust.}",
        r"  \label{tab:t4-" + BENCH_SHORT[benchmark] + r"-delta}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── §4.1 Baseline headroom (Type 5) ────────────────────────────────────────


def write_table_baseline_headroom(df: pd.DataFrame) -> str:
    """4 benchmarks × 4 instr-langs + row mean. Baseline accuracy per cell.

    Picks baseline_rep (same across all condition rows for the same
    (benchmark, data, instr, template), so dedupe before averaging). Aggregates
    across template + data_lang within each (benchmark, instr) pair.

    Paper section: §4.1 (motivates the low-resource-language framing).
    """
    grid = baseline_grid(df, "baseline_rep")  # benchmark × instr
    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * (len(LANG_ORDER) + 1) + r"}",
        r"    \toprule",
        "    Benchmark & " + " & ".join(f"instr={l}" for l in LANG_ORDER) + r" & mean \\",
        r"    \midrule",
    ]
    for b in BENCH_ORDER:
        cells = [fmt_pct(grid.loc[b, l] if l in grid.columns else float("nan"))
                 for l in LANG_ORDER]
        row_mean = grid.loc[b, LANG_ORDER].mean()
        cells.append(fmt_pct(row_mean))
        lines.append(f"    \\textsc{{{BENCH_LABEL[b]}}} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Baseline accuracy of Tiny Aya by benchmark and instruction "
        r"language, under refined extractor. The lowest-resource language "
        r"(Urdu) has the lowest baseline across every benchmark, providing the "
        r"largest representational headroom for fine-tuning to fill.}",
        r"  \label{tab:baseline-headroom}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── §4.3 Matched-language ladder (Type 6) ─────────────────────────────────


_MATCHED_LADDER_CONDS_BASE: list[tuple[str, str]] = [
    ("baseline", r"\textit{Baseline}"),
    ("condition-1-en-5k", "Cond 1 (en, 5k)"),
    ("condition-1-en-20k", "Cond 1 (en, 20k)"),
]


def _matched_ladder_conditions(lang: str) -> list[tuple[str, str]]:
    rows = list(_MATCHED_LADDER_CONDS_BASE)
    rows.append((f"condition-2-{lang}-5k", f"Cond 2 ({lang}, 5k)"))
    rows.append((f"condition-2-{lang}-20k", f"Cond 2 ({lang}, 20k)"))
    if lang == "zh":
        rows.append(("condition-3-zh-5k", "Cond 3 (zh, 5k)"))
    rows.append((f"condition-5-{lang}-5k", f"Cond 5 ({lang}, 5k)"))
    return rows


def _matched_per_seed_stats(
    df: pd.DataFrame, condition: str, benchmark: str, lang: str
) -> tuple[float, float, int]:
    """For (cond, benchmark, data=lang, instr=lang), compute (mean, std, n_seeds)
    of per-seed accuracies, where each seed's value is averaged across templates."""
    sub = df[
        (df.condition == condition) & (df.benchmark == benchmark)
        & (df.data == lang) & (df.instr == lang)
    ]
    per_seed = sub.groupby("seed")["cond_rep"].mean()
    n = len(per_seed)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if n > 1 else float("nan"), n


def _matched_baseline(df: pd.DataFrame, benchmark: str, lang: str) -> float:
    sub = df[
        (df.benchmark == benchmark) & (df.data == lang) & (df.instr == lang)
    ].drop_duplicates(subset=["template"])
    return float(sub["baseline_rep"].mean()) if len(sub) else float("nan")


def fmt_delta_with_std(delta: float, std: float) -> str:
    if pd.isna(delta):
        return "--"
    pp = round(delta * 100.0, 1)
    if pp == 0:
        pp = 0.0
    base = f"+{pp:.1f}" if pp >= 0 else f"{MINUS}{abs(pp):.1f}"
    if pd.isna(std):
        return base
    return base + r"\,$\pm$\," + f"{std * 100:.1f}"


def write_table_matched_ladder(df: pd.DataFrame, lang: str) -> str:
    """Per-language matched-diagonal ladder: data == instr == lang.

    Baseline row shows absolute accuracy (no Δ); subsequent rows show
    Δ-vs-baseline ± seed-std where multi-seed.

    Paper section: §4.3 (headline cond-2 matched-language gains).
    """
    conds = _matched_ladder_conditions(lang)
    lang_name = {"es": "Spanish", "zh": "Chinese", "ur": "Urdu"}[lang]

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(BENCH_ORDER) + r"}",
        r"    \toprule",
        "    Condition & " + " & ".join(
            f"\\textsc{{{BENCH_LABEL[b]}}}" for b in BENCH_ORDER) + r" \\",
        r"    \midrule",
    ]
    # Baseline row: absolute %, no std
    baseline_row_cells = [
        fmt_pct(_matched_baseline(df, b, lang)) for b in BENCH_ORDER
    ]
    lines.append(r"    \textit{Baseline} & " + " & ".join(baseline_row_cells) + r" \\")
    for cond_id, cond_label in conds[1:]:  # skip baseline (already emitted)
        cells: list[str] = []
        base = {b: _matched_baseline(df, b, lang) for b in BENCH_ORDER}
        for b in BENCH_ORDER:
            mean_acc, std_acc, n_seeds = _matched_per_seed_stats(df, cond_id, b, lang)
            if n_seeds == 0:
                cells.append("--")
                continue
            delta = mean_acc - base[b]
            cells.append(fmt_delta_with_std(delta, std_acc))
        lines.append(f"    {cond_label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Matched-language ladder for " + lang_name + r": each "
        r"non-baseline row is the condition's mean $\Delta$ accuracy vs "
        r"baseline on matched-diagonal cells (data $=$ instr $=$ "
        + lang + r"), under refined extractor, with $\pm$std across seeds "
        r"where multi-seed data is available. Baseline row shows absolute "
        r"accuracy (no $\Delta$).}",
        r"  \label{tab:matched-ladder-" + lang + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── §4.4 Cond-2 vs Cond-5 head-to-head (Type 7) ───────────────────────────


def write_table_cond2_vs_cond5(df: pd.DataFrame) -> str:
    """12 rows (3 langs × 4 benchmarks), comparing cond-2-{L}-5k vs cond-5-{L}-5k
    at matched-instruction grain. instr == target_lang; mean across seed,
    template, data_lang.

    Paper section: §4.4 (cond-5 head-to-head presentation).
    """
    rows_data: list[tuple[str, str, str, float, float, float]] = []
    for lang in ("es", "zh", "ur"):
        for b in BENCH_ORDER:
            row: dict[str, float] = {}
            for which, cond in (("c2", f"condition-2-{lang}-5k"),
                                ("c5", f"condition-5-{lang}-5k")):
                sub = df[(df.condition == cond) & (df.benchmark == b) & (df.instr == lang)]
                row[which] = float(sub["delta_rep"].mean()) if len(sub) else float("nan")
            gap = (row["c2"] - row["c5"]) if not (pd.isna(row["c2"]) or pd.isna(row["c5"])) else float("nan")
            rows_data.append((lang, b, "", row["c2"], row["c5"], gap))

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Lang & Benchmark & Cond 2 (5k) $\Delta$ & Cond 5 (5k) $\Delta$ & Gap (C2$-$C5) \\",
        r"    \midrule",
    ]
    prev_lang = None
    for lang, b, _, c2, c5, gap in rows_data:
        lang_cell = lang if lang != prev_lang else ""
        prev_lang = lang
        c2_s = fmt_delta(c2)
        c5_s = fmt_delta(c5)
        gap_s = fmt_delta(gap)
        if not pd.isna(gap) and abs(gap) > 0.05:
            gap_s = bold(gap_s)
        lines.append(f"    {lang_cell} & \\textsc{{{BENCH_LABEL[b]}}} & {c2_s} & {c5_s} & {gap_s} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Translation-aggressiveness controlled comparison: "
        r"\textbf{Cond 2} (keyword-only translation) vs \textbf{Cond 5} "
        r"(combined keyword + LLM natural-language translation) at "
        r"matched-instruction grain (instr $=$ target language). Both "
        r"conditions train on the same 5{,}000 source files from "
        r"\texttt{bigcode/the-stack-v2-dedup}; only the translation pipeline "
        r"differs. Values are mean $\Delta$ accuracy vs baseline under "
        r"refined extractor, in percentage points (mean across seed, "
        r"template, data\_lang). Bold gap marks $|$gap$| > 5$\,pp.}",
        r"  \label{tab:cond2-vs-cond5}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── §4.6 Cross-lingual transfer (Type 8) ──────────────────────────────────


_CROSS_LINGUAL_CONDS: list[tuple[str, str]] = [
    ("condition-1-en-5k", "Cond 1 (en, 5k)"),
    ("condition-2-es-5k", "Cond 2 (es, 5k)"),
    ("condition-2-zh-5k", "Cond 2 (zh, 5k)"),
    ("condition-2-ur-5k", "Cond 2 (ur, 5k)"),
    ("condition-5-es-5k", "Cond 5 (es, 5k)"),
    ("condition-5-zh-5k", "Cond 5 (zh, 5k)"),
    ("condition-5-ur-5k", "Cond 5 (ur, 5k)"),
]


def write_table_cross_lingual_transfer(df: pd.DataFrame) -> str:
    """Effect of target-language fine-tuning on English evaluation cells.

    Filter: instr == en. Aggregation: mean delta_rep across seed, template,
    data_lang, grouped by (condition, benchmark).

    Paper section: §4.6 (cross-lingual transfer, secondary finding).
    """
    sub = df[df.instr == "en"]

    def cell(cond: str, benchmark: str) -> float:
        s = sub[(sub.condition == cond) & (sub.benchmark == benchmark)]
        return float(s["delta_rep"].mean()) if len(s) else float("nan")

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(BENCH_ORDER) + r"}",
        r"    \toprule",
        "    Condition & " + " & ".join(
            f"\\textsc{{{BENCH_LABEL[b]}}} $\\Delta$" for b in BENCH_ORDER) + r" \\",
        r"    \midrule",
    ]
    for cond_id, cond_label in _CROSS_LINGUAL_CONDS:
        cells: list[str] = []
        for b in BENCH_ORDER:
            v = cell(cond_id, b)
            s = fmt_delta(v)
            if not pd.isna(v):
                if v > 0.02:
                    s = bold(s)
                elif v < -0.02:
                    s = r"\textit{" + s + r"}"
            cells.append(s)
        lines.append(f"    {cond_label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Cross-lingual transfer: effect of fine-tuning "
        r"conditions on English evaluation (instr $=$ en cells), under "
        r"refined extractor. Cond-2-ur-5k improves English Belebele and "
        r"English XNLI more than Cond-1-en-5k does --- a surprising "
        r"secondary finding suggesting that low-resource code fine-tuning "
        r"provides representational signal beyond the matched language. "
        r"Values are mean $\Delta$ accuracy vs baseline in percentage "
        r"points; bold marks $\Delta > +2$\,pp, italic marks $\Delta < -2$\,pp.}",
        r"  \label{tab:cross-lingual-transfer}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── §4.7 Mirror sign-flip catalogue (Type 9) ──────────────────────────────


def _flip_rows_sib200(df: pd.DataFrame) -> pd.DataFrame:
    """SIB-200 win→loss flips at (condition × benchmark) aggregate grain.
    Aggregate mean delta_orig and delta_rep over all sub-cells per
    (cond, benchmark=sib200); a flip needs sign change AND |delta_rep|>0.005.
    n_cells = count of finest sub-cells (seed × template × data × instr)
    contributing to that condition's SIB-200 aggregate."""
    sub = df[(df.condition != "baseline") & (df.benchmark == "sib200")]
    agg = (sub.groupby("condition")
           .agg(delta_orig=("delta_orig", "mean"),
                delta_rep=("delta_rep", "mean"),
                n_cells=("delta_rep", "count"))
           .reset_index())
    keep = []
    for _, r in agg.iterrows():
        o, rep = r["delta_orig"], r["delta_rep"]
        if pd.isna(o) or pd.isna(rep):
            continue
        if abs(rep) <= 0.005:
            continue
        if (o > 0 and rep < 0) and o > 0:
            keep.append(r)
    return pd.DataFrame(keep).sort_values("delta_rep")


def write_table_mirror_flips(df: pd.DataFrame) -> str:
    """SIB-200 sign-flip catalogue: condition-level cells where the refined
    extractor flipped Δ from positive (orig) to negative (refined), i.e.
    Rule-A over-credit corrections.

    Earlier drafts paired this with an XNLI loss→win sub-table to demonstrate
    symmetric corrections in both directions. Dropped because the XNLI side
    has no clean count at any single aggregation grain that matches the
    paper-prose intuition: at (cond × benchmark) grain the XNLI aggregate
    has 0 flips (corrections wash out when averaged across instr × seed ×
    data), at (cond × instr × template) grain there are 2, at finest per-row
    grain there are 7 (6 of which concentrate in cond-1-en-* template-2
    instr=zh, supporting the qualitative claim but not a clean enumeration).

    The remaining §4.7 methodology argument is supported by Sub-A alone:
    refinement removed over-credit on SIB-200 in 5 specific condition rows,
    and the prose in §4.7 can describe the qualitative XNLI correction
    pattern separately without a parallel table that doesn't quite line up.

    Paper section: §4.7 (methodology defense — extractor correction
    catalogue).
    """
    a = _flip_rows_sib200(df)

    cond_pretty = {cid: lbl.replace(r"\textit{", "").replace("}", "")
                   for cid, lbl in CONDITIONS}

    def fmt_4(x: float) -> str:
        if pd.isna(x):
            return "--"
        # Force 4 decimals; for negatives use math-mode minus
        if x < 0:
            return f"{MINUS}{abs(x):.4f}"
        return f"+{x:.4f}"

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Condition & n cells & $\Delta_{\text{orig}}$ & $\Delta_{\text{refined}}$ & $|\text{shift}|$ \\",
        r"    \midrule",
    ]
    for _, r in a.iterrows():
        shift = abs(r["delta_orig"] - r["delta_rep"])
        cond = cond_pretty.get(r["condition"], r["condition"])
        lines.append(
            f"    {cond} & {int(r['n_cells'])} & "
            f"{fmt_4(r['delta_orig'])} & {fmt_4(r['delta_rep'])} & "
            f"{shift:.4f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{\textsc{SIB-200} sign-flip catalogue. Five "
        r"(condition $\times$ benchmark) aggregates where the original "
        r"extractor reported a positive $\Delta$ vs baseline but the "
        r"refined extractor reports a negative $\Delta$ --- i.e. cells "
        r"where the refinement removed Rule-A over-credit. Aggregation "
        r"grain: mean across (instr $\times$ seed $\times$ template $\times$ "
        r"data) within each (condition, benchmark); flip requires "
        r"$|\Delta_{\text{refined}}| > 0.005$ to exclude noise-floor "
        r"wobble. The $|\text{shift}|$ column reports the absolute "
        r"distance between the two extractors' aggregate estimates.}",
        r"  \label{tab:sib200-sign-flips}",
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
    out.append("%")
    out.append("% ─── Accounting (P3 methodology note) ───────────────────────────────")
    out.append("% 21 sessions total = 20 fine-tuned (condition × seed) + 1 baseline")
    out.append("%   - 4 five-K conditions × 3 seeds = 12 multi-seed sessions")
    out.append("%   - 4 twenty-K conditions × 1 seed = 4  single-seed sessions")
    out.append("%   - 1 cond-3-zh-5k × 1 seed         = 1  single-seed session")
    out.append("%   - 3 cond-5 conditions × 1 seed    = 3  single-seed sessions")
    out.append("%   - 1 baseline (seed=none)          = 1  un-fine-tuned reference")
    out.append("%")
    out.append("% 1,664 observations = 1,536 fine-tuned cells + 128 baseline cells")
    out.append("%   - 1,536 from vs_baseline_cells.tsv (one row per fine-tuned")
    out.append("%     (cond × seed × template × benchmark × data × instr))")
    out.append("%   - 128 baseline cells = 64 (benchmark × data × instr) cells")
    out.append("%     × 2 templates (baseline runs without seed variation)")
    out.append("%")
    out.append("% Aggregation policy: when reporting means or stds across the seed")
    out.append("% dimension, the pipeline collapses templates within each seed")
    out.append("% FIRST, then averages across seeds. Per-seed std uses ddof=1.")
    out.append("% This avoids inflating sample size by treating (seed, template)")
    out.append("% as independent observations.")
    out.append("% ────────────────────────────────────────────────────────────────────")
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
    # P2: matched-diagonal Δ variant; sits next to Type-2 native it supplements
    block = write_type2_native_delta(df)
    print(f"[type2:native-delta] rows_emitted={block.count(chr(92) + chr(92))}")
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

    # ─── Tables for paper §4 (sections referenced in each writer's docstring) ─
    out.append("% ─── §4.1 Baseline-accuracy headroom ───")
    out.append("")
    block = write_table_baseline_headroom(df)
    print(f"[s4.1:baseline-headroom] rows_emitted={block.count(chr(92) + chr(92))}")
    out.append(block)
    out.append("")

    out.append("% ─── §4.3 Per-language matched-language ladder ───")
    out.append("")
    for lang in ("es", "zh", "ur"):
        block = write_table_matched_ladder(df, lang)
        print(f"[s4.3:matched-ladder-{lang}] rows_emitted={block.count(chr(92) + chr(92))}")
        out.append(block)
        out.append("")

    out.append("% ─── §4.4 Cond-2 vs Cond-5 head-to-head ───")
    out.append("")
    block = write_table_cond2_vs_cond5(df)
    print(f"[s4.4:cond2-vs-cond5] rows_emitted={block.count(chr(92) + chr(92))}")
    out.append(block)
    out.append("")

    out.append("% ─── §4.6 Cross-lingual transfer ───")
    out.append("")
    block = write_table_cross_lingual_transfer(df)
    print(f"[s4.6:cross-lingual-transfer] rows_emitted={block.count(chr(92) + chr(92))}")
    out.append(block)
    out.append("")

    out.append("% ─── §4.7 SIB-200 sign-flip catalogue ───")
    out.append("")
    block = write_table_mirror_flips(df)
    print(f"[s4.7:sib200-sign-flips] rows_emitted={block.count(chr(92) + chr(92))}")
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

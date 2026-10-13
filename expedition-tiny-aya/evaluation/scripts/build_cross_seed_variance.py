"""Build the Phase-3 cross-seed variance table from the canonical per-cell file.

Reproduces ``expedition-tiny-aya/analysis/phase-3/cross-seed-variance.tsv``
(the companion table to ``cross-seed-variance.md``) from
``phase3/analysis/refined-tables/vs_baseline_cells.tsv`` so the table can be
regenerated whenever a new (condition, seed) session lands, and writes a
generated Markdown companion with the per-seed values behind every headline
aggregate.

Method (identical to the one documented in cross-seed-variance.md):

* Metric: ``delta_rep``, the refined-extractor accuracy delta vs baseline, in
  percentage points. The baseline is seed-invariant, so every per-seed delta
  shares the same baseline.
* Grain: one row per (condition, benchmark, data_lang, instr_lang).
  ``data_lang = ALL`` is the instruction-language aggregate: the unweighted
  cell-mean over every data language x 2 templates, computed per seed. A
  specific ``data_lang`` is the per-language pair: the mean over the 2
  template cells, per seed.
* Cross-seed statistics are taken over the per-seed aggregates: ``mean``,
  sample standard deviation (``std``, ddof=1), ``min``, ``max``. ``std`` is
  ``NA`` when a condition has one seed. Nothing is imputed.

Columns added by this generator (the original nine are unchanged):

* ``seeds``: the seeds behind the row, ascending, comma-separated.
* ``per_seed``: the per-seed aggregate for each seed, ``seed=value`` pairs
  separated by ``;``, in the same order as ``seeds``. With two seeds the
  paper reports these two values (or their mean with the range), never an
  SD-based interval.
* ``provisional``: ``no``, or the comma-separated subset of ``seeds`` whose
  session (``condition/seedN``) matches a ``--provisional`` pattern. A row is
  provisional if any seed behind it is. Patterns are per session because the
  same seed number can be a Phase-3 result on one condition and a later
  batch on another.

Usage (from the repository root):

    # Read vs_baseline_cells.tsv from Hugging Face and write the TSV + Markdown
    # into expedition-tiny-aya/analysis/phase-3/:
    python expedition-tiny-aya/evaluation/scripts/build_cross_seed_variance.py

    # Read a locally rebuilt cells file (for example a staged refresh) and
    # mark the 20k seed-123 sessions provisional:
    python expedition-tiny-aya/evaluation/scripts/build_cross_seed_variance.py \
        --cells /tmp/phase3_refresh/tables/vs_baseline_cells.tsv \
        --provisional '*-20k/seed123'
"""
from __future__ import annotations

import argparse
import fnmatch
import statistics
from pathlib import Path

import pandas as pd

HF_REPO_ID = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"
HF_VS_BASELINE_CELLS = "phase3/analysis/refined-tables/vs_baseline_cells.tsv"

_SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = _SCRIPT_DIR.parent.parent / "analysis" / "phase-3"
OUT_TSV = PHASE3_DIR / "cross-seed-variance.tsv"
OUT_MD = PHASE3_DIR / "cross-seed-variance-by-condition.md"

BENCH_ORDER = ["sib200", "xnli", "csqa", "belebele"]
BENCH_LABEL = {
    "sib200": "SIB-200",
    "xnli": "XNLI",
    "csqa": "X-CSQA",
    "belebele": "Belebele",
}
TSV_COLUMNS = [
    "condition",
    "benchmark",
    "data_lang",
    "instr_lang",
    "n_seeds",
    "mean",
    "std",
    "min",
    "max",
    "seeds",
    "per_seed",
    "provisional",
]


def load_cells(path: Path | None = None) -> pd.DataFrame:
    """Load vs_baseline_cells.tsv (local path or Hugging Face) with seeds as
    strings and delta_rep as a float fraction."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=HF_VS_BASELINE_CELLS,
            )
        )
    df = pd.read_csv(path, sep="\t", dtype={"seed": str})
    if df["delta_rep"].dtype == object:
        df["delta_rep"] = (
            df["delta_rep"].str.replace("+", "", regex=False).astype(float)
        )
    return df


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    return (0, int(seed)) if seed.isdigit() else (1, seed)


def is_provisional(cond: str, seed: str, patterns) -> bool:
    """True when ``condition/seedN`` matches any shell-style pattern."""
    key = f"{cond}/seed{seed}"
    return any(fnmatch.fnmatchcase(key, pat) for pat in patterns)


def per_seed_aggregate(sub: pd.DataFrame) -> pd.Series:
    """Per-seed mean of delta_rep over whatever cells are in ``sub``
    (templates, and data languages for the ALL row), in percentage points,
    indexed by seed in ascending numeric order.

    The mean is taken with ``statistics.mean`` over the per-cell values in
    percentage points: exact rational arithmetic, so the fourth decimal of
    the published table does not depend on float summation order. This is
    the route that reproduces the committed table byte for byte."""
    pp = sub.assign(pp=sub["delta_rep"] * 100.0)
    ps = pp.groupby("seed")["pp"].apply(lambda s: statistics.mean(s.tolist()))
    return ps.reindex(sorted(ps.index, key=_seed_sort_key)).astype(float)


def _nz(x: float) -> float:
    """Normalise negative zero so a null delta prints as +0.0000."""
    return 0.0 if x == 0 else x


def _fmt_signed(x: float) -> str:
    return f"{_nz(x):+.4f}"


def build_rows(df: pd.DataFrame, provisional: set[str]) -> list[dict]:
    """One dict per (condition, benchmark, data_lang, instr_lang) row.

    Row order matches the committed table: condition, benchmark, instr_lang
    ascending, with the ALL row first in each (benchmark, instr_lang) group
    followed by the data languages ascending."""
    rows: list[dict] = []
    fine_tuned = df[df.condition != "baseline"]
    for cond, g_cond in fine_tuned.groupby("condition", sort=True):
        for bench, g_bench in g_cond.groupby("benchmark", sort=True):
            for instr, g_instr in g_bench.groupby("instr", sort=True):
                rows.append(_row(cond, bench, "ALL", instr, g_instr, provisional))
                for data, g_data in g_instr.groupby("data", sort=True):
                    rows.append(_row(cond, bench, data, instr, g_data, provisional))
    return rows


def _row(
    cond: str,
    bench: str,
    data_lang: str,
    instr: str,
    sub: pd.DataFrame,
    provisional: set[str],
) -> dict:
    ps = per_seed_aggregate(sub)
    n = int(len(ps))
    seeds = list(ps.index)
    prov = [s for s in seeds if is_provisional(cond, s, provisional)]
    return {
        "condition": cond,
        "benchmark": bench,
        "data_lang": data_lang,
        "instr_lang": instr,
        "n_seeds": n,
        "mean": _fmt_signed(float(ps.mean())),
        "std": f"{float(ps.std(ddof=1)):.4f}" if n > 1 else "NA",
        "min": _fmt_signed(float(ps.min())),
        "max": _fmt_signed(float(ps.max())),
        "seeds": ",".join(seeds),
        "per_seed": ";".join(f"{s}={_fmt_signed(float(v))}" for s, v in ps.items()),
        "provisional": ",".join(prov) if prov else "no",
    }


def write_tsv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in TSV_COLUMNS) + "\n")


# ─── Generated Markdown companion ─────────────────────────────────────────────


def _target_lang(cond: str) -> str | None:
    """condition-2-ur-5k -> ur; condition-1-en-20k -> en; baseline -> None."""
    parts = cond.split("-")
    return parts[2] if len(parts) >= 4 and parts[0] == "condition" else None


def _fmt_pp(x: float) -> str:
    return f"{_nz(x):+.2f}"


def _headline_line(label: str, cond: str, ps: pd.Series, provisional: set[str]) -> str:
    n = len(ps)
    per_seed = " / ".join(
        f"{_fmt_pp(float(v))}" + ("*" if is_provisional(cond, s, provisional) else "")
        for s, v in ps.items()
    )
    if n == 1:
        stat = f"{_fmt_pp(float(ps.iloc[0]))} (point estimate)"
    elif n == 2:
        stat = (
            f"mean {_fmt_pp(float(ps.mean()))}, range "
            f"{_fmt_pp(float(ps.min()))} to {_fmt_pp(float(ps.max()))}"
        )
    else:
        stat = f"{_fmt_pp(float(ps.mean()))} +/- {float(ps.std(ddof=1)):.2f}"
    seeds = ", ".join(ps.index)
    return f"| {label} | {n} ({seeds}) | {per_seed} | {stat} |"


def build_markdown(df: pd.DataFrame, provisional: set[str], source: str) -> str:
    """Per-condition headline table: for every fine-tuned condition, the
    matched-instruction aggregate (instr = target language, all data
    languages) and the matched diagonal (data = instr = target) on each
    benchmark, with per-seed values."""
    fine_tuned = df[df.condition != "baseline"]
    sessions = fine_tuned.drop_duplicates(subset=["condition", "seed"])
    prov_present = sorted(
        f"{c}/seed{s}" for c, s in zip(sessions.condition, sessions.seed) if is_provisional(c, s, provisional)
    )
    lines = [
        "# Cross-seed variance by condition (generated)",
        "",
        "Generated by `expedition-tiny-aya/evaluation/scripts/build_cross_seed_variance.py`"
        f" from `{source}`. Do not edit by hand; rerun the script.",
        "",
        "Values are `delta_rep` (refined-extractor accuracy delta vs baseline) in"
        " percentage points. Each per-seed value collapses templates first (and data"
        " languages for the aggregate rows), so the seed is the unit of replication."
        " Rows with one seed are point estimates. Rows with two seeds report both"
        " values and the mean with its range, not a standard deviation. Rows with"
        " three seeds report mean +/- sample SD.",
        "",
    ]
    if prov_present:
        lines += [
            f"**Provisional:** {len(prov_present)} session(s) ({', '.join(prov_present)})"
            " are marked `*` and are provisional until the software-stack drift check"
            " between the original and the later evaluation environments is resolved."
            " Do not cite them as final.",
            "",
        ]
    for cond, g in fine_tuned.groupby("condition", sort=True):
        target = _target_lang(cond)
        n_seeds = g.seed.nunique()
        seeds = ", ".join(sorted(set(g.seed), key=_seed_sort_key))
        lines += [
            f"## {cond}",
            "",
            f"Seeds: {seeds} (n = {n_seeds}). Target language: {target}.",
            "",
            "| Cell | n_seeds (seeds) | per-seed | statistic |",
            "|---|---|---|---|",
        ]
        for bench in BENCH_ORDER:
            gb = g[(g.benchmark == bench) & (g.instr == target)]
            if gb.empty:
                continue
            lines.append(
                _headline_line(
                    f"{BENCH_LABEL[bench]} instr={target} aggregate",
                    cond,
                    per_seed_aggregate(gb),
                    provisional,
                )
            )
            gd = gb[gb.data == target]
            if not gd.empty:
                lines.append(
                    _headline_line(
                        f"{BENCH_LABEL[bench]} {target} x {target} diagonal",
                        cond,
                        per_seed_aggregate(gd),
                        provisional,
                    )
                )
        lines.append("")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cells",
        type=Path,
        default=None,
        help="Local vs_baseline_cells.tsv (default: download from Hugging Face main).",
    )
    parser.add_argument("--out-tsv", type=Path, default=OUT_TSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument(
        "--provisional",
        nargs="*",
        default=[],
        metavar="COND/SEED",
        help="Sessions (condition/seedN, shell wildcards allowed) whose results are provisional; flagged in the TSV and Markdown.",
    )
    args = parser.parse_args()

    df = load_cells(args.cells)
    provisional = set(args.provisional)
    rows = build_rows(df, provisional)
    write_tsv(rows, args.out_tsv)
    source = (
        f"a local rebuild of {HF_VS_BASELINE_CELLS}" if args.cells else f"{HF_REPO_ID}:{HF_VS_BASELINE_CELLS}"
    )
    args.out_md.write_text(build_markdown(df, provisional, source), encoding="utf-8")
    n_multi = sum(1 for r in rows if r["n_seeds"] > 1 and r["data_lang"] == "ALL")
    print(
        f"[cross-seed] rows={len(rows)} conditions={df[df.condition != 'baseline'].condition.nunique()} "
        f"seeds={sorted(set(df.seed), key=_seed_sort_key)} provisional_patterns={sorted(provisional) or 'none'}"
    )
    print(f"[cross-seed] multi-seed aggregate rows={n_multi}")
    print(f"[write] {args.out_tsv}")
    print(f"[write] {args.out_md}")


if __name__ == "__main__":
    main()

"""Inspect how a Phase-3 `_results_*.json` was scored — per row, not per cell.

The eval summary tells you a cell's accuracy and parse-failure rate. It does
NOT tell you *how* each row landed there. This tool classifies every row on
two axes so you can see the real breakdown:

  Axis 1 — outcome:
    correct       extractor's prediction == gold
    wrong_label   extractor returned a label, but the wrong one
    parse_fail    extractor returned None — the output wasn't readable at all

  Axis 2 — match_via (which extractor branch produced the prediction):
    SIB-200:
      single          all answer pieces resolved to exactly one category
                      (canonical, alias, native-script, or sci/tech sub-topic)
      multi_category  pieces straddled 2+ categories — model hedged, pred=None
      fallback        no pieces resolved, but a canonical English category name
                      was embedded in the first line ("the answer is travel")
      none            parse failure
    XNLI:
      tier1_english     verbatim English label, word-boundary match
      tier1_native      native-script label word (zh/es/ur)
      tier2_cjk_glued   English label glued to a CJK frame ("假设是entailment")
      tier3_paraphrase  native-prose paraphrase of the relationship
      none              parse failure (incl. Tier 2 with a negation marker)
    X-CSQA / Belebele:
      bare_letter | letter_in_text | answer_prefix | none

  Plus a `multiline` flag — did the model emit content past the first line?

Why this matters: a cell at 70% accuracy made entirely of `correct/single`
rows is a clean result. The same 70% made of `correct/tier3_paraphrase` rows
means the model only gets credit because of the prose-paraphrase rescue — a
very different story for the paper. And a 30% parse-fail cell is a
parser-extension lead, not a model failure.

The tool calls the live extractor (imported from `reparse_results.py` — the
paper-grade refined Phase-3 scorer on main) for `pred`, so the prediction is
faithful by construction. The `match_via` stage label is derived by re-using
the extractor's own constants/helpers. A self-test asserts the instrumented
classifier always agrees with the live extractor on the prediction itself
(run with `--self-test`).

Usage:
    # Whole file — per-cell breakdown tables:
    python inspect_failures.py path/to/baseline_seednone_results_template1.json

    # One cell, with 5 example raw_outputs per bucket:
    python inspect_failures.py RESULTS.json \\
        --cell template1_sib200_data=ur_instr=ur --samples 5

    # Only the parse-failures, grouped by output prefix:
    python inspect_failures.py RESULTS.json --benchmark sib200 \\
        --outcome parse_fail --samples 3 --group-prefix 30

    # Frequency table — pool many files, count every distinct answer:
    python inspect_failures.py RESULTS_T1.json RESULTS_T2.json \\
        --benchmark xnli --aggregate --min-count 5 --output xnli-forms.tsv

    # Pass an HF path instead of a local file — it downloads first:
    python inspect_failures.py \\
        phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json

Pure read-only — never writes or uploads anything (except the optional
--output TSV in aggregate mode).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reparse_results import (
    EXTRACTOR_NAMES,
    NATIVE_LABEL_MAP,
    SIB200_STRIP,
    SIB200_TERM_TO_CATEGORY,
    XNLI_LABEL_RES,
    XNLI_LABELS,
    XNLI_TIER2_NEGATION,
    _sib200_split,
    benchmark_from_key,
    extract_choice,
    extract_sib200_category,
    extract_xnli_label,
)

HF_REPO_ID = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"


# ----------------------------------------------------------------------------
# Classifiers — each calls the LIVE extractor (imported from reparse_results)
# for `pred` so the prediction is faithful by construction, then derives a
# coarse `match_via` by re-using the extractor's own helpers/constants.
#
# match_via vocabulary:
#   SIB-200:  single | multi_category | fallback | none
#   XNLI:     tier1_english | tier1_native | tier2_cjk_glued | tier3_paraphrase
#             | none
#   choice:   bare_letter | letter_in_text | answer_prefix | none
# ----------------------------------------------------------------------------
def classify_sib200(raw_output: str) -> tuple[str | None, str]:
    """`pred` from the live extractor; `match_via` derived from how many
    distinct categories the answer's pieces resolve to."""
    pred = extract_sib200_category(raw_output)
    first_line = raw_output.strip().split("\n")[0].strip().strip(SIB200_STRIP)
    pieces = _sib200_split(first_line) if first_line else []
    cats = {SIB200_TERM_TO_CATEGORY.get(p.lower()) for p in pieces}
    cats.discard(None)
    if len(cats) >= 2:
        return pred, "multi_category"  # hedge — pred is None
    if len(cats) == 1:
        return pred, "single"
    if pred is not None:
        return pred, "fallback"  # 0 pieces resolved but canonical scan hit
    return pred, "none"


def classify_xnli(raw_output: str) -> tuple[str | None, str]:
    """`pred` from the live extractor; `match_via` is which tier produced it."""
    pred = extract_xnli_label(raw_output)
    if pred is None:
        return None, "none"
    first_line = raw_output.strip().split("\n")[0].strip()
    fll = first_line.lower()
    if any(r.search(fll) for r in XNLI_LABEL_RES.values()):
        return pred, "tier1_english"
    if any(native.lower() in fll for native in NATIVE_LABEL_MAP):
        return pred, "tier1_native"
    negated = any(neg in first_line for neg in XNLI_TIER2_NEGATION)
    if not negated and any(label in fll for label in XNLI_LABELS):
        return pred, "tier2_cjk_glued"
    return pred, "tier3_paraphrase"


def classify_choice(raw_output: str, choices: str) -> tuple[str | None, str]:
    """`pred` from the live extractor; `match_via` is the letter-match stage."""
    pred = extract_choice(raw_output, choices=choices)
    if pred is None:
        return None, "none"
    first_line = raw_output.strip().upper().split("\n")[0].strip()
    if first_line == pred:
        return pred, "bare_letter"
    if re.search(rf"\b{re.escape(pred)}\b", first_line):
        return pred, "letter_in_text"
    return pred, "answer_prefix"


# Benchmark → classifier callable with a uniform (raw_output,) signature.
def _classifier_for(benchmark: str):
    if benchmark == "sib200":
        return classify_sib200
    if benchmark == "xnli":
        return classify_xnli
    if benchmark == "csqa":
        return lambda raw: classify_choice(raw, "ABCDE")
    if benchmark == "belebele":
        return lambda raw: classify_choice(raw, "ABCD")
    raise ValueError(f"Unknown benchmark {benchmark!r}")


# ----------------------------------------------------------------------------
# Row classification
# ----------------------------------------------------------------------------
def classify_row(benchmark: str, raw_output: str, gold: str) -> dict:
    """Classify one row on both axes. Returns a dict with keys:
    outcome, match_via, multiline, pred, gold."""
    classifier = _classifier_for(benchmark)
    pred, match_via = classifier(raw_output)
    if pred is None:
        outcome = "parse_fail"
    elif pred == gold:
        outcome = "correct"
    else:
        outcome = "wrong_label"
    multiline = len(raw_output.strip().split("\n")) > 1
    return {
        "outcome": outcome,
        "match_via": match_via,
        "multiline": multiline,
        "pred": pred,
        "gold": gold,
    }


def classify_cell(cell_key: str, rows: list[dict]) -> dict:
    """Classify every row in one cell. Returns aggregate counts + the
    classified rows (for sample printing)."""
    benchmark = benchmark_from_key(cell_key)
    classified = []
    for row in rows:
        c = classify_row(benchmark, row["raw_output"], row["gold"])
        c["raw_output"] = row["raw_output"]
        classified.append(c)

    bucket_counts: Counter = Counter()
    multiline_count = 0
    for c in classified:
        bucket_counts[(c["outcome"], c["match_via"])] += 1
        if c["multiline"]:
            multiline_count += 1

    n = len(classified)
    return {
        "cell": cell_key,
        "benchmark": benchmark,
        "n": n,
        "bucket_counts": bucket_counts,
        "multiline_count": multiline_count,
        "accuracy": (
            sum(1 for c in classified if c["outcome"] == "correct") / n if n else 0.0
        ),
        "parse_fail_rate": (
            sum(1 for c in classified if c["outcome"] == "parse_fail") / n if n else 0.0
        ),
        "classified_rows": classified,
    }


# ----------------------------------------------------------------------------
# Surface-form aggregation — pool every row across cells/files and tally
# how often the model emits each distinct answer, with its outcome split.
# This is the "how many times does the model say X" frequency artifact.
# ----------------------------------------------------------------------------
def _first_line(raw_output: str) -> str:
    """The first line, whitespace-stripped. Used as the aggregation cache
    key. Note: each live extractor applies its OWN additional normalization
    on top of this — SIB-200 strips `SIB200_STRIP` (quotes/punctuation),
    choice extractors uppercase. So two raw_outputs like "travel" and
    "travel." (or "A" and "a" for choice) get separate cache entries even
    though they classify identically. The split is intentional: it keeps
    surface-form variants distinguishable in the frequency table, at the
    cost of a few extra cache rows."""
    return raw_output.strip().split("\n")[0].strip()


def aggregate_surface_forms(
    datasets: list[dict],
    only_benchmarks: set[str] | None = None,
) -> list[dict]:
    """Pool every row across all cells of all given result files and group by
    (benchmark, first_line) — see `_first_line` for the exact cache-key
    definition. Within a group every row produces the same `pred` +
    `match_via` (extractors are deterministic on the same input); only the
    `outcome` varies (it depends on each row's `gold`). Surface-form
    variants that the extractor would normalize away (e.g. trailing
    punctuation for SIB-200, casing for choice) appear as separate rows in
    the table — that's deliberate, so the frequency artifact preserves
    every distinct emission shape the model produced.

    Returns a list of dicts sorted by total count descending, each with:
      benchmark, first_line, total, correct, wrong_label, parse_fail,
      multiline, n_cells, pred, match_via
    """
    agg: dict[tuple, dict] = {}
    # Classification depends only on the first line — cache on it.
    classify_cache: dict[tuple, tuple] = {}

    for data in datasets:
        for key, rows in data.items():
            if key in {"summary", "parse_failure_rates"}:
                continue
            if not isinstance(rows, list) or not rows or "raw_output" not in rows[0]:
                continue
            bench = benchmark_from_key(key)
            if only_benchmarks and bench not in only_benchmarks:
                continue
            classifier = _classifier_for(bench)
            for row in rows:
                raw = row["raw_output"]
                fl = _first_line(raw)
                cache_key = (bench, fl)
                if cache_key not in classify_cache:
                    classify_cache[cache_key] = classifier(raw)
                pred, match_via = classify_cache[cache_key]

                gold = row["gold"]
                if pred is None:
                    outcome = "parse_fail"
                elif pred == gold:
                    outcome = "correct"
                else:
                    outcome = "wrong_label"

                entry = agg.setdefault(
                    cache_key,
                    {
                        "benchmark": bench,
                        "first_line": fl,
                        "total": 0,
                        "correct": 0,
                        "wrong_label": 0,
                        "parse_fail": 0,
                        "multiline": 0,
                        "pred": pred,
                        "match_via": match_via,
                        "_cells": set(),
                    },
                )
                entry["total"] += 1
                entry[outcome] += 1
                entry["_cells"].add(key)
                if len(raw.strip().split("\n")) > 1:
                    entry["multiline"] += 1

    out = []
    for entry in agg.values():
        entry["n_cells"] = len(entry.pop("_cells"))
        out.append(entry)
    out.sort(key=lambda r: r["total"], reverse=True)
    return out


def print_surface_form_table(rows: list[dict], min_count: int) -> None:
    """Print the aggregated surface-form frequency table to the terminal."""
    shown = [r for r in rows if r["total"] >= min_count]
    hidden = len(rows) - len(shown)
    print(
        f"\n{'total':>7} {'correct':>8} {'wrong':>7} {'fail':>7} "
        f"{'bench':<9} {'pred':<20} {'match_via':<16} first_line"
    )
    print("-" * 110)
    for r in shown:
        pred = "—" if r["pred"] is None else str(r["pred"])
        print(
            f"{r['total']:>7} {r['correct']:>8} {r['wrong_label']:>7} "
            f"{r['parse_fail']:>7} {r['benchmark']:<9} {pred[:20]:<20} "
            f"{r['match_via']:<16} {r['first_line'][:60]!r}"
        )
    print(
        f"\n{len(shown)} distinct forms shown (count ≥ {min_count}); "
        f"{hidden} rarer forms hidden."
    )


def write_surface_form_tsv(rows: list[dict], out_path: Path) -> None:
    """Write the full aggregated table as TSV — the durable frequency
    artifact (every form, no min-count filter)."""
    header = [
        "benchmark",
        "first_line",
        "total",
        "correct",
        "wrong_label",
        "parse_fail",
        "multiline",
        "n_cells",
        "pred",
        "match_via",
    ]
    lines = ["\t".join(header)]
    for r in rows:
        # Tabs/newlines can't occur in first_line (it's already one stripped
        # line), but guard anyway.
        fl = r["first_line"].replace("\t", " ").replace("\n", " ")
        lines.append(
            "\t".join(
                str(x)
                for x in [
                    r["benchmark"],
                    fl,
                    r["total"],
                    r["correct"],
                    r["wrong_label"],
                    r["parse_fail"],
                    r["multiline"],
                    r["n_cells"],
                    "" if r["pred"] is None else r["pred"],
                    r["match_via"],
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} surface forms → {out_path}")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
# Sort buckets so the table reads correct → wrong → fail, cleanest match first.
_OUTCOME_ORDER = {"correct": 0, "wrong_label": 1, "parse_fail": 2}
_VIA_ORDER = {
    # cleanest / most direct match first
    "single": 0,
    "tier1_english": 0,
    "tier1_native": 1,
    "bare_letter": 0,
    "letter_in_text": 2,
    "answer_prefix": 3,
    "tier2_cjk_glued": 4,
    "tier3_paraphrase": 5,
    "fallback": 6,
    "multi_category": 7,
    "none": 9,
}


def _bucket_sort_key(bucket: tuple[str, str]) -> tuple[int, int]:
    outcome, via = bucket
    return (_OUTCOME_ORDER.get(outcome, 9), _VIA_ORDER.get(via, 8))


def print_cell_report(report: dict, samples: int, group_prefix: int) -> None:
    n = report["n"]
    print(f"\n{report['cell']}  ·  n={n}")
    print(
        f"  accuracy={report['accuracy']:.3f}  "
        f"parse_fail={report['parse_fail_rate']:.3f}  "
        f"multiline={report['multiline_count']}/{n} "
        f"({(report['multiline_count'] / n if n else 0):.1%})"
    )
    print(f"  {'bucket':<28} {'count':>7} {'pct':>7}")
    print(f"  {'-' * 44}")
    for bucket in sorted(report["bucket_counts"], key=_bucket_sort_key):
        count = report["bucket_counts"][bucket]
        label = f"{bucket[0]}/{bucket[1]}"
        print(f"  {label:<28} {count:>7} {count / n if n else 0:>6.1%}")

    if samples > 0:
        _print_samples(report, samples, group_prefix)


def _print_samples(report: dict, samples: int, group_prefix: int) -> None:
    """For each non-empty bucket, print up to `samples` example raw_outputs.
    If group_prefix > 0, cluster by the first N chars of raw_output first so
    repeated surface forms collapse into one line with a count."""
    by_bucket: dict[tuple, list[str]] = defaultdict(list)
    for c in report["classified_rows"]:
        by_bucket[(c["outcome"], c["match_via"])].append(c["raw_output"])

    for bucket in sorted(by_bucket, key=_bucket_sort_key):
        outputs = by_bucket[bucket]
        print(f"\n  ── {bucket[0]}/{bucket[1]}  ({len(outputs)} rows) ──")
        if group_prefix > 0:
            prefix_counts = Counter(_oneline(o)[:group_prefix] for o in outputs)
            for prefix, count in prefix_counts.most_common(samples):
                print(f"     {count:>4}×  {prefix!r}")
        else:
            for raw in outputs[:samples]:
                print(f"     {_oneline(raw)[:200]!r}")


def _oneline(text: str) -> str:
    """Collapse a raw_output to a single line for compact display."""
    return " ⏎ ".join(line for line in text.strip().split("\n") if line.strip())


# ----------------------------------------------------------------------------
# Input resolution — local path, or download from HF
# ----------------------------------------------------------------------------
def resolve_results_file(path_arg: str) -> Path:
    """If `path_arg` is a local file, use it. Otherwise treat it as an HF
    repo path and download it. Returns the local Path."""
    local = Path(path_arg)
    if local.is_file():
        return local
    # Treat as HF path.
    if not path_arg.startswith("phase3/"):
        raise SystemExit(
            f"{path_arg!r} is not a local file and doesn't look like an HF "
            "results path (expected to start with 'phase3/')."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub not installed — can't download from HF. Either "
            "install it or pass a local file path."
        ) from e
    print(f"Downloading {path_arg} from {HF_REPO_ID}...")
    return Path(
        hf_hub_download(repo_id=HF_REPO_ID, filename=path_arg, repo_type=HF_REPO_TYPE)
    )


# ----------------------------------------------------------------------------
# Self-test — confirm the instrumented classifiers agree with the live
# extractors on the prediction itself (not just the stage label).
# ----------------------------------------------------------------------------
def run_self_test(results_path: Path) -> int:
    """Cross-check every row: classify_*'s pred must equal the live extractor's
    pred. Returns the number of mismatches (0 = pass)."""
    extractors = {
        "sib200": extract_sib200_category,
        "xnli": extract_xnli_label,
        "csqa": lambda t: extract_choice(t, choices="ABCDE"),
        "belebele": lambda t: extract_choice(t, choices="ABCD"),
    }
    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)
    mismatches = 0
    checked = 0
    for key, rows in data.items():
        if key in {"summary", "parse_failure_rates"}:
            continue
        if not isinstance(rows, list) or not rows or "raw_output" not in rows[0]:
            continue
        bench = benchmark_from_key(key)
        classifier = _classifier_for(bench)
        live = extractors[bench]
        for row in rows:
            raw = row["raw_output"]
            instrumented_pred, _ = classifier(raw)
            live_pred = live(raw)
            checked += 1
            if instrumented_pred != live_pred:
                mismatches += 1
                if mismatches <= 10:
                    print(
                        f"  MISMATCH [{bench}] instrumented={instrumented_pred!r} "
                        f"live={live_pred!r} raw={_oneline(raw)[:80]!r}"
                    )
    status = "PASS" if mismatches == 0 else "FAIL"
    print(f"\nself-test {status}: {checked} rows checked, {mismatches} mismatches")
    return mismatches


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(
        description=doc.splitlines()[0] if doc else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="One or more local paths to _results_*.json, or HF paths under "
        "phase3/ (downloaded automatically). Multiple files are pooled in "
        "--aggregate mode.",
    )
    parser.add_argument(
        "--cell",
        default=None,
        help="Only inspect this exact cell key, e.g. "
        "'template1_sib200_data=ur_instr=ur'.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(EXTRACTOR_NAMES),
        default=None,
        help="Only inspect cells for this benchmark (default: all).",
    )
    parser.add_argument(
        "--outcome",
        choices=["correct", "wrong_label", "parse_fail"],
        default=None,
        help="When printing samples, restrict to rows with this outcome.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Print up to N example raw_outputs per bucket (default: 0 = "
        "counts only).",
    )
    parser.add_argument(
        "--group-prefix",
        type=int,
        default=0,
        help="When printing samples, cluster by the first N chars of each "
        "raw_output and show counts instead of individual rows. Useful for "
        "spotting recurring surface forms in a parse_fail bucket.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Pool every row across all given files and print a frequency "
        "table: each distinct answer the model emitted, how many times, and "
        "its correct / wrong / parse-fail split. The 'how many times does the "
        "model say X' artifact.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="In --aggregate mode, only print forms emitted at least this "
        "many times (default: 1). The TSV from --output is never filtered.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="In --aggregate mode, also write the full (unfiltered) frequency "
        "table as TSV to this path — the durable artifact for the paper.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Cross-check the instrumented classifiers against the live "
        "extractors on every row, then exit. Non-zero exit on any mismatch.",
    )
    args = parser.parse_args()

    if (args.output or args.min_count != 1) and not args.aggregate:
        parser.error("--output and --min-count only apply with --aggregate")
    if args.aggregate and (
        args.cell or args.outcome or args.samples or args.group_prefix
    ):
        parser.error(
            "--cell, --outcome, --samples, --group-prefix are per-cell-mode "
            "flags and do not apply with --aggregate"
        )

    results_paths = [resolve_results_file(p) for p in args.results_files]

    if args.self_test:
        total_mismatches = sum(run_self_test(p) for p in results_paths)
        raise SystemExit(0 if total_mismatches == 0 else 1)

    only = {args.benchmark} if args.benchmark else None

    # --- Aggregate mode: pool all files, emit a surface-form frequency table.
    if args.aggregate:
        datasets = []
        for p in results_paths:
            with p.open(encoding="utf-8") as f:
                datasets.append(json.load(f))
        print(f"=== Aggregating {len(datasets)} file(s) ===")
        rows = aggregate_surface_forms(datasets, only_benchmarks=only)
        print_surface_form_table(rows, min_count=args.min_count)
        if args.output:
            write_surface_form_tsv(rows, args.output)
        return

    # --- Default mode: per-cell breakdown, one file at a time.
    for results_path in results_paths:
        with results_path.open(encoding="utf-8") as f:
            data = json.load(f)

        cell_keys = [
            k
            for k, v in data.items()
            if k not in {"summary", "parse_failure_rates"}
            and isinstance(v, list)
            and v
            and "raw_output" in v[0]
        ]
        if args.cell:
            cell_keys = [k for k in cell_keys if k == args.cell]
        if args.benchmark:
            cell_keys = [
                k for k in cell_keys if benchmark_from_key(k) == args.benchmark
            ]

        print(f"\n=== {results_path.name} ===")
        if not cell_keys:
            print("  (no matching cells in this file)")
            continue
        print(f"Inspecting {len(cell_keys)} cell(s)")

        for cell_key in sorted(cell_keys):
            report = classify_cell(cell_key, data[cell_key])
            if args.outcome:
                report["classified_rows"] = [
                    c for c in report["classified_rows"] if c["outcome"] == args.outcome
                ]
            print_cell_report(
                report, samples=args.samples, group_prefix=args.group_prefix
            )


if __name__ == "__main__":
    main()

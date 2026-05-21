"""Inspect how a Phase-3 `_results_*.json` was scored — per row, not per cell.

The eval summary tells you a cell's accuracy and parse-failure rate. It does
NOT tell you *how* each row landed there. This tool classifies every row on
two axes so you can see the real breakdown:

  Axis 1 — outcome:
    correct       extractor's prediction == gold
    wrong_label   extractor returned a label, but the wrong one
    parse_fail    extractor returned None — the output wasn't readable at all

  Axis 2 — match_via (how much work the extractor did to land a prediction):
    SIB-200:
      exact        first line IS the canonical category, verbatim
      normalized   needed case-fold and/or quote/punctuation stripping
      substring    canonical category embedded in a longer first line
      alias        matched a known alias (e.g. "sport" -> "sports")
      rule_a       matched the lenient "science/<X>" prefix rule
      rule_b       matched a native-script equivalent (Urdu/Chinese/Spanish)
      rule_c       matched a bare subcategory token (template-2 strips prefix)
      none         parse failure
    XNLI:
      exact / english_substring / native_exact / native_substring / none
    X-CSQA / Belebele:
      bare_letter / letter_in_text / answer_prefix / none

  Plus a `multiline` flag — did the model emit content past the first line?

Why this matters: a cell at 70% accuracy made entirely of `correct/exact`
rows is a clean result. The same 70% made of `correct/rule_b` rows means the
model only gets credit because of the native-script rescue — a very different
story for the paper. And a 30% parse-fail cell is a parser-extension lead,
not a model failure.

The tool re-implements the extractors' staged matching with instrumentation
so it can report *which* stage produced each prediction. A self-test asserts
the instrumented classifier always agrees with the live extractor on the
prediction itself (run with `--self-test`).

Usage:
    # Whole file — per-cell breakdown tables:
    python inspect_failures.py path/to/baseline_seednone_results_template1.json

    # One cell, with 5 example raw_outputs per bucket:
    python inspect_failures.py RESULTS.json \\
        --cell template1_sib200_data=ur_instr=ur --samples 5

    # Only the parse-failures, grouped by output prefix:
    python inspect_failures.py RESULTS.json --benchmark sib200 \\
        --outcome parse_fail --samples 3 --group-prefix 30

    # Pass an HF path instead of a local file — it downloads first:
    python inspect_failures.py \\
        phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json

Reads `run_eval_single.py` from next to this script (or /kaggle/working/) for
the extractor constants. Pure read-only — never writes or uploads anything.
"""

from __future__ import annotations

import argparse
import ast
import json
import re as _re
from collections import Counter, defaultdict
from pathlib import Path

HF_REPO_ID = "legesher/language-decoded-experiments"
HF_REPO_TYPE = "dataset"

# ----------------------------------------------------------------------------
# Extractor source — AST-load constants + functions from run_eval_single.py
# without triggering its heavy top-level imports (torch, unsloth, ...).
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

_WANTED = {
    "SIB200_CATEGORIES",
    "SIB200_ALIASES",
    "SIB200_SCITECH_NATIVE",
    "SIB200_SCITECH_BARE_SUBCATEGORIES",
    "XNLI_LABEL_RES",
    "NATIVE_LABEL_MAP",
    "extract_sib200_category",
    "extract_xnli_label",
    "extract_choice",
}


def _find_extractor_source() -> Path | None:
    for candidate in (
        HERE / "run_eval_single.py",
        Path("/kaggle/working/run_eval_single.py"),
    ):
        if candidate.exists():
            return candidate
    return None


def load_extractor_namespace() -> dict:
    """Exec just the extractor functions + their helper constants into an
    isolated namespace and return it. Raises SystemExit with a clear message
    if `run_eval_single.py` can't be found."""
    src = _find_extractor_source()
    if src is None:
        raise SystemExit(
            "Couldn't find run_eval_single.py. Expected next to this file or at "
            "/kaggle/working/. On Kaggle the launcher writes it; locally extract "
            "it from evaluate.ipynb cell 3."
        )
    tree = ast.parse(src.read_text())
    subset: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _WANTED
        ):
            subset.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _WANTED for t in node.targets
        ):
            subset.append(node)
    ns: dict = {"re": _re}
    exec(compile(ast.Module(body=subset, type_ignores=[]), str(src), "exec"), ns)
    return ns


# ----------------------------------------------------------------------------
# Instrumented classifiers — replay each extractor's staged matching and
# report which stage produced the prediction. Each returns (pred, match_via).
# The `multiline` flag is computed once by the caller.
#
# These MUST agree with the live extractors on `pred`. `--self-test` checks it.
# ----------------------------------------------------------------------------
_SIB200_PUNCT = " .,:;!?()[]{}\"'"


def classify_sib200(raw_output: str, ns: dict) -> tuple[str | None, str]:
    """Mirror of extract_sib200_category with stage instrumentation."""
    first_line_ws = raw_output.strip().split("\n")[0].strip()
    first_line_punct = first_line_ws.strip(_SIB200_PUNCT)
    fl_lower = first_line_punct.lower()

    # Rule A — science/<anything>. The extractor checks this BEFORE the
    # canonical-category loop, so a verbatim "science/technology" technically
    # matches here. Distinguish that from a genuine lenient rescue
    # (science/AI, science/physics, science /technology) so the taxonomy
    # doesn't mislabel a perfect canonical answer as rule-rescued.
    if fl_lower.startswith("science/") or fl_lower.startswith("science /"):
        if first_line_ws == "science/technology":
            return "science/technology", "exact"
        if fl_lower == "science/technology":
            return "science/technology", "normalized"
        return "science/technology", "rule_a"
    # Rule B — native-script equivalents
    for phrase in ns["SIB200_SCITECH_NATIVE"]:
        if phrase.lower() in fl_lower:
            return "science/technology", "rule_b"
    # Rule C — bare subcategory tokens
    if fl_lower in ns["SIB200_SCITECH_BARE_SUBCATEGORIES"]:
        return "science/technology", "rule_c"
    # Canonical category match (substring containment, as the extractor does)
    for category in ns["SIB200_CATEGORIES"]:
        if category in fl_lower:
            if first_line_ws == category:
                via = "exact"
            elif fl_lower == category:
                via = "normalized"
            else:
                via = "substring"
            return category, via
    # Alias lookup
    alias = ns["SIB200_ALIASES"].get(fl_lower)
    if alias is not None:
        return alias, "alias"
    return None, "none"


def classify_xnli(raw_output: str, ns: dict) -> tuple[str | None, str]:
    """Mirror of extract_xnli_label with stage instrumentation."""
    first_line = raw_output.strip().split("\n")[0].strip()
    fl_lower = first_line.lower()
    for label, label_re in ns["XNLI_LABEL_RES"].items():
        if label_re.search(fl_lower):
            via = "exact" if fl_lower == label else "english_substring"
            return label, via
    for native, english in ns["NATIVE_LABEL_MAP"].items():
        if native.lower() in fl_lower:
            via = "native_exact" if fl_lower == native.lower() else "native_substring"
            return english, via
    return None, "none"


def classify_choice(raw_output: str, choices: str) -> tuple[str | None, str]:
    """Mirror of extract_choice with stage instrumentation."""
    text = raw_output.strip().upper()
    first_line = text.split("\n")[0].strip()
    choice_class = _re.escape(choices)
    m = _re.search(rf"\b([{choice_class}])\b", first_line)
    if m:
        via = "bare_letter" if first_line == m.group(1) else "letter_in_text"
        return m.group(1), via
    m = _re.search(rf"ANSWER\s*[:\-]?\s*([{choice_class}])", first_line)
    if m:
        return m.group(1), "answer_prefix"
    return None, "none"


# Benchmark → classifier callable with a uniform (raw_output, ns) signature.
# The choice classifiers ignore `ns` (they need no extractor constants) but
# accept it so callers can dispatch without special-casing.
def _classifier_for(benchmark: str):
    if benchmark == "sib200":
        return lambda raw, ns: classify_sib200(raw, ns)
    if benchmark == "xnli":
        return lambda raw, ns: classify_xnli(raw, ns)
    if benchmark == "csqa":
        return lambda raw, _ns: classify_choice(raw, "ABCDE")
    if benchmark == "belebele":
        return lambda raw, _ns: classify_choice(raw, "ABCD")
    raise ValueError(f"Unknown benchmark {benchmark!r}")


BENCHMARKS = ("belebele", "csqa", "sib200", "xnli")


def benchmark_from_key(cell_key: str) -> str:
    """`template1_sib200_data=ur_instr=ur` → `sib200`."""
    for bench in BENCHMARKS:
        if f"_{bench}_" in cell_key:
            return bench
    raise ValueError(f"Couldn't infer benchmark from {cell_key!r}")


# ----------------------------------------------------------------------------
# Row classification
# ----------------------------------------------------------------------------
def classify_row(benchmark: str, raw_output: str, gold: str, ns: dict) -> dict:
    """Classify one row on both axes. Returns a dict with keys:
    outcome, match_via, multiline, pred, gold."""
    classifier = _classifier_for(benchmark)
    pred, match_via = classifier(raw_output, ns)
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


def classify_cell(cell_key: str, rows: list[dict], ns: dict) -> dict:
    """Classify every row in one cell. Returns aggregate counts + the
    classified rows (for sample printing)."""
    benchmark = benchmark_from_key(cell_key)
    classified = []
    for row in rows:
        c = classify_row(benchmark, row["raw_output"], row["gold"], ns)
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
# Reporting
# ----------------------------------------------------------------------------
# Sort buckets so the table reads correct → wrong → fail, cleanest match first.
_OUTCOME_ORDER = {"correct": 0, "wrong_label": 1, "parse_fail": 2}
_VIA_ORDER = {
    "exact": 0,
    "native_exact": 0,
    "bare_letter": 0,
    "normalized": 1,
    "english_substring": 2,
    "native_substring": 2,
    "substring": 2,
    "letter_in_text": 2,
    "answer_prefix": 3,
    "alias": 4,
    "rule_a": 5,
    "rule_b": 6,
    "rule_c": 7,
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
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed — can't download from HF. Either "
            "install it or pass a local file path."
        )
    print(f"Downloading {path_arg} from {HF_REPO_ID}...")
    return Path(
        hf_hub_download(repo_id=HF_REPO_ID, filename=path_arg, repo_type=HF_REPO_TYPE)
    )


# ----------------------------------------------------------------------------
# Self-test — confirm the instrumented classifiers agree with the live
# extractors on the prediction itself (not just the stage label).
# ----------------------------------------------------------------------------
def run_self_test(results_path: Path, ns: dict) -> int:
    """Cross-check every row: classify_*'s pred must equal the live extractor's
    pred. Returns the number of mismatches (0 = pass)."""
    extractors = {
        "sib200": ns["extract_sib200_category"],
        "xnli": ns["extract_xnli_label"],
        "csqa": lambda t: ns["extract_choice"](t, choices="ABCDE"),
        "belebele": lambda t: ns["extract_choice"](t, choices="ABCD"),
    }
    with results_path.open() as f:
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
            instrumented_pred, _ = classifier(raw, ns)
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
        "results_file",
        help="Local path to a _results_*.json, or an HF path under phase3/ "
        "(downloaded automatically).",
    )
    parser.add_argument(
        "--cell",
        default=None,
        help="Only inspect this exact cell key, e.g. "
        "'template1_sib200_data=ur_instr=ur'.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
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
        "--self-test",
        action="store_true",
        help="Cross-check the instrumented classifiers against the live "
        "extractors on every row, then exit. Non-zero exit on any mismatch.",
    )
    args = parser.parse_args()

    ns = load_extractor_namespace()
    results_path = resolve_results_file(args.results_file)

    if args.self_test:
        raise SystemExit(0 if run_self_test(results_path, ns) == 0 else 1)

    with results_path.open() as f:
        data = json.load(f)

    # Select cells.
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
        if not cell_keys:
            raise SystemExit(f"No cell matched --cell={args.cell!r}")
    if args.benchmark:
        cell_keys = [k for k in cell_keys if benchmark_from_key(k) == args.benchmark]
        if not cell_keys:
            raise SystemExit(f"No cells for --benchmark={args.benchmark!r}")

    print(f"=== {results_path.name} ===")
    print(f"Inspecting {len(cell_keys)} cell(s)")

    for cell_key in sorted(cell_keys):
        report = classify_cell(cell_key, data[cell_key], ns)
        if args.outcome:
            # Restrict the printed samples to one outcome by filtering the
            # classified rows down before reporting.
            report["classified_rows"] = [
                c for c in report["classified_rows"] if c["outcome"] == args.outcome
            ]
        print_cell_report(report, samples=args.samples, group_prefix=args.group_prefix)


if __name__ == "__main__":
    main()

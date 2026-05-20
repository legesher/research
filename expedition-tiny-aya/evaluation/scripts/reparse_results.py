"""Re-parse an existing `_results_*.json` against the current extractors.

Run after updating an extractor (e.g., `extract_sib200_category`) to see the
before/after parse-failure rates and accuracies on already-collected results,
without re-running inference on a GPU.

Usage:
    # Print before/after diff table only:
    python reparse_results.py path/to/condition-2-ur-5k_seed42_smoke20_results_template1.json
    python reparse_results.py path/*.json --extractors sib200

    # Also write a sibling `_summary_reparsed_{template}.json` next to each input:
    python reparse_results.py path/*_results_*.json --write-reparsed-summary

The script imports the live extractors from `run_eval_single.py`. To use it
locally (off-Kaggle), drop a copy of `run_eval_single.py` next to this script
or set PYTHONPATH to point at where evaluate.ipynb wrote it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

# Load extractors from `run_eval_single.py` *without* triggering the heavy
# imports at the top of that file (torch, unsloth, kaggle_secrets). We parse
# the source, pull just the extractor function definitions + their helper
# constants via AST, and exec that subset into an isolated namespace. This
# means reparse_results.py runs anywhere stock Python + `re` is available.
import ast
import re as _re

HERE = Path(__file__).resolve().parent
_candidates = [HERE / "run_eval_single.py", Path("/kaggle/working/run_eval_single.py")]
_source_path = next((p for p in _candidates if p.exists()), None)
if _source_path is None:
    raise SystemExit(
        "Couldn't find run_eval_single.py. Expected next to this file or at "
        "/kaggle/working/. On Kaggle the launcher writes it; locally extract "
        "it from evaluate.ipynb cell 3."
    )

_WANTED_NAMES = {
    "SIB200_CATEGORIES",
    "SIB200_ALIASES",
    "SIB200_SCITECH_NATIVE",
    "SIB200_SCITECH_BARE_SUBCATEGORIES",
    "XNLI_LABEL_RES",
    "extract_sib200_category",
    "extract_xnli_label",
    "extract_choice",
}
_tree = ast.parse(_source_path.read_text())
_subset_nodes = []
for node in _tree.body:
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _WANTED_NAMES
    ):
        _subset_nodes.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in _WANTED_NAMES for t in node.targets
    ):
        _subset_nodes.append(node)

_module = ast.Module(body=_subset_nodes, type_ignores=[])
_ns: dict = {"re": _re}
exec(compile(_module, str(_source_path), "exec"), _ns)
extract_sib200_category = _ns["extract_sib200_category"]
extract_xnli_label = _ns["extract_xnli_label"]
extract_choice = _ns["extract_choice"]


EXTRACTORS = {
    "xnli": extract_xnli_label,
    "csqa": lambda t: extract_choice(t, choices="ABCDE"),
    "belebele": lambda t: extract_choice(t, choices="ABCD"),
    "sib200": extract_sib200_category,
}


def benchmark_from_key(cell_key: str) -> str:
    """`template1_sib200_data=ur_instr=ur` → `sib200`."""
    for bench in EXTRACTORS:
        if f"_{bench}_" in cell_key:
            return bench
    raise ValueError(f"Couldn't infer benchmark from {cell_key!r}")


def reparse_file(path: Path, only: set[str] | None = None) -> list[dict]:
    """Re-run extractors on raw_output, return per-cell before/after rows."""
    with path.open() as f:
        data = json.load(f)

    old_summary = data.get("summary", {})
    old_failure_rates = data.get("parse_failure_rates", {})

    rows = []
    for key, items in data.items():
        if key in {"summary", "parse_failure_rates"}:
            continue
        if not isinstance(items, list) or not items or "raw_output" not in items[0]:
            continue
        bench = benchmark_from_key(key)
        if only is not None and bench not in only:
            continue

        extractor = EXTRACTORS[bench]
        new_correct = 0
        new_fail = 0
        for row in items:
            new_pred = extractor(row["raw_output"])
            if new_pred is None:
                new_fail += 1
            elif new_pred == row["gold"]:
                new_correct += 1

        n = len(items)
        old_acc = old_summary.get(f"{key}_acc")
        old_fail = old_failure_rates.get(key)
        rows.append(
            {
                "cell": key,
                "n": n,
                "old_acc": old_acc,
                "new_acc": new_correct / n if n else 0.0,
                "old_fail": old_fail,
                "new_fail": new_fail / n if n else 0.0,
            }
        )
    return rows


def print_diff_table(rows: list[dict]) -> None:
    changed = [r for r in rows if r["new_acc"] != r["old_acc"]]
    if not changed:
        print("No accuracy changes.")
        return

    print(
        f"{'cell':<55} {'old_acc':>8} {'new_acc':>8} {'Δacc':>7} "
        f"{'old_fail':>9} {'new_fail':>9}"
    )
    print("-" * 100)
    for r in sorted(
        changed, key=lambda x: x["new_acc"] - (x["old_acc"] or 0), reverse=True
    ):
        d = r["new_acc"] - (r["old_acc"] or 0)
        print(
            f"{r['cell']:<55} "
            f"{r['old_acc']:>8.3f} {r['new_acc']:>8.3f} {d:>+7.3f} "
            f"{r['old_fail']:>9.3f} {r['new_fail']:>9.3f}"
        )

    avg_delta = sum(r["new_acc"] - (r["old_acc"] or 0) for r in changed) / len(changed)
    print(f"\n{len(changed)} cells changed; mean Δacc = {avg_delta:+.3f}")


def _extractor_provenance() -> dict:
    """Identify which version of `run_eval_single.py` produced the new numbers.

    Falls back to a SHA-256 of the source if we're not inside a git checkout
    (e.g., on Kaggle where the file is written by `%%writefile`).
    """
    assert _source_path is not None  # narrowed; module would have exited otherwise
    src = _source_path
    provenance: dict = {
        "source_path": str(src),
        "content_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    # Repo HEAD SHA — the extractor source is gitignored locally (it's
    # extracted from evaluate.ipynb at runtime), so per-file git log is empty.
    # The repo HEAD identifies the notebook version that produced this source.
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=src.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if head_sha:
            provenance["repo_head_commit"] = head_sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return provenance


def _reparsed_summary_path(input_path: Path) -> Path:
    """`X_results_template1.json` → `X_summary_reparsed_template1.json`."""
    name = input_path.name
    if "_results_" not in name:
        raise ValueError(f"Input filename must contain '_results_': {input_path.name}")
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    return input_path.with_name(new_name)


def build_reparsed_summary(
    input_path: Path, rows: list[dict], only: set[str] | None
) -> dict:
    """Construct the JSON body for a `_summary_reparsed_{template}.json` sibling.

    Mirrors the original `_summary_*.json` schema (`summary` + `parse_failure_rates`
    top-level keys) so downstream analysis can read original and reparsed
    interchangeably. Adds a `reparse_metadata` block recording when, against
    which extractor version, and what changed.
    """
    new_summary: dict[str, float] = {}
    new_failure_rates: dict[str, float] = {}
    delta_per_cell: dict[str, dict] = {}

    for r in rows:
        cell = r["cell"]
        new_summary[f"{cell}_acc"] = r["new_acc"]
        new_failure_rates[cell] = r["new_fail"]
        d_acc = r["new_acc"] - (r["old_acc"] or 0)
        d_fail = r["new_fail"] - (r["old_fail"] or 0)
        if abs(d_acc) > 1e-9 or abs(d_fail) > 1e-9:
            delta_per_cell[cell] = {
                "old_acc": r["old_acc"],
                "new_acc": r["new_acc"],
                "delta_acc": d_acc,
                "old_fail": r["old_fail"],
                "new_fail": r["new_fail"],
                "delta_fail": d_fail,
            }

    return {
        "summary": new_summary,
        "parse_failure_rates": new_failure_rates,
        "reparse_metadata": {
            "reparsed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "original_results_filename": input_path.name,
            "extractors_applied": sorted(only) if only else sorted(EXTRACTORS),
            "extractor_provenance": _extractor_provenance(),
            "cells_changed": len(delta_per_cell),
            "delta_per_cell": delta_per_cell,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=sorted(EXTRACTORS),
        help="Only re-parse these benchmarks (default: all)",
    )
    parser.add_argument(
        "--write-reparsed-summary",
        action="store_true",
        help=(
            "Write a sibling `_summary_reparsed_{template}.json` next to each "
            "input. Preserves original `_summary_*.json` and `_results_*.json` "
            "untouched."
        ),
    )
    args = parser.parse_args()

    only = set(args.extractors) if args.extractors else None
    for path in args.files:
        print(f"\n=== {path.name} ===")
        rows = reparse_file(path, only=only)
        print_diff_table(rows)
        if args.write_reparsed_summary:
            out_path = _reparsed_summary_path(path)
            body = build_reparsed_summary(path, rows, only)
            out_path.write_text(json.dumps(body, indent=2, ensure_ascii=False))
            print(
                f"  → wrote {out_path.name} ({body['reparse_metadata']['cells_changed']} cells changed)"
            )


if __name__ == "__main__":
    main()

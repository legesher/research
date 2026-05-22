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

This module is import-safe even when `run_eval_single.py` is missing — the
extractors load lazily on first use (or at preflight via `verify_extractor_source()`).
That lets the upload driver import this module to reuse path helpers
without triggering a hard exit.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import re as _re
import subprocess
from pathlib import Path
from typing import Callable

# Static metadata that callers may need before the extractor source is
# available — argparse choices, path helpers, etc.
EXTRACTOR_NAMES: tuple[str, ...] = ("belebele", "csqa", "sib200", "xnli")

HERE = Path(__file__).resolve().parent


def _find_extractor_source() -> Path | None:
    """Locate `run_eval_single.py`. Returns None if not found.

    Search order: next to this script, then Kaggle's working dir.
    """
    for candidate in (
        HERE / "run_eval_single.py",
        Path("/kaggle/working/run_eval_single.py"),
    ):
        if candidate.exists():
            return candidate
    return None


_source_path: Path | None = _find_extractor_source()
_extractors_cache: dict[str, Callable] | None = None


def verify_extractor_source() -> Path:
    """Preflight check — raise with a clear message if the extractor source
    can't be located. Callers should invoke this early so users see the
    error before any expensive work (HF API calls, downloads)."""
    if _source_path is None:
        raise SystemExit(
            "Couldn't find run_eval_single.py. Expected next to this file or at "
            "/kaggle/working/. On Kaggle the launcher writes it; locally extract "
            "it from evaluate.ipynb cell 3."
        )
    return _source_path


def _load_extractors() -> dict[str, Callable]:
    """AST-load the extractor functions + their helper constants from
    `run_eval_single.py`. Cached after first call.

    Skips the heavy top-level imports (torch, unsloth, kaggle_secrets) of
    `run_eval_single.py` by exec'ing only the wanted nodes.
    """
    global _extractors_cache
    if _extractors_cache is not None:
        return _extractors_cache

    src = verify_extractor_source()
    wanted_names = {
        # Constants referenced inside extract_sib200_category
        "SIB200_CATEGORIES",
        "SIB200_ALIASES",
        "SIB200_SCITECH_NATIVE",
        "SIB200_SCITECH_BARE_SUBCATEGORIES",
        # Constants referenced inside extract_xnli_label
        "XNLI_LABEL_RES",
        "NATIVE_LABEL_MAP",
        # The three extractor functions themselves
        "extract_sib200_category",
        "extract_xnli_label",
        "extract_choice",
    }
    tree = ast.parse(src.read_text())
    subset_nodes: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted_names
        ):
            subset_nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets
        ):
            subset_nodes.append(node)

    module = ast.Module(body=subset_nodes, type_ignores=[])
    ns: dict = {"re": _re}
    exec(compile(module, str(src), "exec"), ns)
    extract_sib200_category = ns["extract_sib200_category"]
    extract_xnli_label = ns["extract_xnli_label"]
    extract_choice = ns["extract_choice"]

    _extractors_cache = {
        "xnli": extract_xnli_label,
        "csqa": lambda t: extract_choice(t, choices="ABCDE"),
        "belebele": lambda t: extract_choice(t, choices="ABCD"),
        "sib200": extract_sib200_category,
    }
    return _extractors_cache


def benchmark_from_key(cell_key: str) -> str:
    """`template1_sib200_data=ur_instr=ur` → `sib200`. Pure string parsing,
    doesn't need extractors loaded."""
    for bench in EXTRACTOR_NAMES:
        if f"_{bench}_" in cell_key:
            return bench
    raise ValueError(f"Couldn't infer benchmark from {cell_key!r}")


def reparsed_summary_path_local(input_path: Path) -> Path:
    """`X_results_template1.json` → `X_summary_reparsed_template1.json`.

    Single source of truth for path mangling; used by both the local-file
    workflow (`--write-reparsed-summary`) and the HF round-trip driver."""
    name = input_path.name
    if "_results_" not in name:
        raise ValueError(f"Input filename must contain '_results_': {input_path.name}")
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    return input_path.with_name(new_name)


def reparsed_summary_path_remote(remote_results_path: str) -> str:
    """Same substitution as `reparsed_summary_path_local`, but for HF-style
    forward-slash paths (`phase3/conditions/<cond>/seed<N>/X_results_*.json`).
    Returns the modified path as a string, preserving the parent directory."""
    name = remote_results_path.split("/")[-1]
    if "_results_" not in name:
        raise ValueError(f"Input filename must contain '_results_': {name}")
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    if "/" in remote_results_path:
        parent = remote_results_path.rsplit("/", 1)[0]
        return f"{parent}/{new_name}"
    return new_name


def reparse_file(path: Path, only: set[str] | None = None) -> list[dict]:
    """Re-run extractors on raw_output, return per-cell before/after rows.

    `only` filters which benchmarks are scored (display filter for the diff
    table). When writing a reparsed summary you'd typically pass `only=None`
    so every cell in the original is recomputed and the schema stays
    consistent with the source."""
    extractors = _load_extractors()

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

        extractor = extractors[bench]
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

    Always includes the source path + content sha256. Adds `repo_head_commit`
    when this script is running inside a git checkout (otherwise, e.g., on
    Kaggle where the file is written by `%%writefile`, the field is omitted)."""
    src = verify_extractor_source()
    provenance: dict = {
        "source_path": str(src),
        "content_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=src.parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if head_sha:
            provenance["repo_head_commit"] = head_sha
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass
    return provenance


def build_reparsed_summary(input_path: Path, rows: list[dict]) -> dict:
    """Construct the JSON body for a `_summary_reparsed_{template}.json` sibling.

    Mirrors the original `_summary_*.json` schema (`summary` + `parse_failure_rates`
    top-level keys) so downstream analysis can read original and reparsed
    interchangeably. Adds a `reparse_metadata` block recording when, against
    which extractor version, and what changed.

    Always treats `rows` as a complete recompute — callers should pass rows
    from `reparse_file(path, only=None)`. If you need a partial recompute
    for the diff-table display, run reparse_file twice (once filtered for
    display, once unfiltered for writing) — main() handles this for you."""
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
            "extractors_applied": list(EXTRACTOR_NAMES),
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
        choices=sorted(EXTRACTOR_NAMES),
        help="Filter the printed diff table to these benchmarks (default: all). "
        "Note: when combined with --write-reparsed-summary, the WRITTEN summary "
        "still includes every benchmark — this flag only filters what's printed.",
    )
    parser.add_argument(
        "--write-reparsed-summary",
        action="store_true",
        help=(
            "Write a sibling `_summary_reparsed_{template}.json` next to each "
            "input. Always re-runs every extractor so the summary stays a "
            "drop-in replacement for the original. Preserves original "
            "`_summary_*.json` and `_results_*.json` untouched."
        ),
    )
    args = parser.parse_args()

    # Preflight — surface "no run_eval_single.py" before doing any I/O work.
    verify_extractor_source()

    display_filter = set(args.extractors) if args.extractors else None
    for path in args.files:
        print(f"\n=== {path.name} ===")
        # For the diff table, optionally filter to selected extractors.
        rows_for_display = reparse_file(path, only=display_filter)
        print_diff_table(rows_for_display)

        if args.write_reparsed_summary:
            # For the written summary, always recompute every benchmark so
            # the output is a complete drop-in for the original summary.
            rows_full = (
                rows_for_display
                if display_filter is None
                else reparse_file(path, only=None)
            )
            out_path = reparsed_summary_path_local(path)
            body = build_reparsed_summary(path, rows_full)
            out_path.write_text(
                json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  → wrote {out_path.name} "
                f"({body['reparse_metadata']['cells_changed']} cells changed)"
            )


if __name__ == "__main__":
    main()

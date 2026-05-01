#!/usr/bin/env python3
"""Cross-cell ``idx`` intersection for cross-condition row-level alignment.

Background
----------
After CORE-1171, every per-(language, condition) cell publishes a
``metadata.csv`` with the ``idx`` column (source parquet row index).
File ``0042.py`` in cond2-ur and cond5-ur originate from the same
``condition-1-en-5k`` source row by construction.

But per-pipeline failure rates differ — cond5's LLM may fail on a row
that cond2's mechanical translator handled cleanly (or vice versa). If
those cells are published as-is, joining on ``idx`` produces a partial
result with NULLs for divergent rows.

This script intersects the successful ``idx`` sets across every
``(language, condition)`` cell, per split, so callers can re-package
each cell filtered to the intersection — guaranteeing identical row
membership across cells in the published HF datasets.

Usage
-----
::

    harmonize_splits.py compute \\
      --cell cond2-ur:train=packaged/condition-2-ur-5k/train/ur \\
      --cell cond2-ur:validation=packaged/condition-2-ur-5k/validation/ur \\
      --cell cond5-ur:train=packaged/condition-5-ur-5k-c4ai/train/ur \\
      --cell cond5-ur:validation=packaged/condition-5-ur-5k-c4ai/validation/ur \\
      --output keep_idx/

Add ``--dry-run`` to skip writing the ``.idx`` files; only ``report.json``
is emitted (useful for "how many rows would be dropped before I commit?").

Outputs
-------
``<output>/{train,validation}.idx`` — newline-separated ``idx`` integers
present in EVERY cell's metadata.csv for that split (sorted ascending).

``<output>/report.json`` — per-cell totals, intersection size,
per-dropped-idx attribution (which cells missed each), so you can debug
why a row dropped without re-running the whole chain.

The ``.idx`` files feed ``package_dataset.py from-files --keep-idx-from
<file>`` (CORE-1172 packager patch) to filter rows during the final
package + push step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CELL_FLAG_RE_DOC = "--cell <name>:<split>=<path>"
EXPECTED_SPLITS = ("train", "validation")


def _parse_cell_flag(spec: str) -> tuple[str, str, Path]:
    """Parse one ``--cell`` value into ``(cell_name, split, path)``.

    Format: ``<name>:<split>=<path>``. Both ``:`` and ``=`` are required.
    Raises ``argparse.ArgumentTypeError`` on malformed input.
    """
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            f"--cell value {spec!r} must look like {CELL_FLAG_RE_DOC}"
        )
    name_split, path_str = spec.split("=", 1)
    name, split = name_split.split(":", 1)
    name = name.strip()
    split = split.strip()
    if not name or not split or not path_str.strip():
        raise argparse.ArgumentTypeError(
            f"--cell value {spec!r} has empty name/split/path component"
        )
    if split not in EXPECTED_SPLITS:
        raise argparse.ArgumentTypeError(
            f"--cell value {spec!r}: split must be one of {EXPECTED_SPLITS}"
        )
    return name, split, Path(path_str.strip())


def _read_cell_idx_set(metadata_path: Path) -> set[int]:
    """Read the ``idx`` column from ``metadata_path`` as a set of ints.

    Skips rows with missing/non-integer idx (logged to stderr).
    """
    if not metadata_path.exists():
        print(
            f"  warning: metadata.csv not found: {metadata_path}",
            file=sys.stderr,
        )
        return set()
    out: set[int] = set()
    with metadata_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "idx" not in (reader.fieldnames or []):
            print(
                f"  warning: no 'idx' column in {metadata_path}",
                file=sys.stderr,
            )
            return set()
        for row in reader:
            raw = row.get("idx")
            if raw in (None, ""):
                continue
            try:
                out.add(int(raw))
            except (TypeError, ValueError):
                continue
    return out


def _compute_intersection(
    cell_sets: dict[str, dict[str, set[int]]],
) -> tuple[dict[str, set[int]], dict[str, dict[int, list[str]]]]:
    """Intersect idx sets per split; track which cells dropped each idx.

    Returns ``(intersections, dropped_attribution)`` where:
    - ``intersections[split]`` is the set of idx present in ALL cells
    - ``dropped_attribution[split][idx]`` lists cells that DIDN'T have idx
      (only present for idx values in the union but not the intersection)
    """
    intersections: dict[str, set[int]] = {}
    dropped_attribution: dict[str, dict[int, list[str]]] = {}

    for split in EXPECTED_SPLITS:
        per_cell = {
            name: splits[split] for name, splits in cell_sets.items() if split in splits
        }
        if not per_cell:
            intersections[split] = set()
            dropped_attribution[split] = {}
            continue

        union: set[int] = set().union(*per_cell.values())
        intersection: set[int] = set.intersection(*per_cell.values())
        intersections[split] = intersection

        attribution: dict[int, list[str]] = {}
        for idx in sorted(union - intersection):
            missing_from = sorted(
                cell for cell, idx_set in per_cell.items() if idx not in idx_set
            )
            attribution[idx] = missing_from
        dropped_attribution[split] = attribution

    return intersections, dropped_attribution


def _build_report(
    cell_sets: dict[str, dict[str, set[int]]],
    intersections: dict[str, set[int]],
    dropped_attribution: dict[str, dict[int, list[str]]],
) -> dict[str, object]:
    """Assemble the report.json structure."""
    per_split: dict[str, object] = {}
    for split in EXPECTED_SPLITS:
        per_cell_counts = {
            name: len(splits.get(split, set()))
            for name, splits in sorted(cell_sets.items())
        }
        intersection = intersections.get(split, set())
        union_size = sum(per_cell_counts.values()) and (
            max(per_cell_counts.values()) if per_cell_counts else 0
        )
        # Real union (not just max — that was a wrong shortcut)
        union_size = len(
            set().union(*(splits.get(split, set()) for splits in cell_sets.values()))
        )
        per_split[split] = {
            "per_cell_count": per_cell_counts,
            "union_size": union_size,
            "intersection_size": len(intersection),
            "dropped_count": union_size - len(intersection),
            "dropped_attribution": {
                str(idx): cells
                for idx, cells in dropped_attribution.get(split, {}).items()
            },
        }
    return {
        "cells": sorted(cell_sets.keys()),
        "splits": list(EXPECTED_SPLITS),
        "per_split": per_split,
    }


def _print_summary(report: dict[str, object]) -> None:
    """Compact stdout summary; full detail lives in report.json."""
    cells = report["cells"]  # type: ignore[index]
    print(f"\nCells: {', '.join(cells)}")  # type: ignore[arg-type]
    for split, stats in report["per_split"].items():  # type: ignore[index,union-attr]
        per_cell = stats["per_cell_count"]  # type: ignore[index]
        print(
            f"\n{split}: "
            f"intersection={stats['intersection_size']} "  # type: ignore[index]
            f"union={stats['union_size']} "  # type: ignore[index]
            f"dropped={stats['dropped_count']}"  # type: ignore[index]
        )
        for cell, count in sorted(per_cell.items()):
            in_int = stats["intersection_size"]  # type: ignore[index]
            shortfall = count - in_int
            marker = "" if shortfall == 0 else f" (cell has {shortfall} extras dropped)"
            print(f"  {cell}: {count} rows{marker}")


def cmd_compute(args: argparse.Namespace) -> int:
    """Compute intersection across ``--cell`` flags and write artifacts."""
    cells_raw: list[tuple[str, str, Path]] = args.cell
    if not cells_raw:
        print("error: at least one --cell value required", file=sys.stderr)
        return 2

    # Group: cell_sets[cell_name][split] = set of idx values
    cell_sets: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for name, split, path in cells_raw:
        if split in cell_sets[name]:
            print(
                f"error: duplicate --cell {name}:{split} entries",
                file=sys.stderr,
            )
            return 2
        metadata_path = path / "metadata.csv"
        cell_sets[name][split] = _read_cell_idx_set(metadata_path)
        print(
            f"  {name}/{split}: {len(cell_sets[name][split])} rows from "
            f"{metadata_path}"
        )

    # Sanity: every cell should contribute to every expected split
    incomplete = [
        (name, split)
        for name, splits in cell_sets.items()
        for split in EXPECTED_SPLITS
        if split not in splits
    ]
    if incomplete:
        for name, split in incomplete:
            print(
                f"warning: cell {name!r} missing --cell {name}:{split}=...",
                file=sys.stderr,
            )

    intersections, dropped_attribution = _compute_intersection(cell_sets)
    report = _build_report(cell_sets, intersections, dropped_attribution)
    _print_summary(report)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    if args.dry_run:
        print("(dry-run: skipping .idx file writes)")
        return 0

    for split, intersection in intersections.items():
        idx_path = output_dir / f"{split}.idx"
        with idx_path.open("w", encoding="utf-8") as f:
            for idx in sorted(intersection):
                f.write(f"{idx}\n")
        print(f"  Wrote {idx_path} ({len(intersection)} idx values)")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the cross-cell idx intersection for cross-condition "
            "row-level alignment of published HF datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser(
        "compute",
        help="Compute intersection across (cell, split) cells",
    )
    compute.add_argument(
        "--cell",
        action="append",
        type=_parse_cell_flag,
        default=[],
        metavar=CELL_FLAG_RE_DOC,
        help=(
            "Add one (cell, split) input. Repeat for every cell × split "
            "combination. Path should point at the per-language transpiled "
            "directory containing metadata.csv (i.e. <output>/<lang>)."
        ),
    )
    compute.add_argument(
        "--output",
        required=True,
        help="Directory for keep_idx/{train,validation}.idx + report.json",
    )
    compute.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip writing .idx files; only emit report.json for inspection",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "compute":
        return cmd_compute(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())

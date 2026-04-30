#!/usr/bin/env python3
"""Compile identifier frequency across the first N source files.

Goal: identify identifiers that repeat across files, so we can decide which
ones belong in the language pack (special / dunder / convention) vs. which
should keep going through the LLM.

For each identifier, reports:
- total_occurrences: how many times it appears across all sampled files
- files_seen_in: how many distinct files contain it
- category: dunder / convention / single_letter / typing_special / other

Categories help triage which identifiers should be:
- preserved as-is (dunders like __init__, __name__, __all__)
- pinned to a known translation in the pack (self, cls, args, kwargs)
- left for the LLM (everything else)

Source can be either a parquet (e.g. ``condition-1-en-5k/train.parquet``)
or a directory of ``.py`` files (e.g. a cond5 populator output dir like
``condition-5-ur-5k-c4ai-aya-expanse-32b/ur/``). The ``--lang`` flag
labels the output for cross-language comparison.

Usage:
    # English source (default parquet)
    python scripts/identifier_frequency_analysis.py --n-files 20

    # Translated Spanish output from a cond5 run
    python scripts/identifier_frequency_analysis.py --lang es --n-files 50 \\
        --source packaged/condition-5-es-5k-c4ai-aya-expanse-32b/es/

    # Larger sample, more rows in report
    python scripts/identifier_frequency_analysis.py --n-files 200 --top 100
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import keyword
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    """argparse type for ``--n-files`` / ``--top``: strictly positive int."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"value must be positive (got {n}); pass at least 1"
        )
    return n


def load_source_files(source: str, n_files: int) -> list[dict[str, Any]]:
    """Load N source files from a parquet OR a directory of ``.py`` files.

    Each returned dict has a ``code`` key (file content as a string) and,
    when sourced from a directory, a ``file_path`` key (relative path of
    the source file). Parquets are expected to have a ``code`` column;
    rows are taken in order, capped at ``n_files``.

    Directory mode walks ``**/*.py``, sorts deterministically, and reads
    each as UTF-8. Files with decode errors are skipped silently.
    """
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"--source does not exist: {source}")

    if source_path.is_file():
        ds = load_dataset("parquet", data_files=str(source_path), split="train")
        sample = ds.select(range(min(n_files, len(ds))))
        return [dict(row) for row in sample]

    # Directory mode: walk *.py files
    py_files = sorted(source_path.glob("**/*.py"))[:n_files]
    rows: list[dict[str, Any]] = []
    for p in py_files:
        try:
            code = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rows.append(
            {
                "code": code,
                "file_path": str(p.relative_to(source_path)),
            }
        )
    return rows


try:
    from datasets import load_dataset
except ImportError as exc:
    print(
        f"error: this script requires the `datasets` package "
        f"({exc.name or 'datasets'} not importable). "
        "Run via `uv run` from the data-pipeline checkout, e.g.:\n"
        "    cd expedition-tiny-aya/data-pipeline\n"
        "    uv run python scripts/identifier_frequency_analysis.py --n-files 20",
        file=sys.stderr,
    )
    raise

# Defaults derived relative to this script's location so the tool is
# portable across checkouts. `parents[1]` resolves to `data-pipeline/`
# (the parent of `scripts/`).
DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = str(
    DATA_PIPELINE_ROOT / "packaged" / "condition-1-en-103k" / "train.parquet"
)

PYTHON_KEYWORDS = set(keyword.kwlist) | set(keyword.softkwlist)
PYTHON_BUILTINS = set(dir(builtins))

CONVENTION_IDENTIFIERS = {
    "self",
    "cls",
    "args",
    "kwargs",
}

TYPING_SPECIALS = {
    "T",
    "K",
    "V",
    "U",
    "P",
    "T_co",
    "T_contra",
    "K_co",
    "V_co",
}


def categorize(name: str) -> str:
    if name.startswith("__") and name.endswith("__") and len(name) > 4:
        return "dunder"
    if name in CONVENTION_IDENTIFIERS:
        return "convention"
    if name in TYPING_SPECIALS:
        return "typing_special"
    if len(name) == 1 and name.isalpha():
        return "single_letter"
    if name.startswith("_") and not name.startswith("__"):
        return "private_convention"
    if name.isupper():
        return "constant_style"
    return "other"


def extract_identifiers(code: str) -> list[str]:
    """Walk the AST and collect every binding/use site that is an identifier.

    Filters out Python keywords and builtins (they are handled by the language
    pack, not the LLM).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.append(node.arg)
        # `import X as Y` / `from M import X as Y`: Y is the actual binding.
        elif isinstance(node, ast.alias) and node.asname is not None:
            names.append(node.asname)
        # `except E as e:` — the bound exception variable.
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.append(node.name)

    return [n for n in names if n not in PYTHON_KEYWORDS and n not in PYTHON_BUILTINS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_PARQUET,
        help=(
            "Parquet file or directory of .py files to analyze "
            f"(default: {DEFAULT_PARQUET})"
        ),
    )
    parser.add_argument(
        "--lang",
        default="en",
        help=(
            "Language code label for the report (default: en). "
            "Just metadata — labels the output and JSON for cross-language "
            "comparison; does not change parsing behavior."
        ),
    )
    parser.add_argument(
        "--n-files",
        type=_positive_int,
        default=20,
        help="Number of files to analyze (default: 20). Must be positive.",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=60,
        help="Number of top-frequency identifiers to print (default: 60).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full results as JSON",
    )
    args = parser.parse_args()

    print(f"Loading {args.n_files} files [{args.lang}] from {args.source}")
    files = load_source_files(args.source, args.n_files)

    total_occurrences: Counter[str] = Counter()
    files_seen_in: dict[str, set[int]] = defaultdict(set)

    parsed_ok = 0
    parsed_fail = 0
    for idx, row in enumerate(files):
        ids = extract_identifiers(row["code"])
        if not ids:
            parsed_fail += 1
            continue
        parsed_ok += 1
        total_occurrences.update(ids)
        for name in set(ids):
            files_seen_in[name].add(idx)

    print(
        f"Parsed {parsed_ok}/{len(files)} files "
        f"({parsed_fail} failed to parse with stdlib ast)"
    )

    rows = []
    for name, total in total_occurrences.most_common():
        rows.append(
            {
                "identifier": name,
                "total_occurrences": total,
                "files_seen_in": len(files_seen_in[name]),
                "files_pct": round(
                    100 * len(files_seen_in[name]) / max(1, parsed_ok), 1
                ),
                "category": categorize(name),
            }
        )

    print(f"\nTop {args.top} identifiers (excluding Python keywords + builtins):\n")
    print(
        f"{'rank':>4}  {'identifier':<30} {'total':>7} {'files':>6} {'pct':>6}  category"
    )
    print("-" * 80)
    for rank, r in enumerate(rows[: args.top], 1):
        print(
            f"{rank:>4}  {r['identifier']:<30} "
            f"{r['total_occurrences']:>7} "
            f"{r['files_seen_in']:>6} "
            f"{r['files_pct']:>5}%  "
            f"{r['category']}"
        )

    print("\nBy category (top 5 per category):")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, lst in sorted(by_cat.items()):
        print(f"\n  {cat}: {len(lst)} unique identifiers")
        for r in lst[:5]:
            print(
                f"    {r['identifier']:<30} "
                f"total={r['total_occurrences']:<5} "
                f"files={r['files_seen_in']}/{parsed_ok}"
            )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "lang": args.lang,
                    "source": args.source,
                    "n_files_sampled": len(files),
                    "n_files_parsed": parsed_ok,
                    "total_unique_identifiers": len(rows),
                    "identifiers": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote full results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

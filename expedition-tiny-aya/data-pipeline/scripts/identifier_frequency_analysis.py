#!/usr/bin/env python3
"""Compile identifier frequency across the first N files of a parquet.

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

Usage:
    python scripts/identifier_frequency_analysis.py --n-files 20
    python scripts/identifier_frequency_analysis.py --n-files 50 --top 100
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import keyword
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

DEFAULT_PARQUET = (
    "/Users/madisonedgar/GitHub/Legesher/research/expedition-tiny-aya/"
    "data-pipeline/packaged/condition-1-en-103k/train.parquet"
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

    return [n for n in names if n not in PYTHON_KEYWORDS and n not in PYTHON_BUILTINS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--n-files", type=int, default=20)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full results as JSON",
    )
    args = parser.parse_args()

    print(f"Loading {args.n_files} files from {args.source_parquet}")
    ds = load_dataset("parquet", data_files=args.source_parquet, split="train")
    sample = ds.select(range(min(args.n_files, len(ds))))
    files = [dict(row) for row in sample]

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

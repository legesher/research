"""Per-gold-category SIB-200 accuracy + confusion structure for Phase-3 (CORE-1383).

Companion to `build_correct_via_constant.py`. That script measures how
concentrated a cell's predictions are (top_pred_share / top_raw_share); this
one asks the complementary question the constant-output findings left open:
is the concentration UNIFORM (dominant category predicted regardless of
gold — the model ignores the passage) or SELECTIVE (correct where gold
matches the constant, degraded elsewhere but still passage-sensitive)?
Answering that requires the predicted-category distribution PER GOLD
category, which no published refined-table carries.

Output grain: one TSV row per
(condition, seed, template, data_lang, instr_lang, gold_category) — the full
7x(7+1) confusion structure, no aggregation. Rollups (per-condition
category accuracy, science/technology attribution with both denominators)
are computed downstream from this TSV, so no threshold-count can inherit
the seed-vs-cell inflation bug class documented in
`analysis/phase-3/aggregation-bug-audit.md`.

Columns per row:
    n_gold        : rows in this cell whose gold label is `gold_category`
    correct       : rows where refined pred == gold_category
    acc_on_gold   : correct / n_gold (per-gold-slice recall)
    pred_<cat>    : predicted-category counts within this gold slice
                    (7 columns, category names slug-cased)
    pred_none     : refined-extractor parse-failures within this gold slice

Predictions use the same refined extractor
(`reparse_results.extract_sib200_category`) that produced the paper's
numbers. Cross-check: per-cell accuracy recomputed here is asserted against
the published `_summary_reparsed_template*.json` for the same cell; any
mismatch is printed loudly.

Usage:
    # Default — CORE-1383 SIB-200 conditions, TSV into analysis/phase-3/:
    python build_sib200_category_breakdown.py

    # Custom condition set / output path:
    python build_sib200_category_breakdown.py --conditions baseline \
        --output /tmp/sib200-category-accuracy.tsv

Downloads are cached by huggingface_hub (`hf_hub_download`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from reparse_results import (
    SIB200_CATEGORIES,
    extract_sib200_category,
    reparsed_summary_path_remote,
)

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"
PHASE3_PREFIX = "phase3/conditions"

# The CORE-1383 verification set: baseline anchor, the least-constant
# condition (cond-2-ur-5k, top_raw_share 0.303), and the two most-constant
# cond-5 conditions per correct-via-constant-findings.md.
DEFAULT_CONDITIONS = (
    "baseline",
    "condition-2-ur-5k",
    "condition-5-ur-5k",
    "condition-5-es-5k",
)

CELL_KEY_RE = re.compile(
    r"^template(?P<template>\d+)_sib200_data=(?P<data>[a-z]+)_instr=(?P<instr>[a-z]+)$"
)


def _slug(category: str) -> str:
    return category.replace("/", "_")


def list_phase3_results_files(api: HfApi, conditions: set[str]) -> list[str]:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    out: list[str] = []
    for f in files:
        if not f.startswith(f"{PHASE3_PREFIX}/"):
            continue
        if "_results_template" not in f or not f.endswith(".json"):
            continue
        parts = f.split("/")
        if len(parts) < 4 or parts[2] not in conditions:
            continue
        out.append(f)
    return sorted(out)


def parse_filename_metadata(remote_path: str) -> tuple[str, str]:
    """Condition from dir, seed from parent dir (`seed42` → `42`) — same
    convention as build_correct_via_constant.py; parent dir is used because
    of known stray seed-mismatched filenames on HF."""
    parts = remote_path.split("/")
    return parts[2], parts[3].removeprefix("seed")


def load_reparsed_summary(remote_results_path: str) -> dict:
    remote = reparsed_summary_path_remote(remote_results_path)
    local = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote)
    with Path(local).open(encoding="utf-8") as f:
        return json.load(f)


def process_file(remote_path: str, condition: str, seed: str) -> tuple[list[dict], int]:
    """One results file → per-(cell, gold_category) confusion rows.
    Returns (rows, n_accuracy_mismatches_vs_published_reparsed_summary)."""
    local = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path)
    with Path(local).open(encoding="utf-8") as f:
        data = json.load(f)
    published = load_reparsed_summary(remote_path)["summary"]

    rows_out: list[dict] = []
    mismatches = 0
    unexpected_golds: Counter = Counter()
    for key, items in data.items():
        m = CELL_KEY_RE.match(key)
        if not m or not isinstance(items, list):
            continue
        template = int(m.group("template"))

        pred_by_gold: dict[str, Counter] = {g: Counter() for g in SIB200_CATEGORIES}
        cell_correct = 0
        for row in items:
            gold = row.get("gold")
            if gold not in pred_by_gold:
                # Guard rather than KeyError: the canonical reparse_file and
                # build_correct_via_constant.py both tolerate a gold outside the
                # expected set, and a whole analysis run should not die on one row.
                unexpected_golds[gold] += 1
                continue
            pred = extract_sib200_category(row["raw_output"])
            pred_by_gold[gold][pred] += 1
            if pred == gold:
                cell_correct += 1

        n_cell = len(items)
        published_acc = published.get(f"{key}_acc")
        recomputed_acc = cell_correct / n_cell if n_cell else 0.0
        if published_acc is None or abs(recomputed_acc - published_acc) > 1e-9:
            mismatches += 1
            print(
                f"  !! accuracy mismatch {condition}/seed{seed}/{key}: "
                f"recomputed {recomputed_acc:.6f} vs published {published_acc}",
                file=sys.stderr,
            )

        for gold in SIB200_CATEGORIES:
            c = pred_by_gold[gold]
            n_gold = sum(c.values())
            row_out = {
                "condition": condition,
                "seed": seed,
                "template": template,
                "benchmark": "sib200",
                "data_lang": m.group("data"),
                "instr_lang": m.group("instr"),
                "extractor": "refined",
                "gold_category": gold,
                "n_gold": n_gold,
                "correct": c[gold],
                "acc_on_gold": c[gold] / n_gold if n_gold else 0.0,
            }
            for cat in SIB200_CATEGORIES:
                row_out[f"pred_{_slug(cat)}"] = c[cat]
            row_out["pred_none"] = c[None]
            rows_out.append(row_out)
    if unexpected_golds:
        # Never drop rows silently: a skipped gold changes every denominator
        # downstream, and an exclusion the operator cannot see reads as
        # "we covered everything".
        print(
            f"  !! {sum(unexpected_golds.values())} row(s) skipped for an "
            f"unexpected gold label in {remote_path}: "
            f"{dict(unexpected_golds)}",
            file=sys.stderr,
        )
    return rows_out, mismatches


COLUMNS = (
    "condition",
    "seed",
    "template",
    "benchmark",
    "data_lang",
    "instr_lang",
    "extractor",
    "gold_category",
    "n_gold",
    "correct",
    "acc_on_gold",
    *(f"pred_{_slug(cat)}" for cat in SIB200_CATEGORIES),
    "pred_none",
)


def write_tsv(rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        rows,
        key=lambda r: (
            r["condition"],
            r["seed"],
            r["template"],
            r["data_lang"],
            r["instr_lang"],
            SIB200_CATEGORIES.index(r["gold_category"]),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                    for c in COLUMNS
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(DEFAULT_CONDITIONS),
        help="Condition dir names to include. Default: the CORE-1383 set "
        "(baseline, cond-2-ur-5k, cond-5-ur-5k, cond-5-es-5k).",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[2]
            / "analysis"
            / "phase-3"
            / "sib200-category-accuracy.tsv"
        ),
        help="TSV output path. Default: analysis/phase-3/sib200-category-accuracy.tsv.",
    )
    args = parser.parse_args()

    api = HfApi()
    files = list_phase3_results_files(api, set(args.conditions))
    if not files:
        print("No Phase-3 results files matched the filter.", file=sys.stderr)
        return 1

    print(f"Processing {len(files)} _results_template*.json files from {REPO_ID}...")
    all_rows: list[dict] = []
    total_mismatches = 0
    for i, remote_path in enumerate(files, 1):
        condition, seed = parse_filename_metadata(remote_path)
        rows, mismatches = process_file(remote_path, condition, seed)
        total_mismatches += mismatches
        print(f"  [{i}/{len(files)}] {condition}/seed{seed} → {len(rows)} gold-slice rows")
        all_rows.extend(rows)

    out_path = Path(args.output)
    write_tsv(all_rows, out_path)
    print(f"\nWrote {len(all_rows)} rows → {out_path}")
    if total_mismatches:
        print(
            f"WARNING: {total_mismatches} cell(s) disagreed with the published "
            "_summary_reparsed_ accuracies — check extractor version.",
            file=sys.stderr,
        )
        return 1
    print("All per-cell accuracies match the published _summary_reparsed_ files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

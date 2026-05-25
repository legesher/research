"""Compute per-cell `correct_via_constant` rates for Phase-3 SIB-200.

Action item F from `analysis/phase-3/post-refined-action-items.md`. Some
fine-tuned conditions (notably `cond-2-ur-5k` and `cond-5-ur-5k`) emit a
near-constant native-script category — e.g. `سائنس/تکنالوجی`
(science/technology) — regardless of passage. A chunk of those rows land
on gold by chance, inflating the cell's "correct" count even under the
refined extractor.

This script quantifies how much of each SIB-200 cell's correct rows can be
explained by the model defaulting to its most-common prediction. The
artifact lives separately from the refined-tables to keep the summary
schema stable — adding `correct_ambiguous` to every reparsed summary would
have been more invasive, and this question only matters for SIB-200.

Per cell (condition × seed × template × data_lang × instr_lang) it
computes two parallel metrics:

  category-level (refined extractor's pred space — collapses many
  surface forms to one canonical SIB-200 category):
    top_pred                       : most common non-None prediction
    top_pred_count, top_pred_share : how many rows / share of n
    correct_via_constant_pred_*    : correct rows whose pred == top_pred

  raw-output-level (the model's first-line surface form — answers F's
  original "near-constant سائنس/تکنالوجی" question more directly):
    top_raw                        : most common first-line raw output
                                     (whitespace-collapsed, single line)
    top_raw_count, top_raw_share   : how many rows / share of n
    correct_via_constant_raw_*     : correct rows whose first line == top_raw

Plus:
    n              : total rows in the cell
    accuracy       : refined-extractor accuracy
    correct_count  : rows where pred == gold under refined

A high `top_pred_share` + high `correct_via_constant_pred_pct` means
predictions concentrate on one category (model hedges toward science/tech).
A high `top_raw_share` + high `correct_via_constant_raw_pct` means the
model literally emits the same surface form regardless of passage — the
stronger evidence the original action item asked about.

Usage:
    # Default — pull every Phase-3 _results_*.json from HF and emit TSV:
    python build_correct_via_constant.py

    # Limit to a subset of conditions:
    python build_correct_via_constant.py --conditions condition-2-ur-5k condition-5-ur-5k

    # Custom output path:
    python build_correct_via_constant.py --output /tmp/correct-via-constant-rates.tsv

The download is cached by huggingface_hub (`hf_hub_download`), so re-runs
against the same HF main hit the local cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from reparse_results import benchmark_from_key, extract_sib200_category

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"
PHASE3_PREFIX = "phase3/conditions"

# template{N}_sib200_data={X}_instr={Y}
CELL_KEY_RE = re.compile(
    r"^template(?P<template>\d+)_sib200_data=(?P<data>[a-z]+)_instr=(?P<instr>[a-z]+)$"
)


def list_phase3_results_files(api: HfApi, conditions_filter: set[str] | None) -> list[str]:
    """Return every `_results_template*.json` under `phase3/conditions/` on
    HF main. Filters to specified conditions when provided."""
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    out: list[str] = []
    for f in files:
        if not f.startswith(f"{PHASE3_PREFIX}/"):
            continue
        if "_results_template" not in f or not f.endswith(".json"):
            continue
        # f is like phase3/conditions/condition-2-ur-5k/seed42/condition-2-ur-5k_seed42_results_template1.json
        parts = f.split("/")
        if len(parts) < 4:
            continue
        condition = parts[2]
        if conditions_filter is not None and condition not in conditions_filter:
            continue
        out.append(f)
    return sorted(out)


def parse_filename_metadata(remote_path: str) -> tuple[str, str]:
    """`phase3/conditions/<cond>/seed<N>/...results_template<T>.json` → (condition, seed).

    Seed is normalized to drop the `seed` prefix (e.g. `seed42` → `42`,
    `seednone` → `none`) so this TSV's `seed` column joins cleanly with the
    other Phase-3 analysis tables (cells.tsv, conclusion_flips.tsv, etc.)
    which use the bare-number convention. Source is parent dir, not filename,
    because there are known stray seed-mismatched files on HF."""
    parts = remote_path.split("/")
    condition = parts[2]
    seed = parts[3].removeprefix("seed")
    return condition, seed


_RAW_OUTPUT_TRIM_RE = re.compile(r"\s+")


def _normalize_raw_output(raw: str) -> str:
    """Trim + collapse whitespace + truncate at first newline. The raw
    outputs we want to detect as repeats are short native-script tokens
    like `سائنس/تکنالوجی`; the model often appends a multiline English
    explanation after that. Collapsing to the first line on whitespace
    boundary is what distinguishes "same answer, different filler" from
    actual answer diversity."""
    first_line = raw.split("\n", 1)[0]
    return _RAW_OUTPUT_TRIM_RE.sub(" ", first_line).strip()


def cell_stats(rows: list[dict]) -> dict | None:
    """Compute per-cell `correct_via_constant` stats. Returns None if the
    cell has no rows.

    Two parallel metrics:

      * **category-level** — `top_pred` is the most-common refined-extractor
        category. Captures "model concentrates predictions on one category"
        across different surface forms.
      * **raw-output-level** — `top_raw` is the most-common first-line raw
        output (trimmed + whitespace-normalized). Captures the original
        F-spec "model emits a near-constant native-script surface form"
        question — the cond-2-ur-5k `سائنس/تکنالوجی`-shape concern.
    """
    n = len(rows)
    if n == 0:
        return None

    pred_counter: Counter[str] = Counter()
    raw_counter: Counter[str] = Counter()
    correct_preds: list[str] = []
    correct_raws: list[str] = []
    for row in rows:
        raw_full = row.get("raw_output", "")
        raw_norm = _normalize_raw_output(raw_full)
        pred = extract_sib200_category(raw_full)
        gold = row.get("gold")

        if pred is not None:
            pred_counter[pred] += 1
        if raw_norm:
            raw_counter[raw_norm] += 1
        if pred is not None and pred == gold:
            correct_preds.append(pred)
            correct_raws.append(raw_norm)

    if pred_counter:
        top_pred, top_pred_count = pred_counter.most_common(1)[0]
    else:
        top_pred, top_pred_count = None, 0

    if raw_counter:
        top_raw, top_raw_count = raw_counter.most_common(1)[0]
    else:
        top_raw, top_raw_count = "", 0

    correct_count = len(correct_preds)
    correct_via_constant_pred_count = (
        sum(1 for p in correct_preds if p == top_pred) if top_pred is not None else 0
    )
    correct_via_constant_raw_count = (
        sum(1 for r in correct_raws if r == top_raw) if top_raw else 0
    )
    return {
        "n": n,
        "accuracy": correct_count / n,
        # category-level (refined extractor's pred space)
        "top_pred": top_pred or "",
        "top_pred_count": top_pred_count,
        "top_pred_share": top_pred_count / n,
        # raw-output-level (the model's actual surface form, first line only)
        "top_raw": top_raw,
        "top_raw_count": top_raw_count,
        "top_raw_share": top_raw_count / n,
        # correct-row analysis
        "correct_count": correct_count,
        "correct_via_constant_pred_count": correct_via_constant_pred_count,
        "correct_via_constant_pred_pct": (
            correct_via_constant_pred_count / correct_count if correct_count else 0.0
        ),
        "correct_via_constant_raw_count": correct_via_constant_raw_count,
        "correct_via_constant_raw_pct": (
            correct_via_constant_raw_count / correct_count if correct_count else 0.0
        ),
    }


def process_file(remote_path: str, condition: str, seed: str) -> list[dict]:
    """Download (or hit cache) one results file; return a list of per-cell
    stat rows for its SIB-200 cells."""
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_path,
    )
    with Path(local).open(encoding="utf-8") as f:
        data = json.load(f)

    rows_out: list[dict] = []
    for key, items in data.items():
        if key in {"summary", "parse_failure_rates"}:
            continue
        if not isinstance(items, list) or not items or "raw_output" not in items[0]:
            continue
        try:
            bench = benchmark_from_key(key)
        except ValueError:
            continue
        if bench != "sib200":
            continue
        m = CELL_KEY_RE.match(key)
        if not m:
            continue
        template = int(m.group("template"))
        stats = cell_stats(items)
        if stats is None:
            continue
        rows_out.append(
            {
                "condition": condition,
                "seed": seed,
                "template": template,
                "data_lang": m.group("data"),
                "instr_lang": m.group("instr"),
                **stats,
            }
        )
    return rows_out


def write_tsv(rows: list[dict], out_path: Path) -> None:
    columns = [
        "condition",
        "seed",
        "template",
        "data_lang",
        "instr_lang",
        "n",
        "accuracy",
        "top_pred",
        "top_pred_count",
        "top_pred_share",
        "top_raw",
        "top_raw_count",
        "top_raw_share",
        "correct_count",
        "correct_via_constant_pred_count",
        "correct_via_constant_pred_pct",
        "correct_via_constant_raw_count",
        "correct_via_constant_raw_pct",
    ]
    rows = sorted(
        rows,
        key=lambda r: (r["condition"], r["seed"], r["template"], r["data_lang"], r["instr_lang"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    f"{r[c]:.4f}"
                    if isinstance(r[c], float)
                    else str(r[c])
                    for c in columns
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Restrict to these condition dir names (e.g. condition-2-ur-5k). "
        "Default: all Phase-3 conditions.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/phase3_correct_via_constant/correct-via-constant-rates.tsv",
        help=(
            "TSV output path. Default: /tmp/phase3_correct_via_constant/. The "
            "TSV is published to HF at "
            "phase3/analysis/refined-tables/correct-via-constant-rates.tsv "
            "(HF is canonical); re-run this script and upload via "
            "upload_analysis_tables.py to refresh."
        ),
    )
    args = parser.parse_args()

    api = HfApi()
    cond_filter = set(args.conditions) if args.conditions else None
    files = list_phase3_results_files(api, cond_filter)
    if not files:
        print("No Phase-3 results files matched the filter.", file=sys.stderr)
        return 1

    print(f"Processing {len(files)} _results_template*.json files from {REPO_ID}...")
    all_rows: list[dict] = []
    for i, remote_path in enumerate(files, 1):
        condition, seed = parse_filename_metadata(remote_path)
        try:
            stats_rows = process_file(remote_path, condition, seed)
        except Exception as exc:
            print(f"  [{i}/{len(files)}] {remote_path}: SKIPPED ({exc})", file=sys.stderr)
            continue
        print(f"  [{i}/{len(files)}] {condition}/{seed} → {len(stats_rows)} sib200 cells")
        all_rows.extend(stats_rows)

    if not all_rows:
        print("No SIB-200 cells found.", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    write_tsv(all_rows, out_path)
    print(f"\nWrote {len(all_rows)} cell rows → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

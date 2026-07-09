"""Per-gold-label XNLI prediction distributions for Phase-3 (CORE-1383).

Verification item: the draft claims Tiny Aya "rarely predicts the neutral
XNLI label" and the refined-decision ledger records that every CJK-glued
Tier-2 label form observed on the baseline is `entailment`. Neither claim
has row-level backing in the published refined-tables, which only carry
per-cell accuracies. This script recomputes the full predicted-vs-gold
structure from the per-row `_results_template*.json` files on HF, using the
same refined extractor (`reparse_results.extract_xnli_label`) that produced
the paper's numbers.

Output grain: one TSV row per
(condition, seed, template, data_lang, instr_lang, gold_label) — the full
confusion matrix, no aggregation. Rollups (label distributions split by
instruction language, bias-vs-learning comparisons) are computed downstream
from this TSV so that no threshold-count can inherit the seed-vs-cell
inflation bug class documented in `analysis/phase-3/aggregation-bug-audit.md`.

Per row it also records HOW each resolved prediction was matched, using a
tier-tagged twin of `extract_xnli_label` (asserted equal to the untagged
original on every row):

    via_tier1a_english   : verbatim English label, word-boundary match
    via_tier1b_native    : native zh/es/ur label word
    via_tier2_glued      : English label embedded without a word boundary
                           (predominantly CJK sentence frames; see ledger)
    via_tier3_paraphrase : native-prose paraphrase pattern

plus, for the Tier-2 rows specifically:

    tier2_pred_entailment : how many Tier-2 rows resolved to `entailment`
                            (the ledger's all-entailment observation)
    tier2_cjk_frame       : how many Tier-2 first lines contain a CJK char

Cross-check: per-cell accuracy recomputed here is asserted against the
published `_summary_reparsed_template*.json` for the same cell; any
mismatch is printed loudly (it would mean local extractor drift vs the
sha-pinned version that produced the published numbers).

Usage:
    # Default — key CORE-1383 conditions, TSV into analysis/phase-3/:
    python build_xnli_label_bias.py

    # Custom condition set / output path:
    python build_xnli_label_bias.py --conditions baseline condition-5-zh-5k \
        --output /tmp/xnli-label-distributions.tsv

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
    NATIVE_LABEL_MAP,
    XNLI_LABEL_RES,
    XNLI_LABELS,
    XNLI_PARAPHRASE_RES,
    XNLI_TIER2_NEGATION,
    extract_xnli_label,
    reparsed_summary_path_remote,
)

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"
PHASE3_PREFIX = "phase3/conditions"

# The CORE-1383 verification set: baseline + the anchor condition + the
# cond-2 / cond-5 5k conditions for the three non-English languages.
DEFAULT_CONDITIONS = (
    "baseline",
    "condition-1-en-5k",
    "condition-2-ur-5k",
    "condition-2-es-5k",
    "condition-2-zh-5k",
    "condition-5-ur-5k",
    "condition-5-es-5k",
    "condition-5-zh-5k",
)

CELL_KEY_RE = re.compile(
    r"^template(?P<template>\d+)_xnli_data=(?P<data>[a-z]+)_instr=(?P<instr>[a-z]+)$"
)

_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿]")

TIER_NAMES = ("tier1a_english", "tier1b_native", "tier2_glued", "tier3_paraphrase")


def extract_xnli_label_tagged(text: str) -> tuple[str | None, str | None]:
    """Twin of `reparse_results.extract_xnli_label` that also returns which
    tier matched. Logic must stay line-for-line equivalent; the caller
    asserts agreement with the untagged original on every row."""
    first_line = text.strip().split("\n")[0].strip()
    first_line_lower = first_line.lower()

    for label, label_re in XNLI_LABEL_RES.items():
        if label_re.search(first_line_lower):
            return label, "tier1a_english"
    for native, english in NATIVE_LABEL_MAP.items():
        if native.lower() in first_line_lower:
            return english, "tier1b_native"
    if not any(neg in first_line for neg in XNLI_TIER2_NEGATION):
        for label in XNLI_LABELS:
            if label in first_line_lower:
                return label, "tier2_glued"
    for pat, label in XNLI_PARAPHRASE_RES:
        if pat.search(first_line):
            return label, "tier3_paraphrase"
    return None, None


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
    """One results file → per-(cell, gold_label) confusion rows.
    Returns (rows, n_accuracy_mismatches_vs_published_reparsed_summary)."""
    local = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path)
    with Path(local).open(encoding="utf-8") as f:
        data = json.load(f)
    published = load_reparsed_summary(remote_path)["summary"]

    rows_out: list[dict] = []
    mismatches = 0
    for key, items in data.items():
        m = CELL_KEY_RE.match(key)
        if not m or not isinstance(items, list):
            continue
        template = int(m.group("template"))

        # gold → Counter over predicted label / None, tier bookkeeping
        pred_by_gold: dict[str, Counter] = {g: Counter() for g in XNLI_LABELS}
        tier_by_gold: dict[str, Counter] = {g: Counter() for g in XNLI_LABELS}
        tier2_entail_by_gold: Counter = Counter()
        tier2_cjk_by_gold: Counter = Counter()
        cell_correct = 0
        for row in items:
            gold = row["gold"]
            pred, tier = extract_xnli_label_tagged(row["raw_output"])
            assert pred == extract_xnli_label(row["raw_output"]), (
                f"tagged extractor diverged from reparse_results on {key}"
            )
            pred_by_gold[gold][pred] += 1
            if tier is not None:
                tier_by_gold[gold][tier] += 1
            if tier == "tier2_glued":
                if pred == "entailment":
                    tier2_entail_by_gold[gold] += 1
                first_line = row["raw_output"].strip().split("\n")[0]
                if _CJK_RE.search(first_line):
                    tier2_cjk_by_gold[gold] += 1
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

        for gold in XNLI_LABELS:
            c = pred_by_gold[gold]
            t = tier_by_gold[gold]
            n_gold = sum(c.values())
            rows_out.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "template": template,
                    "benchmark": "xnli",
                    "data_lang": m.group("data"),
                    "instr_lang": m.group("instr"),
                    "extractor": "refined",
                    "gold": gold,
                    "n_gold": n_gold,
                    "pred_entailment": c["entailment"],
                    "pred_neutral": c["neutral"],
                    "pred_contradiction": c["contradiction"],
                    "pred_none": c[None],
                    "acc_on_gold": c[gold] / n_gold if n_gold else 0.0,
                    "via_tier1a_english": t["tier1a_english"],
                    "via_tier1b_native": t["tier1b_native"],
                    "via_tier2_glued": t["tier2_glued"],
                    "via_tier3_paraphrase": t["tier3_paraphrase"],
                    "tier2_pred_entailment": tier2_entail_by_gold[gold],
                    "tier2_cjk_frame": tier2_cjk_by_gold[gold],
                }
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
    "gold",
    "n_gold",
    "pred_entailment",
    "pred_neutral",
    "pred_contradiction",
    "pred_none",
    "acc_on_gold",
    "via_tier1a_english",
    "via_tier1b_native",
    "via_tier2_glued",
    "via_tier3_paraphrase",
    "tier2_pred_entailment",
    "tier2_cjk_frame",
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
            XNLI_LABELS.index(r["gold"]),
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
        "(baseline, cond-1-en-5k, cond-2-{ur,es,zh}-5k, cond-5-{ur,es,zh}-5k).",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[2]
            / "analysis"
            / "phase-3"
            / "xnli-label-distributions.tsv"
        ),
        help="TSV output path. Default: analysis/phase-3/xnli-label-distributions.tsv.",
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

"""Compare a smoke re-evaluation of a Phase-3 session with its original results.

The Phase-3 sessions were evaluated in one software environment; later
sessions in another. To measure whether the environment alone moves the
numbers, one original adapter is re-evaluated on a sample of rows in the
later environment (``evaluate.ipynb`` with ``N_SAMPLES`` set, which writes
under ``phase3/smoke/``), and this script compares that smoke run with the
original ``_results_*.json`` files for the same session, row by row.

Rows are aligned on their inputs (every field except the model outputs:
``raw_output``, ``pred``, ``correct``), never on position, so a smoke run
that evaluated a subset or a reordering of the rows still compares the same
prompts. For every cell the report gives the number of aligned rows, the
share of rows whose raw model output is byte-identical, the share whose
inference-time prediction agrees, and the accuracy of both runs on the
aligned rows under the inference-time extractor (the ``correct`` field) and
under the refined extractor (``reparse_results``).

Usage (from the repository root):

    # Local smoke files; the reference session is read from the file names
    # (``<condition>_seed<seed>_...``) and downloaded from Hugging Face main:
    python expedition-tiny-aya/evaluation/scripts/compare_stack_drift.py \\
        --smoke path/to/condition-2-es-20k_seed42_smoke50_results_template1.json \\
                path/to/condition-2-es-20k_seed42_smoke50_results_template2.json \\
        --out drift.md

    # Smoke files on Hugging Face (a folder under phase3/smoke/). The eval
    # notebook uploads them as a pull request, so pass --revision refs/pr/N
    # until that PR is merged:
    python expedition-tiny-aya/evaluation/scripts/compare_stack_drift.py \\
        --smoke-hf phase3/smoke/condition-2-es-20k/seed42 --revision refs/pr/59 --out drift.md

The verdict says "no drift" only when every aligned raw output is
byte-identical and the inference-time accuracy is unchanged on every cell;
anything else is reported as drift with the counts behind it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_seed_results import (  # noqa: E402
    EXPERIMENTS_REPO as REPO_ID,
    MODEL_OUTPUT_KEYS,
    load_results,
    session_paths,
)
from reparse_results import _load_extractors, benchmark_from_key  # noqa: E402

REPO_TYPE = "dataset"
SESSION_RE = re.compile(r"^(?P<cond>.+?)_seed(?P<seed>[A-Za-z0-9]+)_.*results_(?P<template>template\d+)\.json$")
REFERENCE_RE = re.compile(r"^(?P<cond>[^/]+)/seed(?P<seed>[A-Za-z0-9]+)$")


def input_key(row: dict) -> tuple:
    """Everything the model was given, as a hashable key. Unlike
    ``refresh_seed_results._inputs`` this keeps ``gold`` in the key on
    purpose: XNLI has premise and hypothesis pairs that occur twice with
    different labels, and a row with a changed label must never align with
    its original."""
    return tuple(sorted((k, json.dumps(v, sort_keys=True, ensure_ascii=False)) for k, v in row.items() if k not in MODEL_OUTPUT_KEYS))


def align(smoke_rows: list[dict], ref_rows: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    """Pair each smoke row with the reference row that has the same inputs.
    Duplicate inputs are consumed in order. Returns (pairs, unmatched)."""
    pool: dict[tuple, list[dict]] = {}
    for r in ref_rows:
        pool.setdefault(input_key(r), []).append(r)
    pairs, unmatched = [], 0
    for s in smoke_rows:
        bucket = pool.get(input_key(s))
        if bucket:
            pairs.append((s, bucket.pop(0)))
        else:
            unmatched += 1
    return pairs, unmatched


def compare_cell(cell: str, smoke_rows: list[dict], ref_rows: list[dict], extractor) -> dict:
    pairs, unmatched = align(smoke_rows, ref_rows)
    n = len(pairs)
    same_raw = sum(1 for s, r in pairs if s.get("raw_output") == r.get("raw_output"))
    same_pred = sum(1 for s, r in pairs if s.get("pred") == r.get("pred"))
    orig_smoke = sum(1 for s, _ in pairs if s.get("correct"))
    orig_ref = sum(1 for _, r in pairs if r.get("correct"))
    ref_smoke = sum(1 for s, _ in pairs if extractor(s["raw_output"]) == s["gold"])
    ref_ref = sum(1 for _, r in pairs if extractor(r["raw_output"]) == r["gold"])

    def pct(x: int) -> float:
        return 100.0 * x / n if n else float("nan")

    return {
        "cell": cell,
        "n": n,
        "unmatched": unmatched,
        "raw_match_rows": same_raw,
        "raw_match_pct": pct(same_raw),
        "pred_match_pct": pct(same_pred),
        "orig_acc_smoke": pct(orig_smoke),
        "orig_acc_ref": pct(orig_ref),
        "refined_acc_smoke": pct(ref_smoke),
        "refined_acc_ref": pct(ref_ref),
    }


def compare_files(smoke: dict, ref: dict, extractors: dict | None = None) -> tuple[list[dict], list[str]]:
    """Compare every cell in ``smoke`` with the same cell in ``ref`` (one
    template's results dicts). Returns (per-cell results, problems)."""
    extractors = extractors or _load_extractors()
    results, problems = [], []
    for cell, rows in smoke.items():
        if cell in {"summary", "parse_failure_rates"} or not isinstance(rows, list):
            continue
        if cell not in ref:
            problems.append(f"{cell}: not in the reference session")
            continue
        res = compare_cell(cell, rows, ref[cell], extractors[benchmark_from_key(cell)])
        if res["unmatched"]:
            problems.append(f"{cell}: {res['unmatched']} smoke rows have no reference row with the same inputs")
            if rows and ref[cell]:
                smoke_fields, ref_fields = set(rows[0]), set(ref[cell][0])
                if smoke_fields != ref_fields:
                    problems.append(
                        f"{cell}: row fields differ; only in smoke: {sorted(smoke_fields - ref_fields) or 'none'}; "
                        f"only in reference: {sorted(ref_fields - smoke_fields) or 'none'}"
                    )
        if res["n"] == 0:
            problems.append(f"{cell}: no rows aligned")
        results.append(res)
    return results, problems


def summarize(results: list[dict]) -> dict:
    aligned = [r for r in results if r["n"]]
    n = sum(r["n"] for r in aligned)
    raw_rows = sum(r["raw_match_rows"] for r in aligned)
    deltas_ref = [r["refined_acc_smoke"] - r["refined_acc_ref"] for r in aligned]
    deltas_orig = [r["orig_acc_smoke"] - r["orig_acc_ref"] for r in aligned]
    return {
        "cells": len(aligned),
        "cells_total": len(results),
        "rows": n,
        "raw_match_rows": raw_rows,
        "raw_match_pct": 100.0 * raw_rows / n if n else float("nan"),
        "cells_refined_changed": sum(1 for d in deltas_ref if abs(d) > 1e-9),
        "max_abs_delta_refined_pp": max((abs(d) for d in deltas_ref), default=0.0),
        "mean_delta_refined_pp": sum(deltas_ref) / len(deltas_ref) if deltas_ref else 0.0,
        "max_abs_delta_orig_pp": max((abs(d) for d in deltas_orig), default=0.0),
    }


def verdict(summary: dict) -> str:
    if summary["rows"] == 0:
        return "no rows aligned; nothing can be concluded"
    differing = summary["rows"] - summary["raw_match_rows"]
    if differing == 0 and summary["max_abs_delta_orig_pp"] == 0:
        return (
            "no drift: every aligned raw output is byte-identical and the inference-time accuracy "
            "is unchanged on every cell, so the environment change does not move the refined numbers"
        )
    if differing == 0:
        return (
            "raw outputs identical but the inference-time extractor moved: largest inference-time "
            f"accuracy change {summary['max_abs_delta_orig_pp']:.2f} pp; refined numbers are unaffected"
        )
    return (
        f"drift: {differing} of {summary['rows']} aligned raw outputs differ; "
        f"refined accuracy moves in {summary['cells_refined_changed']} of {summary['cells']} aligned cells, "
        f"largest change {summary['max_abs_delta_refined_pp']:.2f} pp"
    )


def render_markdown(
    per_template: dict[str, tuple[list[dict], list[str]]], session: str, sources: dict[str, str] | None = None
) -> str:
    all_results = [r for res, _ in per_template.values() for r in res]
    s = summarize(all_results)
    lines = [
        f"# Stack drift check: {session}",
        "",
        f"Verdict: {verdict(s)}.",
        "",
    ]
    if sources:
        lines += ["Smoke files: " + "; ".join(f"{t}: {name}" for t, name in sorted(sources.items())), ""]
    lines += [
        f"Aligned rows: {s['rows']} across {s['cells']} aligned cells ({s['cells_total']} compared). "
        f"Raw outputs byte-identical: {s['raw_match_rows']} of {s['rows']} ({s['raw_match_pct']:.2f}%). "
        f"Refined accuracy differs in {s['cells_refined_changed']} cells (largest change {s['max_abs_delta_refined_pp']:.2f} pp, "
        f"mean {s['mean_delta_refined_pp']:+.2f} pp); inference-time accuracy largest change {s['max_abs_delta_orig_pp']:.2f} pp.",
        "",
        "| Cell | Aligned | Unmatched | Raw match % | Pred match % | Orig acc smoke | Orig acc ref | Refined acc smoke | Refined acc ref |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for template in sorted(per_template):
        res, problems = per_template[template]
        for r in sorted(res, key=lambda r: r["cell"]):
            lines.append(
                f"| {r['cell']} | {r['n']} | {r['unmatched']} | {r['raw_match_pct']:.1f} | {r['pred_match_pct']:.1f} | "
                f"{r['orig_acc_smoke']:.1f} | {r['orig_acc_ref']:.1f} | {r['refined_acc_smoke']:.1f} | {r['refined_acc_ref']:.1f} |"
            )
    problems = [p for _, probs in per_template.values() for p in probs]
    if problems:
        lines += ["", "Problems:", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _reference_path(cond: str, seed: str, template: str) -> str:
    """The original results file of the session, as refresh_seed_results lays it out."""
    return next(p for p in session_paths(cond, seed)["results"] if p.endswith(f"_results_{template}.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", nargs="*", type=Path, default=[], help="Local smoke results files (one per template).")
    parser.add_argument("--smoke-hf", default=None, help="Folder on the experiments dataset holding the smoke results, e.g. phase3/smoke/condition-2-es-20k/seed42.")
    parser.add_argument("--reference", default=None, help="Override the reference session as condition/seedN (default: from the smoke file names).")
    parser.add_argument("--revision", default=None, help="Dataset revision for --smoke-hf, e.g. refs/pr/59 while the smoke upload PR is open (default: main).")
    parser.add_argument("--out", type=Path, default=None, help="Write the Markdown report here (default: print only).")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    reference = None
    if args.reference:
        m = REFERENCE_RE.match(args.reference.strip("/"))
        if not m:
            raise SystemExit(f"--reference must look like condition-2-es-20k/seed42, got {args.reference!r}")
        reference = (m["cond"], m["seed"])

    smoke_paths: list[Path] = list(args.smoke)
    if args.smoke_hf:
        prefix = args.smoke_hf.rstrip("/") + "/"
        files = [
            f
            for f in HfApi().list_repo_files(REPO_ID, repo_type=REPO_TYPE, revision=args.revision)
            if f.startswith(prefix) and "_results_" in f
        ]
        if not files:
            raise SystemExit(f"no _results_ files under {args.smoke_hf} on {REPO_ID} at {args.revision or 'main'}")
        smoke_paths += [Path(hf_hub_download(REPO_ID, f, repo_type=REPO_TYPE, revision=args.revision)) for f in sorted(files)]
    if not smoke_paths:
        raise SystemExit("give --smoke files or --smoke-hf folder")

    extractors = _load_extractors()
    per_template: dict[str, tuple[list[dict], list[str]]] = {}
    sources: dict[str, str] = {}
    session = None
    for p in smoke_paths:
        m = SESSION_RE.match(p.name)
        if not m:
            raise SystemExit(f"cannot read condition, seed and template from {p.name}")
        cond, seed, template = m["cond"], m["seed"], m["template"]
        if reference:
            cond, seed = reference
        if template in per_template:
            raise SystemExit(f"two smoke files for {template}: {sources[template]} and {p.name}; pass one set of smoke files")
        session = session or f"{cond}/seed{seed}"
        ref = load_results(Path(hf_hub_download(REPO_ID, _reference_path(cond, seed, template), repo_type=REPO_TYPE)))
        per_template[template] = compare_files(load_results(p), ref, extractors)
        sources[template] = p.name
    report = render_markdown(per_template, session or "?", sources)
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"[write] {args.out}")


if __name__ == "__main__":
    main()

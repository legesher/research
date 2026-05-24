"""Upload Phase-3 analysis tables to `legesher/language-decoded-experiments`.

Companion to `upload_reparsed_summaries.py`. Where that script uploads
per-session refined summaries, this one uploads CROSS-SESSION
AGGREGATIONS — the TSVs produced by `build_comparison.py`,
`build_vs_baseline.py`, `build_framework_comparison.py`, and the
baseline-forms TSVs produced by `inspect_failures.py --aggregate`.

The HF layout this populates:

    phase3/analysis/
        refined-tables/                                  ← the refined extractor's view
            cells.tsv                                    (from build_comparison)
            summary_by_{benchmark,instr_lang,...}.tsv    (from build_comparison)
            summary_{bench_x_instr,cond_x_bench}.tsv     (from build_comparison)
            overall_stats.json                           (from build_comparison)
            vs_baseline_cells.tsv                        (from build_vs_baseline)
            vs_baseline_by_{condition,...}.tsv           (from build_vs_baseline)
            conclusion_flips.tsv                         (from build_vs_baseline)
            framework_{template,same_language,...}.tsv   (from build_framework_comparison)
        surface-form-tables/                             ← baseline-model output inventories
            {sib200,xnli,belebele,csqa}-baseline-forms.tsv  (from inspect_failures --aggregate)

Why this lives on HF, not in the GitHub repo:

The TSV contents are fully determined by `HF dataset main state × the
extractor sha in reparse_results.py × the build script shas`. Committing
them to GitHub creates a sync burden (every HF update should trigger a
regenerate-and-commit cycle, or the GitHub TSVs go stale silently). HF
already tracks dataset history, so the TSVs' provenance is automatic
when they live there.

Usage:

    # Plan-only — list what would be uploaded without doing it:
    python upload_analysis_tables.py --dry-run

    # Default — opens a discussion PR (never commits direct to main):
    python upload_analysis_tables.py

    # Override the input directory (defaults to /tmp/phase3_analysis_output):
    PHASE3_ANALYSIS_TABLES_DIR=/some/local/path python upload_analysis_tables.py

Auth:
    Reads from HF auth cache (huggingface-cli login). Token must have WRITE
    scope on the `legesher/language-decoded-experiments` dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"

# Where the build scripts write by default. Override via env var.
INPUT_DIR = Path(os.environ.get("PHASE3_ANALYSIS_TABLES_DIR", "/tmp/phase3_analysis_output"))

# Target directory on the HF dataset.
HF_ANALYSIS_ROOT = "phase3/analysis"

# Filename -> HF subdirectory. Filenames not matched are placed under
# refined-tables/ by default.
SURFACE_FORM_PREFIXES = ("sib200-", "xnli-", "belebele-", "csqa-")


def classify_file(name: str) -> str:
    """Return the HF subdirectory this file should land in.

    `refined-tables/` holds the cross-session aggregates produced by the
    three build scripts (they encode the refined extractor's view of the
    experiments). `surface-form-tables/` holds the per-benchmark baseline-
    model output inventories produced by inspect_failures --aggregate.
    """
    if any(name.startswith(p) and "baseline-forms" in name for p in SURFACE_FORM_PREFIXES):
        return "surface-form-tables"
    return "refined-tables"


def discover_files(input_dir: Path) -> list[tuple[Path, str]]:
    """Yield (local_path, target_subdir) for every TSV/JSON in input_dir."""
    pairs: list[tuple[Path, str]] = []
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix not in (".tsv", ".json"):
            continue
        pairs.append((f, classify_file(f.name)))
    return pairs


def print_plan(pairs: list[tuple[Path, str]]) -> None:
    print(f"Source: {INPUT_DIR}")
    print(f"Target: {REPO_ID}:{HF_ANALYSIS_ROOT}/")
    print()
    by_subdir: dict[str, list[Path]] = {}
    for p, sub in pairs:
        by_subdir.setdefault(sub, []).append(p)
    for sub in sorted(by_subdir):
        print(f"  {sub}/ ({len(by_subdir[sub])} files):")
        for p in by_subdir[sub]:
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:<55} {size_kb:>8.1f} KB")
    print()
    print(f"Total: {len(pairs)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit. No network calls beyond auth check.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="add phase3/analysis/ — TSVs from build_*.py + inspect_failures --aggregate",
    )
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit(
            f"Input directory not found: {INPUT_DIR}\n"
            f"Generate TSVs first via:\n"
            f"  PHASE3_OUT_DIR={INPUT_DIR} python build_comparison.py\n"
            f"  PHASE3_OUT_DIR={INPUT_DIR} python build_vs_baseline.py\n"
            f"  PHASE3_OUT_DIR={INPUT_DIR} python build_framework_comparison.py\n"
            f"  inspect_failures.py ... --aggregate --output {INPUT_DIR}/<bench>-baseline-forms.tsv"
        )

    pairs = discover_files(INPUT_DIR)
    if not pairs:
        raise SystemExit(f"No .tsv/.json files in {INPUT_DIR}")

    print_plan(pairs)

    if args.dry_run:
        print()
        print("--dry-run set; not uploading. Exiting.")
        return

    # Heavy imports deferred until after dry-run check.
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{HF_ANALYSIS_ROOT}/{sub}/{p.name}",
            path_or_fileobj=str(p),
        )
        for p, sub in pairs
    ]

    print()
    print(f"Creating HF discussion PR with {len(operations)} files...")
    commit_info = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=args.commit_message,
        commit_description=(
            "Adds cross-session analysis TSVs under `phase3/analysis/`.\n\n"
            "Layout:\n"
            "- `refined-tables/` — the refined extractor's view of every cell:\n"
            "  raw cells.tsv, single-axis + crosstab rollups, condition-vs-\n"
            "  baseline rollups, conclusion-flip catalogue, and the comparison-\n"
            "  framework TSVs (template, same-language, data-volume, cross-\n"
            "  language impact, coding-constant decomposition, seed variance,\n"
            "  template robustness, per-benchmark breakdown, parse-failure\n"
            "  recovery).\n"
            "- `surface-form-tables/` — per-benchmark baseline-model output\n"
            "  inventories generated by `inspect_failures.py --aggregate`\n"
            "  against the baseline model's raw outputs.\n\n"
            "**Why on HF, not in the GitHub repo:** these TSVs are derived\n"
            "from the refined extractor (`reparse_results.py` on legesher/\n"
            "research main) applied to the HF-canonical raw outputs. Their\n"
            "contents are fully determined by `HF dataset main state × the\n"
            "extractor sha × the build script shas`. Hosting them on HF keeps\n"
            "source-of-truth in one place and avoids the sync burden of\n"
            "committing generated artefacts to git.\n\n"
            "Generated by `expedition-tiny-aya/evaluation/scripts/build_*.py`\n"
            "and `inspect_failures.py --aggregate` in the research repo."
        ),
        create_pr=True,
    )

    print()
    print(f"✓ HF PR created: {commit_info.pr_url}")
    print(f"  PR number:    {commit_info.pr_num}")
    print(f"  Revision:     {commit_info.pr_revision}")
    print()
    print("Review on HF; merge when satisfied.")


if __name__ == "__main__":
    main()

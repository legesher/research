#!/usr/bin/env python3
"""Build a durable per-idx ledger for the Cond-5 datasets (ur, zh, es).

Why this exists
---------------
The cond5 translation runs record their per-file outcome in each cell's
``summary.json``, but that file is the only authoritative record of *what
was attempted* — and it lives in the git-ignored ``packaged/`` tree, is
overwritten on every ``--resume`` run, and is not on the Hugging Face Hub
(HF holds only the published successes).

This script flattens that state into one small CSV that IS safe to commit
to git: a machine-independent provenance record of exactly which source
idx was attempted, skipped, succeeded, or failed for each language and
split. It is also the data behind the paper's "attempted N source files,
M produced valid translations" framing.

Source of truth — the on-disk files, not summary.json
------------------------------------------------------
``summary.json`` is unreliable across ``--resume`` boundaries: ur was run
with an older populator that does not carry prior successes forward as
``resumed`` rows, so its ``per_file`` undercounts valid translations. The
on-disk artifacts ARE cumulative and resume-safe:

* ``{idx}.py``                       -> a translation succeeded (VALID)
* ``{idx}.error.txt`` and no ``.py`` -> attempted, failed (InvalidCodeError)
* neither                            -> never attempted (outside the
                                        ``--idx-allowlist`` for zh / es)

So valid / failed / not-attempted are derived from the filesystem. The
``summary.json`` per_file entries are used only to *enrich* rows with
``ast`` / ``elapsed_seconds`` / ``input_chars`` / ``output_chars`` where
available (best-effort; blank when the summary lacks them).

Outputs
-------
``analysis/cond5-idx-ledger.csv`` — one row per (lang, split, idx)
``analysis/cond5-idx-ledger.md``  — column docs + per-(lang,split) summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# analysis/scripts/ -> analysis/ -> expedition-tiny-aya/
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
DATA_PIPELINE = ANALYSIS_DIR.parent / "data-pipeline"
PACKAGED = DATA_PIPELINE / "packaged"
SOURCE_PYTHON = DATA_PIPELINE / "source-python"

LANGS = ("ur", "zh", "es")
SPLITS = ("train", "validation")
MODEL_SLUG = "c4ai-aya-expanse-32b"

FIELDS = [
    "lang",
    "split",
    "idx",
    "status",
    "attempted",
    "valid",
    "ast",
    "elapsed_seconds",
    "input_chars",
    "output_chars",
    "file_path",
]


def _cell_dir(lang: str, split: str) -> Path:
    return PACKAGED / f"condition-5-{lang}-5k-{MODEL_SLUG}" / split / lang


def _idx_set(cell: Path, suffix: str) -> set[int]:
    """Idx values of files matching ``*<suffix>`` in ``cell``."""
    out: set[int] = set()
    if not cell.is_dir():
        return out
    for f in cell.glob(f"*{suffix}"):
        stem = f.name[: -len(suffix)]
        try:
            out.add(int(stem))
        except ValueError:
            continue
    return out


def _source_idx_universe(split: str) -> list[int]:
    """All source idx values for a split, from source-python/<split>/manifest.csv."""
    manifest = SOURCE_PYTHON / split / "manifest.csv"
    idxs: list[int] = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                idxs.append(int(row["idx"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(idxs)


def _summary_meta(lang: str, split: str) -> dict[int, dict[str, object]]:
    """Best-effort {idx: {ast, elapsed_seconds, input_chars, output_chars}}."""
    path = PACKAGED / f"condition-5-{lang}-5k-{MODEL_SLUG}" / split / "summary.json"
    meta: dict[int, dict[str, object]] = {}
    if not path.exists():
        return meta
    data = json.loads(path.read_text(encoding="utf-8"))
    per_file = data.get("by_language", {}).get(lang, {}).get("per_file", [])
    for entry in per_file:
        try:
            idx = int(entry["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        meta[idx] = {
            "ast": entry.get("ast", ""),
            "elapsed_seconds": entry.get("elapsed_seconds", ""),
            "input_chars": entry.get("input_chars", ""),
            "output_chars": entry.get("output_chars", ""),
            "file_path": entry.get("file_path", ""),
        }
    return meta


def main() -> int:
    rows: list[dict[str, object]] = []
    summary: dict[tuple[str, str], dict[str, int]] = {}

    for lang in LANGS:
        for split in SPLITS:
            cell = _cell_dir(lang, split)
            universe = _source_idx_universe(split)
            valid_idx = _idx_set(cell, ".py")
            err_idx = _idx_set(cell, ".error.txt")
            failed_idx = err_idx - valid_idx  # failed and never recovered
            meta = _summary_meta(lang, split)

            counts = {"total": len(universe), "valid": 0, "failed": 0, "skipped": 0}
            for idx in universe:
                if idx in valid_idx:
                    status, attempted, valid = "valid", True, True
                    counts["valid"] += 1
                elif idx in failed_idx:
                    status, attempted, valid = "failed", True, False
                    counts["failed"] += 1
                else:
                    status, attempted, valid = "not_attempted", False, False
                    counts["skipped"] += 1
                m = meta.get(idx, {})
                rows.append(
                    {
                        "lang": lang,
                        "split": split,
                        "idx": idx,
                        "status": status,
                        "attempted": attempted,
                        "valid": valid,
                        "ast": m.get("ast", ""),
                        "elapsed_seconds": m.get("elapsed_seconds", ""),
                        "input_chars": m.get("input_chars", ""),
                        "output_chars": m.get("output_chars", ""),
                        "file_path": m.get("file_path", ""),
                    }
                )
            counts["attempted"] = counts["valid"] + counts["failed"]
            summary[(lang, split)] = counts
            print(f"{lang}/{split}: {counts}")

    rows.sort(key=lambda r: (r["lang"], r["split"], r["idx"]))

    csv_path = ANALYSIS_DIR / "cond5-idx-ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    lines = [
        "# Cond-5 idx ledger",
        "",
        "Per-source-row provenance for the Cond-5 datasets (`c4ai-aya-expanse-32b`).",
        "Generated by `analysis/scripts/build_cond5_idx_ledger.py`. Companion data:",
        "`cond5-idx-ledger.csv`.",
        "",
        "Status is derived from the on-disk artifacts (resume-safe), NOT from",
        "`summary.json` — see the script docstring for why.",
        "",
        "## Per-(language, split) coverage",
        "",
        "| Language | Split | Source rows | Attempted | Valid | Failed | Not attempted |",
        "| -------- | ----- | ----------- | --------- | ----- | ------ | ------------- |",
    ]
    for lang in LANGS:
        for split in SPLITS:
            c = summary.get((lang, split))
            if not c:
                continue
            lines.append(
                f"| {lang} | {split} | {c['total']} | {c['attempted']} | "
                f"{c['valid']} | {c['failed']} | {c['skipped']} |"
            )
    lines += [
        "",
        "- **Attempted** = an LLM translation was run (`.py` or `.error.txt` on disk).",
        "- **Valid** = produced a translated `.py`.",
        "- **Failed** = attempted but raised `InvalidCodeError` (`.error.txt`, no `.py`).",
        "- **Not attempted** = neither artifact; the idx was outside the",
        "  `--idx-allowlist` (zh/es were constrained to ur-succeeded idxs).",
        "",
        "`ur` ran the full 5,000 source rows with no allowlist (zero not-attempted).",
        "`zh` and `es` were constrained to ur's succeeded idxs — their *not-attempted*",
        "counts are the un-run remainder, recoverable by running those idxs later.",
        "",
        "## Columns (`cond5-idx-ledger.csv`)",
        "",
        "`lang`, `split`, `idx`, `status` (valid/failed/not_attempted), `attempted`,",
        "`valid`, `ast` (pass/fail/skipped, best-effort from summary.json — blank when",
        "unavailable, e.g. ur resume rows), `elapsed_seconds`, `input_chars`,",
        "`output_chars`, `file_path`.",
    ]
    md_path = ANALYSIS_DIR / "cond5-idx-ledger.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

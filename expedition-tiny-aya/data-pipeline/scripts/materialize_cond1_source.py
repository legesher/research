#!/usr/bin/env python3
"""Materialize ``condition-1-en-5k`` parquet rows as on-disk Python files.

Writes deterministic-named ``.py`` files plus a ``manifest.csv`` (with
``idx, filename, file_path, license`` — same schema as
``populate_cond5_datasets``'s ``metadata.csv``) so downstream pipelines
(``batch_transpile.py`` for cond2, future condition populators) can
preserve provenance and produce HF datasets that share the same source
file membership across conditions and languages.

Default source: ``hf://datasets/legesher/language-decoded-data/data/condition-1-en-5k``
Default output layout::

    <output>/train/{0000.py..4499.py, manifest.csv}
    <output>/validation/{0000.py..0499.py, manifest.csv}

Filename width is the cond5 default: ``{idx:03d}.py`` (3-digit minimum,
expands naturally for >999 files), so ``batch_transpile`` outputs and
cond5 outputs share the same on-disk basenames given the same source row.

Usage::

    python materialize_cond1_source.py --output ./source-python
    python materialize_cond1_source.py \\
        --source hf://datasets/legesher/language-decoded-data/data/condition-1-en-5k \\
        --output ./source-python \\
        --splits train,validation
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from datasets import load_dataset

DEFAULT_SOURCE_URI = (
    "hf://datasets/legesher/language-decoded-data/data/condition-1-en-5k"
)
DEFAULT_SPLITS = "train,validation"
MANIFEST_FIELDS = ["idx", "filename", "file_path", "license"]
# HF naming first, local-export naming second. The HF datasets library
# rejects globs across hf:// URIs, so try each pattern explicitly.
SPLIT_PARQUET_CANDIDATES = [
    "{split}-00000-of-00001.parquet",  # HF-pushed sharded layout
    "{split}.parquet",  # local export from to_parquet()
]


def _resolve_split_parquet(source_uri: str, split: str) -> str:
    """Return a working parquet URI for ``split`` under ``source_uri``.

    Tries each candidate naming scheme in order; raises if none load.
    """
    last_err: Exception | None = None
    for tmpl in SPLIT_PARQUET_CANDIDATES:
        candidate = f"{source_uri}/{tmpl.format(split=split)}"
        try:
            load_dataset("parquet", data_files=candidate, split="train", streaming=True)
            return candidate
        except (FileNotFoundError, OSError) as exc:
            last_err = exc
            continue
    raise FileNotFoundError(
        f"no parquet for split={split!r} under {source_uri} "
        f"(tried {[t.format(split=split) for t in SPLIT_PARQUET_CANDIDATES]})"
    ) from last_err


def _materialize_split(
    source_uri: str,
    split: str,
    output_dir: Path,
    filename_width: int,
) -> int:
    """Materialize one split. Returns row count written."""
    try:
        parquet_uri = _resolve_split_parquet(source_uri, split)
    except FileNotFoundError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 0

    print(f"  loading {split} from {parquet_uri}")
    ds = load_dataset("parquet", data_files=parquet_uri, split="train")

    out = output_dir / split
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for idx, row in enumerate(ds):
            filename = f"{idx:0{filename_width}d}.py"
            (out / filename).write_text(row["code"], encoding="utf-8")
            writer.writerow(
                {
                    "idx": idx,
                    "filename": filename,
                    "file_path": row["file_path"],
                    "license": row["license"],
                }
            )

    print(f"  {split}: wrote {len(ds)} files + manifest to {out}")
    return len(ds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize parquet rows as .py files + manifest.csv for "
            "downstream pipelines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_URI,
        help=(
            f"Source URI prefix (default: {DEFAULT_SOURCE_URI}). "
            "Resolved per split via {source}/{split}-00000-of-00001.parquet."
        ),
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for .py files + manifests"
    )
    parser.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help=f"Comma-separated splits to materialize (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--filename-width",
        type=int,
        default=3,
        help=(
            "Minimum width for {idx:0Nd}.py filename. Default 3 matches "
            "cond5's populate_cond5_datasets output convention."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print(f"Splits: {splits}")
    print(f"Filename width: {{idx:0{args.filename_width}d}}.py")

    total = 0
    for split in splits:
        total += _materialize_split(args.source, split, output_dir, args.filename_width)

    print(f"\nDone — {total} files materialized across {len(splits)} split(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

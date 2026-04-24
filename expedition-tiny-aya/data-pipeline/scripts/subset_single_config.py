#!/usr/bin/env python3
"""Create a deterministic subset from a single HuggingFace dataset config.

Takes one source config (e.g., ``condition-1-en-103k``), samples N records
deterministically via sorted file_paths plus seeded random.sample(), splits
90/10 train/validation, and pushes as a new config (e.g., ``condition-1-en-5k``).

The sampling is designed to be cross-condition-compatible: if you run the
same (seed, size) against a different config that shares the same file_path
set (e.g., keyword-transpiled variants of the same source pool), the
resulting subsets will contain identical file_paths. This preserves the
experimental property that ``condition-*-Nk`` configs pick the same files,
just translated differently.

Relationship to ``create_subset.py``:
``create_subset.py`` assumes multiple pre-existing condition configs on HF
and takes their file_path intersection before sampling. Use this script
when only one condition exists yet (Phase 3 bootstrap). Once conditions
2 / 3 / 4 exist, prefer ``create_subset.py`` for the cross-condition
intersection safety check.

Usage:
    # Manifest-only (streaming, no data download)
    python scripts/subset_single_config.py \\
        --source-dataset legesher/language-decoded-data \\
        --source-config condition-1-en-103k \\
        --target-config condition-1-en-5k \\
        --size 5000 \\
        --manifest-only

    # Full local run (download + filter + split + save locally)
    python scripts/subset_single_config.py \\
        --source-dataset legesher/language-decoded-data \\
        --source-config condition-1-en-103k \\
        --target-config condition-1-en-5k \\
        --size 5000 \\
        --output ./subset-5k/

    # Local + push to HF
    python scripts/subset_single_config.py \\
        --source-dataset legesher/language-decoded-data \\
        --source-config condition-1-en-103k \\
        --target-config condition-1-en-5k \\
        --size 5000 \\
        --output ./subset-5k/ \\
        --push legesher/language-decoded-data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIO = 0.1  # 10 % validation, matches create_subset.py / Phase-2
DEFAULT_SIZE = 5000


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_source(dataset_id: str, config: str) -> DatasetDict:
    """Download all splits of a single config from HF."""
    print(f"Loading {dataset_id} config={config} ...")
    ds = load_dataset(dataset_id, config)
    if not isinstance(ds, DatasetDict):
        raise RuntimeError(
            f"{dataset_id}:{config} did not return a DatasetDict (got {type(ds)})."
        )
    return ds


def extract_file_paths(ds: DatasetDict) -> list[str]:
    """Combine all splits and return every ``file_path`` value."""
    combined = concatenate_datasets([ds[split] for split in ds])
    return list(combined["file_path"])


def extract_file_paths_streaming(dataset_id: str, config: str) -> list[str]:
    """Stream the config and collect only ``file_path`` (avoids full download)."""
    ds = load_dataset(dataset_id, config, streaming=True)
    paths: list[str] = []
    for split_name in ds:
        for row in ds[split_name]:
            paths.append(row["file_path"])  # type: ignore[index]
    return paths


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------


def check_no_duplicates(paths: list[str], config: str) -> set[str]:
    """Assert the source config has no duplicate file_paths. Abort if any found."""
    path_set = set(paths)
    if len(paths) == len(path_set):
        print(f"  {config}: {len(path_set):,} unique file_paths")
        return path_set

    dupes: set[str] = set()
    seen: set[str] = set()
    for p in paths:
        if p in seen:
            dupes.add(p)
        seen.add(p)
    print(
        f"FATAL: {config} has {len(paths) - len(path_set)} duplicate file_paths",
        file=sys.stderr,
    )
    for d in sorted(dupes)[:10]:
        print(f"  {d}", file=sys.stderr)
    if len(dupes) > 10:
        print(f"  ... and {len(dupes) - 10} more", file=sys.stderr)
    sys.exit(1)


def select_subset(paths: set[str], size: int, seed: int) -> list[str]:
    """Deterministically pick ``size`` file_paths from ``paths``.

    Sorts alphabetically first so the selection is immune to HF row
    reordering, then samples via ``random.Random(seed).sample()``. The
    result is re-sorted alphabetically to give downstream code a stable
    iteration order.
    """
    sorted_paths = sorted(paths)

    if size > len(sorted_paths):
        print(
            f"WARNING: requested {size} but only {len(sorted_paths)} available. "
            f"Using all {len(sorted_paths)}.",
            file=sys.stderr,
        )
        return sorted_paths

    rng = random.Random(seed)
    selected = rng.sample(sorted_paths, size)
    selected.sort()
    return selected


def save_manifest(
    selected: list[str],
    output_dir: Path,
    source_dataset: str,
    source_config: str,
    target_config: str,
    size: int,
    seed: int,
) -> Path:
    """Persist the selection plus reproducibility parameters as JSON."""
    manifest = {
        "source_dataset": source_dataset,
        "source_config": source_config,
        "target_config": target_config,
        "subset_size": len(selected),
        "requested_size": size,
        "seed": seed,
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_paths": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "subset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print(
        f"  {len(selected):,} file_paths selected from {source_config} "
        f"with seed={seed}"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Filter + split
# ---------------------------------------------------------------------------


def filter_and_split(
    ds: DatasetDict,
    selected: set[str],
    seed: int,
) -> DatasetDict:
    """Filter the source to the selected file_paths and produce train/validation."""
    combined = concatenate_datasets([ds[split] for split in ds])

    filtered = combined.filter(
        lambda row: row["file_path"] in selected,
        desc="Filtering",
    )

    # Sort by file_path so the subset is ordered identically across different
    # source configs that happen to share the same file_paths.
    filtered = filtered.sort("file_path")

    splits = filtered.train_test_split(
        test_size=DEFAULT_SPLIT_RATIO,
        seed=seed,
    )
    return DatasetDict(
        {
            "train": splits["train"],
            "validation": splits["test"],
        }
    )


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(subset: DatasetDict, expected_size: int) -> dict:
    """Check subset row count, uniqueness of file_paths, optional token stats."""
    combined = concatenate_datasets([subset["train"], subset["validation"]])
    total = len(combined)
    train_size = len(subset["train"])
    val_size = len(subset["validation"])
    path_set = set(combined["file_path"])

    print("\n--- Validation ---")

    size_ok = total == expected_size
    print(
        f"  rows: {total} (train={train_size}, val={val_size}) "
        f"[{'OK' if size_ok else 'FAIL'}]"
    )

    dupes = total - len(path_set)
    if dupes:
        print(f"  FAIL: {dupes} duplicate file_paths in subset", file=sys.stderr)
    else:
        print(f"  file_paths: {len(path_set)} unique [OK]")

    token_stats: dict | None = None
    if "token_count" in combined.column_names:
        raw_counts = combined["token_count"]
        counts = [c for c in raw_counts if c is not None]
        if counts:
            total_tokens = sum(counts)
            token_stats = {
                "total": total_tokens,
                "avg": round(total_tokens / len(counts), 1),
                "min": min(counts),
                "max": max(counts),
            }
            print(
                f"  tokens: total={total_tokens:,}, "
                f"avg={token_stats['avg']:.0f}, "
                f"min={token_stats['min']}, "
                f"max={token_stats['max']}"
            )

    passed = size_ok and dupes == 0
    if passed:
        print("\nAll validations PASSED")
    else:
        print("\nValidation FAILED", file=sys.stderr)

    return {
        "expected_size": expected_size,
        "total": total,
        "train": train_size,
        "validation": val_size,
        "duplicate_file_paths": dupes,
        "unique_file_paths": len(path_set),
        "token_stats": token_stats,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic subset of a single HF dataset config. "
            "Use when only one condition exists yet; switch to "
            "create_subset.py once multiple conditions are available."
        ),
    )
    parser.add_argument(
        "--source-dataset",
        default="legesher/language-decoded-data",
        help="HF dataset ID to subset (default: legesher/language-decoded-data)",
    )
    parser.add_argument(
        "--source-config",
        required=True,
        help="Source config name (e.g., condition-1-en-103k)",
    )
    parser.add_argument(
        "--target-config",
        required=True,
        help="Target config name for the subset (e.g., condition-1-en-5k)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of file_paths to select (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic selection (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local output dir (default: ./subset-{target-config}/)",
    )
    parser.add_argument(
        "--push",
        default=None,
        help="HF dataset ID to push the subset to (optional)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Stream file_paths + write manifest only — no download or push",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output) if args.output else Path(f"./subset-{args.target_config}")
    )

    if args.manifest_only:
        print(
            f"Streaming file_paths from {args.source_dataset}:{args.source_config} ..."
        )
        paths = extract_file_paths_streaming(args.source_dataset, args.source_config)
        unique = check_no_duplicates(paths, args.source_config)
        selected = select_subset(unique, args.size, args.seed)
        save_manifest(
            selected,
            output_dir,
            args.source_dataset,
            args.source_config,
            args.target_config,
            args.size,
            args.seed,
        )
        print("\n--manifest-only: stopping after manifest generation.")
        return

    # Full mode
    ds = load_source(args.source_dataset, args.source_config)
    paths = extract_file_paths(ds)
    unique = check_no_duplicates(paths, args.source_config)

    selected = select_subset(unique, args.size, args.seed)
    save_manifest(
        selected,
        output_dir,
        args.source_dataset,
        args.source_config,
        args.target_config,
        args.size,
        args.seed,
    )

    subset = filter_and_split(ds, set(selected), args.seed)
    report = validate(subset, len(selected))

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report: {report_path}")

    if not report["passed"]:
        print("\nAborting save — validation failures detected.", file=sys.stderr)
        sys.exit(1)

    # Save locally
    config_dir = output_dir / args.target_config
    config_dir.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(config_dir))
    for split_name in subset:
        parquet_path = config_dir / f"{split_name}.parquet"
        subset[split_name].to_parquet(str(parquet_path))
    print(f"\nSaved locally: {config_dir}")

    # Push to HF
    if args.push:
        print(f"\nPushing to {args.push}...")
        subset.push_to_hub(
            args.push,
            config_name=args.target_config,
            data_dir=f"data/{args.target_config}",
            private=False,
        )
        print(
            f"Pushed: "
            f"https://huggingface.co/datasets/{args.push}/tree/main/data/{args.target_config}"
        )

    print("\nComplete.")


if __name__ == "__main__":
    main()

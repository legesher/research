#!/usr/bin/env python3
"""Create a deterministic subset of files from the language-decoded-data dataset.

Ensures the SAME files are used across all experimental conditions (condition-1-en,
condition-2-zh, condition-2-es, condition-2-ur) by selecting via file_path intersection
with a fixed random seed.

Linked issue: AYA-173

Requirements:
    pip install datasets tqdm

Usage:
    # Verify file_path consistency + generate manifest only
    python scripts/create_subset.py \\
        --dataset legesher/language-decoded-data \\
        --size 5000 \\
        --manifest-only

    # Create full subset locally
    python scripts/create_subset.py \\
        --dataset legesher/language-decoded-data \\
        --size 5000 \\
        --output ./subset-5k/

    # Create subset and push to HuggingFace as new configs
    python scripts/create_subset.py \\
        --dataset legesher/language-decoded-data \\
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (aligned with package_dataset.py)
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIO = 0.1  # 10% validation
DEFAULT_SIZE = 5000

CONFIGS = [
    "condition-1-en",
    "condition-2-zh",
    "condition-2-es",
    "condition-2-ur",
]


# ---------------------------------------------------------------------------
# Step 1: Verify file_path consistency across configs
# ---------------------------------------------------------------------------


def load_all_configs(dataset_id: str) -> dict[str, DatasetDict]:
    """Load all configs from the HF dataset."""
    configs = {}
    for config in tqdm(CONFIGS, desc="Loading configs"):
        configs[config] = load_dataset(dataset_id, config)
    return configs


def extract_file_paths(ds: DatasetDict) -> list[str]:
    """Extract all file_path values from a DatasetDict (combining splits)."""
    combined = concatenate_datasets([ds[split] for split in ds])
    return combined["file_path"]


def verify_file_paths(all_configs: dict[str, DatasetDict]) -> dict[str, set[str]]:
    """Verify file_path consistency across all configs.

    Returns dict of {config_name: set of file_paths}.
    Aborts if configs don't match or duplicates are found.
    """
    path_sets: dict[str, set[str]] = {}

    for config_name, ds in all_configs.items():
        paths = extract_file_paths(ds)

        # Check for duplicates within this config
        path_set = set(paths)
        if len(paths) != len(path_set):
            seen = set()
            dupes = set()
            for p in paths:
                if p in seen:
                    dupes.add(p)
                seen.add(p)

            # Count occurrences efficiently
            dupe_counts: dict[str, int] = {}
            for p in paths:
                if p in dupes:
                    dupe_counts[p] = dupe_counts.get(p, 0) + 1

            print(
                f"FATAL: {config_name} has {len(paths) - len(path_set)} duplicate file_paths:",
                file=sys.stderr,
            )
            for d in sorted(dupes)[:10]:
                print(f"  {d} (appears {dupe_counts[d]}x)", file=sys.stderr)
            if len(dupes) > 10:
                print(f"  ... and {len(dupes) - 10} more", file=sys.stderr)
            sys.exit(1)

        path_sets[config_name] = path_set
        print(f"  {config_name}: {len(path_set)} unique file_paths")

    _report_and_intersect(path_sets)
    return path_sets


def _report_and_intersect(path_sets: dict[str, set[str]]) -> set[str]:
    """Check if all configs have identical file_paths. Report differences.

    Returns the intersection of all path sets. If configs differ, prints
    detailed diagnostics but does NOT abort — the caller samples from the
    intersection, which is the safe set.
    """
    all_sets = list(path_sets.values())
    intersection = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    reference_config = CONFIGS[0]
    reference_set = path_sets[reference_config]

    all_match = True
    for config_name in CONFIGS[1:]:
        other_set = path_sets[config_name]
        if reference_set != other_set:
            all_match = False
            only_in_ref = reference_set - other_set
            only_in_other = other_set - reference_set
            print(f"\n  WARNING: {reference_config} vs {config_name}:")
            if only_in_ref:
                print(f"    {len(only_in_ref)} paths only in {reference_config}:")
                for p in sorted(only_in_ref)[:5]:
                    print(f"      {p}")
            if only_in_other:
                print(f"    {len(only_in_other)} paths only in {config_name}:")
                for p in sorted(only_in_other)[:5]:
                    print(f"      {p}")

    if all_match:
        print(
            f"\nAll {len(CONFIGS)} configs have identical file_path sets "
            f"({len(reference_set)} files)"
        )
    else:
        sizes = {name: len(s) for name, s in path_sets.items()}
        print(
            f"\n  Configs have DIFFERENT file_path sets: {sizes}"
            f"\n  Using INTERSECTION of all configs: {len(intersection)} files"
            f"\n  (This ensures only files present in ALL conditions are selected)"
        )

    return intersection


def _verify_file_paths_streaming(dataset_id: str) -> dict[str, set[str]]:
    """Verify file_path consistency using streaming (no full download).

    Used by --manifest-only to avoid downloading all data.
    Reports mismatches but returns all path sets (caller uses intersection).
    """
    path_sets: dict[str, set[str]] = {}

    for config_name in tqdm(CONFIGS, desc="Streaming file_paths"):
        ds = load_dataset(dataset_id, config_name, streaming=True)
        paths: list[str] = []
        for split_name in ds:
            for row in ds[split_name]:
                paths.append(row["file_path"])  # type: ignore[index]

        path_set = set(paths)
        if len(paths) != len(path_set):
            print(
                f"WARNING: {config_name} has {len(paths) - len(path_set)} duplicate file_paths",
                file=sys.stderr,
            )

        path_sets[config_name] = path_set
        print(f"  {config_name}: {len(path_set)} unique file_paths")

    # Check if all match; if not, report and compute intersection
    _report_and_intersect(path_sets)
    return path_sets


# ---------------------------------------------------------------------------
# Step 2: Select subset deterministically
# ---------------------------------------------------------------------------


def select_subset(file_paths: set[str], size: int, seed: int) -> list[str]:
    """Select a deterministic subset of file_paths.

    Sorts alphabetically first (immune to HF row reordering), then samples.
    """
    sorted_paths = sorted(file_paths)

    if size > len(sorted_paths):
        print(
            f"Warning: requested {size} files but only {len(sorted_paths)} available. "
            f"Using all {len(sorted_paths)} files.",
            file=sys.stderr,
        )
        return sorted_paths

    rng = random.Random(seed)
    selected = rng.sample(sorted_paths, size)
    # Sort the selection for consistent ordering
    selected.sort()
    return selected


def save_manifest(
    selected_paths: list[str],
    output_dir: Path,
    dataset_id: str,
    size: int,
    seed: int,
) -> Path:
    """Save the subset manifest as JSON."""
    manifest = {
        "source_dataset": dataset_id,
        "subset_size": len(selected_paths),
        "requested_size": size,
        "seed": seed,
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configs": CONFIGS,
        "file_paths": selected_paths,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "subset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")
    print(f"  {len(selected_paths)} file_paths selected with seed={seed}")
    return manifest_path


# ---------------------------------------------------------------------------
# Step 3: Filter and re-package
# ---------------------------------------------------------------------------


def filter_and_split(
    ds: DatasetDict,
    selected_set: set[str],
    seed: int,
) -> DatasetDict:
    """Filter a DatasetDict to the selected file_paths and re-split."""
    # Combine all splits
    combined = concatenate_datasets([ds[split] for split in ds])

    # Filter to selected file_paths
    filtered = combined.filter(
        lambda row: row["file_path"] in selected_set,
        desc="Filtering",
    )

    # Sort by file_path for consistent ordering across configs
    filtered = filtered.sort("file_path")

    # Re-split train/validation
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
# Step 4: Validate
# ---------------------------------------------------------------------------


def validate_subsets(
    subsets: dict[str, DatasetDict],
    expected_size: int,
) -> dict:
    """Exhaustive validation across all subset configs.

    Returns a validation report dict.
    """
    report: dict = {
        "expected_size": expected_size,
        "configs": {},
        "file_path_match": False,
        "code_en_match": False,
        "all_passed": False,
    }

    all_passed = True

    # Pre-compute combined datasets once per config (avoids repeated concatenation)
    combined_datasets = {}
    for config_name, ds in subsets.items():
        combined_datasets[config_name] = concatenate_datasets(
            [ds["train"], ds["validation"]]
        )

    # --- Check 1: Row counts ---
    print("\n--- Validation ---")
    for config_name, ds in subsets.items():
        train_size = len(ds["train"])
        val_size = len(ds["validation"])
        total = train_size + val_size
        report["configs"][config_name] = {
            "total": total,
            "train": train_size,
            "validation": val_size,
        }
        status = "OK" if total == expected_size else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(
            f"  {config_name}: {total} rows (train={train_size}, val={val_size}) [{status}]"
        )

    # --- Check 2: No duplicate file_paths ---
    path_sets: dict[str, set[str]] = {}
    for config_name, combined in combined_datasets.items():
        paths = combined["file_path"]
        path_set = set(paths)
        path_sets[config_name] = path_set
        if len(paths) != len(path_set):
            print(
                f"  FAIL: {config_name} has {len(paths) - len(path_set)} duplicate file_paths"
            )
            report["configs"][config_name]["duplicates"] = len(paths) - len(path_set)
            all_passed = False
        else:
            report["configs"][config_name]["duplicates"] = 0

    # --- Check 3: file_path set equality ---
    reference_config = CONFIGS[0]
    ref_paths = path_sets[reference_config]

    paths_match = True
    for config_name in CONFIGS[1:]:
        if ref_paths != path_sets[config_name]:
            paths_match = False
            print(
                f"  FAIL: file_path mismatch between {reference_config} and {config_name}"
            )
            all_passed = False

    report["file_path_match"] = paths_match
    if paths_match:
        print(f"  file_path sets: MATCH across all {len(CONFIGS)} configs")

    # --- Check 4: code_en content match (exhaustive) ---
    # Build lookup from condition-1-en: {file_path: code}
    ref_combined = combined_datasets[reference_config]
    ref_lookup = {
        row["file_path"]: row["code"] for row in ref_combined  # type: ignore[index]
    }

    code_en_match = True
    mismatches = []
    for config_name in CONFIGS[1:]:
        combined = combined_datasets[config_name]
        for row in tqdm(
            combined, desc=f"Verifying code_en ({config_name})", leave=False
        ):
            fp = row["file_path"]  # type: ignore[index]
            expected_code = ref_lookup.get(fp)
            actual_code_en = row["code_en"]  # type: ignore[index]

            if expected_code is None:
                mismatches.append(
                    {
                        "config": config_name,
                        "file_path": fp,
                        "reason": "file_path not found in condition-1-en",
                    }
                )
                code_en_match = False
            elif expected_code != actual_code_en:
                mismatches.append(
                    {
                        "config": config_name,
                        "file_path": fp,
                        "reason": "code_en content differs",
                        "expected_len": len(expected_code),
                        "actual_len": len(actual_code_en),
                        "expected_preview": expected_code[:100],
                        "actual_preview": actual_code_en[:100],
                    }
                )
                code_en_match = False

    report["code_en_match"] = code_en_match
    report["code_en_mismatches"] = len(mismatches)

    if code_en_match:
        total_checks = expected_size * (len(CONFIGS) - 1)
        print(
            f"  code_en content: MATCH ({total_checks} comparisons across {len(CONFIGS) - 1} configs)"
        )
    else:
        print(f"  FAIL: {len(mismatches)} code_en mismatches found")
        for m in mismatches[:5]:
            print(f"    {m['config']}: {m['file_path']} — {m['reason']}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")
        all_passed = False

    # --- Token count stats ---
    print("\n--- Token Count Statistics ---")
    for config_name, combined in combined_datasets.items():
        raw_counts = combined["token_count"]
        counts = [c for c in raw_counts if c is not None]
        none_count = len(raw_counts) - len(counts)
        if none_count > 0:
            print(
                f"  WARNING: {config_name} has {none_count} rows with null token_count"
            )
        total = sum(counts)
        avg = total / len(counts) if counts else 0
        min_c = min(counts) if counts else 0
        max_c = max(counts) if counts else 0
        report["configs"][config_name]["token_stats"] = {
            "total": total,
            "avg": round(avg, 1),
            "min": min_c,
            "max": max_c,
        }
        print(
            f"  {config_name}: {total:,} total tokens, avg={avg:.0f}, min={min_c}, max={max_c}"
        )

    report["all_passed"] = all_passed

    if all_passed:
        print("\nAll validations PASSED")
    else:
        print("\nSome validations FAILED — see details above", file=sys.stderr)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic subset of the language-decoded-data dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="legesher/language-decoded-data",
        help="HuggingFace dataset ID (default: legesher/language-decoded-data)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of files to select (default: {DEFAULT_SIZE})",
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
        help="Output directory for subset datasets (default: ./subset-{size}/)",
    )
    parser.add_argument(
        "--push",
        default=None,
        help="Push subsets to this HF dataset ID as new configs (e.g., condition-1-en-5k)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only verify file_paths and generate manifest — do not download/filter data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output) if args.output else Path(f"./subset-{args.size}")

    # Step 1: Load and verify
    print(f"Loading dataset: {args.dataset}")

    if args.manifest_only:
        # Streaming mode: only fetch file_paths without downloading full data
        print("  (streaming mode — downloading file_paths only)")
        path_sets = _verify_file_paths_streaming(args.dataset)

        # Step 2: Select from intersection (safe set across all configs)
        intersection = path_sets[CONFIGS[0]].copy()
        for s in list(path_sets.values())[1:]:
            intersection &= s
        selected = select_subset(intersection, args.size, args.seed)
        save_manifest(selected, output_dir, args.dataset, args.size, args.seed)
        print("\n--manifest-only: stopping after manifest generation.")
        return

    # Full mode: download all data
    all_configs = load_all_configs(args.dataset)

    print("\nVerifying file_path consistency...")
    path_sets = verify_file_paths(all_configs)

    # Step 2: Select from intersection (safe set across all configs)
    intersection = path_sets[CONFIGS[0]].copy()
    for s in list(path_sets.values())[1:]:
        intersection &= s
    selected = select_subset(intersection, args.size, args.seed)
    save_manifest(selected, output_dir, args.dataset, args.size, args.seed)

    # Step 3: Filter and re-package
    selected_set = set(selected)
    subsets: dict[str, DatasetDict] = {}

    print(f"\nFiltering {len(CONFIGS)} configs to {len(selected)} files...")
    for config_name in tqdm(CONFIGS, desc="Processing configs"):
        ds = all_configs[config_name]
        subsets[config_name] = filter_and_split(ds, selected_set, args.seed)

    # Step 4: Validate
    report = validate_subsets(subsets, len(selected))

    # Save validation report
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report: {report_path}")

    if not report["all_passed"]:
        print("\nAborting save — validation failures detected.", file=sys.stderr)
        sys.exit(1)

    # Save locally
    for config_name, ds in subsets.items():
        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(config_dir))

        for split_name in ds:
            parquet_path = config_dir / f"{split_name}.parquet"
            ds[split_name].to_parquet(str(parquet_path))

        print(f"  Saved {config_name} to {config_dir}")

    # Push to HF if requested
    if args.push:
        print(f"\nPushing to {args.push}...")
        suffix = f"-{args.size // 1000}k"
        for config_name, ds in subsets.items():
            new_config = f"{config_name}{suffix}"
            ds.push_to_hub(
                args.push,
                config_name=new_config,
                data_dir=f"data/{new_config}",
                private=True,
            )
            print(f"  Pushed {new_config}")
        print(f"Done! Dataset at https://huggingface.co/datasets/{args.push}")

    print("\nComplete.")


if __name__ == "__main__":
    main()

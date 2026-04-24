#!/usr/bin/env python3
"""Create a deterministic subset of files from a HuggingFace dataset.

Designed for multi-condition experimental pools where all conditions
share the same ``file_path`` set (e.g., an English source and its
keyword-translated variants). Selects a subset via the ``file_path``
intersection with a fixed random seed, then writes one subset dataset
per source config (naming configurable via ``--target-configs``).

Also works with a single source config, in which case the intersection
is trivially that one config's file_paths — useful for the Phase 3
bootstrap scenario where only one condition has been populated yet.

Linked issue: AYA-173 (original); Phase 3 configurability follow-up.

Requirements:
    pip install datasets tqdm

Usage (single config, Phase 3 bootstrap — current recommended form):
    python scripts/create_subset.py \\
        --dataset legesher/language-decoded-data \\
        --source-configs condition-1-en-103k \\
        --target-configs condition-1-en-5k \\
        --size 5000 \\
        --output ./subset-5k/ \\
        --push legesher/language-decoded-data

Usage (multi-config, once conditions 2/3/4 are transpiled):
    python scripts/create_subset.py \\
        --dataset legesher/language-decoded-data \\
        --source-configs <cond-1>,<cond-2>,<cond-3>,<cond-4> \\
        --target-configs <cond-1-5k>,<cond-2-5k>,<cond-3-5k>,<cond-4-5k> \\
        --size 5000 \\
        --output ./subset-5k/ \\
        --push legesher/language-decoded-data

    (Phase 2 used e.g. ``condition-1-en``, ``condition-2-zh`` — those
    configs were renamed on HF to ``phase-2-the-stack-v1-condition-*``
    and are no longer a valid source for new subsets; substitute the
    actual current config names.)

Manifest-only mode (streaming, no data download):
    python scripts/create_subset.py \\
        --source-configs condition-1-en-103k \\
        --target-configs condition-1-en-5k \\
        --size 5000 \\
        --manifest-only
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


# ---------------------------------------------------------------------------
# Step 1: Verify file_path consistency across configs
# ---------------------------------------------------------------------------


def load_all_configs(
    dataset_id: str, source_configs: list[str]
) -> dict[str, DatasetDict]:
    """Load the given source configs from the HF dataset."""
    configs: dict[str, DatasetDict] = {}
    for config in tqdm(source_configs, desc="Loading configs"):
        configs[config] = load_dataset(dataset_id, config)
    return configs


def extract_file_paths(ds: DatasetDict) -> list[str]:
    """Extract all file_path values from a DatasetDict (combining splits)."""
    combined = concatenate_datasets([ds[split] for split in ds])
    return combined["file_path"]


def verify_file_paths(
    all_configs: dict[str, DatasetDict], source_configs: list[str]
) -> dict[str, set[str]]:
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

    _report_and_intersect(path_sets, source_configs)
    return path_sets


def _report_and_intersect(
    path_sets: dict[str, set[str]], configs: list[str]
) -> set[str]:
    """Check if all configs have identical file_paths. Report differences.

    Returns the intersection of all path sets. If configs differ, prints
    detailed diagnostics but does NOT abort — the caller samples from the
    intersection, which is the safe set.
    """
    all_sets = list(path_sets.values())
    intersection = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    # The first config passed on the CLI acts as the reference; every other
    # config is compared against it for mismatch reporting.
    reference_config = configs[0]
    reference_set = path_sets[reference_config]

    all_match = True
    for config_name in configs[1:]:
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
            f"\nAll {len(configs)} configs have identical file_path sets "
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


def _verify_file_paths_streaming(
    dataset_id: str, source_configs: list[str]
) -> dict[str, set[str]]:
    """Verify file_path consistency using streaming (no full download).

    Used by --manifest-only to avoid downloading all data.
    Reports mismatches but returns all path sets (caller uses intersection).
    """
    path_sets: dict[str, set[str]] = {}

    for config_name in tqdm(source_configs, desc="Streaming file_paths"):
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
    _report_and_intersect(path_sets, source_configs)
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
    source_configs: list[str],
    target_configs: list[str],
    size: int,
    seed: int,
) -> Path:
    """Save the subset manifest as JSON."""
    manifest = {
        "source_dataset": dataset_id,
        "source_configs": source_configs,
        "target_configs": target_configs,
        "subset_size": len(selected_paths),
        "requested_size": size,
        "seed": seed,
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
    source_configs: list[str],
) -> dict:
    """Exhaustive validation across all subset configs.

    ``source_configs[0]`` is used as the reference against which every other
    config is compared (file_path equality, code_en content). The ``code_en``
    equality check is skipped when only one config is provided (nothing to
    compare against).

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
    reference_config = source_configs[0]
    ref_paths = path_sets[reference_config]

    paths_match = True
    for config_name in source_configs[1:]:
        if ref_paths != path_sets[config_name]:
            paths_match = False
            print(
                f"  FAIL: file_path mismatch between {reference_config} and {config_name}"
            )
            all_passed = False

    report["file_path_match"] = paths_match
    if paths_match:
        if len(source_configs) > 1:
            print(f"  file_path sets: MATCH across all {len(source_configs)} configs")
        else:
            print("  file_path sets: single config, no cross-config check")

    # --- Check 4: code_en content match (exhaustive) ---
    # Build lookup from reference config: {file_path: code}
    ref_combined = combined_datasets[reference_config]
    ref_lookup = {
        row["file_path"]: row["code"] for row in ref_combined  # type: ignore[index]
    }

    code_en_match = True
    mismatches = []
    for config_name in source_configs[1:]:
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
                        "reason": f"file_path not found in {reference_config}",
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
        total_checks = expected_size * (len(source_configs) - 1)
        if total_checks:
            print(
                f"  code_en content: MATCH "
                f"({total_checks} comparisons across {len(source_configs) - 1} configs)"
            )
        else:
            print("  code_en content: single config, no cross-config check")
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


def _split_csv(value: str) -> list[str]:
    """Parse a comma-separated argparse value into a list of non-empty tokens."""
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic subset of a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="legesher/language-decoded-data",
        help="HuggingFace dataset ID (default: legesher/language-decoded-data)",
    )
    parser.add_argument(
        "--source-configs",
        required=True,
        type=_split_csv,
        help=(
            "Comma-separated list of source config names on the dataset. "
            "Example (Phase 3 bootstrap, single config): 'condition-1-en-103k'. "
            "For multi-config runs, list every condition that should share the "
            "same subset file_path set, in consistent ordering with "
            "--target-configs."
        ),
    )
    parser.add_argument(
        "--target-configs",
        required=True,
        type=_split_csv,
        help=(
            "Comma-separated list of target config names to push as, one per "
            "source config, in the same order. Example (Phase 3 bootstrap): "
            "'condition-1-en-5k'."
        ),
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
    args = parser.parse_args()

    if len(args.source_configs) != len(args.target_configs):
        parser.error(
            f"--source-configs ({len(args.source_configs)}) and "
            f"--target-configs ({len(args.target_configs)}) must have "
            f"the same number of entries"
        )
    return args


def main() -> None:
    args = parse_args()

    source_configs: list[str] = args.source_configs
    target_configs: list[str] = args.target_configs
    # Pairing is positional and was validated in parse_args.
    target_by_source = dict(zip(source_configs, target_configs, strict=True))

    output_dir = Path(args.output) if args.output else Path(f"./subset-{args.size}")

    # Step 1: Load and verify
    print(f"Loading dataset: {args.dataset}")
    print(f"  source configs: {source_configs}")
    print(f"  target configs: {target_configs}")

    if args.manifest_only:
        # Streaming mode: only fetch file_paths without downloading full data
        print("  (streaming mode — downloading file_paths only)")
        path_sets = _verify_file_paths_streaming(args.dataset, source_configs)

        # Step 2: Select from intersection (safe set across all configs)
        intersection = path_sets[source_configs[0]].copy()
        for s in list(path_sets.values())[1:]:
            intersection &= s
        selected = select_subset(intersection, args.size, args.seed)
        save_manifest(
            selected,
            output_dir,
            args.dataset,
            source_configs,
            target_configs,
            args.size,
            args.seed,
        )
        print("\n--manifest-only: stopping after manifest generation.")
        return

    # Full mode: download all data
    all_configs = load_all_configs(args.dataset, source_configs)

    print("\nVerifying file_path consistency...")
    path_sets = verify_file_paths(all_configs, source_configs)

    # Step 2: Select from intersection (safe set across all configs)
    intersection = path_sets[source_configs[0]].copy()
    for s in list(path_sets.values())[1:]:
        intersection &= s
    selected = select_subset(intersection, args.size, args.seed)
    save_manifest(
        selected,
        output_dir,
        args.dataset,
        source_configs,
        target_configs,
        args.size,
        args.seed,
    )

    # Step 3: Filter and re-package
    selected_set = set(selected)
    subsets: dict[str, DatasetDict] = {}

    print(f"\nFiltering {len(source_configs)} configs to {len(selected)} files...")
    for config_name in tqdm(source_configs, desc="Processing configs"):
        ds = all_configs[config_name]
        subsets[config_name] = filter_and_split(ds, selected_set, args.seed)

    # Step 4: Validate
    report = validate_subsets(subsets, len(selected), source_configs)

    # Save validation report
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report: {report_path}")

    if not report["all_passed"]:
        print("\nAborting save — validation failures detected.", file=sys.stderr)
        sys.exit(1)

    # Save locally (directories keyed by TARGET config name for clarity)
    for source_name, ds in subsets.items():
        target_name = target_by_source[source_name]
        config_dir = output_dir / target_name
        config_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(config_dir))

        for split_name in ds:
            parquet_path = config_dir / f"{split_name}.parquet"
            ds[split_name].to_parquet(str(parquet_path))

        print(f"  Saved {source_name} → {target_name} at {config_dir}")

    # Push to HF if requested
    if args.push:
        print(f"\nPushing to {args.push}...")
        for source_name, ds in subsets.items():
            target_name = target_by_source[source_name]
            ds.push_to_hub(
                args.push,
                config_name=target_name,
                data_dir=f"data/{target_name}",
                private=False,
            )
            print(f"  Pushed {target_name}")
        print(f"Done! Dataset at https://huggingface.co/datasets/{args.push}")

    print("\nComplete.")


if __name__ == "__main__":
    main()

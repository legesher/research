#!/usr/bin/env python3
"""Create the condition-3-zh-5k blended dataset.

Blends ALL native Chinese code (from language-decoded-community) with a
sample of transpiled code (from condition-2-zh in language-decoded-data),
targeting 5,000 total files with a 90/10 train/validation split.

The transpiled sample excludes file_paths already used in condition-2-zh-5k
to keep training data independent across conditions.

Linked issue: AYA-163

Requirements:
    pip install datasets transformers huggingface_hub tqdm pandas

Usage:
    # Create blend locally
    python scripts/create_condition3_blend.py \
        --output ./condition3-zh-5k/

    # Create blend and push to HuggingFace
    python scripts/create_condition3_blend.py \
        --output ./condition3-zh-5k/ \
        --push legesher/language-decoded-data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIO = 0.1  # 10% validation
DEFAULT_SIZE = 5000

NATIVE_DATASET = "legesher/language-decoded-community"
NATIVE_CONFIG = "zh"

TRANSPILED_DATASET = "legesher/language-decoded-data"
TRANSPILED_CONFIG = "condition-2-zh"

DEFAULT_TOKENIZER = "CohereLabs/tiny-aya-base"

# Manifest of file_paths in condition-2-zh-5k (to exclude from sampling)
SUBSET_MANIFEST = Path(__file__).parent.parent / "subset-5k" / "subset_manifest.json"

# Output schema columns
OUTPUT_COLUMNS = [
    "file_path",
    "code",
    "code_en",
    "language",
    "license",
    "token_count",
    "source_type",
]


# ---------------------------------------------------------------------------
# Tokenizer helpers (from package_dataset.py)
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer from HuggingFace."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print(
                f"Error: {model_name} is a gated model. Accept the license at "
                f"https://huggingface.co/{model_name} then run `huggingface-cli login`.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    return tokenizer


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Count tokens for a single text string."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------


def load_native_data() -> pd.DataFrame:
    """Load all native Chinese code from the community repo."""
    print("Loading native data from language-decoded-community...")
    ds = load_dataset(NATIVE_DATASET, NATIVE_CONFIG)

    # Combine train + validation into one DataFrame
    frames = []
    for split_name in ds:
        df = ds[split_name].to_pandas()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} native files")
    return combined


def load_transpiled_data() -> pd.DataFrame:
    """Load all transpiled Chinese code from condition-2-zh."""
    print("Loading transpiled data from condition-2-zh...")
    ds = load_dataset(TRANSPILED_DATASET, TRANSPILED_CONFIG)

    frames = []
    for split_name in ds:
        df = ds[split_name].to_pandas()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} transpiled files")
    return combined


def load_excluded_paths() -> set[str]:
    """Load file_paths from condition-2-zh-5k manifest to exclude."""
    if not SUBSET_MANIFEST.exists():
        print(f"  Warning: manifest not found at {SUBSET_MANIFEST}, no exclusions")
        return set()

    with open(SUBSET_MANIFEST) as f:
        manifest = json.load(f)

    excluded = set(manifest["file_paths"])
    print(f"  Loaded {len(excluded)} file_paths to exclude (from condition-2-zh-5k)")
    return excluded


# ---------------------------------------------------------------------------
# Step 2: Transform and blend
# ---------------------------------------------------------------------------


def transform_native(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    """Transform native data to the output schema."""
    print("Tokenizing native files...")
    token_counts = []
    for content in tqdm(df["content"], desc="Tokenizing native"):
        token_counts.append(count_tokens(tokenizer, content))

    return pd.DataFrame(
        {
            "file_path": df["filename"].values,
            "code": df["content"].values,
            "code_en": [None] * len(df),
            "language": ["zh"] * len(df),
            "license": df["license"].values,
            "token_count": token_counts,
            "source_type": ["native"] * len(df),
        }
    )


def sample_transpiled(
    df: pd.DataFrame,
    n: int,
    excluded: set[str],
    seed: int,
) -> pd.DataFrame:
    """Sample n transpiled files, excluding specified file_paths."""
    # Filter out excluded paths
    available = df[~df["file_path"].isin(excluded)].copy()
    print(
        f"  Transpiled pool: {len(df)} total, {len(excluded)} excluded, "
        f"{len(available)} available"
    )

    if n > len(available):
        print(
            f"  Warning: need {n} but only {len(available)} available. Using all.",
            file=sys.stderr,
        )
        n = len(available)

    # Deterministic sampling (sort first, then sample)
    available = available.sort_values("file_path").reset_index(drop=True)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(available)), n))
    sampled = available.iloc[indices].reset_index(drop=True)
    print(f"  Sampled {len(sampled)} transpiled files (seed={seed})")
    return sampled


def transform_transpiled(df: pd.DataFrame) -> pd.DataFrame:
    """Transform transpiled data to the output schema."""
    return pd.DataFrame(
        {
            "file_path": df["file_path"].values,
            "code": df["code"].values,
            "code_en": df["code_en"].values,
            "language": df["language"].values,
            "license": df["license"].values,
            "token_count": df["token_count"].values,
            "source_type": ["transpiled"] * len(df),
        }
    )


# ---------------------------------------------------------------------------
# Step 3: Split and validate
# ---------------------------------------------------------------------------


def split_dataset(df: pd.DataFrame, seed: int, split_ratio: float) -> DatasetDict:
    """Split combined DataFrame into train/validation DatasetDict."""
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.sort("file_path")

    splits = ds.train_test_split(test_size=split_ratio, seed=seed)
    return DatasetDict(
        {
            "train": splits["train"],
            "validation": splits["test"],
        }
    )


def validate(ds: DatasetDict, expected_size: int) -> dict:
    """Validate the blended dataset."""
    report: dict = {"all_passed": True, "checks": []}

    combined = concatenate_datasets([ds["train"], ds["validation"]])
    total = len(combined)

    # Check 1: Total size
    size_ok = total == expected_size
    report["checks"].append(
        {
            "name": "total_size",
            "expected": expected_size,
            "actual": total,
            "ok": size_ok,
        }
    )
    if not size_ok:
        report["all_passed"] = False
    print(
        f"  Total size: {total} (expected {expected_size}) [{'OK' if size_ok else 'FAIL'}]"
    )

    # Check 2: Split sizes
    train_expected = int(expected_size * (1 - DEFAULT_SPLIT_RATIO))
    val_expected = expected_size - train_expected
    print(f"  Train: {len(ds['train'])}, Validation: {len(ds['validation'])}")

    # Check 3: No duplicate file_paths
    paths = combined["file_path"]
    unique = set(paths)
    dupes = len(paths) - len(unique)
    dupes_ok = dupes == 0
    report["checks"].append(
        {"name": "no_duplicates", "duplicates": dupes, "ok": dupes_ok}
    )
    if not dupes_ok:
        report["all_passed"] = False
    print(f"  Duplicates: {dupes} [{'OK' if dupes_ok else 'FAIL'}]")

    # Check 4: source_type values
    source_types = combined["source_type"]
    native_count = sum(1 for s in source_types if s == "native")
    transpiled_count = sum(1 for s in source_types if s == "transpiled")
    print(f"  Native: {native_count}, Transpiled: {transpiled_count}")
    report["native_count"] = native_count
    report["transpiled_count"] = transpiled_count

    # Check 5: code_en nullability
    code_en_values = combined["code_en"]
    native_with_code_en = sum(
        1
        for s, c in zip(source_types, code_en_values)
        if s == "native" and c is not None
    )
    transpiled_without_code_en = sum(
        1
        for s, c in zip(source_types, code_en_values)
        if s == "transpiled" and c is None
    )
    code_en_ok = native_with_code_en == 0 and transpiled_without_code_en == 0
    report["checks"].append(
        {
            "name": "code_en_nullability",
            "native_with_code_en": native_with_code_en,
            "transpiled_without_code_en": transpiled_without_code_en,
            "ok": code_en_ok,
        }
    )
    if not code_en_ok:
        report["all_passed"] = False
    print(
        f"  code_en nullability: native_with_code_en={native_with_code_en}, "
        f"transpiled_without_code_en={transpiled_without_code_en} "
        f"[{'OK' if code_en_ok else 'FAIL'}]"
    )

    # Check 6: No null token counts
    null_tokens = sum(1 for t in combined["token_count"] if t is None)
    tokens_ok = null_tokens == 0
    report["checks"].append(
        {"name": "no_null_tokens", "null_count": null_tokens, "ok": tokens_ok}
    )
    if not tokens_ok:
        report["all_passed"] = False
    print(f"  Null token_count: {null_tokens} [{'OK' if tokens_ok else 'FAIL'}]")

    # Token stats
    counts = [c for c in combined["token_count"] if c is not None]
    report["token_stats"] = {
        "total": sum(counts),
        "avg": round(sum(counts) / len(counts), 1) if counts else 0,
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
    }
    print(
        f"  Token stats: total={report['token_stats']['total']:,}, "
        f"avg={report['token_stats']['avg']:.0f}, "
        f"min={report['token_stats']['min']}, max={report['token_stats']['max']}"
    )

    status = "PASSED" if report["all_passed"] else "FAILED"
    print(f"\n  Validation: {status}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the condition-3-zh-5k blended dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Target total files (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the blended dataset",
    )
    parser.add_argument(
        "--push",
        default=None,
        help="Push to this HF dataset ID as condition-3-zh-5k config",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer model (default: {DEFAULT_TOKENIZER})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 1: Load data
    native_df = load_native_data()
    transpiled_df = load_transpiled_data()
    excluded = load_excluded_paths()

    # Step 2: Calculate blend
    native_count = len(native_df)
    transpiled_needed = args.size - native_count

    if transpiled_needed <= 0:
        print(
            f"Native data ({native_count}) >= target ({args.size}). "
            f"Sampling {args.size} native files only.",
        )
        # Would need to subsample native — unlikely for zh
        transpiled_needed = 0

    print(
        f"\nBlend: {native_count} native + {transpiled_needed} transpiled = {args.size}"
    )

    # Step 3: Transform native data
    native_transformed = transform_native(native_df, tokenizer)

    # Step 4: Sample and transform transpiled data
    if transpiled_needed > 0:
        transpiled_sampled = sample_transpiled(
            transpiled_df, transpiled_needed, excluded, args.seed
        )
        transpiled_transformed = transform_transpiled(transpiled_sampled)
    else:
        transpiled_transformed = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Step 5: Combine and split
    print("\nCombining and splitting...")
    combined = pd.concat(
        [native_transformed, transpiled_transformed], ignore_index=True
    )
    ds = split_dataset(combined, args.seed, DEFAULT_SPLIT_RATIO)

    print(f"\nDataset: {ds}")

    # Step 6: Validate
    print("\n--- Validation ---")
    report = validate(ds, args.size)

    if not report["all_passed"]:
        print("\nAborting — validation failures.", file=sys.stderr)
        sys.exit(1)

    # Step 7: Save locally
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ds:
        parquet_path = output_dir / f"{split_name}.parquet"
        ds[split_name].to_parquet(str(parquet_path))
        print(f"Saved {split_name} to {parquet_path}")

    # Save manifest
    manifest = {
        "target_size": args.size,
        "actual_size": len(combined),
        "seed": args.seed,
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "native": {
            "source_dataset": NATIVE_DATASET,
            "config": NATIVE_CONFIG,
            "count": native_count,
        },
        "transpiled": {
            "source_dataset": TRANSPILED_DATASET,
            "config": TRANSPILED_CONFIG,
            "count": transpiled_needed,
            "excluded_from": str(SUBSET_MANIFEST),
            "excluded_count": len(excluded),
        },
        "validation": report,
    }

    manifest_path = output_dir / "condition3_blend_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Step 8: Push to HF if requested
    if args.push:
        config_name = "condition-3-zh-5k"
        print(f"\nPushing to {args.push} as {config_name}...")
        ds.push_to_hub(
            args.push,
            config_name=config_name,
            data_dir=f"data/{config_name}",
            private=False,
            commit_message=f"feat: add {config_name} blended dataset ({native_count} native + {transpiled_needed} transpiled)",
        )
        print(f"Done! https://huggingface.co/datasets/{args.push}")

    print("\nComplete.")


if __name__ == "__main__":
    main()

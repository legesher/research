#!/usr/bin/env python3
"""Package transpiled corpora as HuggingFace Datasets with metadata.

Takes transpiled Python files (from batch_transpile.py) and their English
originals, tokenizes with the target model's tokenizer, and packages
everything as a HuggingFace Dataset in Parquet format.

Can also retokenize an existing HuggingFace dataset with the correct
tokenizer (e.g., replacing proxy token counts with actual model counts).

Requirements:
    pip install datasets transformers huggingface_hub tqdm

Usage:
    # Package transpiled files into a new HF dataset
    python package_dataset.py from-files \\
        --transpiled ./transpiled-ur/ \\
        --originals ./source-python/ \\
        --language ur \\
        --output ./packaged-ur/

    # Retokenize an existing HF dataset with the correct tokenizer
    python package_dataset.py retokenize \\
        --dataset legesher/python-ur-transpiled \\
        --output ./retokenized-ur/

    # Retokenize and push back to HuggingFace
    python package_dataset.py retokenize \\
        --dataset legesher/python-ur-transpiled \\
        --output ./retokenized-ur/ \\
        --push legesher/python-ur-transpiled

    # Use a specific tokenizer (default: CohereLabs/tiny-aya-base)
    python package_dataset.py retokenize \\
        --dataset legesher/python-ur-transpiled \\
        --tokenizer CohereLabs/tiny-aya-base \\
        --output ./retokenized-ur/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER = "CohereLabs/tiny-aya-base"
DEFAULT_SPLIT_RATIO = 0.1  # 10% validation
DEFAULT_SEED = 42

REQUIRED_COLUMNS = [
    "code",
    "code_en",
    "language",
    "file_path",
    "license",
    "token_count",
]


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer from HuggingFace, with clear error on gated models."""
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
# Retokenize subcommand
# ---------------------------------------------------------------------------


def retokenize(args: argparse.Namespace) -> None:
    """Retokenize an existing HF dataset with the correct tokenizer."""
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    config = getattr(args, "config_name", None)
    print(
        f"Loading dataset: {args.dataset}" + (f" (config={config})" if config else "")
    )
    ds = load_dataset(args.dataset, config)

    # Validate schema
    for split_name in ds:
        cols = ds[split_name].column_names
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            print(
                f"Warning: split '{split_name}' missing columns: {missing}",
                file=sys.stderr,
            )

    def _retokenize_row(example):
        example["token_count"] = count_tokens(tokenizer, example["code"])
        return example

    # Show before/after stats for first split
    first_split = list(ds.keys())[0]
    sample_before = ds[first_split].select(range(min(5, len(ds[first_split]))))
    old_counts = [r["token_count"] for r in sample_before]

    print("Retokenizing...")
    ds = ds.map(_retokenize_row, desc="Retokenizing")

    sample_after = ds[first_split].select(range(min(5, len(ds[first_split]))))
    new_counts = [r["token_count"] for r in sample_after]

    print("\nSample token count comparison (first 5 rows):")
    print(f"  {'Old':>8}  {'New':>8}  {'Diff':>8}")
    for old, new in zip(old_counts, new_counts):
        if old is None:
            print(f"  {'null':>8}  {new:>8}  {'(new)':>8}")
        else:
            diff = new - old
            print(f"  {old:>8}  {new:>8}  {diff:>+8}")

    # Compute aggregate stats
    for split_name in ds:
        counts = ds[split_name]["token_count"]
        total = sum(counts)
        avg = total / len(counts) if counts else 0
        print(
            f"\n{split_name}: {len(counts)} rows, {total:,} total tokens, {avg:.0f} avg tokens/file"
        )

    # Save locally
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output))
    print(f"\nSaved to {output}")

    # Also save as parquet for easy inspection
    for split_name in ds:
        parquet_path = output / f"{split_name}.parquet"
        ds[split_name].to_parquet(str(parquet_path))
        print(f"  Parquet: {parquet_path}")

    # Push to HuggingFace if requested
    if args.push:
        print(f"\nPushing to {args.push}...")
        if args.config_name:
            ds.push_to_hub(
                args.push,
                config_name=args.config_name,
                data_dir=f"data/{args.config_name}",
                private=False,
            )
        else:
            ds.push_to_hub(args.push, private=False)
        print(f"Done! Dataset live at https://huggingface.co/datasets/{args.push}")


# ---------------------------------------------------------------------------
# Package from files subcommand
# ---------------------------------------------------------------------------


def package_from_files(args: argparse.Namespace) -> None:
    """Package transpiled + original files into a HF dataset."""
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    transpiled_dir = Path(args.transpiled)
    originals_dir = Path(args.originals)

    if not transpiled_dir.exists():
        print(
            f"Error: transpiled directory not found: {transpiled_dir}", file=sys.stderr
        )
        sys.exit(1)

    # Collect file pairs (transpiled + original)
    transpiled_files = sorted(transpiled_dir.glob("**/*.py"))
    print(f"Found {len(transpiled_files)} transpiled files")

    # Load metadata CSV if available (from batch_transpile.py output)
    metadata_path = transpiled_dir / "metadata.csv"
    file_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_metadata[row.get("file_path", row.get("filename", ""))] = row

    rows = []
    skipped = 0

    for tf in tqdm(transpiled_files, desc="Packaging"):
        # Read transpiled code
        try:
            code = tf.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue

        # Find matching original
        relative = tf.relative_to(transpiled_dir)
        original_path = originals_dir / relative
        if original_path.exists():
            try:
                code_en = original_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                code_en = ""
        else:
            code_en = ""

        # Get metadata
        meta = file_metadata.get(str(relative), {})
        license_name = meta.get("license", args.default_license or "unknown")
        file_path_str = meta.get("file_path", str(relative))

        # Tokenize
        token_count = count_tokens(tokenizer, code)

        rows.append(
            {
                "code": code,
                "code_en": code_en,
                "language": args.language,
                "file_path": file_path_str,
                "license": license_name,
                "token_count": token_count,
            }
        )

    print(f"Packaged {len(rows)} files, skipped {skipped}")

    if not rows:
        print("Error: no files packaged", file=sys.stderr)
        sys.exit(1)

    # Create dataset and split
    dataset = Dataset.from_list(rows)
    splits = dataset.train_test_split(
        test_size=DEFAULT_SPLIT_RATIO,
        seed=DEFAULT_SEED,
    )
    ds = DatasetDict(
        {
            "train": splits["train"],
            "validation": splits["test"],
        }
    )

    # Stats
    for split_name in ds:
        counts = ds[split_name]["token_count"]
        total = sum(counts)
        avg = total / len(counts) if counts else 0
        print(
            f"{split_name}: {len(counts)} rows, {total:,} total tokens, {avg:.0f} avg tokens/file"
        )

    # Save
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output))
    print(f"\nSaved to {output}")

    for split_name in ds:
        parquet_path = output / f"{split_name}.parquet"
        ds[split_name].to_parquet(str(parquet_path))
        print(f"  Parquet: {parquet_path}")

    if args.push:
        print(f"\nPushing to {args.push}...")
        if args.config_name:
            ds.push_to_hub(
                args.push,
                config_name=args.config_name,
                data_dir=f"data/{args.config_name}",
                private=False,
            )
        else:
            ds.push_to_hub(args.push, private=False)
        print(f"Done! Dataset live at https://huggingface.co/datasets/{args.push}")

    # Save run metadata
    run_meta = {
        "language": args.language,
        "tokenizer": args.tokenizer,
        "total_files": len(rows),
        "skipped_files": skipped,
        "split_ratio": DEFAULT_SPLIT_RATIO,
        "seed": DEFAULT_SEED,
        "train_rows": len(ds["train"]),
        "validation_rows": len(ds["validation"]),
        "train_tokens": sum(ds["train"]["token_count"]),
        "validation_tokens": sum(ds["validation"]["token_count"]),
    }
    meta_path = output / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package transpiled corpora as HuggingFace Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- retokenize ---
    retok = subparsers.add_parser(
        "retokenize",
        help="Retokenize an existing HF dataset with the correct tokenizer",
    )
    retok.add_argument(
        "--dataset",
        required=True,
        help="HF dataset ID (e.g., legesher/python-ur-transpiled)",
    )
    retok.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer model (default: {DEFAULT_TOKENIZER})",
    )
    retok.add_argument("--output", required=True, help="Local output directory")
    retok.add_argument(
        "--push", default=None, help="Push to this HF dataset ID after retokenizing"
    )
    retok.add_argument(
        "--config-name",
        default=None,
        help="Config/subset name within umbrella dataset (e.g., condition-1-en)",
    )

    # --- from-files ---
    files = subparsers.add_parser(
        "from-files",
        help="Package transpiled files into a new HF dataset",
    )
    files.add_argument(
        "--transpiled", required=True, help="Directory of transpiled Python files"
    )
    files.add_argument(
        "--originals", required=True, help="Directory of original English Python files"
    )
    files.add_argument(
        "--language", required=True, help="Language code (e.g., ur, zh, es)"
    )
    files.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer model (default: {DEFAULT_TOKENIZER})",
    )
    files.add_argument("--output", required=True, help="Local output directory")
    files.add_argument("--push", default=None, help="Push to this HF dataset ID")
    files.add_argument(
        "--default-license", default=None, help="Default license if not in metadata"
    )
    files.add_argument(
        "--config-name",
        default=None,
        help="Config/subset name within umbrella dataset (e.g., condition-1-en)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "retokenize":
        retokenize(args)
    elif args.command == "from-files":
        package_from_files(args)


if __name__ == "__main__":
    main()

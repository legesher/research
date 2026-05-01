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
    # Package transpiled files into a new HF dataset (random 90/10 split)
    python package_dataset.py from-files \\
        --transpiled ./transpiled-ur/ \\
        --originals ./source-python/ \\
        --language ur \\
        --output ./packaged-ur/

    # Retokenize a specific config within an umbrella dataset
    python package_dataset.py retokenize \\
        --dataset legesher/language-decoded-data \\
        --config-name condition-1-en \\
        --output ./retokenized/condition-1-en/

    # Retokenize and push back to HuggingFace
    python package_dataset.py retokenize \\
        --dataset legesher/language-decoded-data \\
        --config-name condition-1-en \\
        --output ./retokenized/condition-1-en/ \\
        --push legesher/language-decoded-data

    # Use a specific tokenizer (default: CohereLabs/tiny-aya-base)
    python package_dataset.py retokenize \\
        --dataset legesher/language-decoded-data \\
        --config-name condition-1-en \\
        --tokenizer CohereLabs/tiny-aya-base \\
        --output ./retokenized/condition-1-en/

    # Package with pre-split train + validation directories (preserves an
    # upstream split, e.g. for cross-language consistency where every
    # language must share the same train/val membership)
    python package_dataset.py from-files \\
        --train-transpiled ./packaged/cond5-ur/train/ur \\
        --train-originals  ./packaged/cond5-ur/train/ur.originals \\
        --validation-transpiled ./packaged/cond5-ur/validation/ur \\
        --validation-originals  ./packaged/cond5-ur/validation/ur.originals \\
        --language ur \\
        --output ./packaged-ur/ \\
        --config-name condition-5-ur-5k-c4ai-aya-expanse-32b \\
        --push legesher/language-decoded-data
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

    config = args.config_name
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

    def _retokenize_row(example: dict) -> dict:
        return {**example, "token_count": count_tokens(tokenizer, example["code"])}

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
        raw_counts = ds[split_name]["token_count"]
        counts = [c for c in raw_counts if c is not None]
        if len(counts) < len(raw_counts):
            print(
                f"\n  WARNING: {split_name} has {len(raw_counts) - len(counts)} rows "
                f"with null token_count (stats from {len(counts)}/{len(raw_counts)} rows)"
            )
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


def _load_metadata_csv(transpiled_dir: Path) -> dict[str, dict[str, str]]:
    """Load metadata.csv from a transpiled dir, keyed by filename or file_path.

    Prefer ``filename`` as the lookup key when populated — the cond5 schema
    uses ``filename: 000.py`` for lookup AND
    ``file_path: <stack-v2-attribution>`` so source provenance survives onto
    the published HF dataset's ``file_path`` column. Older CSVs without
    ``filename`` fall back to ``file_path`` for the key (preserving
    backward compatibility with batch_transpile.py output).

    populate_cond5_datasets writes metadata.csv as UTF-8; the ``file_path``
    column may include non-ASCII characters, so open with explicit encoding
    + newline="" so csv.DictReader handles \\r\\n correctly across locales.
    """
    metadata_path = transpiled_dir / "metadata.csv"
    if not metadata_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(metadata_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup_key = row.get("filename") or row.get("file_path") or ""
            out[lookup_key] = row
    return out


def _build_rows(
    transpiled_dir: Path,
    originals_dir: Path,
    language: str,
    tokenizer: AutoTokenizer,
    default_license: str | None,
    desc: str = "Packaging",
) -> tuple[list[dict[str, object]], int]:
    """Walk a transpiled dir and produce HF dataset rows + skipped count.

    Pairs each transpiled ``*.py`` with its English counterpart in
    ``originals_dir`` (via path relative to ``transpiled_dir``), reads the
    metadata.csv if present, and tokenizes the transpiled code.
    """
    if not transpiled_dir.exists():
        print(
            f"Error: transpiled directory not found: {transpiled_dir}", file=sys.stderr
        )
        sys.exit(1)

    transpiled_files = sorted(transpiled_dir.glob("**/*.py"))
    print(f"  {desc}: found {len(transpiled_files)} files in {transpiled_dir}")

    file_metadata = _load_metadata_csv(transpiled_dir)

    rows: list[dict[str, object]] = []
    skipped = 0

    for tf in tqdm(transpiled_files, desc=desc):
        try:
            code = tf.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue

        relative = tf.relative_to(transpiled_dir)
        original_path = originals_dir / relative
        if original_path.exists():
            try:
                code_en = original_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                code_en = ""
        else:
            code_en = ""

        # Fall back to basename when CSV keys are basenames (cond5
        # populator's `filename` column) but `relative` is a nested path
        # (e.g. batch_transpile.py output).
        meta = (
            file_metadata.get(str(relative)) or file_metadata.get(relative.name) or {}
        )
        license_name = meta.get("license", default_license or "unknown")
        file_path_str = meta.get("file_path", str(relative))
        # Propagate `idx` (source parquet row index) from metadata.csv when
        # available — the cond5 populator and the materialize_cond1_source
        # manifest both populate it. Lets cross-condition joins use a
        # deterministic integer key instead of string-matching file_path.
        # Stored as int for clean parquet typing; -1 sentinel when absent.
        idx_raw = meta.get("idx")
        try:
            idx_val = int(idx_raw) if idx_raw not in (None, "") else -1
        except (TypeError, ValueError):
            idx_val = -1

        token_count = count_tokens(tokenizer, code)

        rows.append(
            {
                "code": code,
                "code_en": code_en,
                "language": language,
                "file_path": file_path_str,
                "license": license_name,
                "idx": idx_val,
                "token_count": token_count,
            }
        )

    return rows, skipped


def _print_split_stats(ds: DatasetDict) -> None:
    for split_name in ds:
        counts = ds[split_name]["token_count"]
        total = sum(counts)
        avg = total / len(counts) if counts else 0
        print(
            f"{split_name}: {len(counts)} rows, {total:,} total tokens, "
            f"{avg:.0f} avg tokens/file"
        )


def _save_and_push(
    ds: DatasetDict,
    output_dir: Path,
    push_target: str | None,
    config_name: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"\nSaved to {output_dir}")

    for split_name in ds:
        parquet_path = output_dir / f"{split_name}.parquet"
        ds[split_name].to_parquet(str(parquet_path))
        print(f"  Parquet: {parquet_path}")

    if push_target:
        print(f"\nPushing to {push_target}...")
        if config_name:
            ds.push_to_hub(
                push_target,
                config_name=config_name,
                data_dir=f"data/{config_name}",
                private=False,
            )
        else:
            ds.push_to_hub(push_target, private=False)
        print(f"Done! Dataset live at https://huggingface.co/datasets/{push_target}")


def package_from_files(args: argparse.Namespace) -> None:
    """Package transpiled + original files into a HF dataset.

    Two modes:

    * **Random split** (legacy): pass ``--transpiled`` + ``--originals``
      and the script does its own ``train_test_split(0.1)`` with
      ``seed=DEFAULT_SEED``.
    * **Pre-split** (new): pass ``--train-transpiled`` /
      ``--train-originals`` / ``--validation-transpiled`` /
      ``--validation-originals`` and the script builds the DatasetDict
      directly. Use this whenever the upstream split must be preserved —
      e.g. cross-language consistency where every language packs the same
      source-file membership into ``train`` vs ``validation``.
    """
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    pre_split = args.train_transpiled is not None

    if pre_split:
        train_rows, train_skipped = _build_rows(
            Path(args.train_transpiled),
            Path(args.train_originals),
            args.language,
            tokenizer,
            args.default_license,
            desc="Packaging train",
        )
        val_rows, val_skipped = _build_rows(
            Path(args.validation_transpiled),
            Path(args.validation_originals),
            args.language,
            tokenizer,
            args.default_license,
            desc="Packaging validation",
        )
        print(
            f"Packaged {len(train_rows)} train + {len(val_rows)} validation files, "
            f"skipped {train_skipped + val_skipped}"
        )
        if not train_rows or not val_rows:
            print(
                "Error: pre-split mode requires non-empty train AND validation",
                file=sys.stderr,
            )
            sys.exit(1)
        ds = DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "validation": Dataset.from_list(val_rows),
            }
        )
        split_meta: dict[str, object] = {
            "split_source": "pre-split",
            "train_transpiled": str(args.train_transpiled),
            "train_originals": str(args.train_originals),
            "validation_transpiled": str(args.validation_transpiled),
            "validation_originals": str(args.validation_originals),
        }
        total_files = len(train_rows) + len(val_rows)
        skipped = train_skipped + val_skipped
    else:
        rows, skipped = _build_rows(
            Path(args.transpiled),
            Path(args.originals),
            args.language,
            tokenizer,
            args.default_license,
        )
        print(f"Packaged {len(rows)} files, skipped {skipped}")
        if not rows:
            print("Error: no files packaged", file=sys.stderr)
            sys.exit(1)
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
        split_meta = {
            "split_source": "random",
            "split_ratio": DEFAULT_SPLIT_RATIO,
            "seed": DEFAULT_SEED,
            "transpiled": str(args.transpiled),
            "originals": str(args.originals),
        }
        total_files = len(rows)

    _print_split_stats(ds)

    output = Path(args.output)
    _save_and_push(ds, output, args.push, args.config_name)

    run_meta: dict[str, object] = {
        "language": args.language,
        "tokenizer": args.tokenizer,
        "total_files": total_files,
        "skipped_files": skipped,
        "train_rows": len(ds["train"]),
        "validation_rows": len(ds["validation"]),
        "train_tokens": sum(ds["train"]["token_count"]),
        "validation_tokens": sum(ds["validation"]["token_count"]),
        **split_meta,
    }
    meta_path = output / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
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
        "--transpiled",
        default=None,
        help=(
            "Directory of transpiled Python files (random-split mode). "
            "Mutually exclusive with --train-transpiled / "
            "--validation-transpiled."
        ),
    )
    files.add_argument(
        "--originals",
        default=None,
        help="Directory of original English Python files (random-split mode).",
    )
    files.add_argument(
        "--train-transpiled",
        default=None,
        help=(
            "Directory of transpiled train files (pre-split mode). When set, "
            "--train-originals / --validation-transpiled / "
            "--validation-originals are also required and the script skips "
            "its internal train_test_split — use this to preserve an "
            "upstream split (e.g. cross-language consistency)."
        ),
    )
    files.add_argument(
        "--train-originals",
        default=None,
        help="Directory of original English train files (pre-split mode).",
    )
    files.add_argument(
        "--validation-transpiled",
        default=None,
        help="Directory of transpiled validation files (pre-split mode).",
    )
    files.add_argument(
        "--validation-originals",
        default=None,
        help="Directory of original English validation files (pre-split mode).",
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

    args = parser.parse_args()

    if args.command == "from-files":
        _validate_from_files_args(parser, args)

    return args


def _validate_from_files_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Enforce mode invariants for ``from-files``.

    Two modes:

    * Random split: ``--transpiled`` + ``--originals`` (both required).
    * Pre-split: all four of ``--train-transpiled``, ``--train-originals``,
      ``--validation-transpiled``, ``--validation-originals``.

    Modes are mutually exclusive — mixing flags from both is a user error.
    """
    pre_split_flags = {
        "--train-transpiled": args.train_transpiled,
        "--train-originals": args.train_originals,
        "--validation-transpiled": args.validation_transpiled,
        "--validation-originals": args.validation_originals,
    }
    pre_split_set = [name for name, val in pre_split_flags.items() if val is not None]
    random_set = [
        name
        for name, val in {
            "--transpiled": args.transpiled,
            "--originals": args.originals,
        }.items()
        if val is not None
    ]

    if pre_split_set and random_set:
        parser.error(
            f"from-files: cannot mix random-split flags ({', '.join(random_set)}) "
            f"with pre-split flags ({', '.join(pre_split_set)})."
        )

    if pre_split_set:
        missing = [name for name, val in pre_split_flags.items() if val is None]
        if missing:
            parser.error(
                f"from-files (pre-split mode): missing required flags: "
                f"{', '.join(missing)}."
            )
        return

    if not (args.transpiled and args.originals):
        parser.error(
            "from-files: provide either --transpiled + --originals "
            "(random-split mode) OR all four of --train-transpiled, "
            "--train-originals, --validation-transpiled, "
            "--validation-originals (pre-split mode)."
        )


def main() -> None:
    args = parse_args()
    if args.command == "retokenize":
        retokenize(args)
    elif args.command == "from-files":
        package_from_files(args)


if __name__ == "__main__":
    main()

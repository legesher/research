#!/usr/bin/env python3
"""Analyze natural language distribution in The Stack's Python subset.

Streams Python files from Hugging Face's The Stack dataset, extracts comments
and docstrings, runs language detection, and outputs per-language statistics.

Requirements:
    pip install datasets langdetect huggingface_hub

Usage:
    python analyze_stack_languages.py
    python analyze_stack_languages.py --sample-size 1000
    python analyze_stack_languages.py --sample-size 50000 --output results.csv

Note:
    - Requires accepting The Stack's terms of use on Hugging Face and being
      logged in via `huggingface-cli login`.
    - Default dataset is bigcode/the-stack-dedup (v1). The Stack v2
      (bigcode/the-stack-v2) has a different schema and may require
      adjustments to the data_dir parameter.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
import tokenize
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Make langdetect deterministic
DetectorFactory.seed = 0

# Minimum characters in extracted text to attempt language detection
MIN_TEXT_LENGTH = 50


def extract_comments_and_docstrings(source: str) -> str:
    """Extract comments and docstrings from Python source code.

    Uses Python's tokenize module for comments and ast module for docstrings.
    Returns concatenated natural language text.
    """
    texts: list[str] = []

    # Extract comments via tokenize
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_string, *_ in tokens:
            if tok_type == tokenize.COMMENT:
                comment = tok_string.lstrip("#").strip()
                if comment:
                    texts.append(comment)
    except tokenize.TokenizeError:
        pass

    # Extract docstrings via AST
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    texts.append(docstring)
    except SyntaxError:
        pass

    return "\n".join(texts)


def detect_language(text: str) -> str | None:
    """Detect the natural language of text using langdetect.

    Returns ISO 639-1 language code or None if detection fails.
    """
    if len(text) < MIN_TEXT_LENGTH:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def print_results(
    lang_counts: Counter,
    total_files: int,
    files_with_text: int,
    files_detected: int,
    errors: int,
) -> None:
    """Print formatted results table to stdout."""
    print("\n" + "=" * 65)
    print("  Natural Language Distribution in The Stack (Python subset)")
    print("=" * 65)
    print(f"\n  Files sampled:        {total_files:>8,}")
    print(f"  Files with text:      {files_with_text:>8,}")
    print(f"  Files detected:       {files_detected:>8,}")
    print(f"  Detection failures:   {errors:>8,}")
    print(f"  Below min length:     {files_with_text - files_detected - errors:>8,}")

    if not lang_counts:
        print("\n  No languages detected.")
        return

    total_detected = sum(lang_counts.values())
    if total_detected == 0:
        print("\n  No languages detected.")
        return

    print(f"\n  {'Language':<12} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 12}")

    for lang, count in lang_counts.most_common():
        pct = count / total_detected * 100
        print(f"  {lang:<12} {count:>8,} {pct:>11.2f}%")

    en_count = lang_counts.get("en", 0)
    non_en = total_detected - en_count
    print(f"\n  English:     {en_count:>8,} ({en_count / total_detected * 100:.1f}%)")
    print(f"  Non-English: {non_en:>8,} ({non_en / total_detected * 100:.1f}%)")
    print(f"  Total:       {total_detected:>8,}")
    print("=" * 65)


def save_csv(
    lang_counts: Counter,
    output_path: Path,
) -> None:
    """Save per-language results to CSV file."""
    total_detected = sum(lang_counts.values())
    if total_detected == 0:
        print("\n  No data to save.", file=sys.stderr)
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "count", "percentage"])
        for lang, count in lang_counts.most_common():
            pct = count / total_detected * 100
            writer.writerow([lang, count, f"{pct:.2f}"])

    print(f"\n  Results saved to: {output_path}")


def save_metadata(
    lang_counts: Counter,
    total_files: int,
    files_with_text: int,
    files_detected: int,
    errors: int,
    dataset: str,
    output_path: Path,
) -> None:
    """Save run metadata as JSON sidecar alongside CSV output."""
    total_detected = sum(lang_counts.values())
    meta = {
        "dataset": dataset,
        "total_files_sampled": total_files,
        "files_with_text": files_with_text,
        "files_with_detection": files_detected,
        "detection_failures": errors,
        "below_min_length": files_with_text - files_detected - errors,
        "total_detected": total_detected,
        "english_count": lang_counts.get("en", 0),
        "non_english_count": total_detected - lang_counts.get("en", 0),
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze natural language distribution in The Stack's Python subset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of Python files to sample (default: 10,000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional; metadata saved as .meta.json sidecar)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcode/the-stack-dedup",
        help="Hugging Face dataset ID (default: bigcode/the-stack-dedup)",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (Python subset, streaming)...")
    print(f"Sample size: {args.sample_size:,} files\n")

    try:
        ds = load_dataset(
            args.dataset,
            data_dir="data/python",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        print(
            "\nMake sure you have:\n"
            "  1. Accepted The Stack's terms at https://huggingface.co/datasets/bigcode/the-stack-dedup\n"
            "  2. Logged in via: huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    lang_counts: Counter = Counter()
    total_files = 0
    files_with_text = 0
    files_detected = 0
    errors = 0

    for sample in ds.take(args.sample_size):
        total_files += 1
        content = sample.get("content", "")

        if total_files % 500 == 0:
            print(f"  Processed {total_files:,}/{args.sample_size:,} files...", flush=True)

        text = extract_comments_and_docstrings(content)
        if not text.strip():
            continue
        files_with_text += 1

        lang = detect_language(text)
        if lang is None:
            if len(text) >= MIN_TEXT_LENGTH:
                errors += 1
            continue

        files_detected += 1
        lang_counts[lang] += 1

    print_results(lang_counts, total_files, files_with_text, files_detected, errors)

    if args.output:
        output_path = Path(args.output)
        save_csv(lang_counts, output_path)
        save_metadata(
            lang_counts, total_files, files_with_text, files_detected, errors,
            args.dataset, output_path,
        )


if __name__ == "__main__":
    main()

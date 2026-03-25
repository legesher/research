#!/usr/bin/env python3
"""Re-score XNLI predictions from existing HuggingFace results.

Downloads result JSONs from legesher/language-decoded-experiments, re-applies
the corrected first-line-only label extraction, updates summary accuracy, and
optionally uploads corrected files back to HuggingFace.

Fixes applied:
  1. First-line-only extraction — prevents code leakage on subsequent lines
     from corrupting predictions (e.g. Cond 2-ur Urdu outputs containing
     Legesher keywords like تصدیق(entailment) on line 2).
  2. Expanded native label map — catches Urdu paraphrases like لازم آتی ہے.
  3. Case-insensitive native matching — handles mixed-case model outputs.

Usage:
  # Dry run (download, re-score, print diffs, save locally):
  python rescore_xnli.py

  # Upload corrected files back to HuggingFace:
  python rescore_xnli.py --upload

  # Process only specific conditions:
  python rescore_xnli.py --conditions baseline condition-2-ur-5k

  # Save corrected files to a custom directory:
  python rescore_xnli.py --output-dir ./rescored
"""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"

# All condition directories on HuggingFace
CONDITIONS = [
    "baseline",
    "condition-1-en",
    "condition-1-en-5k",
    "condition-2-zh-5k",
    "condition-2-es-5k",
    "condition-2-ur-5k",
    "condition-3-zh-5k",
]

# Result file names within each condition directory
RESULT_FILES = [
    "english_prompt_results.json",
    "native_prompt_results.json",
]

NATIVE_LABEL_MAP = {
    # Chinese
    "蕴含": "entailment",
    "蕴涵": "entailment",
    "矛盾": "contradiction",
    "中立": "neutral",
    # Spanish
    "implicación": "entailment",
    "implicacion": "entailment",
    "contradicción": "contradiction",
    "contradiccion": "contradiction",
    # Urdu
    "لازمی": "entailment",
    "لازم آتی ہے": "entailment",
    "انضمامیت": "entailment",
    "تردید": "contradiction",
    "غیرجانبدار": "neutral",
}


def extract_xnli_label(text: str) -> str | None:
    """Extract XNLI label from model output using first line only.

    Reads only the first line of output to avoid code leakage on subsequent
    lines (e.g. Legesher keywords containing 'entailment' inside assert calls).
    """
    first_line = text.strip().split("\n")[0].strip()
    first_line_lower = first_line.lower()

    # Try English labels first
    for label in ["entailment", "contradiction", "neutral"]:
        if re.search(rf"\b{label}\b", first_line_lower):
            return label

    # Try native language labels (case-insensitive)
    for native, english in NATIVE_LABEL_MAP.items():
        if native.lower() in first_line_lower:
            return english

    return None


def rescore_xnli_results(data: dict) -> dict:
    """Re-score XNLI entries in a results JSON using corrected extraction.

    Returns a new dict with updated predictions and summary accuracy for XNLI.
    Non-XNLI benchmarks are passed through unchanged.
    """
    data = json.loads(json.dumps(data))  # deep copy

    for benchmark_key in list(data.keys()):
        if "xnli" not in benchmark_key.lower():
            continue

        benchmark = data[benchmark_key]

        # The JSON structure stores per-example results as a flat list
        # directly under the benchmark key (e.g. data["xnli_zh"] = [...]),
        # not nested inside a "results" sub-key.
        if isinstance(benchmark, list):
            results = benchmark
        elif isinstance(benchmark, dict):
            results = benchmark.get("results", benchmark.get("data", []))
        else:
            continue

        if not results:
            continue

        correct = 0
        total = 0
        changed = 0

        for entry in results:
            raw_output = entry.get("raw_output", "")
            if not raw_output:
                continue

            old_pred = entry.get("pred")
            new_pred = extract_xnli_label(raw_output)
            gold = entry.get("gold")

            if new_pred != old_pred:
                changed += 1

            entry["pred"] = new_pred
            entry["correct"] = new_pred == gold

            total += 1
            correct += int(entry["correct"])

        new_accuracy = correct / total if total else 0.0

        # Update summary — keys use "_acc" suffix (e.g. "xnli_zh_acc")
        summary_key = f"{benchmark_key}_acc"
        if "summary" in data and summary_key in data["summary"]:
            old_accuracy = data["summary"][summary_key]
            data["summary"][summary_key] = round(new_accuracy, 6)
        elif isinstance(benchmark, dict) and "accuracy" in benchmark:
            old_accuracy = benchmark["accuracy"]
            benchmark["accuracy"] = round(new_accuracy, 6)
        else:
            old_accuracy = None

        print(
            f"  {benchmark_key}: "
            f"{changed}/{total} predictions changed, "
            f"accuracy {old_accuracy} -> {new_accuracy:.4f}"
        )

    return data


def download_result(condition: str, filename: str, cache_dir: Path) -> Path | None:
    """Download a single result JSON from HuggingFace."""
    # Try multiple path patterns used across conditions
    path_patterns = [
        f"conditions/{condition}/results/{filename}",
        f"conditions/{condition}/{filename}",
        f"{condition}/results/{filename}",
        f"{condition}/{filename}",
    ]

    for path in path_patterns:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=path,
                repo_type=REPO_TYPE,
                cache_dir=str(cache_dir / ".hf_cache"),
            )
            return Path(local_path)
        except Exception:
            continue

    return None


def process_condition(condition: str, cache_dir: Path, output_dir: Path) -> list[dict]:
    """Download, re-score, and save results for one condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    changes = []

    for filename in RESULT_FILES:
        local_path = download_result(condition, filename, cache_dir)
        if local_path is None:
            print(f"  {filename}: not found, skipping")
            continue

        print(f"\n  Processing {filename}:")

        with open(local_path, encoding="utf-8") as f:
            data = json.load(f)

        rescored = rescore_xnli_results(data)

        # Save locally
        out_path = output_dir / condition / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rescored, f, indent=2, ensure_ascii=False)
            f.write("\n")

        changes.append(
            {
                "condition": condition,
                "filename": filename,
                "local_path": str(out_path),
                "hf_path": f"conditions/{condition}/results/{filename}",
            }
        )

    return changes


def upload_results(changes: list[dict]) -> None:
    """Upload re-scored results back to HuggingFace."""
    api = HfApi()

    print(f"\n{'='*60}")
    print("Uploading to HuggingFace...")
    print(f"{'='*60}")

    for change in changes:
        print(f"  Uploading {change['hf_path']}...")
        api.upload_file(
            path_or_fileobj=change["local_path"],
            path_in_repo=change["hf_path"],
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=(
                f"rescore: fix XNLI label extraction for {change['condition']}\n\n"
                "Applied first-line-only extraction to prevent code leakage\n"
                "corruption and expanded native label map for Urdu paraphrases."
            ),
        )

    print(f"\nUploaded {len(changes)} files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score XNLI predictions from HuggingFace results."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Specific conditions to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rescored_results"),
        help="Directory to save re-scored JSONs (default: ./rescored_results)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload corrected files back to HuggingFace",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = args.conditions or CONDITIONS
    output_dir = args.output_dir.resolve()
    cache_dir = Path.home() / ".cache" / "rescore_xnli"

    # Validate condition names
    for cond in conditions:
        if cond not in CONDITIONS:
            print(f"Warning: '{cond}' not in known conditions: {CONDITIONS}")

    all_changes = []
    for condition in conditions:
        changes = process_condition(condition, cache_dir, output_dir)
        all_changes.extend(changes)

    print(f"\n{'='*60}")
    print(f"Re-scored {len(all_changes)} files -> {output_dir}")
    print(f"{'='*60}")

    if args.upload:
        if not all_changes:
            print("No files to upload.")
            return
        upload_results(all_changes)
    else:
        print(
            "\nDry run complete. Use --upload to push corrected files to HuggingFace."
        )


if __name__ == "__main__":
    main()

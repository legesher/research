#!/usr/bin/env python3
"""Stress test Legesher's transpiler on real-world Python files from BigCode datasets.

Streams Python files from HuggingFace BigCode datasets (the-stack-dedup, starcoderdata,
the-stack, etc.), translates each with both TokenTranslator and TreeSitterTranslator
backends, validates round-trip correctness, and produces a structured report.

Requirements:
    pip install datasets huggingface_hub psutil
    # Plus legesher-core and legesher-i18n from workspace

Usage:
    python stress_test_transpiler.py
    python stress_test_transpiler.py --sample-size 100 --language zh
    python stress_test_transpiler.py --sample-size 1000 --report results.json
    python stress_test_transpiler.py --backends token --skip-roundtrip
    python stress_test_transpiler.py --dataset bigcode/starcoderdata --data-dir python
    python stress_test_transpiler.py --dataset bigcode/the-stack-v2-dedup --config Python

Note: Requires accepting the dataset's terms of use on HuggingFace and being
logged in via `huggingface-cli login` or setting HF_TOKEN.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import statistics
import sys
import time
import tokenize
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import legesher_core
from datasets import load_dataset

from legesher_core import TokenTranslator, TreeSitterTranslator


@dataclass
class FileResult:
    """Result of translating a single file with one backend."""

    file_index: int
    file_size_bytes: int
    line_count: int
    backend: str
    language: str
    forward_time_us: float = 0.0
    reverse_time_us: float | None = None
    translated_different: bool = False
    translatable_tokens: int = 0
    round_trip_match: str = "skipped"  # exact, mismatch, error, skipped
    mismatch_first_line: int | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class FilterStats:
    """Statistics about dataset filtering."""

    total_streamed: int = 0
    accepted: int = 0
    syntax_error: int = 0
    too_short: int = 0
    too_large: int = 0
    encoding_error: int = 0


@dataclass
class StressTestReport:
    """Full stress test report."""

    timestamp: str = ""
    dataset: str = ""
    language: str = ""
    backends: list[str] = field(default_factory=list)
    sample_size: int = 0
    filter_stats: dict = field(default_factory=dict)
    backend_results: dict = field(default_factory=dict)
    divergence_count: int = 0
    divergence_examples: list[int] = field(default_factory=list)
    memory_peak_mb: float = 0.0
    wall_time_sec: float = 0.0
    legesher_core_version: str = ""
    python_version: str = ""


def count_translatable_tokens(
    content: str, keyword_map: dict[str, str], builtin_map: dict[str, str]
) -> int:
    """Count NAME tokens in source that match a keyword or builtin."""
    count = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        for tok_type, tok_string, *_ in tokens:
            if tok_type == tokenize.NAME:
                if tok_string in keyword_map or tok_string in builtin_map:
                    count += 1
    except tokenize.TokenError:
        pass
    return count


def find_first_diff_line(original: str, result: str) -> int | None:
    """Find the first line number where two strings differ."""
    orig_lines = original.splitlines()
    result_lines = result.splitlines()
    for i, (a, b) in enumerate(zip(orig_lines, result_lines)):
        if a != b:
            return i + 1
    if len(orig_lines) != len(result_lines):
        return min(len(orig_lines), len(result_lines)) + 1
    return None


def stream_python_files(
    dataset_id: str,
    sample_size: int,
    min_lines: int,
    max_bytes: int,
    data_dir: str = "data/python",
    config_name: str | None = None,
):
    """Stream and filter Python files from a HuggingFace code dataset.

    Yields dicts with content, index, size_bytes, line_count.
    Also returns filter stats via the FilterStats object.
    """
    if config_name:
        print(f"Loading dataset: {dataset_id} (config={config_name}, streaming)...")
    else:
        print(f"Loading dataset: {dataset_id} (data_dir={data_dir}, streaming)...")
    print(f"Target: {sample_size:,} valid files (min {min_lines} lines, max {max_bytes:,} bytes)\n")

    try:
        load_kwargs: dict = dict(
            split="train",
            streaming=True,
            trust_remote_code=False,  # Security: block remote code execution
        )
        if config_name:
            load_kwargs["name"] = config_name
        else:
            load_kwargs["data_dir"] = data_dir
        ds = load_dataset(dataset_id, **load_kwargs)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        print(
            "\nMake sure you have:\n"
            "  1. Accepted The Stack's terms at "
            "https://huggingface.co/datasets/bigcode/the-stack-dedup\n"
            "  2. Logged in via: huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = FilterStats()
    accepted = 0

    for sample in ds:
        stats.total_streamed += 1
        content = sample.get("content", "")

        # Size check
        size_bytes = len(content.encode("utf-8", errors="replace"))
        if size_bytes > max_bytes:
            stats.too_large += 1
            continue

        # Line count check
        lines = content.splitlines()
        line_count = len(lines)
        if line_count < min_lines:
            stats.too_short += 1
            continue

        # Valid Python check
        try:
            ast.parse(content)
        except SyntaxError:
            stats.syntax_error += 1
            continue
        except (ValueError, TypeError):
            stats.encoding_error += 1
            continue

        accepted += 1
        stats.accepted = accepted
        yield {
            "content": content,
            "index": accepted,
            "size_bytes": size_bytes,
            "line_count": line_count,
        }, stats

        if accepted >= sample_size:
            break

    if accepted < sample_size:
        print(
            f"\n  Warning: Only found {accepted} valid files "
            f"after streaming {stats.total_streamed:,}",
            file=sys.stderr,
        )


def setup_translators(
    language: str, backends: list[str]
) -> dict[str, tuple]:
    """Create forward + reverse translators for requested backends."""
    translators = {}
    if "token" in backends:
        fwd = TokenTranslator.from_language_pack(language)
        rev = fwd.reverse()
        translators["token"] = (fwd, rev)
    if "tree-sitter" in backends:
        fwd = TreeSitterTranslator.from_language_pack(language)
        rev = fwd.reverse()
        translators["tree-sitter"] = (fwd, rev)
    return translators


def translate_and_validate(
    content: str,
    fwd,
    rev,
    backend_name: str,
    language: str,
    file_index: int,
    file_size_bytes: int,
    line_count: int,
    keyword_map: dict[str, str],
    builtin_map: dict[str, str],
    skip_roundtrip: bool = False,
) -> tuple[FileResult, str | None]:
    """Translate a file and validate round-trip correctness."""
    result = FileResult(
        file_index=file_index,
        file_size_bytes=file_size_bytes,
        line_count=line_count,
        backend=backend_name,
        language=language,
    )

    # Count translatable tokens
    result.translatable_tokens = count_translatable_tokens(
        content, keyword_map, builtin_map
    )

    # Forward translation
    translated = None
    try:
        t0 = time.perf_counter_ns()
        translated = fwd.translate_code(content, "en", language)
        t1 = time.perf_counter_ns()
        result.forward_time_us = (t1 - t0) / 1_000
    except Exception as e:
        result.error_type = type(e).__name__
        result.error_message = str(e)[:200]
        result.forward_time_us = 0
        return result, None

    # Check if translation actually changed the code
    result.translated_different = translated != content

    # Round-trip validation
    if skip_roundtrip:
        result.round_trip_match = "skipped"
        return result, translated

    try:
        t0 = time.perf_counter_ns()
        round_tripped = rev.translate_code(translated, language, "en")
        t1 = time.perf_counter_ns()
        result.reverse_time_us = (t1 - t0) / 1_000

        if round_tripped == content:
            result.round_trip_match = "exact"
        else:
            result.round_trip_match = "mismatch"
            result.mismatch_first_line = find_first_diff_line(
                content, round_tripped
            )
    except Exception as e:
        result.round_trip_match = "error"
        result.error_type = type(e).__name__
        result.error_message = str(e)[:200]

    return result, translated


def compute_percentiles(values: list[float]) -> dict:
    """Compute timing statistics for a list of values."""
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "min": values_sorted[0],
        "median": statistics.median(values_sorted),
        "mean": statistics.mean(values_sorted),
        "p95": values_sorted[int(n * 0.95)] if n >= 20 else values_sorted[-1],
        "p99": values_sorted[int(n * 0.99)] if n >= 100 else values_sorted[-1],
        "max": values_sorted[-1],
    }


def aggregate_backend_results(results: list[FileResult]) -> dict:
    """Aggregate results for a single backend."""
    total = len(results)
    errors = [r for r in results if r.error_type is not None]
    successes = [r for r in results if r.error_type is None]
    translated = [r for r in successes if r.translated_different]
    silent_failures = [
        r for r in successes
        if not r.translated_different and r.translatable_tokens > 0
    ]

    forward_times = [r.forward_time_us for r in successes]
    reverse_times = [
        r.reverse_time_us for r in successes if r.reverse_time_us is not None
    ]

    round_trip_exact = sum(1 for r in successes if r.round_trip_match == "exact")
    round_trip_mismatch = sum(1 for r in successes if r.round_trip_match == "mismatch")
    round_trip_error = sum(1 for r in successes if r.round_trip_match == "error")
    round_trip_skipped = sum(1 for r in successes if r.round_trip_match == "skipped")

    total_forward_sec = sum(forward_times) / 1_000_000 if forward_times else 0

    error_breakdown = {}
    for r in errors:
        key = r.error_type or "unknown"
        error_breakdown[key] = error_breakdown.get(key, 0) + 1

    return {
        "total_files": total,
        "success_count": len(successes),
        "error_count": len(errors),
        "error_breakdown": error_breakdown,
        "translated_count": len(translated),
        "silent_failure_count": len(silent_failures),
        "silent_failure_indices": [r.file_index for r in silent_failures[:10]],
        "throughput_files_per_sec": (
            len(successes) / total_forward_sec if total_forward_sec > 0 else 0
        ),
        "forward_time_us": compute_percentiles(forward_times),
        "reverse_time_us": compute_percentiles(reverse_times),
        "round_trip": {
            "exact": round_trip_exact,
            "mismatch": round_trip_mismatch,
            "error": round_trip_error,
            "skipped": round_trip_skipped,
            "exact_pct": (
                round_trip_exact / len(successes) * 100 if successes else 0
            ),
        },
    }


def print_backend_summary(name: str, stats: dict) -> None:
    """Print summary for one backend."""
    print(f"\n  Backend: {name}")
    print(f"  {'─' * 45}")
    print(
        f"    Success rate:       "
        f"{stats['success_count']}/{stats['total_files']} "
        f"({stats['success_count'] / stats['total_files'] * 100:.1f}%)"
    )
    print(
        f"    Translated:         "
        f"{stats['translated_count']}/{stats['success_count']}"
    )
    if stats["silent_failure_count"] > 0:
        print(
            f"    Silent failures:    "
            f"{stats['silent_failure_count']} ⚠"
        )
    print(
        f"    Throughput:         "
        f"{stats['throughput_files_per_sec']:,.0f} files/sec"
    )

    ft = stats["forward_time_us"]
    if ft:
        print(
            f"    Forward time:       "
            f"min={ft['min']:.0f}μs  "
            f"median={ft['median']:.0f}μs  "
            f"p95={ft['p95']:.0f}μs  "
            f"max={ft['max']:.0f}μs"
        )

    rt = stats["round_trip"]
    if rt["skipped"] < stats["success_count"]:
        evaluated = rt["exact"] + rt["mismatch"] + rt["error"]
        print(
            f"    Round-trip exact:   "
            f"{rt['exact']}/{evaluated} ({rt['exact_pct']:.1f}%)"
        )
        if rt["mismatch"] > 0:
            print(f"    Round-trip mismatch: {rt['mismatch']}")
        if rt["error"] > 0:
            print(f"    Round-trip error:   {rt['error']}")

    if stats["error_count"] > 0:
        print(f"    Errors:             {stats['error_breakdown']}")


def print_report(
    filter_stats: FilterStats,
    report: StressTestReport,
) -> None:
    """Print full report to console."""
    print("\n" + "=" * 65)
    print("  Legesher Transpiler Stress Test Results")
    print("=" * 65)

    fs = filter_stats
    print(f"\n  Dataset:              {report.dataset} (Python)")
    print(f"  Files processed:      {fs.accepted:,}")
    print(f"  Files streamed:       {fs.total_streamed:,}")
    print(
        f"  Files filtered:       {fs.total_streamed - fs.accepted:,} "
        f"(syntax: {fs.syntax_error}, too_short: {fs.too_short}, "
        f"too_large: {fs.too_large}, encoding: {fs.encoding_error})"
    )

    for backend_name, stats in report.backend_results.items():
        print_backend_summary(backend_name, stats)

    if report.divergence_count > 0:
        print(f"\n  Backend divergence:   {report.divergence_count} files")
        if report.divergence_examples:
            print(
                f"    Example indices:    {report.divergence_examples[:5]}"
            )

    print(f"\n  Memory peak:          {report.memory_peak_mb:.1f} MB")
    print(f"  Wall time:            {report.wall_time_sec:.1f} sec")
    print("=" * 65)


def save_report(
    report: StressTestReport,
    results_by_backend: dict[str, list[FileResult]],
    report_path: Path,
) -> None:
    """Save full report as JSON."""
    output = {
        "summary": {
            "timestamp": report.timestamp,
            "dataset": report.dataset,
            "language": report.language,
            "backends": report.backends,
            "sample_size": report.sample_size,
            "filter_stats": report.filter_stats,
            "memory_peak_mb": report.memory_peak_mb,
            "wall_time_sec": report.wall_time_sec,
            "divergence_count": report.divergence_count,
            "divergence_examples": report.divergence_examples,
            "legesher_core_version": report.legesher_core_version,
            "python_version": report.python_version,
        },
        "backend_results": report.backend_results,
        "per_file_results": {
            backend: [asdict(r) for r in file_results]
            for backend, file_results in results_by_backend.items()
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Report saved to: {report_path}")


def save_translated_files(
    output_dir: Path,
    translated_files: dict[str, dict[int, str]],
) -> None:
    """Save translated files to output directory."""
    for backend, files in translated_files.items():
        backend_dir = output_dir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        for idx, content in files.items():
            (backend_dir / f"file_{idx:04d}.py").write_text(
                content, encoding="utf-8"
            )
    print(f"  Translated files saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress test Legesher's transpiler on real-world Python files "
            "from The Stack dataset."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of valid Python files to process (default: 1000)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Target language code (default: zh)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="both",
        choices=["token", "tree-sitter", "both"],
        help="Which backends to test (default: both)",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines for a file to be included (default: 5)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=100_000,
        help="Maximum file size in bytes (default: 100000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save translated files (optional)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path for JSON report (default: research/reports/stress_test_{dataset}_{lang}_{size}.json)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcode/the-stack-dedup",
        help="Hugging Face dataset ID (default: bigcode/the-stack-dedup)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/python",
        help="Dataset subdirectory for Python files (default: data/python)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name (e.g., 'Python' for the-stack-v2-dedup). Overrides --data-dir.",
    )
    parser.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="Skip round-trip validation (faster)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Default report path includes dataset, language and sample size
    if args.report is None:
        reports_dir = Path("research/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        # Include dataset short name if not the default
        dataset_name = args.dataset.split("/")[-1]
        dataset_slug = f"{dataset_name}_" if dataset_name != "the-stack-dedup" else ""
        args.report = str(reports_dir / f"stress_test_{dataset_slug}{args.language}_{args.sample_size}.json")

    # Determine backends
    if args.backends == "both":
        backends = ["token", "tree-sitter"]
    else:
        backends = [args.backends]

    # Start memory tracking
    tracemalloc.start()
    wall_start = time.monotonic()

    # Setup translators
    print(f"Setting up translators for '{args.language}' ({', '.join(backends)})...")
    translators = setup_translators(args.language, backends)

    # Get keyword/builtin maps from the first translator for token counting
    first_backend = next(iter(translators.values()))
    fwd_translator = first_backend[0]
    keyword_map = fwd_translator.mapper.keyword_map
    builtin_map = fwd_translator.mapper.builtin_map
    print(
        f"  Loaded: {len(keyword_map)} keywords, {len(builtin_map)} builtins\n"
    )

    # Storage for results
    results_by_backend: dict[str, list[FileResult]] = {
        b: [] for b in backends
    }
    translated_outputs: dict[str, dict[int, str]] = {
        b: {} for b in backends
    }
    divergence_count = 0
    divergence_examples: list[int] = []
    final_filter_stats = FilterStats()

    # Process files
    for file_data, filter_stats in stream_python_files(
        args.dataset, args.sample_size, args.min_lines, args.max_bytes,
        data_dir=args.data_dir,
        config_name=args.dataset_config,
    ):
        final_filter_stats = filter_stats
        content = file_data["content"]
        idx = file_data["index"]
        size_bytes = file_data["size_bytes"]
        line_count = file_data["line_count"]

        if idx % 100 == 0 or idx <= 10:
            _, peak = tracemalloc.get_traced_memory()
            print(
                f"  [{idx:>{len(str(args.sample_size))}}/{args.sample_size}] "
                f"Processing... (peak mem: {peak / 1_048_576:.1f} MB)",
                flush=True,
            )

        # Translate with each backend
        backend_translations = {}
        for backend_name, (fwd, rev) in translators.items():
            result, translated = translate_and_validate(
                content=content,
                fwd=fwd,
                rev=rev,
                backend_name=backend_name,
                language=args.language,
                file_index=idx,
                file_size_bytes=size_bytes,
                line_count=line_count,
                keyword_map=keyword_map,
                builtin_map=builtin_map,
                skip_roundtrip=args.skip_roundtrip,
            )
            results_by_backend[backend_name].append(result)

            # Store translated output for divergence check and optional save
            if translated is not None:
                backend_translations[backend_name] = translated
                if args.output_dir:
                    translated_outputs[backend_name][idx] = translated

            if args.verbose and result.error_type:
                print(
                    f"    [{backend_name}] Error on file {idx}: "
                    f"{result.error_type}: {result.error_message}"
                )

        # Check backend divergence (only if both backends ran)
        if len(backend_translations) == 2:
            vals = list(backend_translations.values())
            if vals[0] != vals[1]:
                divergence_count += 1
                if len(divergence_examples) < 20:
                    divergence_examples.append(idx)

    # Gather final metrics
    wall_time = time.monotonic() - wall_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Build report
    report = StressTestReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        dataset=args.dataset,
        language=args.language,
        backends=backends,
        sample_size=args.sample_size,
        filter_stats=asdict(final_filter_stats),
        memory_peak_mb=peak_memory / 1_048_576,
        wall_time_sec=wall_time,
        divergence_count=divergence_count,
        divergence_examples=divergence_examples,
        legesher_core_version=legesher_core.__version__,
        python_version=sys.version,
    )

    # Aggregate per-backend
    for backend_name, file_results in results_by_backend.items():
        report.backend_results[backend_name] = aggregate_backend_results(
            file_results
        )

    # Output
    print_report(final_filter_stats, report)
    save_report(report, results_by_backend, Path(args.report))

    if args.output_dir:
        save_translated_files(Path(args.output_dir), translated_outputs)

    # Force exit — HuggingFace datasets keeps background threads alive after streaming
    sys.exit(0)


if __name__ == "__main__":
    main()

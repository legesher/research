#!/usr/bin/env python3
"""Batch transpilation wrapper for Legesher — Language Decoded project.

Wraps Legesher's translators to process large corpora (50K+ files) with
multiprocessing, error logging, retry logic, and checkpoint/resume support.
Supports token-based, tree-sitter, and dual-backend comparison modes.

Requirements:
    pip install legesher-core legesher-i18n tqdm

Usage:
    # Transpile a directory of Python files to Chinese
    python batch_transpile.py ./source-python zh --output ./transpiled-zh

    # Transpile to multiple languages
    python batch_transpile.py ./source-python zh,am,ur --output ./transpiled

    # Resume after interruption
    python batch_transpile.py ./source-python zh --output ./transpiled-zh --resume

    # Stream from HuggingFace dataset
    python batch_transpile.py --hf-dataset bigcode/the-stack-dedup zh --output ./transpiled-zh --hf-limit 50000

    # Customize workers and batch size
    python batch_transpile.py ./source-python zh --output ./transpiled-zh --workers 8 --batch-size 500

    # Use tree-sitter backend
    python batch_transpile.py ./source-python zh --output ./transpiled-zh --backend tree-sitter

    # Compare both backends
    python batch_transpile.py ./source-python zh --output ./transpiled-zh --backend both

    # Enable pre-transpilation syntax validation
    python batch_transpile.py ./source-python zh --output ./transpiled-zh --validate-syntax

    # Process languages in parallel
    python batch_transpile.py ./source-python zh,am,ur --output ./transpiled --parallel-languages
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from legesher_core import create_translator_from_language_pack
from legesher_core.exceptions import TranslationError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranspileResult:
    """Result of transpiling a single file."""

    file_path: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    retries: int = 0
    elapsed_ms: float = 0.0
    backend: Optional[str] = None
    backends_match: Optional[bool] = None


@dataclass
class BatchStats:
    """Aggregate stats for a batch run."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    retried: int = 0
    elapsed_sec: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Worker globals (set via pool initializer, avoids per-file translator creation)
# ---------------------------------------------------------------------------

_worker_translator = None       # primary backend
_worker_translator_ts = None    # tree-sitter (only for --backend both)
_worker_max_retries: int = 3
_worker_max_file_size: int = 1_048_576
_worker_min_file_size: int = 10
_worker_validate_syntax: bool = False
_worker_backend: str = "token"


def _init_worker(
    target_lang: str,
    backend: str,
    max_file_size: int,
    min_file_size: int,
    validate_syntax: bool,
    max_retries: int,
) -> None:
    """Pool initializer — creates translators once per worker process."""
    global _worker_translator, _worker_translator_ts
    global _worker_max_retries, _worker_max_file_size, _worker_min_file_size
    global _worker_validate_syntax, _worker_backend

    _worker_max_retries = max_retries
    _worker_max_file_size = max_file_size
    _worker_min_file_size = min_file_size
    _worker_validate_syntax = validate_syntax
    _worker_backend = backend

    if backend in ("token", "both"):
        _worker_translator = create_translator_from_language_pack(
            "python", target_lang, use_tree_sitter=False,
        )
    if backend in ("tree-sitter", "both"):
        _worker_translator_ts = create_translator_from_language_pack(
            "python", target_lang, use_tree_sitter=True,
        )
    if backend == "tree-sitter":
        _worker_translator = _worker_translator_ts
        _worker_translator_ts = None


# ---------------------------------------------------------------------------
# Pre-transpilation sanitization
# ---------------------------------------------------------------------------

def _sanitize_content(content: str, file_path: str) -> tuple[str, Optional[str]]:
    """Sanitize file content before transpilation.

    Returns (cleaned_content, skip_reason). skip_reason is None if OK.
    Uses worker-level config globals for thresholds.
    """
    # Null byte check — likely a binary file
    if "\x00" in content:
        return "", "binary file (contains null bytes)"

    # Newline normalization
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Optional AST validation
    if _worker_validate_syntax:
        try:
            ast.parse(content, filename=file_path)
        except SyntaxError as e:
            return "", f"syntax error: {e}"

    return content, None


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _safe_output_path(source: str, input_base: Path, output_base: Path) -> Optional[Path]:
    """Compute output path and verify it stays within output_base.

    Returns the resolved output Path, or None if traversal is detected.
    """
    source_path = Path(source).resolve()
    input_base_resolved = input_base.resolve()
    output_base_resolved = output_base.resolve()

    try:
        rel = source_path.relative_to(input_base_resolved)
    except ValueError:
        return None

    out = (output_base_resolved / rel).resolve()
    if not out.is_relative_to(output_base_resolved):
        return None

    return out


# ---------------------------------------------------------------------------
# Worker function (runs in child process)
# ---------------------------------------------------------------------------

def _transpile_file(args: tuple) -> TranspileResult:
    """Transpile a single file. Designed to run in a multiprocessing pool.

    For 'both' mode, args is:
        (source_path, token_output_path, ts_output_path)
    Otherwise:
        (source_path, output_path)

    Worker config (backend, retries, etc.) comes from pool-initializer globals.
    """
    if _worker_backend == "both":
        source_path, token_output_path, ts_output_path = args
        return _transpile_file_both(source_path, token_output_path, ts_output_path)
    else:
        source_path, output_path = args
        return _transpile_file_single(source_path, output_path)


def _transpile_file_single(source_path: str, output_path: str) -> TranspileResult:
    """Transpile with a single backend (token or tree-sitter)."""
    start = time.perf_counter()

    if _worker_translator is None:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="InitError",
            error_message="Worker translator not initialized",
            elapsed_ms=elapsed,
            backend=_worker_backend,
        )

    # File size checks (before reading content)
    try:
        file_size = os.path.getsize(source_path)
    except OSError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="OSError",
            error_message=str(e),
            elapsed_ms=elapsed,
            backend=_worker_backend,
        )

    if file_size > _worker_max_file_size:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="FileTooLarge",
            error_message=f"File size {file_size} exceeds max {_worker_max_file_size}",
            elapsed_ms=elapsed,
            backend=_worker_backend,
        )

    if file_size < _worker_min_file_size:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="FileTooSmall",
            error_message=f"File size {file_size} below min {_worker_min_file_size}",
            elapsed_ms=elapsed,
            backend=_worker_backend,
        )

    last_error = None
    retries = 0
    for attempt in range(1, _worker_max_retries + 1):
        try:
            source_code = Path(source_path).read_text(encoding="utf-8")

            # Sanitize content
            source_code, skip_reason = _sanitize_content(source_code, source_path)
            if skip_reason is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return TranspileResult(
                    file_path=source_path,
                    success=False,
                    error_type="Sanitization",
                    error_message=skip_reason,
                    elapsed_ms=elapsed,
                    backend=_worker_backend,
                )

            translated = str(_worker_translator.translate_code(source_code, "en", _worker_backend.replace("-", "_") if _worker_backend == "tree-sitter" else _worker_backend))

            # Empty output detection
            if not translated or not translated.strip():
                elapsed = (time.perf_counter() - start) * 1000
                return TranspileResult(
                    file_path=source_path,
                    success=False,
                    error_type="EmptyOutput",
                    error_message="Translation produced empty/whitespace-only output",
                    elapsed_ms=elapsed,
                    backend=_worker_backend,
                )

            # Write output
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(translated, encoding="utf-8")

            elapsed = (time.perf_counter() - start) * 1000
            return TranspileResult(
                file_path=source_path,
                success=True,
                retries=attempt - 1,
                elapsed_ms=elapsed,
                backend=_worker_backend,
            )

        except UnicodeDecodeError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TranspileResult(
                file_path=source_path,
                success=False,
                error_type="UnicodeDecodeError",
                error_message=str(e),
                elapsed_ms=elapsed,
                backend=_worker_backend,
            )

        except (TranslationError, SyntaxError) as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TranspileResult(
                file_path=source_path,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                elapsed_ms=elapsed,
                backend=_worker_backend,
            )

        except Exception as e:
            last_error = e
            retries = attempt
            continue

    # All retries exhausted
    elapsed = (time.perf_counter() - start) * 1000
    return TranspileResult(
        file_path=source_path,
        success=False,
        error_type=type(last_error).__name__,
        error_message=str(last_error),
        error_traceback=traceback.format_exc(),
        retries=retries,
        elapsed_ms=elapsed,
        backend=_worker_backend,
    )


def _transpile_file_both(
    source_path: str,
    token_output_path: str,
    ts_output_path: str,
) -> TranspileResult:
    """Transpile with BOTH backends, write to separate dirs, compare outputs."""
    start = time.perf_counter()

    if _worker_translator is None or _worker_translator_ts is None:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="InitError",
            error_message="Worker translators not initialized for both-mode",
            elapsed_ms=elapsed,
            backend="both",
        )

    # File size checks
    try:
        file_size = os.path.getsize(source_path)
    except OSError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="OSError",
            error_message=str(e),
            elapsed_ms=elapsed,
            backend="both",
        )

    if file_size > _worker_max_file_size:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="FileTooLarge",
            error_message=f"File size {file_size} exceeds max {_worker_max_file_size}",
            elapsed_ms=elapsed,
            backend="both",
        )

    if file_size < _worker_min_file_size:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="FileTooSmall",
            error_message=f"File size {file_size} below min {_worker_min_file_size}",
            elapsed_ms=elapsed,
            backend="both",
        )

    # Read and sanitize once
    try:
        source_code = Path(source_path).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="UnicodeDecodeError",
            error_message=str(e),
            elapsed_ms=elapsed,
            backend="both",
        )

    source_code, skip_reason = _sanitize_content(source_code, source_path)
    if skip_reason is not None:
        elapsed = (time.perf_counter() - start) * 1000
        return TranspileResult(
            file_path=source_path,
            success=False,
            error_type="Sanitization",
            error_message=skip_reason,
            elapsed_ms=elapsed,
            backend="both",
        )

    # --- Token backend ---
    token_ok = True
    token_result = ""
    ts_error_msg: Optional[str] = None
    try:
        token_result = str(_worker_translator.translate_code(source_code, "en", "token"))
        if not token_result or not token_result.strip():
            token_ok = False
            token_result = ""
        else:
            out = Path(token_output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(token_result, encoding="utf-8")
    except Exception as e:
        token_ok = False
        token_result = ""

    # --- Tree-sitter backend ---
    ts_ok = True
    ts_result = ""
    try:
        ts_result = str(_worker_translator_ts.translate_code(source_code, "en", "tree_sitter"))
        if not ts_result or not ts_result.strip():
            ts_ok = False
            ts_result = ""
        else:
            out = Path(ts_output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(ts_result, encoding="utf-8")
    except Exception as e:
        ts_ok = False
        ts_error_msg = str(e)

    # Compare
    outputs_match = (token_ok and ts_ok and token_result == ts_result)

    elapsed = (time.perf_counter() - start) * 1000

    # At least one backend must succeed
    overall_success = token_ok or ts_ok

    return TranspileResult(
        file_path=source_path,
        success=overall_success,
        error_type=None if overall_success else "BothFailed",
        error_message=ts_error_msg if not overall_success else None,
        elapsed_ms=elapsed,
        backend="both",
        backends_match=outputs_match,
    )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-processed file paths from checkpoint."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        return {line.strip() for line in f if line.strip()}


def _append_checkpoint(checkpoint_path: Path, file_path: str) -> None:
    """Append a successfully processed file to checkpoint."""
    with open(checkpoint_path, "a") as f:
        f.write(file_path + "\n")


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def _init_error_log(error_log_path: Path) -> None:
    """Initialize error CSV with headers."""
    with open(error_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "error_type", "error_message", "retries", "traceback"])


def _append_error_log(error_log_path: Path, result: TranspileResult) -> None:
    """Append a failed result to the error CSV."""
    with open(error_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.file_path,
            result.error_type,
            result.error_message,
            result.retries,
            result.error_traceback or "",
        ])


# ---------------------------------------------------------------------------
# Comparison CSV (for --backend both)
# ---------------------------------------------------------------------------

def _init_comparison_csv(path: Path) -> None:
    """Initialize comparison CSV with headers."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "token_ok", "treesitter_ok", "outputs_match", "treesitter_error"])


def _append_comparison_csv(path: Path, result: TranspileResult) -> None:
    """Append a comparison row for a both-mode result."""
    # Determine per-backend success from the result fields
    # In both mode, success=True means at least one backend worked
    token_ok = result.success  # conservative; real detail is in backends_match
    ts_ok = result.success
    ts_error = ""

    if result.error_type == "BothFailed":
        token_ok = False
        ts_ok = False
        ts_error = result.error_message or ""
    elif result.backends_match is False and result.success:
        # At least one succeeded but they differ — we don't track which failed
        # in the result, so mark both as OK (outputs just differ)
        token_ok = True
        ts_ok = True

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result.file_path,
            token_ok,
            ts_ok,
            result.backends_match if result.backends_match is not None else "",
            ts_error,
        ])


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_local_files(input_dir: Path) -> list[Path]:
    """Discover all .py files in a directory recursively.

    Resolves symlinks and rejects any that point outside the input directory.
    """
    input_resolved = input_dir.resolve()
    results: list[Path] = []
    for p in sorted(input_dir.rglob("*.py")):
        resolved = p.resolve()
        if not resolved.is_relative_to(input_resolved):
            print(f"  Warning: skipping symlink escaping input dir: {p}", file=sys.stderr)
            continue
        results.append(p)
    return results


def discover_hf_files(
    dataset_id: str,
    output_dir: Path,
    limit: int,
    data_dir: str = "data/python",
) -> list[Path]:
    """Stream files from a HuggingFace dataset and save locally.

    Returns list of saved file paths.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: `datasets` package required for HF streaming.", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming {limit:,} files from {dataset_id}...")
    ds = load_dataset(dataset_id, data_dir=data_dir, split="train", streaming=True)

    source_dir = output_dir / "_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Content size limit: reject files larger than 10 MB to prevent disk exhaustion
    max_content_bytes = 10 * 1024 * 1024

    saved: list[Path] = []
    for i, sample in enumerate(tqdm(ds.take(limit), total=limit, desc="Downloading")):
        content = sample.get("content", "")

        # Guard against oversized entries that could exhaust disk
        if len(content.encode("utf-8", errors="replace")) > max_content_bytes:
            print(f"  Skipping oversized entry ({len(content):,} chars)", file=sys.stderr)
            continue

        # Sanitize hexsha: allow only alphanumeric chars to prevent path traversal
        raw_hexsha = sample.get("hexsha", f"file_{i:06d}")
        hexpath = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw_hexsha))
        if not hexpath:
            hexpath = f"file_{i:06d}"

        file_path = source_dir / f"{hexpath}.py"
        file_path.write_text(content, encoding="utf-8")
        saved.append(file_path)

    print(f"Downloaded {len(saved):,} files to {source_dir}")
    return saved


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------

def run_batch(
    files: list[Path],
    target_lang: str,
    output_dir: Path,
    input_base: Path,
    workers: int = 4,
    batch_size: int = 200,
    max_retries: int = 3,
    resume: bool = False,
    backend: str = "token",
    max_file_size: int = 1_048_576,
    min_file_size: int = 10,
    validate_syntax: bool = False,
) -> BatchStats:
    """Run batch transpilation with multiprocessing.

    Args:
        files: List of source Python file paths
        target_lang: Target language code (e.g., "zh", "am", "ur")
        output_dir: Root output directory
        input_base: Base directory for calculating relative paths
        workers: Number of multiprocessing workers
        batch_size: Files per processing batch (for memory efficiency)
        max_retries: Max retry attempts for transient failures
        resume: Whether to resume from checkpoint
        backend: Backend to use — "token", "tree-sitter", or "both"
        max_file_size: Maximum file size in bytes (skip larger files)
        min_file_size: Minimum file size in bytes (skip smaller files)
        validate_syntax: Whether to validate Python syntax before transpilation

    Returns:
        BatchStats with aggregate results
    """
    # Determine output directories based on backend mode
    token_output_dir: Optional[Path] = None
    ts_output_dir: Optional[Path] = None
    comparison_csv_path: Optional[Path] = None

    if backend == "both":
        token_output_dir = output_dir / f"{target_lang}-token"
        ts_output_dir = output_dir / f"{target_lang}-tree-sitter"
        comparison_csv_path = output_dir / f"{target_lang}-comparison.csv"
        token_output_dir.mkdir(parents=True, exist_ok=True)
        ts_output_dir.mkdir(parents=True, exist_ok=True)
        # Use the token dir for checkpoint/error log
        lang_output_dir = token_output_dir
    else:
        lang_output_dir = output_dir / target_lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = lang_output_dir / ".checkpoint"
    error_log_path = lang_output_dir / "errors.csv"

    # Load checkpoint if resuming
    completed = _load_checkpoint(checkpoint_path) if resume else set()
    if completed:
        print(f"Resuming: {len(completed):,} files already processed")

    # Filter out already-processed files
    pending = [f for f in files if str(f) not in completed]
    if not pending:
        print(f"All {len(files):,} files already processed for {target_lang}")
        return BatchStats(total=len(files), skipped=len(files))

    # Initialize error log
    if not resume or not error_log_path.exists():
        _init_error_log(error_log_path)

    # Initialize comparison CSV for both mode
    if backend == "both":
        assert comparison_csv_path is not None
        if not resume or not comparison_csv_path.exists():
            _init_comparison_csv(comparison_csv_path)

    # Build work items with security-checked paths
    work_items = []
    for f in pending:
        if backend == "both":
            assert token_output_dir is not None
            assert ts_output_dir is not None
            token_out = _safe_output_path(str(f), input_base, token_output_dir)
            ts_out = _safe_output_path(str(f), input_base, ts_output_dir)
            if token_out is None or ts_out is None:
                print(f"  Warning: skipping file with unsafe path: {f}", file=sys.stderr)
                continue
            work_items.append((str(f), str(token_out), str(ts_out)))
        else:
            out_path = _safe_output_path(str(f), input_base, lang_output_dir)
            if out_path is None:
                print(f"  Warning: skipping file with unsafe path: {f}", file=sys.stderr)
                continue
            work_items.append((str(f), str(out_path)))

    stats = BatchStats(
        total=len(files),
        skipped=len(completed),
    )
    start_time = time.time()

    # Process in batches to limit memory
    num_workers = min(workers, len(work_items))
    backend_label = backend if backend != "both" else "token+tree-sitter"
    print(f"\nTranspiling {len(work_items):,} files to {target_lang} "
          f"[{backend_label}] ({num_workers} workers)")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(target_lang, backend, max_file_size, min_file_size, validate_syntax, max_retries),
    ) as pool:
        for i in range(0, len(work_items), batch_size):
            batch = work_items[i : i + batch_size]

            results = list(tqdm(
                pool.imap_unordered(_transpile_file, batch),
                total=len(batch),
                desc=f"{target_lang} [{i + 1}–{min(i + len(batch), len(work_items))}]",
                unit="file",
            ))

            for result in results:
                if result.success:
                    stats.success += 1
                    _append_checkpoint(checkpoint_path, result.file_path)
                else:
                    stats.failed += 1
                    _append_error_log(error_log_path, result)
                    err_type = result.error_type or "Unknown"
                    stats.errors_by_type[err_type] = stats.errors_by_type.get(err_type, 0) + 1

                if result.retries > 0:
                    stats.retried += 1

                # Write comparison CSV row in both mode
                if backend == "both":
                    assert comparison_csv_path is not None
                    _append_comparison_csv(comparison_csv_path, result)

    stats.elapsed_sec = time.time() - start_time
    return stats


def _run_language(
    lang: str,
    files: list[Path],
    output_dir: Path,
    input_base: Path,
    workers: int,
    batch_size: int,
    max_retries: int,
    resume: bool,
    backend: str,
    max_file_size: int,
    min_file_size: int,
    validate_syntax: bool,
) -> tuple[str, BatchStats]:
    """Run transpilation for a single language. Used by parallel-languages mode."""
    stats = run_batch(
        files=files,
        target_lang=lang,
        output_dir=output_dir,
        input_base=input_base,
        workers=workers,
        batch_size=batch_size,
        max_retries=max_retries,
        resume=resume,
        backend=backend,
        max_file_size=max_file_size,
        min_file_size=min_file_size,
        validate_syntax=validate_syntax,
    )
    return lang, stats


def _parallel_language_worker(
    lang: str,
    files: list[Path],
    output_dir: Path,
    input_base: Path,
    workers: int,
    batch_size: int,
    max_retries: int,
    resume: bool,
    backend: str,
    max_file_size: int,
    min_file_size: int,
    validate_syntax: bool,
    result_queue: mp.Queue,
) -> None:
    """Top-level worker for parallel language processing (must be picklable)."""
    try:
        result_lang, result_stats = _run_language(
            lang, files, output_dir, input_base,
            workers=workers, batch_size=batch_size, max_retries=max_retries,
            resume=resume, backend=backend, max_file_size=max_file_size,
            min_file_size=min_file_size, validate_syntax=validate_syntax,
        )
        result_queue.put((result_lang, result_stats))
    except Exception:
        result_queue.put((lang, BatchStats(total=len(files), failed=len(files))))


# ---------------------------------------------------------------------------
# Summary & metadata
# ---------------------------------------------------------------------------

def print_summary(lang: str, stats: BatchStats) -> None:
    """Print formatted summary for a language run."""
    print(f"\n{'=' * 55}")
    print(f"  {lang} Transpilation Summary")
    print(f"{'=' * 55}")
    print(f"  Total files:      {stats.total:>8,}")
    print(f"  Successful:       {stats.success:>8,}")
    print(f"  Failed:           {stats.failed:>8,}")
    print(f"  Skipped (resume): {stats.skipped:>8,}")
    print(f"  Retried:          {stats.retried:>8,}")
    print(f"  Time:             {stats.elapsed_sec:>7.1f}s")

    if stats.success > 0 and stats.elapsed_sec > 0:
        rate = stats.success / stats.elapsed_sec
        print(f"  Throughput:       {rate:>7.1f} files/sec")

    if stats.errors_by_type:
        print(f"\n  Errors by type:")
        for err_type, count in sorted(stats.errors_by_type.items(), key=lambda x: -x[1]):
            print(f"    {err_type:<30s} {count:>6,}")

    print(f"{'=' * 55}")


def save_run_metadata(output_dir: Path, lang: str, stats: BatchStats) -> None:
    """Save run metadata as JSON sidecar."""
    meta = {
        "language": lang,
        "total_files": stats.total,
        "successful": stats.success,
        "failed": stats.failed,
        "skipped_resumed": stats.skipped,
        "retried": stats.retried,
        "elapsed_seconds": round(stats.elapsed_sec, 2),
        "throughput_files_per_sec": round(stats.success / stats.elapsed_sec, 2) if stats.elapsed_sec > 0 else 0,
        "errors_by_type": stats.errors_by_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_dir / lang / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transpilation wrapper for Legesher (Language Decoded project).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_transpile.py ./source-python zh --output ./transpiled
  python batch_transpile.py ./source-python zh,am,ur --output ./transpiled
  python batch_transpile.py ./source-python zh --output ./transpiled --resume
  python batch_transpile.py --hf-dataset bigcode/the-stack-dedup zh --output ./transpiled --hf-limit 50000
  python batch_transpile.py ./source-python zh --output ./transpiled --backend tree-sitter
  python batch_transpile.py ./source-python zh --output ./transpiled --backend both
  python batch_transpile.py ./source-python zh,am,ur --output ./transpiled --parallel-languages
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input directory of Python files (omit if using --hf-dataset)",
    )
    parser.add_argument(
        "languages",
        help="Target language code(s), comma-separated (e.g., zh,am,ur)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output root directory (subdirs created per language)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help=f"Number of worker processes (default: {max(1, mp.cpu_count() - 1)})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Files per processing batch (default: 200)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient failures (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-processed files)",
    )
    parser.add_argument(
        "--hf-dataset",
        help="HuggingFace dataset ID to stream from (e.g., bigcode/the-stack-dedup)",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=50_000,
        help="Max files to download from HF dataset (default: 50,000)",
    )
    parser.add_argument(
        "--hf-data-dir",
        default="data/python",
        help="Data directory within HF dataset (default: data/python)",
    )
    parser.add_argument(
        "--backend",
        choices=["token", "tree-sitter", "both"],
        default="token",
        help="Translator backend: token (default), tree-sitter, or both (compare)",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1_048_576,
        help="Maximum file size in bytes to process (default: 1048576 = 1MB)",
    )
    parser.add_argument(
        "--min-file-size",
        type=int,
        default=10,
        help="Minimum file size in bytes to process (default: 10)",
    )
    parser.add_argument(
        "--validate-syntax",
        action="store_true",
        help="Validate Python syntax (ast.parse) before transpilation",
    )
    parser.add_argument(
        "--parallel-languages",
        action="store_true",
        help="Process multiple languages in parallel (one process group per language)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input and not args.hf_dataset:
        parser.error("Either provide an input directory or use --hf-dataset")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if not languages:
        print("Error: No valid language codes provided", file=sys.stderr)
        sys.exit(1)

    # Discover files
    if args.hf_dataset:
        files = discover_hf_files(
            args.hf_dataset, output_dir, args.hf_limit, args.hf_data_dir,
        )
        input_base = output_dir / "_source"
    else:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: {args.input} is not a directory", file=sys.stderr)
            sys.exit(1)
        files = discover_local_files(input_dir)
        input_base = input_dir

    if not files:
        print("No .py files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files):,} Python files")
    print(f"Target languages: {', '.join(languages)}")
    print(f"Backend: {args.backend}")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}, Max retries: {args.max_retries}")
    print(f"File size limits: {args.min_file_size}–{args.max_file_size} bytes")
    if args.validate_syntax:
        print("Syntax validation: enabled")

    # Run transpilation for each language
    all_stats: dict[str, BatchStats] = {}

    if args.parallel_languages and len(languages) > 1:
        # Parallel language processing: one mp.Process per language
        workers_per_lang = max(1, args.workers // len(languages))
        print(f"\nParallel language mode: {len(languages)} languages, "
              f"{workers_per_lang} workers each")

        processes: list[tuple[str, mp.Process, mp.Queue]] = []
        for lang in languages:
            q: mp.Queue = mp.Queue()
            p = mp.Process(
                target=_parallel_language_worker,
                args=(
                    lang, list(files), output_dir, input_base,
                    workers_per_lang, args.batch_size, args.max_retries,
                    args.resume, args.backend, args.max_file_size,
                    args.min_file_size, args.validate_syntax, q,
                ),
            )
            processes.append((lang, p, q))
            p.start()

        # Collect results
        for lang, p, q in processes:
            p.join()
            try:
                result_lang, stats = q.get_nowait()
                all_stats[result_lang] = stats
            except Exception:
                all_stats[lang] = BatchStats(total=len(files), failed=len(files))

        # Print summaries after all complete
        for lang in languages:
            if lang in all_stats:
                print_summary(lang, all_stats[lang])
                save_run_metadata(output_dir, lang, all_stats[lang])
    else:
        # Sequential language processing (default)
        for lang in languages:
            stats = run_batch(
                files=files,
                target_lang=lang,
                output_dir=output_dir,
                input_base=input_base,
                workers=args.workers,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                resume=args.resume,
                backend=args.backend,
                max_file_size=args.max_file_size,
                min_file_size=args.min_file_size,
                validate_syntax=args.validate_syntax,
            )
            all_stats[lang] = stats
            print_summary(lang, stats)
            save_run_metadata(output_dir, lang, stats)

    # Overall summary
    if len(languages) > 1:
        print(f"\n{'=' * 55}")
        print(f"  Overall Summary ({len(languages)} languages)")
        print(f"{'=' * 55}")
        for lang, stats in all_stats.items():
            rate = stats.success / stats.elapsed_sec if stats.elapsed_sec > 0 else 0
            print(f"  {lang}: {stats.success:,}/{stats.total:,} OK, {stats.failed:,} failed ({rate:.0f} files/sec)")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Populate Condition 5 datasets — full LLM translation of Cond-1 files.

Translates ``condition-1-en-{size}`` Python files into target languages
({es, zh, ur}, etc.) for the AYA-213 Cond-5 datasets, exercising the full
Legesher Cond-5 pipeline:

    TreeSitterTranslator (keywords/builtins/reserved_words via packs)
        +
    {OpenAICompatProvider | CohereAyaProvider}
        (LLM for identifiers/comments/docstrings/strings)
        ↓
    LLMTranslator.translate_code(...)

Outputs land in a layout that ``package_dataset.py from-files`` consumes
directly:

    {output_dir}/{lang}/000.py            (translation)
    {output_dir}/{lang}/metadata.csv      (filename + file_path + license)
    {output_dir}/{lang}.originals/000.py  (English source mirror)

Linked Linear issue: AYA-213 (Cond-5 dataset rebuild — 6 datasets:
3 langs × 2 backends).

Usage examples:
    # Cohere production run (recommended for AYA-213)
    COHERE_API_KEY=$(pass-cli ...) \\
    python scripts/populate_cond5_datasets.py \\
        --provider cohere --cohere-model c4ai-aya-expanse-32b \\
        --target-langs ur --n-files 5000 --concurrency 4 \\
        --source-parquet packaged/condition-1-en-5k/train.parquet \\
        --output-dir packaged/condition-5-ur-5k-c4ai-aya-expanse-32b \\
        --resume

    # Local Ollama dev iteration
    OLLAMA_NUM_PARALLEL=4 ollama serve  # in a separate shell
    python scripts/populate_cond5_datasets.py \\
        --model gemma3:12b --target-langs es \\
        --concurrency 4 --n-files 20

    # Single language, single file (smoke test)
    python scripts/populate_cond5_datasets.py \\
        --provider cohere --target-langs ur --n-files 1
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from legesher_i18n import load_language_pack
from legesher_i18n.api.providers import OpenAICompatProvider
from legesher_i18n.api.providers.cohere_aya import CohereAyaProvider
from legesher_core.tree_sitter.llm_translator import LLMTranslator

logger = logging.getLogger(__name__)

# Defaults derived relative to this script's location so the pilot is
# portable across checkouts. `parents[1]` resolves to `data-pipeline/`
# (the parent of `scripts/`).
DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = str(
    DATA_PIPELINE_ROOT / "packaged" / "condition-1-en-103k" / "train.parquet"
)
DEFAULT_OUTPUT_DIR = str(DATA_PIPELINE_ROOT / "cond5-qwen-pilot")

# Tiny Aya (cohere `tiny-aya-global`) returns identifier translations as
# `original_name_<en>_translated_name_<tgt>` glued strings instead of just
# the bare translated identifier — a structured-format prompt-following
# failure that CORE-950's no-explanation rule didn't catch. We strip the
# wrapper here so output is shaped like every other backend's output.
# Idempotent on backends that don't produce the artifact (c4ai-aya-expanse-32b,
# command-a-translate, gemma3, etc.) — the pattern is specific enough that
# false positives in legitimate code are vanishingly unlikely.
_LABELED_MAPPING_PATTERN = re.compile(
    r"original_name_(\w+?)_translated_name_(\w+)", re.UNICODE
)


def strip_labeled_mappings(code: str) -> tuple[str, int]:
    """Strip the tiny-aya `original_name_X_translated_name_Y` artifact.

    Returns the cleaned code plus the count of substitutions applied.
    Counters surface in the per-file summary so we can see how often
    a given backend triggered the artifact.
    """
    return _LABELED_MAPPING_PATTERN.subn(r"\2", code)


def write_metadata_csv(output_dir: Path, per_file: list[dict[str, Any]]) -> Path | None:
    """Emit a metadata.csv that ``package_dataset.py from-files`` consumes.

    Schema:
    - ``filename`` (e.g. ``000.py``): used by ``package_dataset`` as the
      lookup key against the relative path of each translated file. The
      companion patch in this PR makes ``package_dataset`` prefer this
      column over ``file_path`` when building its metadata dict.
    - ``file_path``: the **source attribution** (Stack v2 path) carried
      forward into the published HF dataset's ``file_path`` column so
      provenance survives the trip from local run → HF.
    - ``license`` and ``idx`` round out the row.

    Resumed entries are included so a fully-resumed run still produces a
    complete metadata.csv that ``package_dataset`` can consume. Runtime
    errors are skipped — those files have no valid output to package.
    """
    rows = [r for r in per_file if r["status"] in ("ok", "resumed")]
    if not rows:
        return None
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["filename", "file_path", "license", "idx"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "filename": f"{r['idx']:03d}.py",
                    "file_path": r.get("file_path", ""),
                    "license": r.get("license", ""),
                    "idx": r["idx"],
                }
            )
    return metadata_path


class OpenAICompatBackend:
    """Sync ``TextTranslationBackend`` adapter over async ``OpenAICompatProvider``.

    ``LLMTranslator`` expects a synchronous backend (``translate_text`` /
    ``translate_batch``); ``OpenAICompatProvider.translate`` is async. This
    adapter wraps each call in ``asyncio.run`` and unwraps ``TranslationResult``
    to ``str``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout: float = 180.0,
    ) -> None:
        self._provider = OpenAICompatProvider(
            base_url=base_url, model=model, api_key=api_key, timeout=timeout
        )

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        result = asyncio.run(
            self._provider.translate(text, source_lang, target_lang, context)
        )
        return result.text

    def translate_batch(
        self,
        items: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        async def _run_concurrent() -> list[str]:
            tasks = [
                self._provider.translate(
                    item["text"], source_lang, target_lang, item.get("context")
                )
                for item in items
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return [r.text for r in results]

        return asyncio.run(_run_concurrent())


class CohereBackend:
    """Sync backend adapter over async ``CohereAyaProvider``.

    Same surface as ``OpenAICompatBackend`` but routes through Cohere's hosted
    API (Aya Expanse 32B by default). Useful for AYA-213 production runs where
    local gemma3 is too slow (~16s/call vs ~0.7s/call via Cohere).

    Creates a fresh ``CohereAyaProvider`` (and therefore a fresh
    ``AsyncClientV2``) inside each ``asyncio.run`` invocation. Caching the
    provider across calls would bind the underlying httpx connection pool to
    a now-closed event loop, causing ``RuntimeError: Event loop is closed``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "c4ai-aya-expanse-32b",
    ) -> None:
        self._api_key = api_key
        self._model = model

    def _new_provider(self) -> CohereAyaProvider:
        return CohereAyaProvider(api_key=self._api_key, model=self._model)

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        async def _run() -> str:
            provider = self._new_provider()
            result = await provider.translate(text, source_lang, target_lang, context)
            return result.text

        return asyncio.run(_run())

    def translate_batch(
        self,
        items: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        async def _run_concurrent() -> list[str]:
            provider = self._new_provider()
            tasks = [
                provider.translate(
                    item["text"], source_lang, target_lang, item.get("context")
                )
                for item in items
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return [r.text for r in results]

        return asyncio.run(_run_concurrent())


def reverse_keywords_and_builtins(
    code: str,
    keyword_map: dict[str, str],
    builtin_map: dict[str, str],
) -> str:
    """Substitute target-language keywords/builtins back to their English form.

    Used for AST validation: Python's standard ``ast.parse`` only recognizes
    English keywords (``def``, ``class``, ``from``, etc.). Cond-5 outputs use
    the target language's keywords (``definir``, ``clase``, ``desde``), so
    they fail standard parsing despite being structurally valid Cond-5 Python.
    Reversing keywords/builtins back to English preserves AST structure and
    leaves identifiers untouched (Python identifiers can be any Unicode).

    Best-effort: relies on word-boundary regex. If the LLM emitted the
    translated form verbatim (which the legesher pack contracts encourage),
    this catches it; if the LLM hallucinated synonyms, those won't reverse.
    """
    items = list(keyword_map.items()) + list(builtin_map.items())
    # Longest first to avoid prefix conflicts (e.g. "in" vs "input").
    items.sort(key=lambda kv: -len(kv[1]))
    result = code
    for english, translated in items:
        if not translated or translated == english:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(translated)}(?!\w)", re.UNICODE)
        result = pattern.sub(english, result)
    return result


class _RateLimiter:
    """Per-process rate limiter: enforce a minimum delay between calls.

    Used to proactively match Cohere's per-minute cap on
    ``c4ai-aya-expanse-32b`` (empirically ~2 RPM steady-state, far below
    the published 500 RPM for prod chat keys). Issuing calls faster than
    the cap produces 429 rejections that still bill input tokens —
    waiting between calls is materially cheaper than retrying.
    """

    def __init__(self, min_delay_seconds: float) -> None:
        self.min_delay = max(0.0, min_delay_seconds)
        self._last_call_t = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_delay <= 0:
            return
        with self._lock:
            wait_for = self.min_delay - (time.perf_counter() - self._last_call_t)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_call_t = time.perf_counter()


# CORE-974: circuit-breaker exceptions raised from inside _translate_one to
# unwind the iteration loop cleanly when a budget or wall-time cap is hit.
# Caught at the run_pilot level so a partial summary still lands on disk.
class _WallTimeExceeded(Exception):
    """Raised when ``--max-wall-seconds`` is exceeded mid-run."""


class _BudgetExceeded(Exception):
    """Raised when ``--max-credits-usd`` is exceeded mid-run."""


class _BudgetTracker:
    """Thread-safe cumulative cost tracker with a configurable cap.

    Cost-per-call is an estimate (defaults from AYA-213 spend reconciliation,
    overridable via CLI). For circuit-breaker purposes this is sufficient:
    "off by 2x in the right direction" still prevents runaway spend. Precise
    per-token accounting via ``CohereBackend`` accumulation is a follow-up.
    """

    def __init__(
        self,
        max_credits_usd: float | None,
        cost_per_success_usd: float,
        cost_per_failure_usd: float,
    ) -> None:
        self.max_credits_usd = max_credits_usd
        self.cost_per_success_usd = cost_per_success_usd
        self.cost_per_failure_usd = cost_per_failure_usd
        self._spent_usd = 0.0
        self._lock = threading.Lock()

    def record(self, success: bool) -> float:
        """Increment cumulative spend; return the new total."""
        with self._lock:
            self._spent_usd += (
                self.cost_per_success_usd if success else self.cost_per_failure_usd
            )
            return self._spent_usd

    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    def is_exceeded(self) -> bool:
        if self.max_credits_usd is None:
            return False
        return self.spent_usd() >= self.max_credits_usd


def _positive_int(value: str) -> int:
    """argparse type for ``--n-files``: must be a strictly positive integer.

    Rejecting 0 and negatives at parse time avoids a downstream
    ``ZeroDivisionError`` in the per-language average-chars print and
    keeps every code path that assumes ``len(files) >= 1`` honest.
    """
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"--n-files must be positive (got {n}); pass at least 1"
        )
    return n


def _non_negative_float(value: str) -> float:
    """argparse type for ``--min-call-delay``: must be a non-negative float.

    Rejecting negatives at parse time surfaces user mistakes with a clear
    error rather than silently clamping inside ``_RateLimiter``.
    """
    try:
        x = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if x < 0:
        raise argparse.ArgumentTypeError(
            f"--min-call-delay must be >= 0 (got {x}); use 0 to disable"
        )
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pilot the Cond-5 translation pipeline against a local LLM "
            "(Qwen via Ollama by default)."
        )
    )
    parser.add_argument(
        "--source-parquet",
        default=DEFAULT_PARQUET,
        help="Local parquet to read English source files from",
    )
    parser.add_argument(
        "--n-files",
        type=_positive_int,
        default=20,
        help="Number of files to translate per language (default: 20)",
    )
    parser.add_argument(
        "--target-langs",
        default="es,zh,ur",
        help="Comma-separated target language codes (default: 'es,zh,ur')",
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code (default: 'en')",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b-instruct-q4_K_M",
        help="Ollama model tag (default: qwen2.5:7b-instruct-q4_K_M)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL for Ollama (default: localhost:11434/v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write per-language translated files + summary",
    )
    parser.add_argument(
        "--python-version",
        default="3.13",
        help="Python version for language pack loading (default: 3.13)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds (default: 180; bump for larger models)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of files to translate in parallel (default: 1). "
            "The Ollama server must be started with OLLAMA_NUM_PARALLEL>=N "
            "for this to actually parallelize; otherwise Ollama queues."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "cohere"),
        default="ollama",
        help=(
            "LLM provider (default: ollama). With 'cohere', requires "
            "COHERE_API_KEY in env and uses --cohere-model instead of --model."
        ),
    )
    parser.add_argument(
        "--cohere-model",
        default="c4ai-aya-expanse-32b",
        help=(
            "Cohere model when --provider=cohere "
            "(default: c4ai-aya-expanse-32b; alternatives: "
            "command-a-translate-08-2025, command-a-03-2025)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip files that already have an output `.py` in the per-language "
            "output dir. Required for resumable 5K+ runs that get interrupted."
        ),
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic single retry on transient API/translation errors.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between retries (default: 5).",
    )
    parser.add_argument(
        "--min-call-delay",
        type=_non_negative_float,
        default=None,
        help=(
            "Minimum seconds between consecutive backend calls per process. "
            "When omitted, defaults to 30s for --provider=cohere (matches "
            "c4ai-aya-expanse-32b's empirical ~2 RPM cap) and 0s for "
            "--provider=ollama. Pass an explicit value to override either "
            "default. Applied across all worker threads and shared across "
            "all target languages within a single run."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=_non_negative_float,
        default=None,
        help=(
            "Hard wall-clock cap. Run aborts after N seconds with a partial "
            "summary.json and exit code 4. Default: no cap. CORE-974: "
            "prevents wedged retry loops from running for hours unattended."
        ),
    )
    parser.add_argument(
        "--max-credits-usd",
        type=_non_negative_float,
        default=None,
        help=(
            "Cumulative spend cap (USD, estimated). Run aborts when "
            "estimated_spend >= N with a partial summary.json and exit code "
            "5. Note: --max-credits-usd 0 trips the breaker on the first "
            "recorded call (>= compares both equal and over). Default: no "
            "cap. CORE-974: prevents runaway 429 storms from burning the "
            "credit budget."
        ),
    )
    parser.add_argument(
        "--cost-per-success-usd",
        type=_non_negative_float,
        default=0.05,
        help=(
            "Estimated cost in USD for a successful translation. Used by "
            "--max-credits-usd. Default 0.05 (empirical from AYA-213)."
        ),
    )
    parser.add_argument(
        "--cost-per-failure-usd",
        type=_non_negative_float,
        default=0.025,
        help=(
            "Estimated cost in USD for a failed translation (input tokens "
            "only — 429-rejected calls still bill). Used by "
            "--max-credits-usd. Default 0.025 (empirical from AYA-213)."
        ),
    )
    parser.add_argument(
        "--idx-allowlist",
        type=str,
        default=None,
        help=(
            "Path to a newline-separated file of idxs to translate. When "
            "set, the script processes ONLY rows whose idx (0-based row "
            "number in the source parquet) appears in the file. Other "
            "rows are silently skipped — no LLM call, no .py written, no "
            ".error.txt written. Accepts bare integers (`42`), filename "
            "stems (`042`), or `.py` filenames (`042.py`). Whitespace and "
            "lines starting with `#` are ignored. CORE-1049: enables "
            "cross-lingual idx alignment so subsequent zh/es runs only "
            "translate the same source idxs that succeeded for ur."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def _load_idx_allowlist(path: str) -> set[int]:
    """Parse --idx-allowlist file into a set of integer idxs.

    Accepts (per line):
    - bare integer:        ``42``
    - zero-padded stem:    ``042``
    - filename:            ``042.py``
    - whitespace + ``#`` comments stripped

    Empty lines are skipped silently. An empty allowlist (no parseable
    entries) raises ``ValueError`` so the caller fails fast instead of
    quietly translating zero files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--idx-allowlist file not found: {path}")
    idxs: set[int] = set()
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        # Strip comments and whitespace
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Strip trailing .py
        if line.endswith(".py"):
            line = line[:-3]
        try:
            idxs.add(int(line))
        except ValueError as exc:
            raise ValueError(
                f"--idx-allowlist {path}:{line_no}: cannot parse {raw!r} as an idx"
            ) from exc
    if not idxs:
        raise ValueError(
            f"--idx-allowlist {path} contained no parseable idxs; aborting "
            "(would have translated 0 files)"
        )
    return idxs


def run_pilot(
    source_lang: str,
    target_lang: str,
    files: list[dict[str, Any]],
    backend_factory: Callable[[], OpenAICompatBackend],
    keyword_map: dict[str, str],
    builtin_map: dict[str, str],
    output_dir: Path,
    concurrency: int = 1,
    reserved_word_map: dict[str, str] | None = None,
    resume: bool = False,
    retry: bool = True,
    retry_delay: float = 5.0,
    rate_limiter: _RateLimiter | None = None,
    deadline: float | None = None,
    budget: _BudgetTracker | None = None,
    idx_allowlist: set[int] | None = None,
) -> dict[str, Any]:
    """Translate ``files`` to ``target_lang`` and capture per-file outcomes.

    When ``concurrency > 1``, files are translated in parallel via a
    ``ThreadPoolExecutor``. Each worker thread gets its own backend
    instance via ``threading.local``. Backend lifecycle differs by provider:

    - ``OpenAICompatBackend`` caches its async client across calls; the
      threading.local guard prevents the cached httpx ``AsyncClient`` from
      being shared across threads that each call ``asyncio.run`` (an
      unsupported pattern for httpx's transport pool).
    - ``CohereBackend`` constructs a fresh ``CohereAyaProvider`` (and
      therefore a fresh ``AsyncClientV2``) inside each ``asyncio.run``
      invocation rather than caching it. The threading.local guard is
      effectively a no-op for Cohere, but kept symmetric with the
      OpenAI-compat path so the dispatch logic is uniform.

    ``resume=True`` skips files that already have an output ``.py`` in
    ``output_dir`` (typical use: long runs that got interrupted). ``retry=True``
    retries a failing translation once after ``retry_delay`` seconds — covers
    transient 429s, 5xx, and connection blips on hosted APIs without needing
    to restart the whole run.

    ``rate_limiter`` enforces a minimum gap between consecutive backend
    calls, shared across all worker threads (and across multi-language
    runs when the same instance is reused). Pass ``None`` for no pacing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    # Default to a no-op limiter when caller doesn't supply one (preserves
    # legacy behavior where each run_pilot() invocation manages its own pacing).
    if rate_limiter is None:
        rate_limiter = _RateLimiter(0.0)

    def _backend() -> OpenAICompatBackend:
        b = getattr(local, "backend", None)
        if b is None:
            b = backend_factory()
            local.backend = b
        return b

    # Originals dir is a sibling of the per-language output dir. Layout:
    #   output_root/{lang}/000.py            (translation)
    #   output_root/{lang}.originals/000.py  (English source)
    # package_dataset.py from-files maps these via --transpiled / --originals.
    originals_dir = output_dir.parent / f"{output_dir.name}.originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    def _translate_one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        code_en = row["code"]
        file_path = row.get("file_path") or row.get("metadata_file") or f"row_{idx}"
        license_str = row.get("license") or ""
        out_path = output_dir / f"{idx:03d}.py"
        original_path = originals_dir / f"{idx:03d}.py"

        # CORE-1049: idx-allowlist short-circuit. Cheaper than resume check
        # (set membership vs filesystem stat) and runs first so circuit
        # breakers don't trigger on rows we'd never translate anyway.
        # Returns a distinct status so summary.json can report skipped count.
        if idx_allowlist is not None and idx not in idx_allowlist:
            return {
                "idx": idx,
                "file_path": file_path,
                "license": license_str,
                "status": "skipped_not_in_allowlist",
                "ast": "skipped",
                "elapsed_seconds": 0.0,
                "input_chars": len(code_en),
                "output_chars": 0,
            }

        # CORE-974: circuit breakers checked BEFORE any work for this file.
        # Resume short-circuit (below) is exempt — checking existing output
        # is essentially free and counts as making forward progress.
        if deadline is not None and time.perf_counter() > deadline:
            raise _WallTimeExceeded(f"--max-wall-seconds reached before file idx={idx}")
        if budget is not None and budget.is_exceeded():
            raise _BudgetExceeded(
                f"--max-credits-usd reached "
                f"(spent ~${budget.spent_usd():.2f}) before file idx={idx}"
            )

        # Resume: short-circuit if output already exists from a prior run.
        if resume and out_path.exists():
            # Best-effort: also write the original if it's missing, so a
            # resumed run still produces a complete originals/ tree for
            # package_dataset.
            if not original_path.exists():
                original_path.write_text(code_en, encoding="utf-8")
            # Read the existing translation to compute output_chars in
            # Unicode code points, matching the non-resumed path. Using
            # st_size (bytes) here would over-report by ~3× on RTL/CJK
            # output and skew the run summary.
            try:
                resumed_text = out_path.read_text(encoding="utf-8")
                resumed_output_chars = len(resumed_text)
            except (UnicodeDecodeError, OSError):
                resumed_output_chars = out_path.stat().st_size
            return {
                "idx": idx,
                "file_path": file_path,
                "license": license_str,
                "status": "resumed",
                "ast": "skipped",
                "elapsed_seconds": 0.0,
                "input_chars": len(code_en),
                "output_chars": resumed_output_chars,
            }

        translator = LLMTranslator(
            keyword_map=keyword_map,
            builtin_map=builtin_map,
            backend=_backend(),
            reserved_word_map=reserved_word_map,
        )

        # Translate with one optional retry on any exception. Most production
        # failures are transient (429, 5xx, ReadTimeout); retrying once covers
        # them without a full run restart.
        max_attempts = 2 if retry else 1
        translated: str | None = None
        elapsed = 0.0
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            rate_limiter.wait()
            t0 = time.perf_counter()
            try:
                translated_raw = translator.translate_code(
                    code_en, source_lang, target_lang
                )
                translated = (
                    translated_raw
                    if isinstance(translated_raw, str)
                    else translated_raw[0]
                )
                elapsed = time.perf_counter() - t0
                last_exc = None
                break
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Translate attempt %d/%d failed for %s after %.1fs: "
                        "%s: %s; retrying in %.1fs",
                        attempt,
                        max_attempts,
                        file_path,
                        elapsed,
                        type(exc).__name__,
                        exc,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        if last_exc is not None or translated is None:
            err_path = output_dir / f"{idx:03d}.error.txt"
            err_path.write_text(
                f"file_path: {file_path}\nelapsed: {elapsed:.2f}s\n"
                f"attempts: {max_attempts}\n\n"
                f"{type(last_exc).__name__}: {last_exc}\n",
                encoding="utf-8",
            )
            # CORE-974: record failed attempt for cost-cap accounting.
            if budget is not None:
                budget.record(success=False)
            return {
                "idx": idx,
                "file_path": file_path,
                "status": "runtime_error",
                "error": f"{type(last_exc).__name__}: {last_exc}",
                "elapsed_seconds": round(elapsed, 2),
                "input_chars": len(code_en),
            }

        # Strip tiny-aya labeled-mapping artifact (if present). Idempotent on
        # backends that don't produce it. Count surfaces in per_file output.
        translated, stripped_count = strip_labeled_mappings(translated)

        # `elapsed` is already set to the successful attempt's duration
        # by the for loop above. `out_path` was bound at the top of the
        # function for the resume short-circuit; reuse it for the write.
        en_for_ast = reverse_keywords_and_builtins(translated, keyword_map, builtin_map)
        try:
            ast.parse(en_for_ast)
            ast_status = "pass"
        except SyntaxError as syn_exc:
            ast_status = f"fail: {syn_exc.msg} at line {syn_exc.lineno}"

        out_path.write_text(translated, encoding="utf-8")
        # Mirror the English source so package_dataset.py from-files can pair
        # transpiled ↔ original by relative path (000.py ↔ 000.py).
        original_path.write_text(code_en, encoding="utf-8")

        preview_path = output_dir / f"{idx:03d}.preview.md"
        preview_path.write_text(
            f"# {file_path}\n\n"
            f"AST: **{ast_status}** | elapsed: **{elapsed:.2f}s** | "
            f"input chars: {len(code_en)} | output chars: {len(translated)}\n\n"
            "## English source (first 30 lines)\n\n```python\n"
            + "\n".join(code_en.splitlines()[:30])
            + "\n```\n\n"
            f"## {target_lang} translation (first 30 lines)\n\n```python\n"
            + "\n".join(translated.splitlines()[:30])
            + "\n```\n",
            encoding="utf-8",
        )

        return {
            "idx": idx,
            "file_path": file_path,
            "license": license_str,
            "status": "ok",
            "ast": ast_status,
            "elapsed_seconds": round(elapsed, 2),
            "input_chars": len(code_en),
            "output_chars": len(translated),
            "stripped_labeled_mappings": stripped_count,
        }

    def _safe_translate(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        """Wrap ``_translate_one`` to record success cost on the happy path.

        ``_translate_one`` records failure cost itself (because it has the
        try/except context). Success cost is recorded here to avoid mutating
        ``_translate_one``'s return-dict construction. Sentinel exceptions
        (``_WallTimeExceeded`` / ``_BudgetExceeded``) propagate up to the
        iteration loop unchanged.
        """
        result = _translate_one(idx, row)
        if budget is not None and result.get("status") == "ok":
            budget.record(success=True)
        return result

    per_file: list[dict[str, Any]] = []
    aborted_reason: str | None = None
    wall_t0 = time.perf_counter()

    if concurrency <= 1:
        # Sequential path. Sentinels propagate cleanly out of the for-loop
        # at the next file iteration, no executor lifecycle to manage.
        try:
            for idx, row in enumerate(files):
                per_file.append(_safe_translate(idx, row))
        except _WallTimeExceeded as exc:
            aborted_reason = f"wall_time_exceeded: {exc}"
            logger.warning("Aborted: %s", aborted_reason)
        except _BudgetExceeded as exc:
            aborted_reason = f"budget_exceeded: {exc}"
            logger.warning("Aborted: %s", aborted_reason)
    else:
        # Concurrent path. CORE-974 / PR #44 review: manual executor
        # lifecycle (vs. the natural `with ThreadPoolExecutor(...) as
        # pool:` pattern) so the abort path can call
        # ``shutdown(cancel_futures=True)`` BEFORE the implicit
        # ``shutdown(wait=True)`` runs at block exit. Without this,
        # queued workers would drain one-by-one (each cheap because they
        # re-raise the sentinel at the deadline check, but still seconds
        # of overhead per queued worker), and any in-flight workers'
        # results would never reach ``per_file`` because we'd have
        # already broken out of the ``as_completed`` loop.
        #
        # Limitation: Python cannot cancel a thread that's already past
        # the deadline check, inside ``translator.translate_code(...)``.
        # Those calls finish on their own. The wall-time cap therefore
        # overshoots by up to ``concurrency × longest_inflight_call``
        # seconds — bounded, not zero. The drain block below reconciles
        # ``per_file`` with the .py outputs those in-flight workers
        # write to disk, so summary counts don't under-report relative
        # to ``estimated_spend_usd`` and to files actually present.
        pool = ThreadPoolExecutor(max_workers=concurrency)
        futures: list[Future[dict[str, Any]]] = []
        try:
            futures = [
                pool.submit(_safe_translate, idx, row) for idx, row in enumerate(files)
            ]
            for fut in as_completed(futures):
                # Sentinel propagates here via ``fut.result()`` and is
                # caught below; we then cancel queued + drain in-flight.
                per_file.append(fut.result())
        except _WallTimeExceeded as exc:
            aborted_reason = f"wall_time_exceeded: {exc}"
            logger.warning("Aborted: %s", aborted_reason)
            pool.shutdown(wait=True, cancel_futures=True)
        except _BudgetExceeded as exc:
            aborted_reason = f"budget_exceeded: {exc}"
            logger.warning("Aborted: %s", aborted_reason)
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            pool.shutdown(wait=True)  # idempotent; happy-path shutdown

        # On abort: drain in-flight workers' results so per_file matches
        # what's actually on disk + what budget already counted. On the
        # happy path this loop is a no-op (every future was already
        # collected via as_completed).
        if aborted_reason is not None:
            collected_idxs = {r["idx"] for r in per_file if isinstance(r, dict)}
            for fut in futures:
                if fut.cancelled():
                    continue
                try:
                    r = fut.result()
                except (_WallTimeExceeded, _BudgetExceeded):
                    continue  # the originator(s) — not a per_file row
                except Exception:
                    continue  # _translate_one catches its own; safety net
                if isinstance(r, dict) and r.get("idx") not in collected_idxs:
                    per_file.append(r)
                    collected_idxs.add(r["idx"])

        per_file.sort(key=lambda r: r["idx"])

    wall_seconds = time.perf_counter() - wall_t0

    # Emit metadata.csv so package_dataset.py from-files can pick up
    # license + source attribution per row.
    metadata_path = write_metadata_csv(output_dir, per_file)
    if metadata_path is not None:
        logger.info("Wrote metadata.csv: %s", metadata_path)

    parse_pass = 0
    parse_fail = 0
    runtime_fail = 0
    resumed_count = 0
    skipped_not_in_allowlist_count = 0
    total_input_chars = 0
    total_output_chars = 0
    total_seconds = 0.0
    for r in per_file:
        total_input_chars += r.get("input_chars", 0)
        if r["status"] == "runtime_error":
            runtime_fail += 1
            continue
        if r["status"] == "resumed":
            # Resumed files contribute output_chars but not LLM time or AST
            # status (we trust prior-run validity rather than re-parsing).
            resumed_count += 1
            total_output_chars += r.get("output_chars", 0)
            continue
        if r["status"] == "skipped_not_in_allowlist":
            # CORE-1049: rows filtered out by --idx-allowlist. No work done,
            # no cost incurred, no .py written. Tracked for traceability.
            skipped_not_in_allowlist_count += 1
            continue
        total_output_chars += r.get("output_chars", 0)
        total_seconds += r["elapsed_seconds"]
        if r["ast"] == "pass":
            parse_pass += 1
        else:
            parse_fail += 1

    # Only files actually translated this run count toward the per-file
    # average; resumed/skipped files would skew it to ~0s. When aborted
    # mid-run the processed count comes from per_file, not the original
    # file list.
    processed_count = len(per_file)
    fresh_translated = (
        processed_count - runtime_fail - resumed_count - skipped_not_in_allowlist_count
    )
    return {
        "target_lang": target_lang,
        "n_files": len(files),
        "n_processed": processed_count,
        "concurrency": concurrency,
        "ast_pass": parse_pass,
        "ast_fail": parse_fail,
        "runtime_fail": runtime_fail,
        "resumed": resumed_count,
        "skipped_not_in_allowlist": skipped_not_in_allowlist_count,
        "aborted_reason": aborted_reason,
        "estimated_spend_usd": (
            round(budget.spent_usd(), 4) if budget is not None else None
        ),
        "wall_seconds": round(wall_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "avg_seconds_per_file": (
            round(total_seconds / fresh_translated, 2) if fresh_translated > 0 else 0.0
        ),
        "throughput_files_per_min": (
            round(60 * fresh_translated / wall_seconds, 2)
            if wall_seconds > 0 and fresh_translated > 0
            else 0.0
        ),
        "total_input_chars": total_input_chars,
        "total_output_chars": total_output_chars,
        "per_file": per_file,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet noisy per-request HTTP logs unless --verbose. At ~25 LLM calls per
    # file × 5,000 files = 125K log lines that bury anything actionable.
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    target_langs = [
        lang.strip() for lang in args.target_langs.split(",") if lang.strip()
    ]
    if not target_langs:
        print("error: --target-langs produced an empty list", file=sys.stderr)
        return 2

    if args.concurrency > 1:
        ollama_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        print(
            f"Concurrency: {args.concurrency} files in parallel per language. "
            f"OLLAMA_NUM_PARALLEL={ollama_parallel or '(unset; Ollama default is 1)'}. "
            "If unset or smaller than --concurrency, Ollama will queue requests "
            "and you won't see real speedup."
        )

    print(f"Loading source files from {args.source_parquet}")
    ds = load_dataset("parquet", data_files=args.source_parquet, split="train")
    if args.n_files > len(ds):
        print(
            f"warning: requested {args.n_files} but parquet only has {len(ds)}; "
            f"using all available",
            file=sys.stderr,
        )
        args.n_files = len(ds)
    sample = ds.select(range(args.n_files))
    files = [dict(row) for row in sample]
    print(
        f"  loaded {len(files)} files "
        f"(avg input chars: {sum(len(f['code']) for f in files) // len(files):,})"
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.provider == "cohere":
        cohere_key = os.environ.get("COHERE_API_KEY")
        if not cohere_key:
            print(
                "error: --provider=cohere requires COHERE_API_KEY in environment",
                file=sys.stderr,
            )
            return 2
        active_model = args.cohere_model
        print(f"Provider: Cohere API, model={active_model}")
    else:
        cohere_key = None
        active_model = args.model
        print(f"Provider: Ollama, model={active_model}, url={args.ollama_url}")

    summary: dict[str, Any] = {
        "provider": args.provider,
        "model": active_model,
        "ollama_url": args.ollama_url if args.provider == "ollama" else None,
        "n_files": args.n_files,
        "concurrency": args.concurrency,
        "ollama_num_parallel": (
            os.environ.get("OLLAMA_NUM_PARALLEL") if args.provider == "ollama" else None
        ),
        "source_parquet": args.source_parquet,
        "source_lang": args.source_lang,
        "target_langs": target_langs,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "by_language": {},
    }

    def backend_factory() -> Any:
        if args.provider == "cohere":
            assert cohere_key is not None
            return CohereBackend(api_key=cohere_key, model=args.cohere_model)
        return OpenAICompatBackend(
            base_url=args.ollama_url,
            model=args.model,
            api_key="ollama",
            timeout=args.timeout,
        )

    # CORE-974 (#3): auto-pace cohere when --min-call-delay was not passed.
    # Sentinel `None` from argparse means "use the per-provider default."
    # Existing explicit users (passing `--min-call-delay 0`, `30`, etc.)
    # are unaffected — only the omit-the-flag case picks up the new default.
    if args.min_call_delay is None:
        if args.provider == "cohere":
            args.min_call_delay = 30.0
            print(
                "Auto-set --min-call-delay=30.0 for --provider=cohere "
                "(c4ai-aya-expanse-32b empirical ~2 RPM cap). "
                "Pass --min-call-delay 0 to disable."
            )
        else:
            args.min_call_delay = 0.0

    # One limiter shared across every target language so the cap really is
    # per-process. Constructed here (not inside run_pilot) to ensure
    # `_last_call_t` carries across the lang boundary in multi-lang runs.
    shared_rate_limiter = _RateLimiter(args.min_call_delay)

    # CORE-974 (#1, #2): wall-time and budget circuit breakers, both shared
    # across all target langs in this run so a multi-lang run can't bypass
    # the cap by switching languages.
    deadline: float | None = None
    if args.max_wall_seconds is not None:
        deadline = time.perf_counter() + args.max_wall_seconds
        print(
            f"Wall-time circuit breaker armed: max {args.max_wall_seconds}s "
            f"(exit code 4 if exceeded)."
        )
    budget: _BudgetTracker | None = None
    if args.max_credits_usd is not None:
        budget = _BudgetTracker(
            max_credits_usd=args.max_credits_usd,
            cost_per_success_usd=args.cost_per_success_usd,
            cost_per_failure_usd=args.cost_per_failure_usd,
        )
        print(
            f"Budget circuit breaker armed: max ${args.max_credits_usd} USD "
            f"estimated (success=${args.cost_per_success_usd}/file, "
            f"failure=${args.cost_per_failure_usd}/file). Exit code 5 if exceeded."
        )

    # CORE-1049: load idx allowlist when provided. Fail fast on missing /
    # empty file so we don't silently translate zero rows.
    idx_allowlist: set[int] | None = None
    if args.idx_allowlist is not None:
        try:
            idx_allowlist = _load_idx_allowlist(args.idx_allowlist)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"Idx allowlist loaded: {len(idx_allowlist):,} idxs from "
            f"{args.idx_allowlist}. Rows outside the allowlist will be skipped."
        )

    for lang in target_langs:
        print(
            f"\n=== Translating {len(files)} files: "
            f"{args.source_lang} → {lang} (concurrency={args.concurrency}) ==="
        )
        try:
            pack = load_language_pack(lang, "python", args.python_version)
        except Exception as exc:
            print(
                f"  failed to load language pack for {lang}: {exc}",
                file=sys.stderr,
            )
            summary["by_language"][lang] = {
                "error": f"language_pack_load_failed: {exc}"
            }
            continue

        result = run_pilot(
            source_lang=args.source_lang,
            target_lang=lang,
            files=files,
            backend_factory=backend_factory,
            keyword_map=pack.keywords,
            builtin_map=pack.builtins,
            output_dir=output_root / lang,
            concurrency=args.concurrency,
            reserved_word_map=getattr(pack, "reserved_words", None),
            resume=args.resume,
            retry=not args.no_retry,
            retry_delay=args.retry_delay,
            rate_limiter=shared_rate_limiter,
            deadline=deadline,
            budget=budget,
            idx_allowlist=idx_allowlist,
        )
        summary["by_language"][lang] = result

        spend_str = (
            f" | spent ~${result['estimated_spend_usd']}"
            if result.get("estimated_spend_usd") is not None
            else ""
        )
        print(
            f"  AST pass: {result['ast_pass']}/{result.get('n_processed', len(files))} | "
            f"AST fail: {result['ast_fail']} | "
            f"runtime fail: {result['runtime_fail']} | "
            f"resumed: {result.get('resumed', 0)} | "
            f"skipped (allowlist): {result.get('skipped_not_in_allowlist', 0)} | "
            f"avg {result['avg_seconds_per_file']}s/file | "
            f"wall {result['wall_seconds']}s | "
            f"throughput {result['throughput_files_per_min']} files/min"
            f"{spend_str}"
        )

        # CORE-974: when a circuit breaker fires, surface it and stop
        # processing remaining target langs (the cap is shared, so every
        # subsequent lang would just trip immediately anyway).
        if result.get("aborted_reason"):
            print(
                f"  ABORTED: {result['aborted_reason']} — "
                "stopping multi-lang run before remaining languages.",
                file=sys.stderr,
            )
            break

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(
        json.dumps(
            {
                lang: {
                    "ast_pass": r.get("ast_pass"),
                    "ast_fail": r.get("ast_fail"),
                    "runtime_fail": r.get("runtime_fail"),
                    "resumed": r.get("resumed", 0),
                    "avg_seconds_per_file": r.get("avg_seconds_per_file"),
                    "wall_seconds": r.get("wall_seconds"),
                    "throughput_files_per_min": r.get("throughput_files_per_min"),
                    "aborted_reason": r.get("aborted_reason"),
                    "estimated_spend_usd": r.get("estimated_spend_usd"),
                }
                for lang, r in summary["by_language"].items()
            },
            indent=2,
        )
    )

    # CORE-974: distinct exit codes for circuit-breaker triggers so callers
    # (CI, orchestrator scripts, manual operators) can branch on them.
    for lang_result in summary["by_language"].values():
        reason = (
            lang_result.get("aborted_reason") if isinstance(lang_result, dict) else None
        )
        if reason and reason.startswith("wall_time_exceeded"):
            return 4
        if reason and reason.startswith("budget_exceeded"):
            return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())

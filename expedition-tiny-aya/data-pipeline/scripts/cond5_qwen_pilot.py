#!/usr/bin/env python3
"""Cond-5 translation pilot — Qwen via Ollama, full AYA-191 pipeline.

Translates a small slice of ``condition-1-en-103k`` Python files to
``{zh, es, ur}`` (or any subset) using a local Qwen model served by Ollama,
exercising the full Legesher Cond-5 pipeline:

    TreeSitterTranslator (keywords/builtins via language packs)
        +
    OpenAICompatProvider (LLM for identifiers/comments/docstrings/strings)
        ↓
    LLMTranslator.translate_code(...)

Designed to:
1. Validate that the AYA-191 pipeline runs end-to-end against a local LLM
2. Surface per-language quality issues (e.g., AYA-202 already showed Urdu is
   weak with smaller models)
3. Give a wall-clock estimate before scaling to the 5k subset or 103k pool

Linked Linear issue: AYA-213 (Condition 5 dataset rebuild — 9 datasets)

Usage:
    # Default: 20 files × {es, zh, ur}, qwen2.5:7b-instruct-q4_K_M
    python scripts/cond5_qwen_pilot.py

    # Faster iteration (smaller model, fewer files)
    python scripts/cond5_qwen_pilot.py --model qwen2.5:3b-instruct-q4_K_M --n-files 5

    # One language only
    python scripts/cond5_qwen_pilot.py --target-langs es

    # Different source parquet
    python scripts/cond5_qwen_pilot.py \\
        --source-parquet /path/to/train.parquet --n-files 10

    # Parallel files (requires OLLAMA_NUM_PARALLEL>=N on the server)
    OLLAMA_NUM_PARALLEL=4 ollama serve  # in a separate shell
    python scripts/cond5_qwen_pilot.py --concurrency 4 --n-files 20
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


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
) -> dict[str, Any]:
    """Translate ``files`` to ``target_lang`` and capture per-file outcomes.

    When ``concurrency > 1``, files are translated in parallel via a
    ``ThreadPoolExecutor``. Each worker thread gets its own
    ``OpenAICompatBackend`` (and therefore its own ``httpx.AsyncClient``)
    via ``threading.local`` — sharing one ``AsyncClient`` across threads
    that each call ``asyncio.run`` is unsupported by httpx's transport pool.

    ``resume=True`` skips files that already have an output ``.py`` in
    ``output_dir`` (typical use: long runs that got interrupted). ``retry=True``
    retries a failing translation once after ``retry_delay`` seconds — covers
    transient 429s, 5xx, and connection blips on hosted APIs without needing
    to restart the whole run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def _backend() -> OpenAICompatBackend:
        b = getattr(local, "backend", None)
        if b is None:
            b = backend_factory()
            local.backend = b
        return b

    def _translate_one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        code_en = row["code"]
        file_path = row.get("file_path") or row.get("metadata_file") or f"row_{idx}"
        out_path = output_dir / f"{idx:03d}.py"

        # Resume: short-circuit if output already exists from a prior run.
        if resume and out_path.exists():
            return {
                "idx": idx,
                "file_path": file_path,
                "status": "resumed",
                "ast": "skipped",
                "elapsed_seconds": 0.0,
                "input_chars": len(code_en),
                "output_chars": out_path.stat().st_size,
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
            return {
                "idx": idx,
                "file_path": file_path,
                "status": "runtime_error",
                "error": f"{type(last_exc).__name__}: {last_exc}",
                "elapsed_seconds": round(elapsed, 2),
                "input_chars": len(code_en),
            }

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
            "status": "ok",
            "ast": ast_status,
            "elapsed_seconds": round(elapsed, 2),
            "input_chars": len(code_en),
            "output_chars": len(translated),
        }

    per_file: list[dict[str, Any]] = []
    wall_t0 = time.perf_counter()

    if concurrency <= 1:
        for idx, row in enumerate(files):
            per_file.append(_translate_one(idx, row))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_translate_one, idx, row) for idx, row in enumerate(files)
            ]
            for fut in as_completed(futures):
                per_file.append(fut.result())
        per_file.sort(key=lambda r: r["idx"])

    wall_seconds = time.perf_counter() - wall_t0

    parse_pass = 0
    parse_fail = 0
    runtime_fail = 0
    resumed_count = 0
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
        total_output_chars += r.get("output_chars", 0)
        total_seconds += r["elapsed_seconds"]
        if r["ast"] == "pass":
            parse_pass += 1
        else:
            parse_fail += 1

    # Only files actually translated this run count toward the per-file
    # average; resumed files would skew it to ~0s.
    fresh_translated = len(files) - runtime_fail - resumed_count
    return {
        "target_lang": target_lang,
        "n_files": len(files),
        "concurrency": concurrency,
        "ast_pass": parse_pass,
        "ast_fail": parse_fail,
        "runtime_fail": runtime_fail,
        "resumed": resumed_count,
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
        )
        summary["by_language"][lang] = result

        print(
            f"  AST pass: {result['ast_pass']}/{len(files)} | "
            f"AST fail: {result['ast_fail']} | "
            f"runtime fail: {result['runtime_fail']} | "
            f"resumed: {result.get('resumed', 0)} | "
            f"avg {result['avg_seconds_per_file']}s/file | "
            f"wall {result['wall_seconds']}s | "
            f"throughput {result['throughput_files_per_min']} files/min"
        )

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
                }
                for lang, r in summary["by_language"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

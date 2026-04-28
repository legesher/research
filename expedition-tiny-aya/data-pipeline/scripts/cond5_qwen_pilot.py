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
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from legesher_i18n import load_language_pack
from legesher_i18n.api.providers import OpenAICompatProvider
from legesher_core.tree_sitter.llm_translator import LLMTranslator

DEFAULT_PARQUET = (
    "/Users/madisonedgar/GitHub/Legesher/research/expedition-tiny-aya/"
    "data-pipeline/packaged/condition-1-en-103k/train.parquet"
)
DEFAULT_OUTPUT_DIR = (
    "/Users/madisonedgar/GitHub/Legesher/research/expedition-tiny-aya/"
    "data-pipeline/cond5-qwen-pilot"
)


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
        type=int,
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
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def run_pilot(
    source_lang: str,
    target_lang: str,
    files: list[dict[str, Any]],
    backend: OpenAICompatBackend,
    keyword_map: dict[str, str],
    builtin_map: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Translate ``files`` to ``target_lang`` and capture per-file outcomes."""
    translator = LLMTranslator(
        keyword_map=keyword_map,
        builtin_map=builtin_map,
        backend=backend,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_file: list[dict[str, Any]] = []
    parse_pass = 0
    parse_fail = 0
    runtime_fail = 0
    total_input_chars = 0
    total_output_chars = 0
    total_seconds = 0.0

    for idx, row in enumerate(files):
        code_en = row["code"]
        file_path = row.get("file_path") or row.get("metadata_file") or f"row_{idx}"
        total_input_chars += len(code_en)

        t0 = time.perf_counter()
        try:
            translated_raw = translator.translate_code(
                code_en, source_lang, target_lang
            )
            # translate_code returns str by default; tuple only when
            # return_metadata=True. Narrow for the type checker + safety.
            translated = (
                translated_raw if isinstance(translated_raw, str) else translated_raw[0]
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            runtime_fail += 1
            err_path = output_dir / f"{idx:03d}.error.txt"
            err_path.write_text(
                f"file_path: {file_path}\nelapsed: {elapsed:.2f}s\n\n"
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            per_file.append(
                {
                    "idx": idx,
                    "file_path": file_path,
                    "status": "runtime_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
            continue

        elapsed = time.perf_counter() - t0
        total_seconds += elapsed
        total_output_chars += len(translated)

        # AST validation: reverse-translate keywords/builtins to English first,
        # then ast.parse. Cond-5 outputs use target-language keywords, which
        # standard Python ast.parse doesn't recognize.
        en_for_ast = reverse_keywords_and_builtins(translated, keyword_map, builtin_map)
        try:
            ast.parse(en_for_ast)
            ast_status = "pass"
            parse_pass += 1
        except SyntaxError as syn_exc:
            ast_status = f"fail: {syn_exc.msg} at line {syn_exc.lineno}"
            parse_fail += 1

        out_path = output_dir / f"{idx:03d}.py"
        out_path.write_text(translated, encoding="utf-8")

        # Side-by-side preview file (first 30 lines each side)
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

        per_file.append(
            {
                "idx": idx,
                "file_path": file_path,
                "status": "ok",
                "ast": ast_status,
                "elapsed_seconds": round(elapsed, 2),
                "input_chars": len(code_en),
                "output_chars": len(translated),
            }
        )

    return {
        "target_lang": target_lang,
        "n_files": len(files),
        "ast_pass": parse_pass,
        "ast_fail": parse_fail,
        "runtime_fail": runtime_fail,
        "total_seconds": round(total_seconds, 2),
        "avg_seconds_per_file": (
            round(total_seconds / max(1, len(files) - runtime_fail), 2)
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

    target_langs = [
        lang.strip() for lang in args.target_langs.split(",") if lang.strip()
    ]
    if not target_langs:
        print("error: --target-langs produced an empty list", file=sys.stderr)
        return 2

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

    summary: dict[str, Any] = {
        "model": args.model,
        "ollama_url": args.ollama_url,
        "n_files": args.n_files,
        "source_parquet": args.source_parquet,
        "source_lang": args.source_lang,
        "target_langs": target_langs,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "by_language": {},
    }

    backend = OpenAICompatBackend(
        base_url=args.ollama_url,
        model=args.model,
        api_key="ollama",
    )

    for lang in target_langs:
        print(f"\n=== Translating {len(files)} files: {args.source_lang} → {lang} ===")
        try:
            pack = load_language_pack(lang, "python", args.python_version)
        except Exception as exc:
            print(f"  failed to load language pack for {lang}: {exc}", file=sys.stderr)
            summary["by_language"][lang] = {
                "error": f"language_pack_load_failed: {exc}"
            }
            continue

        result = run_pilot(
            source_lang=args.source_lang,
            target_lang=lang,
            files=files,
            backend=backend,
            keyword_map=pack.keywords,
            builtin_map=pack.builtins,
            output_dir=output_root / lang,
        )
        summary["by_language"][lang] = result

        print(
            f"  AST pass: {result['ast_pass']}/{len(files)} | "
            f"AST fail: {result['ast_fail']} | "
            f"runtime fail: {result['runtime_fail']} | "
            f"avg {result['avg_seconds_per_file']}s/file | "
            f"total {result['total_seconds']}s"
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
                    "avg_seconds_per_file": r.get("avg_seconds_per_file"),
                }
                for lang, r in summary["by_language"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

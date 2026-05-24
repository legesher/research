"""Refined post-hoc extractor for Phase-3 `_results_*.json` files.

Phase 3 of the tiny-aya expedition scores model outputs at two layers:

  1. An inference-time extractor in `evaluate.ipynb` (the `%%writefile
     run_eval_single.py` cell). Scoped to canonical English category labels,
     this runs on Kaggle as each row is generated and writes the live
     `pred`/`correct`/`{cell}_acc` fields into `_results_*.json` /
     `_summary_*.json`. It is FROZEN — it's the exact code that produced the
     Phase-3 corpus and must not change.

  2. This module — a refined post-hoc extractor that revisits the same
     `raw_output` fields with broader scope: native-script answers (Urdu
     سائنس/ٹکنالوجی, Chinese 科学/技术, Spanish ciencia y tecnología), the
     Arabic forms the Urdu-prompted model code-switches into, compound-answer
     hedges, and CJK-glued embeddings the word-boundary regex can't see.
     This refined view surfaces the model's actual multilingual
     category-naming behavior, which the inference-time extractor wasn't
     built to read.

The paper's reported numbers come from this module — running the script
entry point against Phase-3's `_results_*.json` produces sibling
`_summary_reparsed_*.json` files; the originals are never touched.

Usage:
    # Print before/after diff table only:
    python reparse_results.py path/to/condition-2-ur-5k_seed42_smoke20_results_template1.json
    python reparse_results.py path/*.json --extractors sib200

    # Also write a sibling `_summary_reparsed_{template}.json` next to each input:
    python reparse_results.py path/*_results_*.json --write-reparsed-summary

Self-contained: the extractor logic lives entirely in this file. No notebook
read, no AST surgery, no on-disk run_eval_single.py required. Reviewers
reproducing the paper read this single file to understand the scoring; the
content_sha256 in each reparsed summary's provenance block hashes this file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

# Static metadata that callers may need before the extractor source is
# available — argparse choices, path helpers, etc.
EXTRACTOR_NAMES: tuple[str, ...] = ("belebele", "csqa", "sib200", "xnli")

HERE = Path(__file__).resolve().parent


# =============================================================================
# Refined Phase-3 extractors
# =============================================================================
# These produce the numbers the paper reports. Phase 3's evaluate.ipynb is
# frozen at its inference-time extractor (scoped to canonical English labels)
# so `raw_output` matches the run that actually happened on Kaggle. The
# functions below revisit those raw outputs with broader scope — native
# scripts, code-switched forms, compound-answer hedges, CJK-glued
# embeddings — to surface multilingual category-naming behavior the
# inference-time scorer wasn't built to read.

SIB200_CATEGORIES = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)

# Unified surface-form -> category map. Each key is one *piece* of a model
# answer (compound answers are split on separators/conjunctions before
# lookup), mapped to one of the 7 canonical SIB-200 categories. Covers
# English canonical labels, aliases, and science/technology sub-topics, plus
# native-language category words observed in the Phase-3 eval outputs:
# Urdu, Chinese, Spanish, and Arabic (the Urdu-prompted model code-switches
# to Arabic). Native-language forms are confirmed by native speakers — see
# analysis/{urdu,chinese,spanish}-surface-forms-review.md.
#
# Note: the SIB-200 prompt presents the 7 categories in English regardless of
# instruction language. A native-language answer is therefore a deliberate
# *lenient* scoring choice — we credit the model for identifying the topic
# even though it did not answer in the requested English label. The
# strict-vs-lenient gap is reported as an instruction-following measure.
SIB200_TERM_TO_CATEGORY = {
    # --- English canonical + aliases ---
    "science/technology": "science/technology",
    "science": "science/technology",
    "technology": "science/technology",
    "travel": "travel",
    "politics": "politics",
    "sports": "sports",
    "sport": "sports",
    "health": "health",
    "entertainment": "entertainment",
    "geography": "geography",
    # --- English science/technology sub-topics ---
    "physics": "science/technology",
    "chemistry": "science/technology",
    "transportation": "science/technology",
    "telecommunications": "science/technology",
    "internet security": "science/technology",
    "interactive design": "science/technology",
    "ai": "science/technology",
    # --- Urdu (analysis/urdu-surface-forms-review.md) ---
    "سائنس": "science/technology",            # sains - science
    "ٹکنالوجی": "science/technology",  # tiknaloji - technology
    "تکنالوجی": "science/technology",  # taknaloji - technology variant
    "ٹیکنالوجی": "science/technology",  # taiknaloji - technology variant
    "تکنولوجی": "science/technology",  # teknoloji - technology variant
    "علم": "science/technology",                       # ilm - knowledge/science
    "تکنیک": "science/technology",            # teknik - technique
    "انٹرایکٹو ڈیزائن": "science/technology",  # interactive design
    "سیٹیلائٹ فون": "science/technology",  # satellite phone
    "انٹرنیٹ پراکسی": "science/technology",  # internet proxy
    "ٹرانسپورٹیشن": "science/technology",  # transportation
    "سیاست": "politics",                     # siyasat - politics
    "کھیل": "sports",                             # khel - sport/game
    "سپورٹس": "sports",                 # sports - transliteration
    "سفر": "travel",                                   # safar - travel/journey
    "سیاحت": "travel",                       # siyahat - tourism
    "مسافرت": "travel",                 # musafarat - travel
    "مسافر": "travel",                       # musafir - traveller
    "صحت": "health",                                   # sehat - health
    "تفریح": "entertainment",                # tafrih - recreation/entertainment
    # --- Chinese (analysis/chinese-surface-forms-review.md) ---
    "科学": "science/technology",                             # kexue - science
    "技术": "science/technology",                             # jishu - technology
    "公共交通": "travel",                             # gonggong jiaotong - public transport (reviewer decision: travel, not sci/tech)
    "政治": "politics",                                       # zhengzhi - politics
    "体育": "sports",                                         # tiyu - sports
    "旅行": "travel",                                         # luxing - travel
    "旅游": "travel",                                         # luyou - tourism/travel
    "娱乐": "entertainment",                                  # yule - entertainment
    "地理": "geography",                                      # dili - geography
    # --- Spanish (analysis/spanish-surface-forms-review.md) ---
    "ciencia": "science/technology",
    "tecnología": "science/technology",                          # tecnologia
    "tecnologia": "science/technology",                               # accent-stripped variant
    "política": "politics",                                      # politica
    "politica": "politics",                                           # accent-stripped variant
    "deportes": "sports",
    "viajes": "travel",
    "viaje": "travel",
    "salud": "health",
    "entretenimiento": "entertainment",
    # --- Arabic (Urdu-prompted model code-switches; urdu-review section D) ---
    "السياسة": "politics",         # as-siyasa - politics
    "سياسة": "politics",                     # siyasa - politics
    "الرياضة": "sports",           # ar-riyada - sports
    "رياضة": "sports",                       # riyada - sports
    "التكنولوجيا": "science/technology",  # at-tiknolojiya
    "تكنولوجيا": "science/technology",  # tiknolojiya
}

# Conjunction words the model uses to join two terms ("science and technology",
# "ciencia y tecnologia", Chinese science-and-technology forms). Normalised to
# "/" before splitting so a compound's pieces can be classified independently.
SIB200_CONJUNCTIONS = (" and ", " y ", " e ", " و ", "和", "与")

# Characters stripped from the answer (and from each compound piece).
SIB200_STRIP = " \t.,:;!?()[]{}\"'،۔。"


def _sib200_split(line: str) -> list[str]:
    """Split a model answer into its category pieces. Compounds joined by
    '/', a comma, '&', '+', or a conjunction word become separate pieces."""
    s = line
    for conj in SIB200_CONJUNCTIONS:
        s = s.replace(conj, "/")
    parts = re.split(r"[/،,&;+]", s)
    return [p.strip().strip(SIB200_STRIP).strip() for p in parts if p.strip()]


def extract_sib200_category(text: str):
    """Map a model's SIB-200 answer to one of the 7 categories, or None.

    The model's answer is split into pieces (compounds split on separators /
    conjunctions); each piece is resolved against SIB200_TERM_TO_CATEGORY.
    The answer counts only if it references exactly ONE distinct category:

      - exactly 1 distinct category  -> that category
      - 2+ distinct categories       -> None  (the model hedged — a deliberate
                                              parse-failure; see the decision
                                              ledger's compound policy)
      - 0 distinct categories        -> a substring fallback scan for the
                                        canonical English names, else None

    Any number of pieces is fine as long as they all collapse to one category
    (`science / technology / AI / physics` -> all science/technology -> ok);
    the failure trips only on a genuine cross-category split
    (`science / politics`, politics/technology compounds).
    """
    first_line = text.strip().split("\n")[0].strip()
    first_line = first_line.strip(SIB200_STRIP)
    if not first_line:
        return None

    categories = set()
    for piece in _sib200_split(first_line):
        cat = SIB200_TERM_TO_CATEGORY.get(piece.lower())
        if cat is not None:
            categories.add(cat)

    if len(categories) == 1:
        return next(iter(categories))
    if len(categories) >= 2:
        return None  # multi-category hedge

    # Fallback: a canonical English name embedded in a longer first line.
    # Plain substring (matches the Phase-3 inference-time extractor): catches
    # English embeddings like "the answer is travel" AND CJK-glued forms
    # like "答案是travel" — Python's Unicode \b would refuse a boundary
    # between a CJK char (\w in unicode-aware re) and a Latin letter.
    line_lower = first_line.lower()
    found = {cat for cat in SIB200_CATEGORIES if cat in line_lower}
    if len(found) == 1:
        return next(iter(found))
    return None


# XNLI re-scoring is tiered by how directly the model names a label.
# Tier 1: a verbatim English label, or a native label word.
# Tier 2: the literal English label word embedded in a CJK sentence frame
#         (the \b word-boundary regex misses it because CJK chars are \w).
# Tier 3: a native-prose paraphrase of the relationship, no label word at all.
# See analysis/reparse-decision-ledger.md for the full per-tier rationale.
NATIVE_LABEL_MAP = {
    # Chinese
    "蕴含": "entailment", "蕴涵": "entailment",
    "矛盾": "contradiction",
    "中立": "neutral",
    # Spanish ("neutral" matches the English regex by coincidence; add native)
    "implicación": "entailment", "implicacion": "entailment",
    "contradicción": "contradiction", "contradiccion": "contradiction",
    "neutro": "neutral", "neutra": "neutral",
    # Urdu
    "لازمی": "entailment",
    "لازم آتی ہے": "entailment",
    "انضمامیت": "entailment",
    "تردید": "contradiction",
    "غیرجانبدار": "neutral",
}
XNLI_LABELS = ("entailment", "contradiction", "neutral")
XNLI_LABEL_RES = {label: re.compile(rf"\b{label}\b") for label in XNLI_LABELS}

# Tier 2: a negation marker anywhere in the line means the model is *denying*
# the glued label ("没有entailment" = "there is no entailment"). When present,
# the glued word must NOT be taken as the answer — the negated frames are a
# deliberate parse-failure unless a Tier-3 paraphrase pattern also fires
# (see decision ledger, XNLI Tier 2).
XNLI_TIER2_NEGATION = ("没有", "沒有")

# Tier 3: native-prose paraphrases that imply a label without naming it.
# Ordered — first match wins. The mechanism is final; the phrase list is
# PENDING native-speaker confirmation (decision ledger, XNLI Tier 3).
XNLI_PARAPHRASE_RES = [
    (re.compile(r"没有.{0,8}关系"), "neutral"),       # "no relationship"
    (re.compile(r"没有.{0,8}关联"), "neutral"),       # "no association"
    (re.compile(r"没有.{0,8}联系"), "neutral"),       # "no connection"
    (re.compile(r"کوئی.{0,15}تعلق.{0,8}نہیں"), "neutral"),  # Urdu "no (clear) relationship"
    (re.compile(r"否定"), "contradiction"),           # "negation"
    (re.compile(r"直接结果|自然结果|必然结果"), "entailment"),  # "(direct/natural) result"
    (re.compile(r"推论|推断"), "entailment"),         # "inference / corollary"
    (re.compile(r"等同|等价"), "entailment"),         # "equivalent"
]


def extract_xnli_label(text: str):
    """Map a model's XNLI answer to entailment/contradiction/neutral, or None.
    See the tier ordering comment above XNLI_TIER2_NEGATION."""
    first_line = text.strip().split("\n")[0].strip()
    first_line_lower = first_line.lower()

    # Tier 1a — verbatim English label (word-boundary match).
    for label, label_re in XNLI_LABEL_RES.items():
        if label_re.search(first_line_lower):
            return label
    # Tier 1b — native label word.
    for native, english in NATIVE_LABEL_MAP.items():
        if native.lower() in first_line_lower:
            return english
    # Tier 2 — English label glued to a CJK frame. Skipped entirely if the
    # line carries a negation marker (the model is denying the label).
    if not any(neg in first_line for neg in XNLI_TIER2_NEGATION):
        for label in XNLI_LABELS:
            if label in first_line_lower:
                return label
    # Tier 3 — native-prose paraphrase of the relationship.
    for pat, label in XNLI_PARAPHRASE_RES:
        if pat.search(first_line):
            return label
    return None


def extract_choice(text: str, choices: str = "ABCDE"):
    """X-CSQA / Belebele multiple-choice extractor. Unchanged from the
    Phase-3 inference-time extractor in evaluate.ipynb; included here so
    this module is the single source for all four benchmarks' scoring."""
    text = text.strip().upper()
    first_line = text.split("\n")[0].strip()
    choice_class = re.escape(choices)
    match = re.search(rf"\b([{choice_class}])\b", first_line)
    if match:
        return match.group(1)
    match = re.search(rf"ANSWER\s*[:\-]?\s*([{choice_class}])", first_line)
    if match:
        return match.group(1)
    return None


# =============================================================================
# Reparse driver
# =============================================================================

def verify_extractor_source() -> Path:
    """Preflight kept for API compatibility (called by upload_reparsed_summaries.py).
    With inline extractors there's nothing to locate — this module's file IS
    the extractor source — so the call is now a trivial success.
    """
    return Path(__file__).resolve()


def _load_extractors() -> dict[str, Callable]:
    """Return the extractor dict used by `reparse_file`. Extractors are
    module-level functions; the dict adapts `extract_choice`'s `choices`
    argument for each multiple-choice benchmark."""
    return {
        "xnli": extract_xnli_label,
        "csqa": lambda t: extract_choice(t, choices="ABCDE"),
        "belebele": lambda t: extract_choice(t, choices="ABCD"),
        "sib200": extract_sib200_category,
    }


def benchmark_from_key(cell_key: str) -> str:
    """`template1_sib200_data=ur_instr=ur` → `sib200`. Pure string parsing,
    doesn't need extractors loaded."""
    for bench in EXTRACTOR_NAMES:
        if f"_{bench}_" in cell_key:
            return bench
    raise ValueError(f"Couldn't infer benchmark from {cell_key!r}")


def reparsed_summary_path_local(input_path: Path) -> Path:
    """`X_results_template1.json` → `X_summary_reparsed_template1.json`.

    Single source of truth for path mangling; used by both the local-file
    workflow (`--write-reparsed-summary`) and the HF round-trip driver."""
    name = input_path.name
    if "_results_" not in name:
        raise ValueError(f"Input filename must contain '_results_': {input_path.name}")
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    return input_path.with_name(new_name)


def reparsed_summary_path_remote(remote_results_path: str) -> str:
    """Same substitution as `reparsed_summary_path_local`, but for HF-style
    forward-slash paths (`phase3/conditions/<cond>/seed<N>/X_results_*.json`).
    Returns the modified path as a string, preserving the parent directory."""
    name = remote_results_path.split("/")[-1]
    if "_results_" not in name:
        raise ValueError(f"Input filename must contain '_results_': {name}")
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    if "/" in remote_results_path:
        parent = remote_results_path.rsplit("/", 1)[0]
        return f"{parent}/{new_name}"
    return new_name


def reparse_file(path: Path, only: set[str] | None = None) -> list[dict]:
    """Re-run extractors on raw_output, return per-cell before/after rows.

    `only` filters which benchmarks are scored (display filter for the diff
    table). When writing a reparsed summary you'd typically pass `only=None`
    so every cell in the original is recomputed and the schema stays
    consistent with the source."""
    extractors = _load_extractors()

    with path.open() as f:
        data = json.load(f)

    old_summary = data.get("summary", {})
    old_failure_rates = data.get("parse_failure_rates", {})

    rows = []
    for key, items in data.items():
        if key in {"summary", "parse_failure_rates"}:
            continue
        if not isinstance(items, list) or not items or "raw_output" not in items[0]:
            continue
        bench = benchmark_from_key(key)
        if only is not None and bench not in only:
            continue

        extractor = extractors[bench]
        new_correct = 0
        new_fail = 0
        for row in items:
            new_pred = extractor(row["raw_output"])
            if new_pred is None:
                new_fail += 1
            elif new_pred == row["gold"]:
                new_correct += 1

        n = len(items)
        old_acc = old_summary.get(f"{key}_acc")
        old_fail = old_failure_rates.get(key)
        rows.append(
            {
                "cell": key,
                "n": n,
                "old_acc": old_acc,
                "new_acc": new_correct / n if n else 0.0,
                "old_fail": old_fail,
                "new_fail": new_fail / n if n else 0.0,
            }
        )
    return rows


def print_diff_table(rows: list[dict]) -> None:
    changed = [r for r in rows if r["new_acc"] != r["old_acc"]]
    if not changed:
        print("No accuracy changes.")
        return

    print(
        f"{'cell':<55} {'old_acc':>8} {'new_acc':>8} {'Δacc':>7} "
        f"{'old_fail':>9} {'new_fail':>9}"
    )
    print("-" * 100)
    for r in sorted(
        changed, key=lambda x: x["new_acc"] - (x["old_acc"] or 0), reverse=True
    ):
        d = r["new_acc"] - (r["old_acc"] or 0)
        print(
            f"{r['cell']:<55} "
            f"{r['old_acc']:>8.3f} {r['new_acc']:>8.3f} {d:>+7.3f} "
            f"{r['old_fail']:>9.3f} {r['new_fail']:>9.3f}"
        )

    avg_delta = sum(r["new_acc"] - (r["old_acc"] or 0) for r in changed) / len(changed)
    print(f"\n{len(changed)} cells changed; mean Δacc = {avg_delta:+.3f}")


def _extractor_provenance() -> dict:
    """Identify which version of this module produced the numbers.

    Hashes reparse_results.py itself (the extractor source) so anyone with
    the same checkout can verify they're scoring with the same logic.
    `source_path` is the basename — full paths would leak the runner's local
    filesystem layout into the published summary files. Adds
    `repo_head_commit` when running inside a git checkout."""
    src = Path(__file__).resolve()
    provenance: dict = {
        "source_path": src.name,
        "content_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=src.parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if head_sha:
            provenance["repo_head_commit"] = head_sha
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass
    return provenance


def build_reparsed_summary(input_path: Path, rows: list[dict]) -> dict:
    """Construct the JSON body for a `_summary_reparsed_{template}.json` sibling.

    Mirrors the original `_summary_*.json` schema (`summary` + `parse_failure_rates`
    top-level keys) so downstream analysis can read original and reparsed
    interchangeably. Adds a `reparse_metadata` block recording when, against
    which extractor version, and what changed.

    Always treats `rows` as a complete recompute — callers should pass rows
    from `reparse_file(path, only=None)`. If you need a partial recompute
    for the diff-table display, run reparse_file twice (once filtered for
    display, once unfiltered for writing) — main() handles this for you."""
    new_summary: dict[str, float] = {}
    new_failure_rates: dict[str, float] = {}
    delta_per_cell: dict[str, dict] = {}

    for r in rows:
        cell = r["cell"]
        new_summary[f"{cell}_acc"] = r["new_acc"]
        new_failure_rates[cell] = r["new_fail"]
        d_acc = r["new_acc"] - (r["old_acc"] or 0)
        d_fail = r["new_fail"] - (r["old_fail"] or 0)
        if abs(d_acc) > 1e-9 or abs(d_fail) > 1e-9:
            delta_per_cell[cell] = {
                "old_acc": r["old_acc"],
                "new_acc": r["new_acc"],
                "delta_acc": d_acc,
                "old_fail": r["old_fail"],
                "new_fail": r["new_fail"],
                "delta_fail": d_fail,
            }

    return {
        "summary": new_summary,
        "parse_failure_rates": new_failure_rates,
        "reparse_metadata": {
            "reparsed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "original_results_filename": input_path.name,
            "extractors_applied": list(EXTRACTOR_NAMES),
            "extractor_provenance": _extractor_provenance(),
            "cells_changed": len(delta_per_cell),
            "delta_per_cell": delta_per_cell,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=sorted(EXTRACTOR_NAMES),
        help="Filter the printed diff table to these benchmarks (default: all). "
        "Note: when combined with --write-reparsed-summary, the WRITTEN summary "
        "still includes every benchmark — this flag only filters what's printed.",
    )
    parser.add_argument(
        "--write-reparsed-summary",
        action="store_true",
        help=(
            "Write a sibling `_summary_reparsed_{template}.json` next to each "
            "input. Always re-runs every extractor so the summary stays a "
            "drop-in replacement for the original. Preserves original "
            "`_summary_*.json` and `_results_*.json` untouched."
        ),
    )
    args = parser.parse_args()

    display_filter = set(args.extractors) if args.extractors else None
    for path in args.files:
        print(f"\n=== {path.name} ===")
        # For the diff table, optionally filter to selected extractors.
        rows_for_display = reparse_file(path, only=display_filter)
        print_diff_table(rows_for_display)

        if args.write_reparsed_summary:
            # For the written summary, always recompute every benchmark so
            # the output is a complete drop-in for the original summary.
            rows_full = (
                rows_for_display
                if display_filter is None
                else reparse_file(path, only=None)
            )
            out_path = reparsed_summary_path_local(path)
            body = build_reparsed_summary(path, rows_full)
            out_path.write_text(
                json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  → wrote {out_path.name} "
                f"({body['reparse_metadata']['cells_changed']} cells changed)"
            )


if __name__ == "__main__":
    main()

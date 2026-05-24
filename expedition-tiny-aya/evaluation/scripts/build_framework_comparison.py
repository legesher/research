"""Phase-3 comparison-framework artefacts (paper-results-oriented).

Companion to `build_comparison.py` (original-vs-refined-extractor validation)
and `build_vs_baseline.py` (condition-vs-baseline rollups). This script
operationalizes the *paper's* comparison framework so the results section
can be built directly from the emitted TSVs.

The framework has three axes (see memory: feedback-comparison-framework):

  Axis 1 — Within-condition template comparison.
           For each (condition, seed, benchmark, data_lang, instr_lang),
           surface template1_acc vs template2_acc side-by-side with diff.
           → template_comparison.tsv

  Axis 2 — Cross-condition same-language.
           For each target language X ∈ {en, zh, es, ur}, group every
           condition trained on or anchored to X, with baseline and
           cond-1-en-5k as standing anchors.
           → same_language_comparison.tsv

  Axis 3 — Within-language data-volume comparison (5k vs 20k).
           For each (target_lang, template, bench, data_lang, instr_lang)
           cell, show baseline + cond-1-en-5k + cond-2-X-5k + cond-2-X-20k
           with the (20k − 5k) delta.
           → data_volume_comparison.tsv

Every cell is also annotated with `matched_instr` (instr_lang ==
fine_tune_lang) and `matched_diagonal` (data_lang == instr_lang) so the
paper writers can filter directly to the cells the structural findings
talk about (paper-prep-callouts items 7-11).

Inputs:
  Reads HF-snapshot summary files from $PHASE3_SNAPSHOT_ROOT (default
  `/tmp/phase3_reparse/hf_snapshot/phase3/conditions`). Uses ONLY the
  reparsed-extractor numbers (the paper-grade view); the original-
  extractor numbers are excluded here on purpose — for original-vs-refined
  comparison use `build_comparison.py` instead.

Outputs:
  Written into $PHASE3_OUT_DIR (default `/tmp/phase3_reparse`); the
  intended final destination is
  `expedition-tiny-aya/analysis/reparse-tables/`.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(
    os.environ.get(
        "PHASE3_SNAPSHOT_ROOT",
        "/tmp/phase3_reparse/hf_snapshot/phase3/conditions",
    )
)
OUT = Path(os.environ.get("PHASE3_OUT_DIR", "/tmp/phase3_reparse"))

CELL_RE = re.compile(
    r"^template(?P<t>\d+)_(?P<bench>xnli|csqa|sib200|belebele)_data=(?P<data>[a-z]{2})_instr=(?P<instr>[a-z]{2})$"
)
SEED_FROM_NAME = re.compile(r"_seed(?P<seed>[A-Za-z0-9]+)_summary")

# Condition → target-language mapping. cond-1-en is the "coding constant"
# control (English fine-tune); cond-2-X conditions fine-tune on language X;
# cond-3-zh and cond-5-X are additional variations on those targets.
CONDITION_TARGET_LANG = {
    "baseline": None,                  # no fine-tune target
    "condition-1-en-5k": "en",
    "condition-1-en-20k": "en",
    "condition-2-zh-5k": "zh",
    "condition-2-zh-20k": "zh",
    "condition-2-es-5k": "es",
    "condition-2-es-20k": "es",
    "condition-2-ur-5k": "ur",
    "condition-2-ur-20k": "ur",
    "condition-3-zh-5k": "zh",
    "condition-5-zh-5k": "zh",
    "condition-5-es-5k": "es",
    "condition-5-ur-5k": "ur",
}

# Data-volume tags for axis 3.
CONDITION_DATA_VOLUME = {
    "baseline": None,
    "condition-1-en-5k": "5k",
    "condition-1-en-20k": "20k",
    "condition-2-zh-5k": "5k",
    "condition-2-zh-20k": "20k",
    "condition-2-es-5k": "5k",
    "condition-2-es-20k": "20k",
    "condition-2-ur-5k": "5k",
    "condition-2-ur-20k": "20k",
    "condition-3-zh-5k": "5k",
    "condition-5-zh-5k": "5k",
    "condition-5-es-5k": "5k",
    "condition-5-ur-5k": "5k",
}


def load(p):
    return json.loads(p.read_text())


def gather() -> list[dict]:
    """Walk the HF snapshot, return one row per (condition, seed, template,
    benchmark, data, instr) with the refined-extractor accuracy + count +
    correct. Uses the reparsed summary file; falls back to gracefully skip
    sessions without a reparsed sibling (shouldn't happen post HF PR #34)."""
    rows: list[dict] = []
    for cond_dir in sorted(ROOT.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for seed_dir in sorted(cond_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            dir_seed = seed_dir.name.removeprefix("seed") if seed_dir.name.startswith("seed") else seed_dir.name
            for rep in sorted(seed_dir.glob("*_summary_reparsed_template*.json")):
                m_seed = SEED_FROM_NAME.search(rep.name.replace("_reparsed", ""))
                file_seed = m_seed["seed"] if m_seed else dir_seed
                # Filename mismatch with parent dir indicates a stray; skip.
                if dir_seed not in ("none",) and file_seed != dir_seed:
                    continue
                template = rep.stem.rsplit("_", 1)[-1]
                rd = load(rep)
                rep_summary = rd.get("summary", {})
                rep_pf = rd.get("parse_failure_rates", {})
                for k in rep_summary:
                    if not k.endswith("_acc"):
                        continue
                    base = k.removesuffix("_acc")
                    m = CELL_RE.match(base)
                    if not m:
                        continue
                    rows.append({
                        "condition": cond,
                        "seed": file_seed,
                        "template": m["t"],
                        "benchmark": m["bench"],
                        "data": m["data"],
                        "instr": m["instr"],
                        "acc": rep_summary.get(base + "_acc"),
                        "count": rep_summary.get(base + "_count"),
                        "correct": rep_summary.get(base + "_correct"),
                        "pf": rep_pf.get(base),
                    })
    return rows


def annotate(rows: list[dict]) -> list[dict]:
    """Add `matched_instr` (instr == fine_tune_lang), `matched_diagonal`
    (data == instr), target_lang, and data_volume per row."""
    for r in rows:
        target = CONDITION_TARGET_LANG.get(r["condition"])
        r["target_lang"] = target if target else ""
        r["data_volume"] = CONDITION_DATA_VOLUME.get(r["condition"]) or ""
        r["matched_instr"] = "y" if target and r["instr"] == target else "n"
        r["matched_diagonal"] = "y" if r["data"] == r["instr"] else "n"
    return rows


# =============================================================================
# Axis 1 — Within-condition template comparison
# =============================================================================
def write_template_comparison(rows: list[dict], out: Path):
    """One row per (condition, seed, benchmark, data, instr) with template-1
    and template-2 accuracies side-by-side and a t2−t1 diff. Surfaces the
    template-asymmetry pattern (paper-prep-callouts item 11)."""
    pivot: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        key = (r["condition"], r["seed"], r["benchmark"], r["data"], r["instr"])
        pivot[key][f"t{r['template']}"] = r

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "seed", "benchmark", "data", "instr",
            "matched_instr", "matched_diagonal",
            "t1_acc", "t2_acc", "delta_t2_t1",
            "t1_count", "t2_count",
            "t1_correct", "t2_correct",
        ]) + "\n")
        for key in sorted(pivot.keys()):
            row = pivot[key]
            t1 = row.get("t1")
            t2 = row.get("t2")
            if not (t1 and t2):
                continue  # need both templates to compare
            delta = t2["acc"] - t1["acc"]
            f.write("\t".join([
                t1["condition"], t1["seed"], t1["benchmark"], t1["data"], t1["instr"],
                t1["matched_instr"], t1["matched_diagonal"],
                f"{t1['acc']:.4f}", f"{t2['acc']:.4f}", f"{delta:+.4f}",
                str(t1["count"]), str(t2["count"]),
                str(t1["correct"]), str(t2["correct"]),
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# Axis 2 — Cross-condition same-language comparison
# =============================================================================
def write_same_language_comparison(rows: list[dict], out: Path):
    """For each (target_lang, template, benchmark, data, instr) cell, list
    every condition that ran it, anchored against baseline and cond-1-en-5k.

    The point is to make 'how do all conditions trained on Urdu compare to
    each other and to the anchors' a single-glance question per cell.
    """
    # Build a lookup: cell -> {condition -> per-seed-mean acc}
    cell_cond: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["template"], r["benchmark"], r["data"], r["instr"])
        cell_cond[key][r["condition"]].append(r["acc"])

    # Anchor conditions present in every group of comparisons
    ANCHOR_CONDITIONS = ("baseline", "condition-1-en-5k", "condition-1-en-20k")

    with out.open("w") as f:
        f.write("\t".join([
            "target_lang", "template", "benchmark", "data", "instr",
            "matched_diagonal", "condition", "data_volume",
            "matched_instr_for_this_condition",
            "mean_acc_across_seeds", "n_seeds",
        ]) + "\n")
        for target in ("en", "zh", "es", "ur"):
            conditions_for_target = [
                c for c, t in CONDITION_TARGET_LANG.items() if t == target
            ]
            # Always include anchors
            cond_list = list(ANCHOR_CONDITIONS) + [
                c for c in conditions_for_target if c not in ANCHOR_CONDITIONS
            ]
            for cell in sorted(cell_cond.keys()):
                template, bench, data, instr = cell
                matched_diag = "y" if data == instr else "n"
                for cond in cond_list:
                    accs = cell_cond[cell].get(cond, [])
                    if not accs:
                        continue
                    matched_instr_cond = (
                        "y" if (CONDITION_TARGET_LANG.get(cond) == instr) else "n"
                    )
                    f.write("\t".join([
                        target, template, bench, data, instr,
                        matched_diag, cond,
                        CONDITION_DATA_VOLUME.get(cond) or "",
                        matched_instr_cond,
                        f"{mean(accs):.4f}", str(len(accs)),
                    ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# Axis 3 — Within-language data-volume comparison (5k vs 20k)
# =============================================================================
def write_data_volume_comparison(rows: list[dict], out: Path):
    """For each (target_lang, template, benchmark, data, instr) cell, show
    5k vs 20k pairs side-by-side with the (20k − 5k) delta. Includes
    baseline and cond-1-en pairs as anchors so the reader can size the
    delta against unrelated baselines.

    Question this axis answers: does more training data move the needle?
    Small delta = data isn't the bottleneck.
    """
    # group by (target_lang, family_prefix, template, bench, data, instr)
    # family_prefix is "cond-2-X" or "cond-1-en" — i.e., the (cond family, target)
    # pair we want to compare 5k vs 20k within
    FAMILY = {
        "condition-1-en-5k": "condition-1-en",
        "condition-1-en-20k": "condition-1-en",
        "condition-2-zh-5k": "condition-2-zh",
        "condition-2-zh-20k": "condition-2-zh",
        "condition-2-es-5k": "condition-2-es",
        "condition-2-es-20k": "condition-2-es",
        "condition-2-ur-5k": "condition-2-ur",
        "condition-2-ur-20k": "condition-2-ur",
    }

    pivot: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        fam = FAMILY.get(r["condition"])
        if not fam:
            continue
        key = (
            r["target_lang"], fam, r["template"], r["benchmark"], r["data"], r["instr"]
        )
        pivot[key].setdefault(r["data_volume"], []).append(r["acc"])

    with out.open("w") as f:
        f.write("\t".join([
            "target_lang", "family", "template", "benchmark", "data", "instr",
            "matched_instr", "matched_diagonal",
            "acc_5k_mean", "n_seeds_5k",
            "acc_20k_mean", "n_seeds_20k",
            "delta_20k_minus_5k",
        ]) + "\n")
        for key in sorted(pivot.keys()):
            target, fam, template, bench, data, instr = key
            volumes = pivot[key]
            accs_5k = volumes.get("5k", [])
            accs_20k = volumes.get("20k", [])
            if not (accs_5k and accs_20k):
                continue  # only emit rows where both volumes exist
            matched_diag = "y" if data == instr else "n"
            matched_inst = "y" if target == instr else "n"
            mean_5k = mean(accs_5k)
            mean_20k = mean(accs_20k)
            f.write("\t".join([
                target, fam, template, bench, data, instr,
                matched_inst, matched_diag,
                f"{mean_5k:.4f}", str(len(accs_5k)),
                f"{mean_20k:.4f}", str(len(accs_20k)),
                f"{mean_20k - mean_5k:+.4f}",
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# Main
# =============================================================================
def main():
    rows = annotate(gather())
    print(f"Loaded {len(rows)} cell-rows from {ROOT}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_template_comparison(rows, OUT / "framework_template_comparison.tsv")
    write_same_language_comparison(rows, OUT / "framework_same_language_comparison.tsv")
    write_data_volume_comparison(rows, OUT / "framework_data_volume_comparison.tsv")

    print()
    print("Done. The three framework TSVs encode the paper's results axes:")
    print("  Axis 1: framework_template_comparison.tsv   (t1 vs t2 within each condition/seed)")
    print("  Axis 2: framework_same_language_comparison.tsv  (all conditions for each target lang, anchored)")
    print("  Axis 3: framework_data_volume_comparison.tsv  (5k vs 20k, per condition family)")


if __name__ == "__main__":
    main()
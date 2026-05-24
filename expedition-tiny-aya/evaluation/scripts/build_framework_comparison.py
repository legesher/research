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
    benchmark, data, instr).

    Each row carries both the refined-extractor view (acc, count, correct,
    parse-failure rate) AND the inference-time-extractor's parse-failure
    rate (read from the sibling `_summary_template*.json` file). The
    inference-time `pf` is used by view F (parse-failure recovery rate).
    The inference-time `acc` is NOT loaded here; for the cell-by-cell
    original-vs-refined accuracy comparison see `build_comparison.py`.
    """
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

                # Sibling inference-time-extractor summary; needed for the
                # parse-failure recovery view (F). Filename: drop "_reparsed_"
                # to find the inference-time-extractor's matching file.
                orig_path = rep.parent / rep.name.replace("_summary_reparsed_", "_summary_")
                orig_pf: dict = {}
                if orig_path.exists():
                    od = load(orig_path)
                    orig_pf = od.get("parse_failure_rates", {})

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
                        "orig_pf": orig_pf.get(base),
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
# Shared lookup helpers (used by views A, B, E)
# =============================================================================
def build_baseline_lookup(rows: list[dict]) -> dict:
    """{(template, bench, data, instr) -> baseline acc}. Single seed."""
    out = {}
    for r in rows:
        if r["condition"] == "baseline":
            out[(r["template"], r["benchmark"], r["data"], r["instr"])] = r["acc"]
    return out


def build_anchor_lookup(rows: list[dict], anchor_cond: str) -> dict:
    """{(template, bench, data, instr) -> mean acc across that anchor's seeds}.
    Used by view B with anchor_cond="condition-1-en-5k" — the coding constant.
    """
    accum: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["condition"] == anchor_cond:
            accum[(r["template"], r["benchmark"], r["data"], r["instr"])].append(r["acc"])
    return {k: mean(vs) for k, vs in accum.items()}


# =============================================================================
# View A — Cross-language impact matrix
# =============================================================================
def write_cross_language_impact(rows: list[dict], out: Path):
    """For each fine-tune × (data_lang × instr_lang), the (cond − baseline)
    delta under the refined extractor. Surfaces 'fine-tuning on X impacts
    passages in Y by Δpp'. Includes both matched and off-target rows with
    an `off_target` flag so the reader can filter to the cross-language
    view directly.

    The aggregation is across templates × seeds within each
    (condition, data_lang, instr_lang) bucket — the matched_instr /
    matched_diagonal flags refer to that bucket's properties.
    """
    baseline_lookup = build_baseline_lookup(rows)

    # Group condition rows by (condition, data_lang, instr_lang)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["condition"] == "baseline":
            continue
        grouped[(r["condition"], r["data"], r["instr"])].append(r)

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "data_lang", "instr_lang",
            "off_target", "matched_diagonal", "matched_instr",
            "n_cells", "mean_cond_acc", "mean_baseline_acc",
            "delta_vs_baseline",
        ]) + "\n")
        for key in sorted(grouped.keys()):
            cond, data_lang, instr_lang = key
            rs = grouped[key]
            target = CONDITION_TARGET_LANG.get(cond) or ""
            off_target = "y" if (target and data_lang != target) else "n"
            matched_diag = "y" if data_lang == instr_lang else "n"
            matched_inst = "y" if (target and instr_lang == target) else "n"
            cond_accs = [r["acc"] for r in rs]
            # Baseline aggregation across templates for the same data/instr
            baseline_accs = [
                baseline_lookup.get((t, r["benchmark"], data_lang, instr_lang))
                for r in rs for t in [r["template"]]
            ]
            baseline_accs = [b for b in baseline_accs if b is not None]
            if not baseline_accs:
                continue
            mean_cond = mean(cond_accs)
            mean_base = mean(baseline_accs)
            f.write("\t".join([
                cond, target, data_lang, instr_lang,
                off_target, matched_diag, matched_inst,
                str(len(rs)), f"{mean_cond:.4f}", f"{mean_base:.4f}",
                f"{mean_cond - mean_base:+.4f}",
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# View B — Coding-constant decomposition
# =============================================================================
def write_decomposition_vs_baseline(rows: list[dict], out: Path):
    """Decompose (cond − baseline) into 'effect of any fine-tune' (using
    cond-1-en-5k as the anchor — Madison's coding constant) plus 'effect
    of language-specific fine-tune'. Per-cell, per-seed.

      delta_vs_baseline      = cond_acc − baseline_acc
      delta_en_finetune_only = cond_1_en_5k_acc − baseline_acc
      delta_language_specific = cond_acc − cond_1_en_5k_acc

    The decomposition is exact at the cell level:
      delta_vs_baseline = delta_en_finetune_only + delta_language_specific
    so the reader can size which component dominates.
    """
    baseline_lookup = build_baseline_lookup(rows)
    anchor_lookup = build_anchor_lookup(rows, "condition-1-en-5k")

    EXCLUDE = {"baseline", "condition-1-en-5k", "condition-1-en-20k"}

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "seed", "template",
            "benchmark", "data", "instr",
            "matched_diagonal", "matched_instr",
            "baseline_acc", "cond_1_en_5k_acc", "cond_acc",
            "delta_vs_baseline", "delta_en_finetune_only",
            "delta_language_specific",
        ]) + "\n")
        for r in sorted(
            rows,
            key=lambda x: (x["condition"], x["seed"], x["template"],
                           x["benchmark"], x["data"], x["instr"]),
        ):
            if r["condition"] in EXCLUDE:
                continue
            cell = (r["template"], r["benchmark"], r["data"], r["instr"])
            b = baseline_lookup.get(cell)
            a = anchor_lookup.get(cell)
            if b is None or a is None:
                continue
            cond_acc = r["acc"]
            f.write("\t".join([
                r["condition"], r["target_lang"], r["seed"], r["template"],
                r["benchmark"], r["data"], r["instr"],
                r["matched_diagonal"], r["matched_instr"],
                f"{b:.4f}", f"{a:.4f}", f"{cond_acc:.4f}",
                f"{cond_acc - b:+.4f}",
                f"{a - b:+.4f}",
                f"{cond_acc - a:+.4f}",
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# View C — Seed variance / statistical significance
# =============================================================================
def write_seed_variance(rows: list[dict], out: Path):
    """Per (condition, cell) with > 1 seed: mean, std, min, max across
    seeds. Lets the paper report 'mean ± std with n' and judge whether
    deltas exceed the within-condition noise floor.

    For n=1 conditions (single-seed runs), std is reported as 'nan'.
    """
    from statistics import stdev

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["template"], r["benchmark"], r["data"], r["instr"])
        grouped[key].append(r)

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "template",
            "benchmark", "data", "instr",
            "matched_diagonal", "matched_instr",
            "n_seeds", "mean_acc", "std_acc", "min_acc", "max_acc",
            "total_count", "total_correct",
        ]) + "\n")
        for key in sorted(grouped.keys()):
            rs = grouped[key]
            cond = rs[0]["condition"]
            accs = [r["acc"] for r in rs]
            counts = [r["count"] or 0 for r in rs]
            corrects = [r["correct"] or 0 for r in rs]
            std = f"{stdev(accs):.4f}" if len(accs) > 1 else "nan"
            f.write("\t".join([
                cond, rs[0]["target_lang"], rs[0]["template"],
                rs[0]["benchmark"], rs[0]["data"], rs[0]["instr"],
                rs[0]["matched_diagonal"], rs[0]["matched_instr"],
                str(len(rs)), f"{mean(accs):.4f}", std,
                f"{min(accs):.4f}", f"{max(accs):.4f}",
                str(sum(counts)), str(sum(corrects)),
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# View D — Template robustness per condition
# =============================================================================
def write_template_robustness(rows: list[dict], out: Path):
    """For each (condition, seed, benchmark, data, instr) cell that ran
    both templates, the gap |t1_acc − t2_acc|. Aggregated per condition
    and per (condition × benchmark): mean gap, max gap, fraction of cells
    with gap > 0.10 (a brittle-cells count).

    Operationalizes the template-asymmetry structural finding: which
    conditions are robust to prompt-template choice vs which break."""
    # Pivot per-cell to (t1, t2) pairs
    pivot: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["condition"], r["seed"], r["benchmark"], r["data"], r["instr"])
        pivot[key][f"t{r['template']}"] = r["acc"]

    gaps_by_cond_bench: dict[tuple, list[float]] = defaultdict(list)
    gaps_by_cond: dict[str, list[float]] = defaultdict(list)
    for key, t in pivot.items():
        if "t1" in t and "t2" in t:
            cond, _, bench, _, _ = key
            g = abs(t["t1"] - t["t2"])
            gaps_by_cond_bench[(cond, bench)].append(g)
            gaps_by_cond[cond].append(g)

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "benchmark", "n_cells",
            "mean_gap", "median_gap", "max_gap",
            "brittle_cells_gt_0.10", "frac_brittle",
        ]) + "\n")
        # First: per condition row (benchmark="ALL")
        for cond in sorted(gaps_by_cond.keys()):
            gs = gaps_by_cond[cond]
            brittle = sum(1 for x in gs if x > 0.10)
            f.write("\t".join([
                cond, CONDITION_TARGET_LANG.get(cond) or "", "ALL",
                str(len(gs)), f"{mean(gs):.4f}",
                f"{sorted(gs)[len(gs)//2]:.4f}",
                f"{max(gs):.4f}",
                str(brittle), f"{brittle / len(gs):.4f}",
            ]) + "\n")
        # Then: per (condition, benchmark) rows
        for key in sorted(gaps_by_cond_bench.keys()):
            cond, bench = key
            gs = gaps_by_cond_bench[key]
            brittle = sum(1 for x in gs if x > 0.10)
            f.write("\t".join([
                cond, CONDITION_TARGET_LANG.get(cond) or "", bench,
                str(len(gs)), f"{mean(gs):.4f}",
                f"{sorted(gs)[len(gs)//2]:.4f}",
                f"{max(gs):.4f}",
                str(brittle), f"{brittle / len(gs):.4f}",
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# View E — Per-benchmark catastrophic forgetting breakdown
# =============================================================================
def write_benchmark_breakdown(rows: list[dict], out: Path):
    """For each (condition, benchmark, matched_diagonal), the mean (cond
    − baseline) delta. Answers: does the catastrophic-forgetting pattern
    hold across all benchmarks, or only on SIB-200?

    Three subgroups per (condition, benchmark):
      matched_diagonal=y: home turf — strongest forgetting signal.
      matched_diagonal=n: cross-language — interference signal.
      ALL: combined view (for top-line numbers)."""
    baseline_lookup = build_baseline_lookup(rows)

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["condition"] == "baseline":
            continue
        cell = (r["template"], r["benchmark"], r["data"], r["instr"])
        b = baseline_lookup.get(cell)
        if b is None:
            continue
        delta = r["acc"] - b
        grouped[(r["condition"], r["benchmark"], r["matched_diagonal"])].append(delta)
        grouped[(r["condition"], r["benchmark"], "ALL")].append(delta)

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "benchmark", "matched_diagonal",
            "n_cells", "mean_delta_vs_baseline",
            "median_delta", "min_delta", "max_delta",
        ]) + "\n")
        for key in sorted(grouped.keys()):
            cond, bench, md = key
            deltas = grouped[key]
            f.write("\t".join([
                cond, CONDITION_TARGET_LANG.get(cond) or "", bench, md,
                str(len(deltas)),
                f"{mean(deltas):+.4f}",
                f"{sorted(deltas)[len(deltas)//2]:+.4f}",
                f"{min(deltas):+.4f}",
                f"{max(deltas):+.4f}",
            ]) + "\n")
    print(f"Wrote {out}")


# =============================================================================
# View F — Parse-failure recovery rate per condition
# =============================================================================
def write_parse_failure_recovery(rows: list[dict], out: Path):
    """For each cell: how many parse failures did the refined extractor
    recover from the inference-time extractor?

      recovery_pp = orig_pf − refined_pf

    Positive = refined extractor read more answers correctly that the
    inference-time extractor was returning None for. The aggregation
    per (condition, benchmark) quantifies the methodology contribution.

    Cells where orig_pf is missing (e.g., the original summary file was
    unavailable in the snapshot) are skipped silently."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    raw_rows: list[tuple] = []  # for the per-cell view too
    for r in rows:
        if r["orig_pf"] is None or r["pf"] is None:
            continue
        recovery = r["orig_pf"] - r["pf"]
        grouped[(r["condition"], r["benchmark"])].append(recovery)
        grouped[(r["condition"], "ALL")].append(recovery)
        raw_rows.append((r, recovery))

    with out.open("w") as f:
        f.write("\t".join([
            "condition", "target_lang", "benchmark",
            "n_cells", "mean_recovery_pp", "median_recovery",
            "max_recovery", "n_cells_recovery_gt_0.05",
        ]) + "\n")
        for key in sorted(grouped.keys()):
            cond, bench = key
            recs = grouped[key]
            high = sum(1 for x in recs if x > 0.05)
            f.write("\t".join([
                cond, CONDITION_TARGET_LANG.get(cond) or "", bench,
                str(len(recs)),
                f"{mean(recs):+.4f}",
                f"{sorted(recs)[len(recs)//2]:+.4f}",
                f"{max(recs):+.4f}",
                str(high),
            ]) + "\n")
    print(f"Wrote {out}")

    # Also write a per-cell version surfacing the top 50 largest recoveries
    top_cells_path = out.parent / "framework_parse_failure_recovery_top_cells.tsv"
    raw_rows.sort(key=lambda x: -x[1])
    with top_cells_path.open("w") as f:
        f.write("\t".join([
            "rank", "condition", "seed", "template", "benchmark",
            "data", "instr", "matched_diagonal",
            "orig_pf", "refined_pf", "recovery_pp",
        ]) + "\n")
        for rank, (r, recovery) in enumerate(raw_rows[:50], 1):
            f.write("\t".join([
                str(rank), r["condition"], r["seed"], r["template"],
                r["benchmark"], r["data"], r["instr"], r["matched_diagonal"],
                f"{r['orig_pf']:.4f}", f"{r['pf']:.4f}",
                f"{recovery:+.4f}",
            ]) + "\n")
    print(f"Wrote {top_cells_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    rows = annotate(gather())
    print(f"Loaded {len(rows)} cell-rows from {ROOT}")

    OUT.mkdir(parents=True, exist_ok=True)

    # The original three framework axes
    write_template_comparison(rows, OUT / "framework_template_comparison.tsv")
    write_same_language_comparison(rows, OUT / "framework_same_language_comparison.tsv")
    write_data_volume_comparison(rows, OUT / "framework_data_volume_comparison.tsv")

    # The six additional perspectives (paper-prep callouts items 7-12 + methodology)
    write_cross_language_impact(rows, OUT / "framework_cross_language_impact.tsv")
    write_decomposition_vs_baseline(rows, OUT / "framework_decomposition_vs_baseline.tsv")
    write_seed_variance(rows, OUT / "framework_seed_variance.tsv")
    write_template_robustness(rows, OUT / "framework_template_robustness.tsv")
    write_benchmark_breakdown(rows, OUT / "framework_benchmark_breakdown.tsv")
    write_parse_failure_recovery(rows, OUT / "framework_parse_failure_recovery.tsv")

    print()
    print("Done. Framework TSVs encode the paper's results axes + supporting views:")
    print()
    print("  Comparison framework (three axes):")
    print("    framework_template_comparison.tsv     — t1 vs t2 within each (condition, seed)")
    print("    framework_same_language_comparison.tsv — all conditions for each target lang, anchored")
    print("    framework_data_volume_comparison.tsv  — 5k vs 20k, per condition family")
    print()
    print("  Cross-language behavior:")
    print("    framework_cross_language_impact.tsv  — how each fine-tune impacts non-target languages")
    print()
    print("  Statistical decomposition:")
    print("    framework_decomposition_vs_baseline.tsv — separates language-specific effect from generic-finetune effect (cond-1-en-5k anchor)")
    print("    framework_seed_variance.tsv          — mean ± std per cell across seeds (gating for paper significance claims)")
    print()
    print("  Robustness / methodology:")
    print("    framework_template_robustness.tsv    — template-asymmetry quantified per condition")
    print("    framework_benchmark_breakdown.tsv    — does catastrophic forgetting hold across benchmarks?")
    print("    framework_parse_failure_recovery.tsv — extractor methodology contribution (orig_pf - refined_pf)")
    print("    framework_parse_failure_recovery_top_cells.tsv — top 50 cells by recovery magnitude")


if __name__ == "__main__":
    main()
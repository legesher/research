"""Phase-3 condition-vs-baseline analysis under both extractors.

For each (template, benchmark, data, instr) cell that a condition ran:
  baseline_orig  : the baseline model's accuracy under the original extractor
  baseline_rep   : the baseline model's accuracy under the refined extractor
  cond_orig      : the condition's accuracy under the original extractor (per seed)
  cond_rep       : the condition's accuracy under the refined extractor (per seed)

Then compute, per (condition, seed, cell):
  delta_orig = cond_orig - baseline_orig
  delta_rep  = cond_rep  - baseline_rep

And aggregate across cells (and seeds) per condition, per benchmark.

Emit:
  vs_baseline_cells.tsv               one row per cell (per condition × seed)
  vs_baseline_by_condition.tsv        rolled up per condition
  vs_baseline_by_cond_x_bench.tsv     rolled up per (condition, benchmark)
  vs_baseline_by_cond_x_instr.tsv     rolled up per (condition, instr_lang)
  conclusion_flips.tsv                cells where (cond − baseline) flips sign between the
                                       two extractors, with |delta| > 0.01 required on BOTH
                                       sides. Cells with |delta| ≤ 0.01 on either side are
                                       in the noise-floor band and excluded. The 0.01
                                       threshold is below the seed-to-seed reproducibility
                                       noise observed in Phase-3 baseline (~0.03 std on
                                       SIB-200 cond-2-X-5k cells); a flip smaller than that
                                       is noise, not a finding.

Aggregation buffers used:
  ±1e-6 for win/loss/tie counting in the rollups (float-precision noise floor)
  ±0.01 for conclusion-flip detection (paper-claim noise floor)
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Override via PHASE3_SNAPSHOT_ROOT / PHASE3_OUT_DIR env vars when running
# outside the default /tmp/phase3_reparse layout. See phase-3/phase3-refined-evaluation.md §9.
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


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def gather() -> dict:
    """Walk the HF snapshot and return

        {(condition, seed, template): {(t, bench, data, instr): {
            "orig_acc": ..., "rep_acc": ...,
            "orig_pf":  ..., "rep_pf":  ...,
        }}}

    The outer key is one per (condition, seed_from_filename, template_from_filename)
    summary-file pair (original + reparsed); the inner dict is per (t, bench, data, instr)
    cell within that pair. Canonical/stray reconciliation prefers the file whose
    parent directory matches its filename seed (see `_classify_session_files` in
    `upload_reparsed_summaries.py` for the same logic at upload time)."""
    out: dict[tuple, dict] = {}
    canon: dict[tuple, Path] = {}
    for cond_dir in sorted(ROOT.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for seed_dir in sorted(cond_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            dir_seed = (
                seed_dir.name.removeprefix("seed")
                if seed_dir.name.startswith("seed")
                else seed_dir.name
            )
            for orig in sorted(seed_dir.glob("*_summary_template*.json")):
                if "reparsed" in orig.name:
                    continue
                m = SEED_FROM_NAME.search(orig.name)
                file_seed = m["seed"] if m else dir_seed
                template = orig.stem.rsplit("_", 1)[-1]
                key = (cond, file_seed, template)
                if key in canon:
                    if dir_seed == file_seed and (
                        canon[key].parent.name.removeprefix("seed") != file_seed
                    ):
                        canon[key] = orig
                else:
                    canon[key] = orig
    for (cond, seed, template), orig in canon.items():
        rep = orig.parent / orig.name.replace("_summary_", "_summary_reparsed_")
        if not rep.exists():
            continue
        od, rd = load(orig), load(rep)
        os_, rs_ = od.get("summary", {}), rd.get("summary", {})
        opf, rpf = od.get("parse_failure_rates", {}), rd.get("parse_failure_rates", {})
        cells = {}
        for k in os_:
            if not k.endswith("_acc"):
                continue
            base = k.removesuffix("_acc")
            m = CELL_RE.match(base)
            if not m:
                continue
            cells[(m["t"], m["bench"], m["data"], m["instr"])] = {
                "orig_acc": os_.get(base + "_acc"),
                "rep_acc": rs_.get(base + "_acc"),
                "orig_pf": opf.get(base),
                "rep_pf": rpf.get(base),
            }
        out[(cond, seed, template)] = cells
    return out



def agg(rows):
    bo = [r["baseline_orig"] for r in rows]
    br = [r["baseline_rep"] for r in rows]
    co = [r["cond_orig"] for r in rows]
    cr = [r["cond_rep"] for r in rows]
    do = [r["delta_orig"] for r in rows]
    dr = [r["delta_rep"] for r in rows]
    n_wins_o = sum(1 for x in do if x > 1e-6)
    n_loss_o = sum(1 for x in do if x < -1e-6)
    n_wins_r = sum(1 for x in dr if x > 1e-6)
    n_loss_r = sum(1 for x in dr if x < -1e-6)
    n_flip_pos_to_neg = sum(1 for a, b in zip(do, dr) if a > 1e-6 and b < -1e-6)
    n_flip_neg_to_pos = sum(1 for a, b in zip(do, dr) if a < -1e-6 and b > 1e-6)
    return {
        "n": len(rows),
        "mean_baseline_orig": mean(bo),
        "mean_baseline_rep": mean(br),
        "mean_cond_orig": mean(co),
        "mean_cond_rep": mean(cr),
        "mean_delta_orig": mean(do),
        "mean_delta_rep": mean(dr),
        "n_wins_orig": n_wins_o,
        "n_losses_orig": n_loss_o,
        "n_wins_rep": n_wins_r,
        "n_losses_rep": n_loss_r,
        "n_flip_win_to_loss": n_flip_pos_to_neg,
        "n_flip_loss_to_win": n_flip_neg_to_pos,
    }



def write_rollup(path, records, group_key_fn, key_label):
    g = defaultdict(list)
    for r in records:
        g[group_key_fn(r)].append(r)
    with path.open("w") as f:
        cols = list(key_label) if isinstance(key_label, (tuple, list)) else [key_label]
        f.write(
            "\t".join(
                cols
                + [
                    "n_cells",
                    "mean_baseline_orig",
                    "mean_cond_orig",
                    "mean_delta_orig",
                    "mean_baseline_rep",
                    "mean_cond_rep",
                    "mean_delta_rep",
                    "n_wins_orig",
                    "n_losses_orig",
                    "n_wins_rep",
                    "n_losses_rep",
                    "n_flip_win_to_loss",
                    "n_flip_loss_to_win",
                ]
            )
            + "\n"
        )
        for k in sorted(g.keys()):
            a = agg(g[k])
            row_keys = list(k) if isinstance(k, tuple) else [k]
            f.write(
                "\t".join(
                    [str(x) for x in row_keys]
                    + [
                        str(a["n"]),
                        f"{a['mean_baseline_orig']:.4f}",
                        f"{a['mean_cond_orig']:.4f}",
                        f"{a['mean_delta_orig']:+.4f}",
                        f"{a['mean_baseline_rep']:.4f}",
                        f"{a['mean_cond_rep']:.4f}",
                        f"{a['mean_delta_rep']:+.4f}",
                        str(a["n_wins_orig"]),
                        str(a["n_losses_orig"]),
                        str(a["n_wins_rep"]),
                        str(a["n_losses_rep"]),
                        str(a["n_flip_win_to_loss"]),
                        str(a["n_flip_loss_to_win"]),
                    ]
                )
                + "\n"
            )
    print(f"Wrote {path}")



def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    data = gather()

    # Pull out the baseline reference (single row, seed="none")
    baseline_cells: dict = {}
    for (cond, seed, template), cells in data.items():
        if cond == "baseline":
            for k, v in cells.items():
                # k = (t, bench, data, instr) where t is the template *digit* (e.g. "1")
                # parsed from the cell key. The outer `template` from the gather key is
                # the filename suffix (e.g. "template1") and is not the same shape — don't
                # conflate the two.
                baseline_cells[k] = v
    print(f"Baseline cells: {len(baseline_cells)}")

    # For each non-baseline condition, build per-cell records
    records = []
    for (cond, seed, template), cells in data.items():
        if cond == "baseline":
            continue
        for k, v in cells.items():
            b = baseline_cells.get(k)
            if b is None:
                continue
            records.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "template": k[0],
                    "benchmark": k[1],
                    "data": k[2],
                    "instr": k[3],
                    "baseline_orig": b["orig_acc"],
                    "baseline_rep": b["rep_acc"],
                    "cond_orig": v["orig_acc"],
                    "cond_rep": v["rep_acc"],
                    "delta_orig": v["orig_acc"] - b["orig_acc"],
                    "delta_rep": v["rep_acc"] - b["rep_acc"],
                    "baseline_pf_orig": b["orig_pf"],
                    "baseline_pf_rep": b["rep_pf"],
                    "cond_pf_orig": v["orig_pf"],
                    "cond_pf_rep": v["rep_pf"],
                }
            )

    print(f"Total cell-comparison records: {len(records)}")

    # ---- write per-cell TSV ----
    with (OUT / "vs_baseline_cells.tsv").open("w") as f:
        cols = [
            "condition",
            "seed",
            "template",
            "benchmark",
            "data",
            "instr",
            "baseline_orig",
            "cond_orig",
            "delta_orig",
            "baseline_rep",
            "cond_rep",
            "delta_rep",
            "baseline_pf_orig",
            "cond_pf_orig",
            "baseline_pf_rep",
            "cond_pf_rep",
        ]
        f.write("\t".join(cols) + "\n")
        for r in sorted(
            records,
            key=lambda r: (
                r["condition"],
                r["seed"],
                r["template"],
                r["benchmark"],
                r["data"],
                r["instr"],
            ),
        ):
            f.write(
                "\t".join(
                    [
                        r["condition"],
                        r["seed"],
                        r["template"],
                        r["benchmark"],
                        r["data"],
                        r["instr"],
                        f"{r['baseline_orig']:.4f}",
                        f"{r['cond_orig']:.4f}",
                        f"{r['delta_orig']:+.4f}",
                        f"{r['baseline_rep']:.4f}",
                        f"{r['cond_rep']:.4f}",
                        f"{r['delta_rep']:+.4f}",
                        f"{r['baseline_pf_orig']:.4f}",
                        f"{r['cond_pf_orig']:.4f}",
                        f"{r['baseline_pf_rep']:.4f}",
                        f"{r['cond_pf_rep']:.4f}",
                    ]
                )
                + "\n"
            )
    print("Wrote vs_baseline_cells.tsv")


    # ---- rollups ----
    write_rollup(
        OUT / "vs_baseline_by_condition.tsv", records, lambda r: r["condition"], "condition"
    )
    write_rollup(
        OUT / "vs_baseline_by_cond_x_bench.tsv",
        records,
        lambda r: (r["condition"], r["benchmark"]),
        ("condition", "benchmark"),
    )
    write_rollup(
        OUT / "vs_baseline_by_cond_x_instr.tsv",
        records,
        lambda r: (r["condition"], r["instr"]),
        ("condition", "instr"),
    )


    # ---- conclusion flips: cells where the sign of cond-vs-baseline differs between extractors ----
    flips = []
    for r in records:
        sign_o = 1 if r["delta_orig"] > 0.01 else (-1 if r["delta_orig"] < -0.01 else 0)
        sign_r = 1 if r["delta_rep"] > 0.01 else (-1 if r["delta_rep"] < -0.01 else 0)
        if sign_o != sign_r and sign_o != 0 and sign_r != 0:
            flips.append((r, sign_o, sign_r))
    with (OUT / "conclusion_flips.tsv").open("w") as f:
        cols = [
            "condition",
            "seed",
            "template",
            "benchmark",
            "data",
            "instr",
            "baseline_orig",
            "cond_orig",
            "delta_orig",
            "baseline_rep",
            "cond_rep",
            "delta_rep",
            "verdict",
        ]
        f.write("\t".join(cols) + "\n")
        for r, so, sr in sorted(
            flips,
            key=lambda t: (
                t[0]["condition"],
                t[0]["seed"],
                t[0]["template"],
                t[0]["benchmark"],
            ),
        ):
            verdict = "win→loss" if so > 0 and sr < 0 else "loss→win"
            f.write(
                "\t".join(
                    [
                        r["condition"],
                        r["seed"],
                        r["template"],
                        r["benchmark"],
                        r["data"],
                        r["instr"],
                        f"{r['baseline_orig']:.4f}",
                        f"{r['cond_orig']:.4f}",
                        f"{r['delta_orig']:+.4f}",
                        f"{r['baseline_rep']:.4f}",
                        f"{r['cond_rep']:.4f}",
                        f"{r['delta_rep']:+.4f}",
                        verdict,
                    ]
                )
                + "\n"
            )
    print(f"Wrote conclusion_flips.tsv ({len(flips)} flips)")


    # ---- print headlines for the markdown ----
    print()
    print("===== HEADLINE: per-condition vs baseline =====")
    print(
        f"{'condition':<22} {'n':>4}  {'b_orig':>8} {'c_orig':>8} {'Δ_orig':>8}  {'b_rep':>8} {'c_rep':>8} {'Δ_rep':>8}  wins_o/loss_o  wins_r/loss_r  flips"
    )
    print("-" * 130)
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)
    for cond in sorted(by_cond.keys()):
        a = agg(by_cond[cond])
        print(
            f"{cond:<22} {a['n']:>4}  "
            f"{a['mean_baseline_orig']:>8.3f} {a['mean_cond_orig']:>8.3f} {a['mean_delta_orig']:>+8.3f}  "
            f"{a['mean_baseline_rep']:>8.3f} {a['mean_cond_rep']:>8.3f} {a['mean_delta_rep']:>+8.3f}  "
            f"{a['n_wins_orig']:>4}/{a['n_losses_orig']:<4}  "
            f"{a['n_wins_rep']:>4}/{a['n_losses_rep']:<4}  "
            f"w→l={a['n_flip_win_to_loss']} l→w={a['n_flip_loss_to_win']}"
        )

    # Per-bench within condition
    print()
    print("===== per (condition × benchmark) =====")
    print(
        f"{'condition':<22} {'bench':<9} {'n':>4}  {'b_orig':>8} {'c_orig':>8} {'Δ_orig':>8}  {'b_rep':>8} {'c_rep':>8} {'Δ_rep':>8}"
    )
    print("-" * 110)
    by_cb = defaultdict(list)
    for r in records:
        by_cb[(r["condition"], r["benchmark"])].append(r)
    for k in sorted(by_cb.keys()):
        a = agg(by_cb[k])
        print(
            f"{k[0]:<22} {k[1]:<9} {a['n']:>4}  "
            f"{a['mean_baseline_orig']:>8.3f} {a['mean_cond_orig']:>8.3f} {a['mean_delta_orig']:>+8.3f}  "
            f"{a['mean_baseline_rep']:>8.3f} {a['mean_cond_rep']:>8.3f} {a['mean_delta_rep']:>+8.3f}"
        )


if __name__ == "__main__":
    main()

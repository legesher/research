"""Build phase-3 original-vs-reparsed comparison artefacts.

Reads every (original, reparsed) summary pair under
ROOT (defaults to /tmp/phase3_reparse/hf_snapshot/phase3/conditions) and emits
the cells.tsv + rollup TSVs + overall_stats.json into OUT.

To use: either run from a freshly snapshotted layout per
`expedition-tiny-aya/analysis/phase3-reparse-evaluation.md` §9 (Reproducibility),
or override ROOT / OUT below to point at your local mirror. Paths are
deliberately env-var-overridable so the script survives moves of the snapshot.

Output TSVs are intended to live in
expedition-tiny-aya/analysis/reparse-tables/.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(
    os.environ.get(
        "PHASE3_SNAPSHOT_ROOT",
        "/tmp/phase3_reparse/hf_snapshot/phase3/conditions",
    )
)
OUT = Path(os.environ.get("PHASE3_OUT_DIR", "/tmp/phase3_reparse"))

# {condition}/{seed}/{condition}_{seed}_summary[_reparsed]_{template}.json
CELL_RE = re.compile(
    r"^template(?P<t>\d+)_(?P<bench>xnli|csqa|sib200|belebele)_data=(?P<data>[a-z]{2})_instr=(?P<instr>[a-z]{2})$"
)


def load_summary(path: Path):
    with open(path) as f:
        return json.load(f)


SEED_FROM_NAME = re.compile(r"_seed(?P<seed>[A-Za-z0-9]+)_summary")


def iter_cells():
    """Yield one row per (condition, seed, template, benchmark, data, instr).

    Seed is derived from the FILENAME (not the directory) so misfiled summaries
    like `seed42/condition-2-es-5k_seed123_summary_template1.json` are tagged
    with their true seed and de-duplicated against any sibling copy that
    lives in the matching directory.
    """
    canonical: dict[tuple, Path] = {}
    for cond_dir in sorted(ROOT.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name
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
                key = (condition, file_seed, template)
                if key in canonical:
                    existing = canonical[key]
                    existing_dir = existing.parent.name.removeprefix("seed")
                    # prefer the file whose parent-dir seed matches its filename seed
                    if dir_seed == file_seed and existing_dir != file_seed:
                        canonical[key] = orig
                else:
                    canonical[key] = orig

    for (condition, seed, template), orig in sorted(canonical.items()):
        reparsed = orig.parent / orig.name.replace("_summary_", "_summary_reparsed_")
        if not reparsed.exists():
            yield (
                condition,
                seed,
                template,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                True,
            )
            continue
        orig_data = load_summary(orig)
        repar_data = load_summary(reparsed)
        orig_summary = orig_data.get("summary", {})
        repar_summary = repar_data.get("summary", {})
        orig_pf = orig_data.get("parse_failure_rates", {})
        repar_pf = repar_data.get("parse_failure_rates", {})
        # Iterate cells via _acc keys
        for key in orig_summary:
            if not key.endswith("_acc"):
                continue
            base = key.removesuffix("_acc")
            m = CELL_RE.match(base)
            if not m:
                continue
            n = orig_summary.get(base + "_count")
            o_acc = orig_summary.get(base + "_acc")
            r_acc = repar_summary.get(base + "_acc")
            o_pf = orig_pf.get(base)
            r_pf = repar_pf.get(base)
            yield (
                condition,
                seed,
                m["t"],
                m["bench"],
                m["data"],
                m["instr"],
                n,
                o_acc,
                r_acc,
                o_pf,
                r_pf,
                False,
            )


cells = []
gaps = []
for row in iter_cells():
    if row[-1]:
        gaps.append({"condition": row[0], "seed": row[1], "template": row[2]})
    else:
        cells.append(
            {
                "condition": row[0],
                "seed": row[1],
                "template": row[2],
                "benchmark": row[3],
                "data": row[4],
                "instr": row[5],
                "n": row[6],
                "orig_acc": row[7],
                "repar_acc": row[8],
                "orig_pf": row[9],
                "repar_pf": row[10],
                "delta_acc": (
                    (row[8] - row[7])
                    if (row[7] is not None and row[8] is not None)
                    else None
                ),
                "delta_pf": (
                    (row[10] - row[9])
                    if (row[9] is not None and row[10] is not None)
                    else None
                ),
            }
        )

print(f"Cells: {len(cells)}    Gaps: {len(gaps)}")

# ---- write cells.tsv ----
cells_path = OUT / "cells.tsv"
with cells_path.open("w") as f:
    headers = [
        "condition",
        "seed",
        "template",
        "benchmark",
        "data",
        "instr",
        "n",
        "orig_acc",
        "repar_acc",
        "delta_acc",
        "orig_pf",
        "repar_pf",
        "delta_pf",
    ]
    f.write("\t".join(headers) + "\n")
    for c in sorted(
        cells,
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
                    c["condition"],
                    c["seed"],
                    c["template"],
                    c["benchmark"],
                    c["data"],
                    c["instr"],
                    str(c["n"]),
                    f"{c['orig_acc']:.4f}",
                    f"{c['repar_acc']:.4f}",
                    f"{c['delta_acc']:+.4f}",
                    f"{c['orig_pf']:.4f}",
                    f"{c['repar_pf']:.4f}",
                    f"{c['delta_pf']:+.4f}",
                ]
            )
            + "\n"
        )
print(f"Wrote {cells_path}")


# ---- headline stats ----
deltas_acc = [c["delta_acc"] for c in cells]
deltas_pf = [c["delta_pf"] for c in cells]
improved = [c for c in cells if c["delta_acc"] > 1e-6]
regressed = [c for c in cells if c["delta_acc"] < -1e-6]
flat = [c for c in cells if abs(c["delta_acc"]) <= 1e-6]
pf_improved = [c for c in cells if c["delta_pf"] < -1e-6]  # pf going down is good
pf_worsened = [c for c in cells if c["delta_pf"] > 1e-6]

# Cells where acc flat but pf dropped
flat_acc_pf_drop = [
    c for c in cells if abs(c["delta_acc"]) <= 1e-6 and c["delta_pf"] < -1e-6
]
flat_acc_pf_drop_5 = [
    c for c in cells if abs(c["delta_acc"]) <= 1e-6 and c["delta_pf"] < -0.05
]


# ----- by benchmark -----
def group_stats(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    out = {}
    for k, v in g.items():
        d_acc = [x["delta_acc"] for x in v]
        d_pf = [x["delta_pf"] for x in v]
        out[k] = {
            "n_cells": len(v),
            "mean_delta_acc": mean(d_acc),
            "median_delta_acc": median(d_acc),
            "max_delta_acc": max(d_acc),
            "min_delta_acc": min(d_acc),
            "mean_delta_pf": mean(d_pf),
            "median_delta_pf": median(d_pf),
            "max_delta_pf": max(d_pf),
            "min_delta_pf": min(d_pf),
            "n_improved": sum(1 for x in d_acc if x > 1e-6),
            "n_regressed": sum(1 for x in d_acc if x < -1e-6),
            "n_flat": sum(1 for x in d_acc if abs(x) <= 1e-6),
            "mean_orig_acc": mean(x["orig_acc"] for x in v),
            "mean_repar_acc": mean(x["repar_acc"] for x in v),
            "mean_orig_pf": mean(x["orig_pf"] for x in v),
            "mean_repar_pf": mean(x["repar_pf"] for x in v),
        }
    return out


def write_tsv(path: Path, group_rows: dict, key_label: str):
    with path.open("w") as f:
        cols = [
            key_label,
            "n_cells",
            "mean_orig_acc",
            "mean_repar_acc",
            "mean_delta_acc",
            "median_delta_acc",
            "n_improved",
            "n_regressed",
            "n_flat",
            "mean_orig_pf",
            "mean_repar_pf",
            "mean_delta_pf",
            "median_delta_pf",
            "min_delta_acc",
            "max_delta_acc",
            "min_delta_pf",
            "max_delta_pf",
        ]
        f.write("\t".join(cols) + "\n")
        for k in sorted(group_rows.keys()):
            v = group_rows[k]
            f.write(
                "\t".join(
                    [
                        str(k),
                        str(v["n_cells"]),
                        f"{v['mean_orig_acc']:.4f}",
                        f"{v['mean_repar_acc']:.4f}",
                        f"{v['mean_delta_acc']:+.4f}",
                        f"{v['median_delta_acc']:+.4f}",
                        str(v["n_improved"]),
                        str(v["n_regressed"]),
                        str(v["n_flat"]),
                        f"{v['mean_orig_pf']:.4f}",
                        f"{v['mean_repar_pf']:.4f}",
                        f"{v['mean_delta_pf']:+.4f}",
                        f"{v['median_delta_pf']:+.4f}",
                        f"{v['min_delta_acc']:+.4f}",
                        f"{v['max_delta_acc']:+.4f}",
                        f"{v['min_delta_pf']:+.4f}",
                        f"{v['max_delta_pf']:+.4f}",
                    ]
                )
                + "\n"
            )
    print(f"Wrote {path}")


write_tsv(
    OUT / "summary_by_benchmark.tsv", group_stats(cells, "benchmark"), "benchmark"
)
write_tsv(OUT / "summary_by_instr_lang.tsv", group_stats(cells, "instr"), "instr_lang")
write_tsv(OUT / "summary_by_data_lang.tsv", group_stats(cells, "data"), "data_lang")
write_tsv(
    OUT / "summary_by_condition.tsv", group_stats(cells, "condition"), "condition"
)
write_tsv(OUT / "summary_by_template.tsv", group_stats(cells, "template"), "template")

# Also cross-cuts that matter
# benchmark x instr_lang
g = defaultdict(list)
for r in cells:
    g[(r["benchmark"], r["instr"])].append(r)
with (OUT / "summary_bench_x_instr.tsv").open("w") as f:
    f.write(
        "benchmark\tinstr\tn_cells\tmean_orig_acc\tmean_repar_acc\tmean_delta_acc\tmean_orig_pf\tmean_repar_pf\tmean_delta_pf\n"
    )
    for k in sorted(g.keys()):
        v = g[k]
        f.write(
            f"{k[0]}\t{k[1]}\t{len(v)}\t"
            f"{mean(x['orig_acc'] for x in v):.4f}\t{mean(x['repar_acc'] for x in v):.4f}\t"
            f"{mean(x['delta_acc'] for x in v):+.4f}\t"
            f"{mean(x['orig_pf'] for x in v):.4f}\t{mean(x['repar_pf'] for x in v):.4f}\t"
            f"{mean(x['delta_pf'] for x in v):+.4f}\n"
        )
print("Wrote summary_bench_x_instr.tsv")

# condition x benchmark
g2 = defaultdict(list)
for r in cells:
    g2[(r["condition"], r["benchmark"])].append(r)
with (OUT / "summary_cond_x_bench.tsv").open("w") as f:
    f.write(
        "condition\tbenchmark\tn_cells\tmean_orig_acc\tmean_repar_acc\tmean_delta_acc\tmean_orig_pf\tmean_repar_pf\tmean_delta_pf\n"
    )
    for k in sorted(g2.keys()):
        v = g2[k]
        f.write(
            f"{k[0]}\t{k[1]}\t{len(v)}\t"
            f"{mean(x['orig_acc'] for x in v):.4f}\t{mean(x['repar_acc'] for x in v):.4f}\t"
            f"{mean(x['delta_acc'] for x in v):+.4f}\t"
            f"{mean(x['orig_pf'] for x in v):.4f}\t{mean(x['repar_pf'] for x in v):.4f}\t"
            f"{mean(x['delta_pf'] for x in v):+.4f}\n"
        )
print("Wrote summary_cond_x_bench.tsv")

# Top movers
top_up = sorted(cells, key=lambda r: -r["delta_acc"])[:25]
top_down = sorted(cells, key=lambda r: r["delta_acc"])[:25]
top_pf_drop = sorted(cells, key=lambda r: r["delta_pf"])[:25]
top_pf_rise = sorted(cells, key=lambda r: -r["delta_pf"])[:15]


def fmt_cell(c):
    return (
        f"{c['condition']:<22} seed{c['seed']:<4} {c['template']:<4} {c['benchmark']:<9} "
        f"data={c['data']} instr={c['instr']}  n={str(c['n']):<5}  "
        f"acc {c['orig_acc']:.3f}→{c['repar_acc']:.3f} Δ{c['delta_acc']:+.3f}  "
        f"pf {c['orig_pf']:.3f}→{c['repar_pf']:.3f} Δ{c['delta_pf']:+.3f}"
    )


print()
print("---- TOP ACC IMPROVEMENTS (top 25) ----")
for c in top_up:
    print(fmt_cell(c))
print()
print("---- TOP ACC REGRESSIONS (top 25) ----")
for c in top_down:
    print(fmt_cell(c))
print()
print("---- TOP PARSE-FAILURE DROPS (top 25) ----")
for c in top_pf_drop:
    print(fmt_cell(c))
print()
print("---- TOP PARSE-FAILURE RISES (top 15) ----")
for c in top_pf_rise:
    print(fmt_cell(c))


# Headline stats
stats = {
    "n_cells": len(cells),
    "n_improved_acc": len(improved),
    "n_regressed_acc": len(regressed),
    "n_flat_acc": len(flat),
    "n_pf_improved": len(pf_improved),
    "n_pf_worsened": len(pf_worsened),
    "n_flat_acc_pf_drop": len(flat_acc_pf_drop),
    "n_flat_acc_pf_drop_>5pct": len(flat_acc_pf_drop_5),
    "mean_delta_acc": mean(deltas_acc),
    "median_delta_acc": median(deltas_acc),
    "mean_delta_pf": mean(deltas_pf),
    "median_delta_pf": median(deltas_pf),
    "mean_orig_acc": mean(c["orig_acc"] for c in cells),
    "mean_repar_acc": mean(c["repar_acc"] for c in cells),
    "mean_orig_pf": mean(c["orig_pf"] for c in cells),
    "mean_repar_pf": mean(c["repar_pf"] for c in cells),
    "max_delta_acc": max(deltas_acc),
    "min_delta_acc": min(deltas_acc),
    "max_delta_pf_drop": min(deltas_pf),
    "max_delta_pf_rise": max(deltas_pf),
    "gaps": gaps,
}
with (OUT / "overall_stats.json").open("w") as f:
    json.dump(stats, f, indent=2)

print()
print("---- HEADLINE STATS ----")
for k, v in stats.items():
    if k == "gaps":
        print(f"  gaps: {len(v)} missing reparsed summaries")
        continue
    if isinstance(v, float):
        print(f"  {k:30s} {v:+.4f}")
    else:
        print(f"  {k:30s} {v}")

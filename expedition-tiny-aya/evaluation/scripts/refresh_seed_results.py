"""Rerunnable refresh for Phase-3 results as new (condition, seed) sessions land.

One entry point for the sequence in ``analysis/phase-3/data-refresh-cookbook.md``
plus the status and verification steps around it. Every step is idempotent and
reports what changed since the previous run, so the command can be rerun each
time another evaluation finishes.

Subcommands (run from the repository root; ``all`` runs them in order):

  status   For every (condition, seed) in the eval ladder, check Hugging Face
           for the adapter, the raw results, the inference-time summaries and
           the reparsed summaries, plus any open Hugging Face PR that carries
           them. Writes status.md / status.json under --refresh-dir and prints
           the rows whose state changed since the last run.
  reparse  For every session with results on main but no reparsed sibling,
           download the results, run the refined extractor locally into an
           overlay directory, and verify the session against the condition's
           reference session (the first other seed with results on main,
           seed 42 where it exists): same cells, row counts, inputs and gold
           labels. Without a reference session only the cell inventory is
           checked and the report says so.
           With --upload, open one Hugging Face PR per session via
           upload_reparsed_summaries.py and check the PR carries the same
           cells as the local run.
  build    Snapshot the per-session summary JSONs from main, add the local
           overlay, rebuild the refined-tables TSVs (comparison, vs-baseline,
           framework, significance), diff them against main, and regenerate
           analysis/phase-3/cross-seed-variance.{tsv,md}. With --upload, open
           one Hugging Face PR carrying the changed TSVs.
  render   Rerun the five figure scripts and build_phase3_tables.py against
           the rebuilt tables.

Nothing is merged on Hugging Face by this script; --upload only opens PRs.
Sessions matching a --provisional pattern (``condition/seedN``, shell-style
wildcards allowed) are labelled provisional in every output. The pattern is
per session, not per seed: seed 123 of a 5k condition is a Phase-3 result,
while seed 123 of a 20k condition is the later batch.

Usage:

    python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py status
    python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py all \\
        --provisional '*-20k/seed123' 'condition-3-*/seed123' 'condition-5-*/seed123'
    python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py all \\
        --provisional '*-20k/seed123' --upload      # also opens the Hugging Face PRs

The eval ladder is read from ``EVAL_MATRIX`` in evaluate.ipynb, so a seed that
is registered there is expected here; seeds found on Hugging Face but not
registered are reported as unregistered rather than ignored.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_REPO = "legesher/language-decoded-experiments"
LORA_REPO = "legesher/language-decoded-lora"
PHASE3_ROOT = "phase3/conditions"
TABLES_PREFIX = "phase3/analysis/refined-tables"
BENCHMARKS = ("xnli", "csqa", "sib200", "belebele")
TEMPLATES = ("template1", "template2")

SCRIPTS_DIR = Path(__file__).resolve().parent
NOTEBOOK = SCRIPTS_DIR / "evaluate.ipynb"
REPO_ROOT = SCRIPTS_DIR.parent.parent.parent
ANALYSIS_SCRIPTS = REPO_ROOT / "expedition-tiny-aya" / "analysis" / "scripts"
PHASE3_ANALYSIS = REPO_ROOT / "expedition-tiny-aya" / "analysis" / "phase-3"
FIGURE_SCRIPTS = (
    "fig01_extractor_slopegraph",
    "fig02_cell_scatter",
    "fig03_regression_concentration",
    "fig04_signflip_slopegraph",
    "fig05_cond5_rehabilitated",
)
BUILD_SCRIPTS = (
    "build_comparison.py",
    "build_vs_baseline.py",
    "build_framework_comparison.py",
)

# ─── Eval ladder ─────────────────────────────────────────────────────────────


def load_eval_matrix(notebook: Path = NOTEBOOK) -> dict[str, dict]:
    """Read ``EVAL_MATRIX`` from the eval notebook.

    The notebook defines it as a Python dict literal whose subfolder templates
    are f-strings over ``LORA_REPO`` / ``BASE_NS``; the definitions of those
    two names and the dict are executed in an empty namespace. Returns
    ``{condition: {"seeds": [...], "subfolder_template": str | None,
    "dataset_langs": [...], "instruction_langs": [...]}}`` with seeds as
    strings (``"none"`` for the baseline)."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if "EVAL_MATRIX = {" not in src:
            continue
        start = src.index("EVAL_MATRIX = {")
        depth, i = 0, src.index("{", start)
        while True:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        consts = "\n".join(
            m.group(0)
            for m in re.finditer(r'^(LORA_REPO|BASE_NS)\s*=\s*"[^"\n]*"\s*$', src, re.M)
        )
        ns: dict = {}
        exec(consts + "\n" + src[start : i + 1], ns)  # noqa: S102 - our own notebook
        matrix = {}
        for cond, entry in ns["EVAL_MATRIX"].items():
            matrix[cond] = {
                "seeds": ["none" if s is None else str(s) for s in entry["seeds"]],
                "subfolder_template": entry.get("subfolder_template"),
                "dataset_langs": list(entry["dataset_langs"]),
                "instruction_langs": list(entry["instruction_langs"]),
            }
        return matrix
    raise SystemExit(f"EVAL_MATRIX not found in {notebook}")


def expected_cells(entry: dict) -> set[str]:
    """Every ``template{t}_{bench}_data={d}_instr={i}`` cell a session must have."""
    return {
        f"{t}_{b}_data={d}_instr={i}"
        for t in TEMPLATES
        for b in BENCHMARKS
        for d in entry["dataset_langs"]
        for i in entry["instruction_langs"]
    }


def is_provisional(cond: str, seed: str, patterns: set[str] | list[str]) -> bool:
    """True when ``condition/seedN`` matches any shell-style pattern."""
    key = f"{cond}/seed{seed}"
    return any(fnmatch.fnmatchcase(key, pat) for pat in patterns)


def session_paths(cond: str, seed: str) -> dict[str, list[str]]:
    folder = f"{PHASE3_ROOT}/{cond}/seed{seed}"
    prefix = f"{folder}/{cond}_seed{seed}"
    return {
        "results": [f"{prefix}_results_{t}.json" for t in TEMPLATES],
        "summary": [f"{prefix}_summary_{t}.json" for t in TEMPLATES],
        "reparsed": [f"{prefix}_summary_reparsed_{t}.json" for t in TEMPLATES],
    }


# ─── Hugging Face inventory ──────────────────────────────────────────────────


def hf_inventory() -> dict:
    """Files on main of both repos plus, per open PR on the experiments repo,
    the files it adds relative to main."""
    from huggingface_hub import HfApi

    api = HfApi()
    exp_main = set(api.list_repo_files(EXPERIMENTS_REPO, repo_type="dataset"))
    lora_main = set(api.list_repo_files(LORA_REPO))
    open_prs: dict[int, dict] = {}
    for d in api.get_repo_discussions(EXPERIMENTS_REPO, repo_type="dataset"):
        if not d.is_pull_request or d.status != "open":
            continue
        try:
            pr_files = set(
                api.list_repo_files(
                    EXPERIMENTS_REPO, repo_type="dataset", revision=f"refs/pr/{d.num}"
                )
            )
        except Exception as e:  # noqa: BLE001 - a PR ref that cannot be listed is reported, not fatal
            pr_files, note = set(), f"could not list: {type(e).__name__}"
        else:
            note = ""
        open_prs[d.num] = {"title": d.title, "added": pr_files - exp_main, "note": note}
    return {"experiments_main": exp_main, "lora_main": lora_main, "open_prs": open_prs}


def _condition_short(subfolder_template: str) -> str:
    """``tiny-aya-base/condition-3-zh-5k-native-code-seed{seed}`` -> ``condition-3-zh-5k``:
    the condition token the training notebook keeps when it names an adapter
    folder after the data config instead of the eval template."""
    cond_token = subfolder_template.split("/")[-1].split("-seed")[0]
    return "-".join(cond_token.split("-")[:4])


def adapter_state(entry: dict, seed: str, lora_files: set[str]) -> str:
    """present / misnamed (<folder>) / missing / n/a (baseline)."""
    tmpl = entry.get("subfolder_template")
    if not tmpl:
        return "n/a"
    folder = tmpl.format(seed=seed) + "/"
    if any(f.startswith(folder) for f in lora_files):
        return "present"
    # The training notebook names the folder after the data config; cond-3's
    # config lives under a phase-2 prefix, so its adapter can arrive misnamed.
    short = _condition_short(tmpl)
    for f in sorted(lora_files):
        parts = f.split("/")
        if (
            len(parts) >= 2
            and parts[1].endswith(f"seed{seed}")
            and short in parts[1]
            and f"{parts[0]}/{parts[1]}/" != folder
        ):
            return f"misnamed ({parts[1]})"
    return "missing"


def _in_open_pr(paths: list[str], open_prs: dict[int, dict]) -> list[int]:
    return sorted(n for n, pr in open_prs.items() if all(p in pr["added"] for p in paths))


def classify_session(
    cond: str, seed: str, entry: dict, inv: dict, registered: bool, provisional: set[str]
) -> dict:
    paths = session_paths(cond, seed)
    main = inv["experiments_main"]
    counts = {k: sum(p in main for p in v) for k, v in paths.items()}
    prs = {k: _in_open_pr(v, inv["open_prs"]) for k, v in paths.items()}
    adapter = adapter_state(entry, seed, inv["lora_main"])

    if counts["reparsed"] == 2:
        state = "complete"
    elif counts["results"] == 2:
        state = f"reparse PR #{prs['reparsed'][0]} open" if prs["reparsed"] else "needs reparse"
    elif prs["results"]:
        state = f"results PR #{prs['results'][0]} open"
    elif counts["results"] == 1:
        state = "results partial"
    elif adapter == "present" or adapter == "n/a":
        state = "awaiting eval"
    elif adapter.startswith("misnamed"):
        state = "adapter misnamed"
    else:
        state = "awaiting adapter"
    if not registered:
        state += " (unregistered)"
    return {
        "condition": cond,
        "seed": seed,
        "registered": registered,
        "adapter": adapter,
        "results": f"{counts['results']}/2",
        "summary": f"{counts['summary']}/2",
        "reparsed": f"{counts['reparsed']}/2",
        "open_prs": sorted(set(prs["results"] + prs["reparsed"])),
        "state": state,
        "provisional": is_provisional(cond, seed, provisional),
    }


def discover_hf_seeds(cond: str, entry: dict, inv: dict) -> set[str]:
    """Seeds for ``cond`` that exist on Hugging Face in any form (adapter,
    results folder, or open PR), registered or not."""
    seeds: set[str] = set()
    pat = re.compile(rf"^{re.escape(PHASE3_ROOT)}/{re.escape(cond)}/seed([A-Za-z0-9]+)/")
    files = set(inv["experiments_main"])
    for pr in inv["open_prs"].values():
        files |= pr["added"]
    for f in files:
        m = pat.match(f)
        if m:
            seeds.add(m.group(1))
    tmpl = entry.get("subfolder_template")
    if tmpl:
        apat = re.compile("^" + re.escape(tmpl).replace(re.escape("{seed}"), r"([0-9]+)") + "/")
        # Misnamed folders (data-config name instead of the eval template) still
        # carry the condition token and the seed suffix.
        mpat = re.compile(rf"^[^/]+/[^/]*{re.escape(_condition_short(tmpl))}[^/]*-seed([0-9]+)/")
        for f in inv["lora_main"]:
            m = apat.match(f) or mpat.match(f)
            if m:
                seeds.add(m.group(1))
    return seeds


def build_status(matrix: dict, inv: dict, provisional: set[str]) -> list[dict]:
    rows = []
    for cond, entry in matrix.items():
        registered = list(entry["seeds"])
        extra = sorted(discover_hf_seeds(cond, entry, inv) - set(registered), key=_seed_key)
        for seed in registered + extra:
            rows.append(classify_session(cond, seed, entry, inv, seed in registered, provisional))
    return rows


def _seed_key(seed: str) -> tuple[int, int | str]:
    return (0, int(seed)) if seed.isdigit() else (1, seed)


def status_markdown(rows: list[dict], generated_at: str) -> str:
    lines = [
        f"# Phase-3 session status ({generated_at})",
        "",
        "| Condition | Seed | Adapter | Results | Summary | Reparsed | State |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        seed = r["seed"] + (" (provisional)" if r["provisional"] else "")
        lines.append(
            f"| {r['condition']} | {seed} | {r['adapter']} | {r['results']} | "
            f"{r['summary']} | {r['reparsed']} | {r['state']} |"
        )
    ready = [r for r in rows if r["state"] == "complete"]
    pending = [r for r in rows if r["state"] != "complete"]
    lines += ["", f"Ready: {len(ready)} sessions. Pending: {len(pending)}."]
    for r in pending:
        lines.append(f"- {r['condition']} seed {r['seed']}: {r['state']}")
    return "\n".join(lines) + "\n"


def diff_status(prev: list[dict] | None, cur: list[dict]) -> list[str]:
    """Human-readable list of sessions whose state changed since ``prev``."""
    if prev is None:
        return ["first run: no previous status to compare against"]
    before = {(r["condition"], r["seed"]): r for r in prev}
    after = {(r["condition"], r["seed"]): r for r in cur}
    out = []
    for key in sorted(set(before) | set(after), key=lambda k: (k[0], _seed_key(k[1]))):
        b, a = before.get(key), after.get(key)
        if b is None:
            out.append(f"{key[0]} seed {key[1]}: new row, {a['state']}")
        elif a is None:
            out.append(f"{key[0]} seed {key[1]}: row gone (was {b['state']})")
        else:
            changed = [
                f"{f} {b[f]} -> {a[f]}"
                for f in ("adapter", "results", "summary", "reparsed", "state")
                if b[f] != a[f]
            ]
            if changed:
                out.append(f"{key[0]} seed {key[1]}: " + "; ".join(changed))
    return out or ["no change since the last run"]


# ─── Verification of a new session against seed 42 ──────────────────────────


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


MODEL_OUTPUT_KEYS = frozenset({"raw_output", "pred", "correct"})


def _inputs(row: dict) -> dict:
    """Everything the model was given: the row minus its outputs and the gold
    label (gold is compared separately so the report names it)."""
    return {k: v for k, v in row.items() if k not in MODEL_OUTPUT_KEYS and k != "gold"}


def verify_session(
    new: dict[str, dict],
    ref: dict[str, dict] | None,
    expected: set[str],
    reparsed: dict[str, dict] | None = None,
) -> list[str]:
    """Compare a new session's two results files (keyed by template) against
    the reference session's. Returns a list of problems; empty means the
    session passes. Checks: every expected cell present in both, same row
    count per cell, identical input fields (every key except the model
    outputs and gold; benchmarks name their inputs differently, so the whole
    row is compared rather than one field), identical gold label sequence;
    and, when given, an accuracy for every cell in the reparsed summaries.
    With ``ref`` None only the new session's inventory is checked."""
    problems: list[str] = []
    for template in TEMPLATES:
        n = new.get(template)
        if n is None:
            problems.append(f"{template}: missing results file (new)")
            continue
        want = {c for c in expected if c.startswith(template + "_")}
        cells_new = {k for k in n if k.startswith(template + "_")}
        for c in sorted(want - cells_new):
            problems.append(f"{c}: missing in new session")
        if ref is not None:
            r = ref.get(template)
            if r is None:
                problems.append(f"{template}: missing results file (reference)")
            else:
                cells_ref = {k for k in r if k.startswith(template + "_")}
                for c in sorted(want - cells_ref):
                    problems.append(f"{c}: missing in reference session")
                for c in sorted(want & cells_new & cells_ref):
                    rn, rr = n[c], r[c]
                    if len(rn) != len(rr):
                        problems.append(f"{c}: {len(rn)} rows vs {len(rr)} in reference")
                        continue
                    if [_inputs(x) for x in rn] != [_inputs(x) for x in rr]:
                        problems.append(f"{c}: input fields differ from reference")
                    if [x.get("gold") for x in rn] != [x.get("gold") for x in rr]:
                        problems.append(f"{c}: gold labels differ from reference")
        if reparsed is not None:
            rs = reparsed.get(template)
            if rs is None:
                problems.append(f"{template}: reparsed summary missing")
            else:
                have = {k[: -len("_acc")] for k in rs.get("summary", {}) if k.endswith("_acc")}
                for c in sorted(want - have):
                    problems.append(f"{c}: no accuracy in reparsed summary")
    return problems


def reference_seed(cond: str, seed: str, rows: list[dict], entry: dict) -> str | None:
    """The seed to verify ``seed`` against: the first other registered seed of
    ``cond`` whose results are complete on main (seed 42 first when it is one
    of them), or None when the condition has no other evaluated seed."""
    complete = {
        r["seed"] for r in rows if r["condition"] == cond and r["results"] == "2/2" and r["seed"] != seed
    }
    candidates = [s_ for s_ in entry["seeds"] if s_ in complete]
    if "42" in candidates:
        return "42"
    return candidates[0] if candidates else None


# ─── Steps ───────────────────────────────────────────────────────────────────


class Refresh:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dir = Path(args.refresh_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.provisional = set(args.provisional)
        self.overlay = self.dir / "overlay"
        self.snapshot = self.dir / "snapshot"
        self.tables = self.dir / "tables"
        self.main_tables = self.dir / "main-tables"
        self.downloads = self.dir / "downloads"
        self.matrix = load_eval_matrix()
        self._inv: dict | None = None
        self.changes: list[str] = []

    # -- shared --
    @property
    def inv(self) -> dict:
        if self._inv is None:
            print("[hf] listing both repositories and open PRs...")
            self._inv = hf_inventory()
        return self._inv

    def _download(self, path_in_repo: str, revision: str | None = None) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                EXPERIMENTS_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=str(self.downloads if revision is None else self.downloads / revision.replace("/", "_")),
            )
        )

    def _run(self, script: Path, *argv: str, env: dict | None = None) -> str:
        cmd = [sys.executable, str(script), *argv]
        full_env = {**os.environ, **(env or {})}
        print(f"[run] {script.name} {' '.join(argv)}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=full_env, text=True, capture_output=True)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"{script.name} failed with exit code {proc.returncode}")
        return proc.stdout

    # -- status --
    def status(self) -> list[dict]:
        rows = build_status(self.matrix, self.inv, self.provisional)
        now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        state_file = self.dir / "status.json"
        prev = json.loads(state_file.read_text())["rows"] if state_file.exists() else None
        changes = diff_status(prev, rows)
        md = status_markdown(rows, now)
        md += "\n## Changed since last run\n\n" + "\n".join(f"- {c}" for c in changes) + "\n"
        if self.inv["open_prs"]:
            md += "\n## Open Hugging Face PRs on the experiments repo\n\n"
            for n, pr in sorted(self.inv["open_prs"].items()):
                md += f"- #{n}: {pr['title']} ({len(pr['added'])} added files{'; ' + pr['note'] if pr['note'] else ''})\n"
        (self.dir / "status.md").write_text(md, encoding="utf-8")
        if state_file.exists():
            shutil.copy(state_file, self.dir / "status.prev.json")
        state_file.write_text(json.dumps({"generated_at": now, "rows": rows}, indent=2))
        print(md)
        self.changes += [f"status: {c}" for c in changes if not c.startswith("no change")]
        return rows

    # -- reparse --
    def reparse(self, rows: list[dict]) -> list[tuple[str, str]]:
        """Reparse locally every session that has results but no reparsed
        sibling on main. Sessions whose reparsed siblings sit in an open PR are
        reparsed locally too and the PR content is checked against that local
        run; sessions with nothing on main yet are uploaded only with --upload."""
        sys.path.insert(0, str(SCRIPTS_DIR))
        from reparse_results import build_reparsed_summary, reparse_file

        todo = [
            (r["condition"], r["seed"], r["open_prs"])
            for r in rows
            if r["state"].startswith(("needs reparse", "reparse PR"))
        ]
        if not todo:
            print("[reparse] nothing to reparse or verify")
            return []
        done = []
        for cond, seed, prs in todo:
            entry = self.matrix[cond]
            ref_seed = reference_seed(cond, seed, rows, entry)
            print(f"[reparse] {cond} seed {seed} (reference seed {ref_seed or 'none available'})")
            new_results, reparsed = {}, {}
            ref_results: dict[str, dict] | None = {} if ref_seed is not None else None
            for t, remote in zip(TEMPLATES, session_paths(cond, seed)["results"]):
                local = self._download(remote)
                rows_ = reparse_file(local, only=None)
                body = build_reparsed_summary(Path(remote), rows_)
                out = self.overlay / remote.replace("_results_", "_summary_reparsed_")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
                new_results[t] = load_results(local)
                reparsed[t] = body
                print(f"  wrote {out.relative_to(self.dir)} (cells_changed={body['reparse_metadata']['cells_changed']})")
            if ref_results is not None:
                for t, remote in zip(TEMPLATES, session_paths(cond, ref_seed)["results"]):
                    ref_results[t] = load_results(self._download(remote))
            problems = verify_session(new_results, ref_results, expected_cells(entry), reparsed)
            report = self.dir / f"verify_{cond}_seed{seed}.md"
            verdict = "PASS" if not problems else f"FAIL ({len(problems)} problems)"
            against = f"vs seed {ref_seed}" if ref_seed is not None else "(no reference session available)"
            lines = [f"# {cond} seed {seed} {against}: {verdict}", ""]
            if problems:
                lines += [f"- {p}" for p in problems]
            elif ref_seed is not None:
                lines.append(
                    "- every expected cell present; row counts, input fields and gold labels identical to the reference session"
                )
            else:
                lines.append(
                    "- every expected cell present in the results and the reparsed summaries; no other evaluated seed of this condition exists on main, so no cross-session comparison was made"
                )
            print(f"  verification: {verdict} -> {report.name}")
            if problems:
                for p in problems[:20]:
                    print("   ", p)
                report.write_text("\n".join(lines) + "\n", encoding="utf-8")
                raise SystemExit(f"{cond} seed {seed} failed verification; not uploading")
            self.changes.append(f"reparse: {cond} seed {seed} reparsed locally, verification {verdict}")
            if prs:
                for pr in prs:
                    lines += ["", f"## Hugging Face PR #{pr}", ""]
                    lines += [f"- {m}" for m in self._check_pr_matches(cond, seed, pr, reparsed)]
            elif self.args.upload:
                pr = self._upload_reparsed(cond, seed)
                lines += ["", f"## Hugging Face PR #{pr} (opened by this run)", ""]
                lines += [f"- {m}" for m in self._check_pr_matches(cond, seed, pr, reparsed)]
            report.write_text("\n".join(lines) + "\n", encoding="utf-8")
            done.append((cond, seed))
        return done

    def _check_pr_matches(self, cond: str, seed: str, pr: int, local: dict[str, dict]) -> list[str]:
        """Compare the reparsed summaries on ``refs/pr/N`` with the local run:
        same accuracies and parse-failure rates for every cell, and the same
        extractor hash. Raises if they differ."""
        notes = []
        for t, remote in zip(TEMPLATES, session_paths(cond, seed)["reparsed"]):
            body = json.loads(self._download(remote, revision=f"refs/pr/{pr}").read_text(encoding="utf-8"))
            same = (
                body["summary"] == local[t]["summary"]
                and body["parse_failure_rates"] == local[t]["parse_failure_rates"]
            )
            sha_pr = body.get("reparse_metadata", {}).get("extractor_provenance", {}).get("content_sha256")
            sha_local = local[t]["reparse_metadata"]["extractor_provenance"]["content_sha256"]
            n_cells = sum(1 for k in body["summary"] if k.endswith("_acc"))
            msg = (
                f"{Path(remote).name}: {n_cells} cells; "
                f"{'identical to the local reparse' if same else 'DIFFERS from the local reparse'}; "
                f"extractor sha {'matches' if sha_pr == sha_local else 'differs: ' + str(sha_pr)[:12]}"
            )
            print(f"  PR #{pr} {msg}")
            notes.append(msg)
            if not same:
                raise SystemExit(f"PR #{pr} content differs from the local reparse for {remote}")
        self.changes.append(f"verify: Hugging Face PR #{pr} ({cond} seed {seed}) matches the local reparse cell for cell (unmerged)")
        return notes

    def _upload_reparsed(self, cond: str, seed: str) -> int:
        out = self._run(
            SCRIPTS_DIR / "upload_reparsed_summaries.py",
            "--only",
            f"{cond}/seed{seed}",
            "--commit-message",
            f"add reparsed summaries for {cond} seed {seed}",
        )
        m = re.search(r"PR number:\s+(\d+)", out)
        if not m:
            print(out[-2000:])
            raise SystemExit("upload_reparsed_summaries.py did not report a PR number")
        pr = int(m.group(1))
        print(f"[upload] reparsed summaries PR #{pr} opened for {cond} seed {seed}")
        self.changes.append(f"upload: Hugging Face PR #{pr} opened with reparsed summaries for {cond} seed {seed} (unmerged)")
        return pr

    # -- build --
    def build(self) -> list[str]:
        from huggingface_hub import snapshot_download

        print("[build] snapshotting summary JSONs and refined tables from main...")
        if self.snapshot.exists():
            shutil.rmtree(self.snapshot)
        snapshot_download(
            EXPERIMENTS_REPO,
            repo_type="dataset",
            allow_patterns=[f"{PHASE3_ROOT}/*/*/*_summary_*.json"],
            local_dir=str(self.snapshot),
        )
        if self.main_tables.exists():
            shutil.rmtree(self.main_tables)
        snapshot_download(
            EXPERIMENTS_REPO,
            repo_type="dataset",
            allow_patterns=[f"{TABLES_PREFIX}/*"],
            local_dir=str(self.main_tables),
        )
        overlaid, superseded = [], []
        if self.overlay.exists():
            for src in sorted(self.overlay.rglob("*_summary_reparsed_*.json")):
                rel = src.relative_to(self.overlay)
                dst = self.snapshot / rel
                if dst.exists():
                    superseded.append(str(rel))
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                overlaid.append(str(rel))
        if overlaid:
            print(f"[build] overlaid {len(overlaid)} local reparsed summaries not yet on main")
        if superseded:
            print(f"[build] {len(superseded)} overlay files are now on main and were ignored")
        if self.tables.exists():
            shutil.rmtree(self.tables)
        self.tables.mkdir(parents=True)
        env = {
            "PHASE3_SNAPSHOT_ROOT": str(self.snapshot / PHASE3_ROOT),
            "PHASE3_OUT_DIR": str(self.tables),
            "PHASE3_TABLES_DIR": str(self.tables),
            "PHASE3_ANALYSIS_TABLES_DIR": str(self.tables),
        }
        for script in BUILD_SCRIPTS:
            self._run(SCRIPTS_DIR / script, env=env)
        self._run(SCRIPTS_DIR / "build_significance.py", env=env)
        changed = self._diff_tables()
        prov = ["--provisional", *sorted(self.provisional)] if self.provisional else []
        print(self._run(SCRIPTS_DIR / "build_cross_seed_variance.py", "--cells", str(self.tables / "vs_baseline_cells.tsv"), *prov))
        if self.args.upload and changed:
            self._upload_tables(changed, overlaid)
        return changed

    def _diff_tables(self) -> list[str]:
        main_dir = self.main_tables / TABLES_PREFIX
        changed, same, new = [], [], []
        for f in sorted(self.tables.iterdir()):
            if f.name == "overall_stats.json" or f.suffix not in (".tsv", ".json"):
                continue
            ref = main_dir / f.name
            if not ref.exists():
                new.append(f.name)
            elif ref.read_bytes() != f.read_bytes():
                changed.append(f.name)
            else:
                same.append(f.name)
        lines = [f"changed ({len(changed)}): " + ", ".join(changed) if changed else "changed: none"]
        lines.append(f"new ({len(new)}): " + ", ".join(new) if new else "new: none")
        lines.append(f"unchanged: {len(same)}")
        for name in changed:
            ref = main_dir / name
            n_ref = sum(1 for _ in ref.open(encoding="utf-8")) - 1
            n_new = sum(1 for _ in (self.tables / name).open(encoding="utf-8")) - 1
            lines.append(f"  {name}: {n_ref} -> {n_new} data rows")
        text = "\n".join(lines)
        print("[build] refined-tables vs main:\n" + text)
        (self.dir / "changed_tables.txt").write_text(text + "\n", encoding="utf-8")
        if changed or new:
            self.changes.append("build: refined tables differing from main: " + ", ".join(changed + new))
        return changed + new

    TABLES_PR_TITLE = "refined-tables: regenerate"

    def _upload_tables(self, names: list[str], overlaid: list[str]) -> None:
        """Open one Hugging Face PR with the changed TSVs, or add a commit to
        the open PR this script created earlier so reruns do not pile up PRs.
        The PR's status is re-read first: a push to the ref of a merged PR
        would succeed silently and never show up anywhere."""
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        sessions = sorted({"/".join(Path(p).parts[2:4]) for p in overlaid}) if overlaid else []
        prov = (
            f"\n\nProvisional: sessions matching {', '.join(sorted(self.provisional))} were evaluated in a "
            "later software environment than the other sessions. Their rows are provisional until the "
            "evaluation drift between the two environments has been measured."
            if self.provisional
            else ""
        )
        desc = (
            "Regenerated with build_comparison.py, build_vs_baseline.py, build_framework_comparison.py "
            "and build_significance.py from the per-session summaries on main"
            + (f" plus the reparsed summaries for {', '.join(sessions)} (open PRs on this repo)" if sessions else "")
            + f". Files: {', '.join(names)}. Paper tables and figures in the research repo cite these views."
            + prov
        )
        ops = [
            CommitOperationAdd(path_in_repo=f"{TABLES_PREFIX}/{n}", path_or_fileobj=str(self.tables / n))
            for n in names
        ]
        message = self.TABLES_PR_TITLE + (f" with {', '.join(sessions)}" if sessions else "")
        existing = [
            n
            for n, pr in sorted(self.inv["open_prs"].items())
            if pr["title"].startswith(self.TABLES_PR_TITLE)
            and api.get_discussion_details(EXPERIMENTS_REPO, n, repo_type="dataset").status == "open"
        ]
        if existing:
            num = existing[0]
            api.create_commit(
                repo_id=EXPERIMENTS_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message=message,
                commit_description=desc,
                revision=f"refs/pr/{num}",
                create_pr=False,
            )
            print(f"[upload] added a commit to the open refined-tables PR #{num}")
            self.changes.append(f"upload: Hugging Face PR #{num} updated with {len(names)} refined-tables files (unmerged)")
            return
        info = api.create_commit(
            repo_id=EXPERIMENTS_REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=message,
            commit_description=desc,
            create_pr=True,
        )
        print(f"[upload] refined-tables PR #{info.pr_num}: {info.pr_url}")
        self.changes.append(f"upload: Hugging Face PR #{info.pr_num} opened with {len(names)} refined-tables files (unmerged)")

    # -- render --
    def render(self) -> None:
        if not (self.tables / "vs_baseline_cells.tsv").exists():
            raise SystemExit("run `build` first; no rebuilt tables found")
        for fig in FIGURE_SCRIPTS:
            self._run(ANALYSIS_SCRIPTS / f"{fig}.py", "--tables-dir", str(self.tables))
        prov = ["--provisional", *sorted(self.provisional)] if self.provisional else []
        out = self._run(ANALYSIS_SCRIPTS / "build_phase3_tables.py", "--tables-dir", str(self.tables), *prov)
        if out.strip():
            print(out.strip().splitlines()[-1])
        stat = subprocess.run(
            ["git", "diff", "--stat", "--", "expedition-tiny-aya/analysis/figures-phase3/", str(PHASE3_ANALYSIS)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        ).stdout
        print("[render] git diff --stat:\n" + (stat or "  (no changes)"))
        if stat.strip():
            self.changes.append("render: " + stat.strip().splitlines()[-1].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["status", "reparse", "build", "render", "all"])
    parser.add_argument("--refresh-dir", default=os.environ.get("PHASE3_REFRESH_DIR", "/tmp/phase3_refresh"))
    parser.add_argument(
        "--provisional",
        nargs="*",
        default=[],
        metavar="COND/SEED",
        help="Sessions to label provisional, as condition/seedN with shell wildcards.",
    )
    parser.add_argument("--upload", action="store_true", help="Open Hugging Face PRs (never merges).")
    args = parser.parse_args()
    r = Refresh(args)
    if args.step in ("status", "all", "reparse"):
        rows = r.status()
    if args.step in ("reparse", "all"):
        r.reparse(rows)
    if args.step in ("build", "all"):
        r.build()
    if args.step in ("render", "all"):
        r.render()
    print("\n== Changed in this run ==")
    for c in r.changes or ["nothing changed"]:
        print(f"- {c}")
    (r.dir / "last_run_changes.md").write_text("\n".join(f"- {c}" for c in r.changes or ["nothing changed"]) + "\n")


if __name__ == "__main__":
    main()

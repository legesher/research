# Phase-3 data-refresh cookbook

When the underlying per-session JSONs change on HuggingFace (new condition, re-scoring, extractor update), the downstream refined-tables TSVs, paper figures, and result tables all need to regenerate. This is the documented sequence. Each step is run from the repo root.

**When you need this:** new model session uploaded; extractor refined again; bug-class fix that touches refined-tables (e.g. HF PR #38, #40); or any future rebuttal / camera-ready cycle that requires fresh numbers.

**Triggers that DO NOT require this:** GitHub-only docs/code changes that don't touch the data path; figure-style tweaks; caption rewrites.

---

## One command: `refresh_seed_results.py`

`evaluation/scripts/refresh_seed_results.py` runs steps 1 to 5 below as one rerunnable command and reports what changed since its last run. Use it when new (condition, seed) sessions are landing over several days; use the manual steps when you need to intervene between them.

```bash
# Status only: for every (condition, seed) in EVAL_MATRIX, what is on Hugging Face
# (adapter, results, summaries, reparsed summaries, open PRs), plus the diff since the last run.
python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py status

# Everything, no uploads: reparse new sessions locally and verify them against seed 42,
# rebuild the refined tables with the local reparsed summaries overlaid, regenerate
# cross-seed-variance.{tsv,md}, rerun the figures and tables.tex.
python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py all \
  --provisional '*-20k/seed123' 'condition-3-*/seed123' 'condition-5-*/seed123'

# Same, and also open the Hugging Face PRs (one per reparsed session, one for the changed TSVs).
python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py all --provisional ... --upload
```

- `--refresh-dir` (default `/tmp/phase3_refresh`, or `$PHASE3_REFRESH_DIR`) holds `status.md`, `status.json`, one `verify_<condition>_seed<N>.md` per new session, the rebuilt `tables/`, `changed_tables.txt` and `last_run_changes.md`.
- `--provisional` labels sessions (`condition/seedN`, shell wildcards) whose results are not yet final. The label is carried into `cross-seed-variance.tsv` (`provisional` column), `cross-seed-variance-by-condition.md`, the `tables.tex` header and a `% provisional` comment on every table that includes such a session, and the Hugging Face PR descriptions.
- The script never merges on Hugging Face. Sessions whose reparsed summaries sit in an open PR are reparsed locally anyway and the PR content is checked cell for cell against that local run.
- The eval ladder is read from `EVAL_MATRIX` in `evaluate.ipynb`; a seed present on Hugging Face but not registered there is reported as `(unregistered)`.
- It needs `scipy` (for `build_significance.py`) in addition to the packages in step 0.

## 0. Pre-flight

- [ ] Active worktree off `origin/main` (per repo convention; don't run from a worktree that's on an unrelated feature branch).
- [ ] `huggingface_hub` installed and `huggingface-cli login` is current with a write-scope token.
- [ ] A Python environment with `pandas`, `matplotlib`, `scienceplots`, and `huggingface_hub` installed. Examples below assume `python` resolves to that env; substitute your venv path (e.g. `./venv/bin/python`) if needed.

## 1. Regenerate refined-tables TSVs from per-session JSONs

```bash
mkdir -p /tmp/phase3_refresh
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  python expedition-tiny-aya/evaluation/scripts/build_comparison.py
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  python expedition-tiny-aya/evaluation/scripts/build_vs_baseline.py
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  python expedition-tiny-aya/evaluation/scripts/build_framework_comparison.py
```

These read `_summary_reparsed_*.json` files from HF, aggregate, and emit the ~20 TSVs that live at `phase3/analysis/refined-tables/` on HF. Output lands in `$PHASE3_OUT_DIR`. Verify row counts before proceeding — `wc -l /tmp/phase3_refresh/*.tsv` should match expected (`cells.tsv` has 64 rows per 128-cell session plus the baseline; 1,664 + header for the 21 Phase-3 sessions).

Two more generated tables depend on `vs_baseline_cells.tsv` and are rebuilt from it:

```bash
PHASE3_TABLES_DIR=/tmp/phase3_refresh PHASE3_ANALYSIS_TABLES_DIR=/tmp/phase3_refresh \
  python expedition-tiny-aya/evaluation/scripts/build_significance.py      # significance.tsv
python expedition-tiny-aya/evaluation/scripts/build_cross_seed_variance.py \
  --cells /tmp/phase3_refresh/vs_baseline_cells.tsv                        # analysis/phase-3/cross-seed-variance.{tsv,md}
```

## 2. Upload changed TSVs to HF via single-file PR

Use the pattern established by HF PR #38 / #40 / #41: one TSV per HF PR, opened via `huggingface_hub.create_commit(..., create_pr=True)`. Single-file PRs are easier to review and roll back.

```python
from huggingface_hub import CommitOperationAdd, create_commit
create_commit(
    repo_id="legesher/language-decoded-experiments",
    repo_type="dataset",
    operations=[CommitOperationAdd(
        path_in_repo="phase3/analysis/refined-tables/<filename>.tsv",
        path_or_fileobj="/tmp/phase3_refresh/<filename>.tsv",
    )],
    commit_message="fix(refined-tables): <what changed>",
    commit_description="<diff summary + rationale>",
    create_pr=True,
)
```

The diff summary should name (a) the affected column(s), (b) which build script regenerated it, (c) which paper artifacts cite it. Review on HF, merge.

If multiple TSVs changed in coordinated ways (rare — usually only one bug class at a time), bundle them into a single HF PR. Otherwise prefer one-PR-per-TSV.

## 3. Clear local HF cache

```bash
rm -rf ~/.cache/huggingface/hub/datasets--legesher--language-decoded-experiments
```

`hf_hub_download` checks ETags but the cache invalidation is sometimes flaky across cross-machine setups. Clearing forces a fresh pull on the next call. Cheap (~30s to re-download).

## 4. Rerun all 5 figures + the tables

```bash
for fig in \
  fig01_extractor_slopegraph \
  fig02_cell_scatter \
  fig03_regression_concentration \
  fig04_signflip_slopegraph \
  fig05_cond5_rehabilitated; do
  python expedition-tiny-aya/analysis/scripts/$fig.py
done
python expedition-tiny-aya/analysis/scripts/build_phase3_tables.py
```

All scripts default to HF refined-tables; pass `--tables-dir /tmp/phase3_refresh` to read from a local snapshot instead (useful if you're staging changes before the HF PR is merged). `build_phase3_tables.py` also takes `--provisional 'condition/seedN'` patterns; the header accounting (sessions per condition, seeds, cell counts) is computed from the cells file, so adding a seed needs no edit to the generator.

## 5. Diff outputs vs committed

```bash
git diff --stat \
  expedition-tiny-aya/analysis/figures-phase3/ \
  expedition-tiny-aya/analysis/phase-3/tables.tex
```

PDFs and PNGs that show 0-byte diffs mean the data is bit-identical to the last committed run — no figure regeneration needed. Non-zero diffs mean genuine data drift; commit them.

## 6. Audit hand-written numbers

These files contain hand-set numbers that the rerun scripts do NOT update automatically:

- `expedition-tiny-aya/analysis/phase-3/phase3-refined-evaluation.md` — §1 headline table, §3.1 by-benchmark table, §3.2 instr-lang totals + SIB-200×instr crosstab, §3.3 template table, §3.4 condition table, §4 anomalies table, §8.7 conclusion-flip totals. Re-derive from current HF state.
- `expedition-tiny-aya/analysis/phase-3/captions.md` — extreme-cell call-outs in fig 2 caption (currently mentions `−76 pp pf, +51 pp acc` for the headline cell). If that cell's coordinates shift, update the caption text.
- Any private notes or memory files that quote specific TSV numbers — re-verify if the corresponding views regenerated.

Pull current HF values via `curl -sSL "https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/<name>.tsv"` and grep for the cited numbers.

## 7. Commit + PR

Stage scripts, figures, tables, and writeup updates together so the diff is reviewable as one cohesive refresh. Reference the HF PR(s) from step 2 in the GitHub PR description for traceability.

```bash
git add expedition-tiny-aya/analysis/figures-phase3/ \
        expedition-tiny-aya/analysis/phase-3/{tables.tex,captions.md,phase3-refined-evaluation.md}
git -c user.email=<your-id>+<your-username>@users.noreply.github.com commit \
  -m "refresh(phase-3): regenerate against HF main post <PR ref>"
git push -u origin docs/data-refresh-<short-name>
gh pr create --base main --title "refresh(phase-3): ..."
```

The committer-email override is required by this repo's email-privacy restriction: GitHub will reject pushes that expose a private email address. Use *your own* `users.noreply.github.com` address (find yours at <https://github.com/settings/emails>, "Keep my email addresses private"). Set it persistently with `git config user.email <your-noreply>` if you prefer not to pass `-c` each commit.

## Verification checklist

- [ ] Row counts in HF match locally-generated TSVs (post step 1)
- [ ] HF PR(s) merged before figures rerun (step 2 → 3 ordering matters)
- [ ] All 5 fig scripts exit with no `--` cells / no uniqueness assertion failures
- [ ] `build_phase3_tables.py` writes one table block per writer (22 blocks today) and its header accounting matches the session count in `status.md`
- [ ] `cross-seed-variance.tsv` row count is 40 per condition with two instruction languages, 80 with four (560 for the 12 Phase-3 conditions); `n_seeds` matches the status table
- [ ] `git diff --stat` shows expected files only (no unintended drift)
- [ ] Writeup hand-numbers re-checked against current HF state

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Fig script raises `KeyError` on a column | Build script renamed a TSV column | Update fig script's column reference; the renamed column should also appear in the rebuild output |
| `\textbf{--}` in tables.tex | Bold logic firing on missing cells (was a known bug pre-`1179bd1`) | Should not recur — uniqueness + NaN guards are in place; if it does, check `build_phase3_tables.py:render_type3_table` for skipped NaN guards |
| Figure PDF bytes differ but PNG looks identical | matplotlib non-determinism (font metadata, embed timestamps) | Set `mpl.rcParams["pdf.fonttype"] = 42` (already set in `_viz_common.py`); verify scienceplots style is installed |
| HF PR merge button greyed out | TSV diff exceeds HF's review threshold | Split into smaller per-column PRs, or use `huggingface-cli` to bypass review |
| `hf_hub_download` returns old cached TSV after upload | Cache TTL not yet invalidated | Step 3 — `rm -rf ~/.cache/huggingface/hub/datasets--legesher--language-decoded-experiments` |

## Cross-references

- HF PR #38 — `framework_template_robustness.tsv` seed-vs-cell fix (canonical example of the single-TSV-refresh pattern)
- HF PR #40 — `framework_*.tsv` median calculation fix (`statistics.median` vs `sorted//2`)
- HF PR #41 + #42 — `correct-via-constant-rates.tsv` add + seed-format fix
- GitHub commit `b0ec1dc` — View D fix (build_framework_comparison.py `write_template_robustness`)
- `aggregation-bug-audit.md` (this directory) — catalog of remaining seed-vs-cell bug sites; same refresh pattern applies for each fix
- `cross-seed-variance.md` (this directory) — the method behind `cross-seed-variance.tsv`; the TSV and `cross-seed-variance-by-condition.md` are generated by `evaluation/scripts/build_cross_seed_variance.py`
- Paper-prep callouts memory file — list of memory items quoting framework_*.tsv numbers that may need re-verification after a refresh

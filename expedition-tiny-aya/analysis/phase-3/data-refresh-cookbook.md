# Phase-3 data-refresh cookbook

When the underlying per-session JSONs change on HuggingFace (new condition, re-scoring, extractor update), the downstream refined-tables TSVs, paper figures, and result tables all need to regenerate. This is the documented sequence. Each step is run from the repo root.

**When you need this:** new model session uploaded; extractor refined again; bug-class fix that touches refined-tables (e.g. HF PR #38, #40); or any future rebuttal / camera-ready cycle that requires fresh numbers.

**Triggers that DO NOT require this:** GitHub-only docs/code changes that don't touch the data path; figure-style tweaks; caption rewrites.

---

## 0. Pre-flight

- [ ] Active worktree off `origin/main` (per repo convention; don't run from Madison's primary worktree if it's on a branch).
- [ ] `huggingface_hub` installed and `huggingface-cli login` is current (write scope; same token used for HF PR #38 et al.).
- [ ] `/tmp/figvenv/bin/python` exists with pandas, matplotlib, scienceplots, huggingface_hub — or substitute a venv that has them.

## 1. Regenerate refined-tables TSVs from per-session JSONs

```bash
mkdir -p /tmp/phase3_refresh
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  /tmp/figvenv/bin/python expedition-tiny-aya/evaluation/scripts/build_comparison.py
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  /tmp/figvenv/bin/python expedition-tiny-aya/evaluation/scripts/build_vs_baseline.py
PHASE3_OUT_DIR=/tmp/phase3_refresh \
  /tmp/figvenv/bin/python expedition-tiny-aya/evaluation/scripts/build_framework_comparison.py
```

These read `_summary_reparsed_*.json` files from HF, aggregate, and emit the ~20 TSVs that live at `phase3/analysis/refined-tables/` on HF. Output lands in `$PHASE3_OUT_DIR`. Verify row counts before proceeding — `wc -l /tmp/phase3_refresh/*.tsv` should match expected (e.g. `cells.tsv` ≈ 1,664 rows + header).

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
  /tmp/figvenv/bin/python expedition-tiny-aya/analysis/scripts/$fig.py
done
/tmp/figvenv/bin/python expedition-tiny-aya/analysis/scripts/build_phase3_tables.py
```

All scripts default to HF refined-tables; pass `--tables-dir /tmp/phase3_refresh` to read from a local snapshot instead (useful if you're staging changes before the HF PR is merged).

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
- `~/.claude/projects/.../memory/paper-prep-callouts.md` — items #20, #22, #23, #24 quote specific numbers from `framework_*.tsv` views. Items #8, #34, #12 quote per-cell numbers. Re-verify if those views regenerated.

Pull current HF values via `curl -sSL "https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/<name>.tsv"` and grep for the cited numbers.

## 7. Commit + PR

Stage scripts, figures, tables, and writeup updates together so the diff is reviewable as one cohesive refresh. Reference the HF PR(s) from step 2 in the GitHub PR description for traceability.

```bash
git add expedition-tiny-aya/analysis/figures-phase3/ \
        expedition-tiny-aya/analysis/phase-3/{tables.tex,captions.md,phase3-refined-evaluation.md}
git -c user.email=7844510+madiedgar@users.noreply.github.com commit -m "refresh(phase-3): regenerate against HF main post <PR ref>"
git push -u origin docs/data-refresh-<short-name>
gh pr create --base main --title "refresh(phase-3): ..."
```

The committer-email override is required on this repo (email-privacy restriction blocks the default).

## Verification checklist

- [ ] Row counts in HF match locally-generated TSVs (post step 1)
- [ ] HF PR(s) merged before figures rerun (step 2 → 3 ordering matters)
- [ ] All 5 fig scripts exit with no `--` cells / no uniqueness assertion failures
- [ ] `build_phase3_tables.py` reports 14 tables, file size 20–22 KB
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
- Paper-prep callouts memory file — list of memory items quoting framework_*.tsv numbers that may need re-verification after a refresh

# Post-reparse action items — Phase-3

**Originally captured 2026-05-23. Status updated 2026-05-24** after PR #54
landed + HF PR #34 published the full-coverage reparsed dataset.

Items surfaced during a deep critical pass on the extractor extension +
full-suite reparse. The reparse pipeline + refined extractor cover **WHAT
the model said** much better than the inference-time extractor did. The
items below are about making the next-round reparse reproducible,
auditable, and paper-grade.

**Status:**

- ✅ **A.1 done** — HF PR #33 deleted stray seed-123 files from `condition-2-es-5k/seed42/`.
- ⏳ **A.2 pending** — filename-seed-vs-parent-dir guard in `reparse_results.py`.
- ✅ **B done** — obsoleted by PR #54's redirect; extractors are inline in `reparse_results.py`, no AST loader to be confused by a stale `run_eval_single.py`.
- ✅ **C done** — `upload_reparsed_summaries.py` flipped to skip-existing default in PR #54.
- ✅ **D done** — `{key}_count` and `{key}_correct` written to reparsed summaries in PR #54. (Per PR #54's redirect, these land in `build_reparsed_summary` rather than `evaluate.ipynb` cell 3 — the notebook stays frozen for Phase-3 reproducibility.)
- ⏳ **E pending** — cond-5 dataset card banner on `legesher/language-decoded-experiments`.
- ⏳ **F pending** — `correct_via_constant_pct` analysis artefact.
- ⏳ **G in progress** — conclusion-flip audit; the canonical flip list is now in `analysis/reparse-tables/conclusion_flips.tsv` (regenerated against HF main post HF PR #34, 48 flips).
- ⏳ **H pending** — parse-fail floor verification samples.
- ⏳ **I pending** — cond-5-ur-5k +0.515 acc lift sanity sample.

---

## A. Misfiled-summary duplicates in the existing HF dataset

**Problem.** `phase3/conditions/condition-2-es-5k/seed42/` contains both the
legitimate seed42 files **and** stray copies of seed123 files (the seed123
files also live in `seed123/`). Source: derived seed from filename, not
directory, during the analysis pass — had to dedupe against the parent
directory. This is an upload-pipeline bug, not an extractor bug.

**Mitigation status.** [`upload_reparsed_summaries.py`][upload] already
detects this via `_classify_session_files`: it splits `canonical_results`
(filename-seed matches parent-dir seed) from `stray_results` (mismatch).
Stray files are visibly logged (`✗ stray: <name>`) and **excluded** from
reparse, so they don't double-count at upload time. But the existing
dataset still has the pollution baked in, and `reparse_results.py` itself
has no such guard — if you point it at a misfiled file directly, it'll
cheerfully reparse it and produce a misleading sibling.

**Fix.** Two parts:

1. **HF cleanup script.** Write `clean_stray_files.py` that:
   - Uses the same `_classify_session_files` to list stray files across
     every session.
   - Either deletes them or moves to a `_stray/` prefix.
   - Audits all conditions, not just the known `condition-2-es-5k/seed42`.
2. **Reject in `reparse_results.py`.** Assert filename-seed == parent-dir
   seed before consuming a `_results_*.json`; refuse to write a sibling if
   the input is misfiled.

---

## B. Stale-extractor footgun on `reparse_results.py`

**Problem.** Before PR #53, `reparse_results.py` only knew how to read a
pre-extracted `run_eval_single.py`. A gitignored copy from a previous
branch could be silently picked up — this bit twice during the session.

**Status.** **Resolved on main as of 4961cf4 (PR #53 merged).** The
notebook-fallback (`_find_extractor_source` falls back to `evaluate.ipynb`,
`_read_extractor_source` extracts the `%%writefile` cell body) means a
fresh checkout of any branch will use that branch's notebook extractor.

**Remaining work.** None code-side. Make sure any documentation that still
says "extract `run_eval_single.py` from cell 3 first" gets updated.

---

## C. Re-run semantics in `upload_reparsed_summaries.py`

**Problem.** `--skip-existing` is implemented; **default is overwrite**.
A careless re-run with the wrong branch checked out clobbers good numbers.

**Fix (~15 lines).**

- Flip the default to `--skip-existing`; require `--overwrite` explicit.
- Add `--fail-on-existing` mode for paranoia.
- When `--overwrite` is passed, log the extractor `content_sha256` (we have
  a stable one as of PR #53's `_extractor_provenance` change) so re-uploads
  are auditable.

---

## D. `_count` missing from summary cells

**Problem.** At [`evaluate.ipynb` cell 3, line 735][eval], the per-cell
summary writer emits only `summary[f"{key}_acc"] = acc`. No `{key}_count`.
For paper-grade reporting (mean ± std with n, statistical tests), anyone
who wants n per cell currently has to re-open `_results_*.json`. Cheap to
add now; expensive to retrofit after the dataset is consumed.

**Fix (1 line, maybe 2).**

```python
summary[f"{key}_acc"] = acc
summary[f"{key}_count"] = len(rows)
summary[f"{key}_correct"] = int(rows["correct"].sum())  # optional, paranoia-safe
```

`rows` is already in scope. Land on PR #54 as a one-commit amendment, or
right after #54 merges as its own small PR.

---

## E. Cond-5 numbers under the original extractor are extractor-coverage-confounded

**Problem.** `cond-5-zh-5k` SIB-200 cells had 20–35% parse-failure rates
under the original extractor — not the model failing, the strict extractor
refusing to read its native-script answers. Cond-5 original-extractor
numbers must not be cited standalone without a warning.

**Fix.** Dataset card update on `legesher/language-decoded-experiments`:

- Top-of-card banner: "Original `_summary_*.json` numbers under-report
  cond-5 SIB-200 accuracy by 20–35pp because the strict extractor refused
  native-script answers. Cite `_summary_reparsed_*.json` for paper-grade
  numbers."
- Cross-link to [`reparse-decision-ledger.md`][ledger] and
  [`phase3-reparse-evaluation.md`][writeup].

---

## F. Cond-2-ur-5k constant-output rate

**Problem.** Per the decision ledger, the Urdu-tuned cond-2 model emits a
near-constant `سائنس/ٹیکنالوجی` (science/technology) regardless of passage.
Some of its "correct" rows are just that constant landing on gold by
chance — meaning the cell accuracy slightly overstates the model even
under the reparsed extractor. The ledger flagged a planned per-row
`correct_ambiguous` flag, never landed.

**Fix.** Cheap as a separate analysis artifact rather than baking into the
summary writer:

- `analysis/correct-via-constant-rates.tsv` next to the reparse tables.
- Computed from `_results_*.json` + a known-constant-output detector
  (look for the top-1 surface form per cell; if it's ≥80% of outputs and
  also the gold for some rows, flag those as `correct_via_constant`).
- Doesn't require re-running inference.

Keeps the summary schema stable; lets the flag evolve as more
constant-output cases are found.

---

## G. Conclusion-flip cells revise previously-reported numbers

**Problem.** Four (condition, benchmark) SIB-200 cells flip win→loss
against baseline once the extractor is corrected (§8.3 of the writeup):

- `cond-2-es-5k` SIB-200
- `cond-2-es-20k` SIB-200
- `cond-2-zh-20k` SIB-200
- `cond-3-zh-5k` SIB-200

And `cond-2-ur-5k`'s gain deflates 2.4×.

**Fix.** Two parts:

1. **Audit.** `git grep` the research repo, search Linear comments,
   review slide decks for citations of original-extractor SIB-200 numbers.
   Update each.
2. **Dataset card note.** "If you previously cited Phase-3 SIB-200 numbers
   from the original `_summary_*.json` files, please re-read against
   `_summary_reparsed_*.json` — four cells flip and one's gain deflates
   2.4×. See [`conclusion_flips.tsv`][flips]."

A one-line `WHEN_REPORTED_NUMBERS_CHANGE.md` at `analysis/` root would be
nice for the paper-prep workflow.

---

## H. 0.0001 parse-fail floor on X-CSQA / Belebele

**Problem.** Every X-CSQA and Belebele cell has a parse-fail rate of
~0.0001 — a single row's worth, given test-set sizes ~10K. Probably a
genuine unparseable output (model emitted whitespace), but worth
confirming it's not a tokenizer/loader artifact.

**Check.**

```bash
python inspect_failures.py phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json \
    --benchmark csqa --outcome parse_fail --samples 5
```

If it's `<bos>`-only / empty-string / whitespace, expected. If structural,
escalate.

---

## I. Cond-5-ur-5k +0.515 acc lift — eyeball sample

**Problem.** §3 of the writeup shows `cond-5-ur-5k` template-2 SIB-200
`instr=ur` going from 0.049 → 0.564 — a 51-point swing. This is the kind
of headline number reviewers will recheck.

**Check.**

```bash
python inspect_failures.py \
    phase3/conditions/cond-5-ur-5k/seed42/cond-5-ur-5k_seed42_results_template2.json \
    --cell template2_sib200_data=ur_instr=ur --samples 20 --outcome correct
```

If the recovered "correct" rows show native-Urdu answers (سائنس/ٹیکنالوجی-shape)
being matched, recovery is genuine. If they're English answers, something's
off.

---

## Suggested execution order

1. **Add `{key}_count` + `{key}_correct` to PR #54** (one-commit amendment
   to `evaluate.ipynb` cell 3, ~2 lines). → satisfies D.
2. **Merge PR #54.** (PR #53 is already on main as commit 4961cf4.)
3. **Write `clean_stray_files.py`**, run against existing HF dataset,
   audit all conditions. → satisfies A.1.
4. **Add filename-seed guard to `reparse_results.py`**. → satisfies A.2.
5. **Flip upload-script default to `--skip-existing`** + add `--overwrite`
   / `--fail-on-existing`. → satisfies C.
6. **Run full-suite reparse + upload** with the new defaults.
7. **Dataset card refresh** — banners for E + G, "re-read against reparsed
   siblings" callout.
8. **Item H + I verification samples** + the `correct_via_constant_pct`
   artifact (F).

[upload]: ../evaluation/scripts/upload_reparsed_summaries.py
[eval]: ../evaluation/scripts/evaluate.ipynb
[ledger]: reparse-decision-ledger.md
[writeup]: phase3-reparse-evaluation.md
[flips]: reparse-tables/conclusion_flips.tsv

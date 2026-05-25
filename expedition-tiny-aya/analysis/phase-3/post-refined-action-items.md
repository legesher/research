# Refined-extractor action items — Phase-3

**Originally captured 2026-05-23. Status updated 2026-05-24** after PR #54
landed + HF PR #34 published the full-coverage refined dataset.

Items surfaced during a deep critical pass on the extractor extension +
full-suite re-scoring against the refined extractor. The refined extractor
covers **WHAT the model said** much better than the inference-time
extractor did. The items below are about making the next-round
re-scoring reproducible, auditable, and paper-grade.

**Status (updated 2026-05-25 post-PR-#58 / -#59 merges):**

- ✅ **A.1 done** — HF PR #33 deleted stray seed-123 files from `condition-2-es-5k/seed42/`.
- ⏳ **A.2 pending** — filename-seed-vs-parent-dir guard in `reparse_results.py`. (Not paper-blocking; the upload pipeline already skips stray files at upload time via `_classify_session_files` in `upload_reparsed_summaries.py`.)
- ✅ **B done** — obsoleted by PR #54's redirect; extractors are inline in `reparse_results.py`, no AST loader to be confused by a stale `run_eval_single.py`.
- ✅ **C done** — `upload_reparsed_summaries.py` flipped to skip-existing default in PR #54.
- ✅ **D done** — `{key}_count` and `{key}_correct` written to refined summaries in PR #54. (Per PR #54's redirect, these land in `build_reparsed_summary` rather than `evaluate.ipynb` cell 3 — the notebook stays frozen for Phase-3 reproducibility.)
- ✅ **E done** — cond-5 banner content authored on `legesher/language-decoded-experiments` (HF discussion PR #39) + `legesher/language-decoded-lora` (HF discussion PR #9). Banner prose final; the HF discussion PRs are open at the user's discretion to merge — not paper-blocking.
- ✅ **F done** — `build_correct_via_constant.py` in `evaluation/scripts/` + `correct-via-constant-findings.md` (PR #58); `correct-via-constant-rates.tsv` on HF at `phase3/analysis/refined-tables/` (HF PR #41 + #42 seed-format fix). **Finding REVERSED the original framing**: cond-2-ur-5k is the *least* constant-output condition by raw-output share (mean ≈ 0.30); cond-5-zh-5k and cond-5-es-5k are the most (~0.50+). The paper-prose caveat should name cond-5-{zh,es}, not cond-2-ur-5k.
- ✅ **G done** — `WHEN_REPORTED_NUMBERS_CHANGE.md` at top-level `analysis/` (PR #58) reconciles original vs refined extractor numbers + authoritative-source table for paper-prep. Companion conclusion-flip catalogue at `analysis/refined-tables/conclusion_flips.tsv` on HF (48 flips, regenerated against HF main post HF PR #34).
- ✅ **H done** — parse-fail floor verification log in `refined-verification-spot-checks.md` (PR #58). X-CSQA / Belebele parse-fails are genuine model outputs (Urdu phrases, empty strings); not tokenizer artifacts.
- ✅ **I done** — cond-5-ur-5k template2 SIB-200 `instr=ur` spot-check in `refined-verification-spot-checks.md` (PR #58). Recovery is genuine native-Urdu matches (`سائنس`, `سیاست`, `سفر`, `کھیل`); strict extractor was refusing valid Urdu answers. Also surfaced the cond-5 constant-output observation that drove F's reversal.

**New pending (separate from E–I, tracked for paper-prep follow-up):**

- ⏳ **Aggregation-bug-class audit** — see [`aggregation-bug-audit.md`](aggregation-bug-audit.md). Five more sites where the seed-vs-cell aggregation bug recurs (`framework_parse_failure_recovery`, `framework_benchmark_breakdown`, `framework_cross_language_impact`, `build_comparison` rollups, `build_vs_baseline` rollups). Not paper-blocking; deferred until a draft cites the affected columns.
- ⏳ **Fig 1 `is_flip` count bug + caption** — being fixed in PR #60 (`fix/fig01-flip-count`). `_viz_common.py` `COLOR_NEGATIVE` and `COLOR_FLIP_W2L` shared the same vermillion hex; counting via hex set conflated stable-negative lines with win→loss flips. Script-printed counts were 14 SIB-200 / 4 XNLI; true counts are 6 / 2 at (cond × instr) grain.

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
- Cross-link to [`refined-decision-ledger.md`][ledger] and
  [`phase3-refined-evaluation.md`][writeup].

---

## F. Cond-2-ur-5k constant-output rate

**Problem.** Per the decision ledger, the Urdu-tuned cond-2 model emits a
near-constant `سائنس/ٹیکنالوجی` (science/technology) regardless of passage.
Some of its "correct" rows are just that constant landing on gold by
chance — meaning the cell accuracy slightly overstates the model even
under the refined extractor. The ledger flagged a planned per-row
`correct_ambiguous` flag, never landed.

**Fix.** Cheap as a separate analysis artifact rather than baking into the
summary writer:

- `analysis/correct-via-constant-rates.tsv` next to the refined-extractor pass tables.
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
6. **Run full-suite re-scoring + upload** against the refined extractor with the new defaults.
7. **Dataset card refresh** — banners for E + G, "re-read against refined
   siblings" callout.
8. **Item H + I verification samples** + the `correct_via_constant_pct`
   artifact (F).

[upload]: ../evaluation/scripts/upload_reparsed_summaries.py
[eval]: ../evaluation/scripts/evaluate.ipynb
[ledger]: refined-decision-ledger.md
[writeup]: phase3-refined-evaluation.md
[flips]: refined-tables/conclusion_flips.tsv

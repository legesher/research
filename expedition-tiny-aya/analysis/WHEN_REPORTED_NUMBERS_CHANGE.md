# When reported numbers change — Phase 3 paper-prep reconciliation

If a Phase 3 SIB-200 (or X-CSQA / XNLI / Belebele / MGSM) number disagrees across two sources, read this first.

## The two extractors

Phase 3 was scored twice:

1. **Inference-time extractor** — frozen inside [`evaluation/scripts/evaluate.ipynb`](../evaluation/scripts/evaluate.ipynb) cell 3, run during each evaluation pass. Strict; doesn't accept native-script answers (e.g., it refused `سائنس` / `科学` / `ciencia` when gold was `science`).
2. **Refined post-hoc extractor** — [`evaluation/scripts/reparse_results.py`](../evaluation/scripts/reparse_results.py), sha256-pinned (`reparse_metadata.extractor_provenance.content_sha256` is embedded in every reparsed summary). Adds native-label / multi-term-hedge / CJK-glued / native-prose tiers; re-scores existing `_results_*.json` files.

Both extractors run over the same model outputs. They differ only in **what counts as a correct answer**. The notebook stays frozen for Phase 3 reproducibility; the refined extractor is the canonical scorer for paper claims. See [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md) §2–§3 for the recipe and [`phase-3/refined-decision-ledger.md`](phase-3/refined-decision-ledger.md) for what each tier accepts.

## What flipped — sign changes between extractors

Source: [`phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md) §8.3. Four (condition × benchmark) cells flip win→loss against baseline once the extractor is corrected:

| Condition         | Benchmark | Δ (original) | Δ (refined) | Verdict change             |
| ----------------- | --------- | ------------ | ----------- | -------------------------- |
| cond-2-es-5k      | SIB-200   | small win    | small loss  | sign flips                 |
| cond-2-es-20k     | SIB-200   | small win    | small loss  | sign flips                 |
| cond-2-zh-20k     | SIB-200   | small win    | small loss  | sign flips                 |
| cond-3-zh-5k      | SIB-200   | small win    | small loss  | sign flips                 |

One cell flips the other direction (cond-5-zh-5k XNLI), but the magnitude is ±0.01 — likely noise.

The canonical full flip catalogue is [`conclusion_flips.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/conclusion_flips.tsv) on Hugging Face: **48 cells flip** out of 1,536 condition-vs-baseline comparisons (3.1%); 34 of those are SIB-200; all flipped cells have `instr ≠ en`.

## What didn't flip but moved

The most consequential non-flip is **cond-2-ur-5k SIB-200**: +0.205 under the original extractor → **+0.047 under the refined extractor — a 4.4× deflation** that stays positive. At the (`instr=ur`) cell level the same condition shows +0.108 → +0.023 (~4×). The headline "Urdu keyword-swap helps SIB-200" survives in sign but loses most of its magnitude. If a paper figure quotes "+0.2" for cond-2-ur-5k SIB-200, that number is wrong post-refinement.

§8.4 of the writeup explains the mechanism: PR #49's `extract_sib200_category` over-credited Rule A (`science/<anything>` → science) and the `سیاست/تکنالوجی → science/technology` mapping. Fine-tuned models hit those buggy patterns more than the baseline, so removing them shrinks (or flips) the cond-vs-baseline delta. The effect is concentrated on SIB-200; XNLI deflations are smaller and never flip.

## Authoritative source — which number wins where

| Artefact                                                                                                                                                | What it says                                            | Canonical?                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| `_summary_reparsed_*.json` on HF (`phase3/conditions/*/seed*/`)                                                                                          | Refined-extractor per-template-seed accuracy + count    | **Yes** — cite for cell-level paper claims              |
| `phase3/analysis/refined-tables/cells.tsv` on HF                                                                                                         | Refined-extractor cross-session pivot                   | **Yes** — cite for paper tables                         |
| `phase3/analysis/refined-tables/conclusion_flips.tsv` on HF                                                                                              | The 48-cell catalogue                                   | **Yes** — cite when discussing flipped cells            |
| `_summary_*.json` on HF (no `reparsed`)                                                                                                                  | Inference-time extractor accuracy                       | No — retained for provenance only                       |
| Phase 3 writeup ([`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md))                                                          | Both readings side-by-side with narrative               | Authoritative narrative; numbers mirror HF tables       |
| Anything in this repo dated before 2026-05-19 that quotes Phase 3 SIB-200                                                                                | Probably original-extractor                             | Re-check against `cells.tsv` / `conclusion_flips.tsv`   |
| External (slides, Linear, drafts) quoting Phase 3 SIB-200 from `_summary_*.json`                                                                         | Original-extractor numbers — likely wrong post-refinement | Update against refined siblings                        |

## Rule for paper-prep

For any Phase 3 cell-level claim, cite `_summary_reparsed_*.json` (or `cells.tsv`) — not `_summary_*.json`. The inference-time numbers are retained on HF for reproducibility of the notebook scoring, **not** for citation.

If a number you have in a draft was sourced from a `_summary_*.json` file, find the matching `_summary_reparsed_*.json` (same condition / seed / template) and replace it. If the cell appears in `conclusion_flips.tsv`, flag it in the draft as a known flip and re-verify the surrounding claim still holds.

## Pointers

- Refined extractor implementation: [`evaluation/scripts/reparse_results.py`](../evaluation/scripts/reparse_results.py) (sha256-pinned via `_extractor_provenance`)
- Inference-time extractor (frozen): [`evaluation/scripts/evaluate.ipynb`](../evaluation/scripts/evaluate.ipynb) cell 3
- Phase 3 writeup: [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md)
- Decision ledger (what each tier accepts/rejects): [`phase-3/refined-decision-ledger.md`](phase-3/refined-decision-ledger.md)
- Methodology narrative: [`phase-3/sib200-parser-methodology.md`](phase-3/sib200-parser-methodology.md)
- Conclusion flips (HF): [`conclusion_flips.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/conclusion_flips.tsv)
- Cross-session cells pivot (HF): [`cells.tsv`](https://huggingface.co/datasets/legesher/language-decoded-experiments/resolve/main/phase3/analysis/refined-tables/cells.tsv)
- HF dataset card banner: [`legesher/language-decoded-experiments`](https://huggingface.co/datasets/legesher/language-decoded-experiments)

_Last updated: 2026-05-25._

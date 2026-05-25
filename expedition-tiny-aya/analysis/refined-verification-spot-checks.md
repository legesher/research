# Refined-extractor spot-checks — parse-fail floor + cond-5-ur-5k lift

Verification samples for action items H + I in [`phase-3/post-refined-action-items.md`](phase-3/post-refined-action-items.md). Run 2026-05-25 against HF main (post HF PR #34 full-coverage refined dataset).

All commands rerun from the research repo:

```bash
cd expedition-tiny-aya/evaluation/scripts
python3 inspect_failures.py <hf-path-or-local>.json --cell <cell> --outcome <outcome> --samples N
```

## H — X-CSQA / Belebele parse-fail floor

**Question.** Every X-CSQA and Belebele cell shows parse-fail rate ~0.0001 in the refined-evaluation writeup. Confirm the floor is genuine unparseable model output, not a tokenizer or loader artifact.

**Findings — baseline template1 X-CSQA** (`baseline_seednone_results_template1.json --benchmark csqa --outcome parse_fail`):

| Cell                                     | parse_fail | Sample raw_outputs                                                                                     |
| ---------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| `template1_csqa_data=ur_instr=ur`        | 1 / 1000   | `'سیبیس'` (Urdu transliteration, not a valid letter answer)                                            |
| `template1_csqa_data=zh_instr=es`        | 2 / 1000   | 2× `''` (empty string)                                                                                  |
| all other csqa cells                     | 0 / 1000   | —                                                                                                       |

**Findings — baseline Belebele** (template1 + template2):

| Cell (template1 / template2)             | parse_fail | Sample raw_outputs                                                                                     |
| ---------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| Belebele template1 (one cell, n=900)     | 4 / 900    | `'ایلکس اوویچیکن'` (Alex Ovechkin), `'جغرافیہ'` (geography), `'بیلجیئم کا انقلاب (1830'` (Belgian revolution) |
| Belebele template2 (one cell, n=900)     | 3 / 900    | `'ایلکس اوویچیکن'`, `'جغرافیہ'`, `'آپریشن سی لاین'` (Operation Sea Lion)                                |

**Findings — cond-2-ur-5k seed42 template1 X-CSQA**: 0 parse-fails across all cells.

**Verdict — confirmed genuine.** Parse-fails are model outputs that aren't letter answers: Urdu proper nouns/topic words, empty strings, transliterations. Not a tokenizer or loader artifact. The floor is the model occasionally answering in a non-letter form on Urdu cells, plus a handful of empty-string emissions. Concentrated on the baseline; fine-tuned conditions show essentially 0 parse-fails on letter-answer benchmarks.

The 0.0001 *mean* parse-fail rate quoted in [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md) §2 is an across-cell average — most cells are 0, a few baseline Urdu cells are 1–4/900–1000.

## I — cond-5-ur-5k template2 SIB-200 instr=ur sanity sample

**Question.** Action items I originally flagged a 0.049 → 0.564 acc swing on this cell — a 51-point recovery under the refined extractor. Spot-check that the recovered "correct" rows are matched on genuine native-Urdu surface forms, not English answers.

**Current numbers** (post HF PR #34 full-coverage refined dataset):

| | accuracy | parse_fail | match_via=single (correct) |
| - | - | - | - |
| `template2_sib200_data=ur_instr=ur` (cond-5-ur-5k seed42) | 0.549 | 0.186 | 112 / 204 |

The original action-items 0.564 number lines up with the refined-extractor cell accuracy (0.549 here for seed42; mean across seeds may be ~0.55–0.59). The 0.049 number was the strict-extractor result for the same cell — the 50-point swing is confirmed real.

**Spot-check — correct/single buckets** (top 15 surface forms, 112 rows):

| count | surface form (Urdu / mixed)                                        |
| ----- | ------------------------------------------------------------------ |
| 22    | `سائنس/تکنالوجی` (science/technology)                              |
| 19    | `سیاست ⏎ Explanation: The provi…` (politics, prose tail in English) |
| 18    | `سفر ⏎ Explanation: The provide…` (travel)                          |
| 11    | `کھیل ⏎ Explanation: The provid…` (sports)                          |
| 8     | `سیاست ⏎ Explanation: The text …`                                  |
| 5     | `علم/تکنالوجی ⏎ توض…`                                              |
| 4     | `علم/تکنالوجی ⏎ Explanation:`                                       |
| 3     | `کھیل/سپورٹس ⏎ Explanation`                                         |
| 2     | `سیاحت ⏎ Explanation: The provi…` (tourism)                         |
| 2     | `صحت ⏎ Explanation: The provide…` (health)                          |
| 1     | `science/تکنالوجی ⏎ توض…` (English-Urdu mix)                        |
| 1     | `ٹرانسپورٹیشن ⏎ Explanation:`                                       |
| ...   | (and others, all native Urdu category words)                       |

**Verdict — recovery is genuine.** Every correct row is the model emitting an Urdu category name (`سائنس`, `سیاست`, `سفر`, `کھیل`, `صحت`, `سیاحت`, etc.) matched by the refined extractor's native-label tier. The strict inference-time extractor refused these answers because it required English labels. There is no English-answer fakery in the correct/single bucket.

The 19/18/11/8-count clusters all show **multiline outputs** — the model emits the Urdu category word, then `⏎ Explanation: …` in English prose, which the strict extractor's first-line filtering would also have rejected.

**Constant-output observation (relevant to action item F).** The model emits `سائنس/تکنالوجی` (science/technology) 49 times across this 204-row cell (24%); 22 of those land on gold (correct), 27 land off-gold (wrong_label). The cond-2-ur-5k constant-output pattern flagged in F is **also present in cond-5-ur-5k**: at least 22/112 (~20%) of the "correct" rows here are explained by the model defaulting to `سائنس/تکنالوجی` and happening to be right. Action item F should treat cond-5-ur-5k as also affected, not just cond-2-ur-5k.

## Pointers

- [`phase-3/phase3-refined-evaluation.md`](phase-3/phase3-refined-evaluation.md) — the writeup these spot-checks support
- [`phase-3/post-refined-action-items.md`](phase-3/post-refined-action-items.md) — items H and I
- [`evaluation/scripts/inspect_failures.py`](../evaluation/scripts/inspect_failures.py) — the tool used

_Run 2026-05-25 against `phase3/conditions/.../seed42/` on HF main._

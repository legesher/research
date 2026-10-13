"""Unit tests for compare_stack_drift.py.

Stdlib unittest; no network. Rows carry Urdu and Devanagari inputs. These
pin: rows align on their inputs (not on position) and consume duplicates in
order, a row with no reference counterpart is reported rather than
silently dropped, the raw-output match rate and both accuracies are
computed on the aligned rows only, and the verdict says "no drift" only
when every aligned raw output is byte-identical.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_stack_drift as drift  # noqa: E402

CELL = "template1_sib200_data=ur_instr=ur"


def _row(text, gold, raw, pred=None, correct=None):
    r = {"text": text, "template": "template1", "instruction_lang": "ur", "gold": gold, "raw_output": raw}
    r["pred"] = pred if pred is not None else raw
    r["correct"] = correct if correct is not None else (r["pred"] == gold)
    return r


class AlignTests(unittest.TestCase):
    def test_rows_align_on_inputs_not_position(self):
        ref = [_row("یہ سائنس ہے۔", "science/technology", "science/technology"),
               _row("यह खेल है।", "sports", "sports"),
               _row("یہ سفر ہے۔", "travel", "travel")]
        smoke = [ref[2], ref[0]]  # reordered subset
        pairs, unmatched = drift.align(smoke, ref)
        self.assertEqual(unmatched, 0)
        self.assertEqual([r["text"] for _, r in pairs], ["یہ سفر ہے۔", "یہ سائنس ہے۔"])

    def test_changed_gold_or_text_does_not_align(self):
        ref = [_row("یہ سائنس ہے۔", "science/technology", "science/technology")]
        smoke = [_row("یہ سائنس ہے۔", "politics", "science/technology")]
        pairs, unmatched = drift.align(smoke, ref)
        self.assertEqual((len(pairs), unmatched), (0, 1))

    def test_duplicate_inputs_are_consumed_in_order(self):
        ref = [_row("same", "sports", "a"), _row("same", "sports", "b")]
        smoke = [_row("same", "sports", "x"), _row("same", "sports", "y"), _row("same", "sports", "z")]
        pairs, unmatched = drift.align(smoke, ref)
        self.assertEqual([r["raw_output"] for _, r in pairs], ["a", "b"])
        self.assertEqual(unmatched, 1)


class CompareTests(unittest.TestCase):
    def test_rates_are_computed_on_aligned_rows_with_the_refined_extractor(self):
        extractor = drift._load_extractors()["sib200"]
        ref = [
            _row("یہ سائنس ہے۔", "science/technology", "science/technology"),
            _row("यह खेल है।", "sports", "sports"),
            _row("یہ سفر ہے۔", "travel", "health"),
        ]
        smoke = [
            # same raw output
            _row("یہ سائنس ہے۔", "science/technology", "science/technology"),
            # native-script answer: wrong for the inference-time extractor, right for the refined one
            _row("यह खेल है।", "sports", "کھیل", pred=None, correct=False),
            # changed raw output, still wrong
            _row("یہ سفر ہے۔", "travel", "politics"),
        ]
        res = drift.compare_cell(CELL, smoke, ref, extractor)
        self.assertEqual(res["n"], 3)
        self.assertAlmostEqual(res["raw_match_pct"], 100 / 3)
        self.assertAlmostEqual(res["orig_acc_ref"], 200 / 3)
        self.assertAlmostEqual(res["orig_acc_smoke"], 100 / 3)
        self.assertAlmostEqual(res["refined_acc_ref"], 200 / 3)
        self.assertAlmostEqual(res["refined_acc_smoke"], 200 / 3)

    def test_missing_cell_and_unmatched_rows_are_problems(self):
        extractors = drift._load_extractors()
        smoke = {"summary": {}, CELL: [_row("یہ سائنس ہے۔", "science/technology", "x")],
                 "template1_sib200_data=zh_instr=ur": [_row("这是科学。", "science/technology", "x")]}
        ref = {CELL: [_row("یہ سیاست ہے۔", "politics", "politics")]}
        results, problems = drift.compare_files(smoke, ref, extractors)
        self.assertEqual(len(results), 1)
        self.assertIn(f"{CELL}: 1 smoke rows have no reference row with the same inputs", problems)
        self.assertIn(f"{CELL}: no rows aligned", problems)
        self.assertIn("template1_sib200_data=zh_instr=ur: not in the reference session", problems)


class VerdictTests(unittest.TestCase):
    def _results(self, raw_match):
        return [{"cell": CELL, "n": 50, "unmatched": 0, "raw_match_pct": raw_match, "pred_match_pct": 100.0,
                 "orig_acc_smoke": 60.0, "orig_acc_ref": 60.0, "refined_acc_smoke": 62.0, "refined_acc_ref": 60.0}]

    def test_no_drift_only_when_raw_outputs_all_match(self):
        self.assertTrue(drift.verdict(drift.summarize(self._results(100.0))).startswith("no drift"))
        v = drift.verdict(drift.summarize(self._results(98.0)))
        self.assertTrue(v.startswith("drift:"))
        self.assertIn("1 of 1 cells", v)
        self.assertIn("2.0 pp", v)

    def test_markdown_carries_verdict_table_and_problems(self):
        md = drift.render_markdown({"template1": (self._results(98.0), ["x: not in the reference session"])}, "condition-2-es-20k/seed42")
        self.assertIn("# Stack drift check: condition-2-es-20k/seed42", md)
        self.assertIn("Verdict: drift:", md)
        self.assertIn(f"| {CELL} | 50 | 0 | 98.0 | 100.0 | 60.0 | 60.0 | 62.0 | 60.0 |", md)
        self.assertIn("- x: not in the reference session", md)


if __name__ == "__main__":
    unittest.main()

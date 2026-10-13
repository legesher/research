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
import reparse_results  # noqa: E402

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

    def test_every_benchmark_aligns_and_scores_with_its_own_input_fields(self):
        """Parametrized from reparse_results.EXTRACTOR_NAMES: each benchmark's
        rows carry different input fields and the extractor must be the
        benchmark's own."""
        extractors = drift._load_extractors()
        shapes = {
            "sib200": ({"text": "यह विज्ञान है।"}, "science/technology", "science/technology", "deportes"),  # a native alias of sports: wrong under both extractors
            "xnli": ({"premise": "وہ گھر گیا۔", "hypothesis": "وہ باہر ہے۔"}, "contradiction", "contradiction", "neutral"),
            "csqa": ({"stem": "कहाँ?", "A": "घर", "B": "बाज़ार"}, "B", "B", "A"),
            "belebele": ({"passage": "ایک اقتباس۔", "question": "سوال؟"}, "C", "C", "D"),
        }
        self.assertEqual(set(shapes), set(reparse_results.EXTRACTOR_NAMES))
        for bench in reparse_results.EXTRACTOR_NAMES:
            with self.subTest(bench=bench):
                inputs, gold, right, wrong = shapes[bench]
                cell = f"template1_{bench}_data=ur_instr=ur"
                base = {**inputs, "template": "template1", "instruction_lang": "ur", "gold": gold}
                ref = [{**base, "raw_output": right, "pred": right, "correct": True}]
                smoke = [{**base, "raw_output": wrong, "pred": wrong, "correct": False}]
                res = drift.compare_cell(cell, smoke, ref, extractors[bench])
                self.assertEqual((res["n"], res["unmatched"]), (1, 0))
                self.assertEqual(res["raw_match_rows"], 0)
                self.assertEqual((res["refined_acc_ref"], res["refined_acc_smoke"]), (100.0, 0.0))

    def test_field_schema_difference_is_named_when_rows_do_not_align(self):
        extractors = drift._load_extractors()
        smoke = {CELL: [dict(_row("یہ سائنس ہے۔", "science/technology", "x"), row_id=7)]}
        ref = {CELL: [_row("یہ سائنس ہے۔", "science/technology", "x")]}
        _, problems = drift.compare_files(smoke, ref, extractors)
        self.assertIn(f"{CELL}: row fields differ; only in smoke: ['row_id']; only in reference: none", problems)

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
    def _results(self, n=50, raw_match_rows=None, orig_delta=0.0, refined_delta=2.0):
        raw_match_rows = n if raw_match_rows is None else raw_match_rows
        return [{"cell": CELL, "n": n, "unmatched": 0, "raw_match_rows": raw_match_rows,
                 "raw_match_pct": 100.0 * raw_match_rows / n, "pred_match_pct": 100.0,
                 "orig_acc_smoke": 60.0 + orig_delta, "orig_acc_ref": 60.0,
                 "refined_acc_smoke": 60.0 + refined_delta, "refined_acc_ref": 60.0}]

    def test_no_drift_only_when_raw_outputs_all_match(self):
        self.assertTrue(drift.verdict(drift.summarize(self._results(refined_delta=0.0))).startswith("no drift"))
        v = drift.verdict(drift.summarize(self._results(raw_match_rows=49)))
        self.assertTrue(v.startswith("drift: 1 of 50 aligned raw outputs differ"), v)
        self.assertIn("1 of 1 aligned cells", v)
        self.assertIn("2.00 pp", v)

    def test_one_differing_row_in_ten_thousand_is_still_drift(self):
        v = drift.verdict(drift.summarize(self._results(n=10000, raw_match_rows=9999)))
        self.assertTrue(v.startswith("drift: 1 of 10000"), v)

    def test_identical_raw_outputs_with_a_moved_inference_time_extractor_is_not_no_drift(self):
        v = drift.verdict(drift.summarize(self._results(orig_delta=4.0, refined_delta=0.0)))
        self.assertTrue(v.startswith("raw outputs identical but the inference-time extractor moved"), v)
        self.assertIn("4.00 pp", v)

    def test_summary_weights_by_rows_and_counts_only_aligned_cells(self):
        big = self._results(n=50)[0]
        empty = {**self._results(n=50)[0], "cell": "other", "n": 0, "raw_match_rows": 0, "raw_match_pct": float("nan")}
        none_match = {**self._results(n=10, raw_match_rows=0)[0], "cell": "third"}
        s = drift.summarize([big, empty, none_match])
        self.assertEqual((s["rows"], s["raw_match_rows"], s["cells"], s["cells_total"]), (60, 50, 2, 3))
        self.assertAlmostEqual(s["raw_match_pct"], 100.0 * 50 / 60)

    def test_markdown_carries_verdict_sources_table_and_problems(self):
        md = drift.render_markdown(
            {"template1": (self._results(raw_match_rows=49), ["x: not in the reference session"])},
            "condition-2-es-20k/seed42",
            {"template1": "condition-2-es-20k_seed42_smoke50_results_template1.json"},
        )
        self.assertIn("# Stack drift check: condition-2-es-20k/seed42", md)
        self.assertIn("Verdict: drift: 1 of 50", md)
        self.assertIn("Smoke files: template1: condition-2-es-20k_seed42_smoke50_results_template1.json", md)
        self.assertIn(f"| {CELL} | 50 | 0 | 98.0 | 100.0 | 60.0 | 60.0 | 62.0 | 60.0 |", md)
        self.assertIn("- x: not in the reference session", md)


class NamingTests(unittest.TestCase):
    def test_session_regex_reads_smoke_and_baseline_names(self):
        m = drift.SESSION_RE.match("condition-2-es-20k_seed42_smoke50_results_template1.json")
        self.assertEqual((m["cond"], m["seed"], m["template"]), ("condition-2-es-20k", "42", "template1"))
        m = drift.SESSION_RE.match("baseline_seednone_results_template2.json")
        self.assertEqual((m["cond"], m["seed"], m["template"]), ("baseline", "none", "template2"))
        self.assertIsNone(drift.SESSION_RE.match("condition-2-es-20k_seed42_summary_template1.json"))

    def test_reference_regex_and_path(self):
        m = drift.REFERENCE_RE.match("condition-2-es-20k/seed42")
        self.assertEqual((m["cond"], m["seed"]), ("condition-2-es-20k", "42"))
        self.assertIsNone(drift.REFERENCE_RE.match("condition-2-es-20k"))
        self.assertEqual(
            drift._reference_path("condition-2-es-20k", "42", "template2"),
            "phase3/conditions/condition-2-es-20k/seed42/condition-2-es-20k_seed42_results_template2.json",
        )


if __name__ == "__main__":
    unittest.main()

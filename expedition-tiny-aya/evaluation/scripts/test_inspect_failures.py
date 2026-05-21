"""Unit tests for inspect_failures.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_inspect_failures.py -v

Classifier tests need the extractor constants, so they're skipped when
`run_eval_single.py` isn't present next to this file. The `benchmark_from_key`
and reporting-helper tests always run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_failures  # noqa: E402

HAS_SOURCE = inspect_failures._find_extractor_source() is not None


class TestBenchmarkFromKey(unittest.TestCase):
    """Pure string parsing — no extractor source needed."""

    def test_known(self):
        self.assertEqual(
            inspect_failures.benchmark_from_key("template1_sib200_data=ur_instr=ur"),
            "sib200",
        )
        self.assertEqual(
            inspect_failures.benchmark_from_key("template2_xnli_data=en_instr=zh"),
            "xnli",
        )
        self.assertEqual(
            inspect_failures.benchmark_from_key("template1_csqa_data=es_instr=en"),
            "csqa",
        )
        self.assertEqual(
            inspect_failures.benchmark_from_key("template2_belebele_data=zh_instr=ur"),
            "belebele",
        )

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            inspect_failures.benchmark_from_key("template1_mystery_data=en_instr=en")


class TestOnelineHelper(unittest.TestCase):
    def test_collapses_multiline(self):
        self.assertEqual(
            inspect_failures._oneline("science/technology\n解释: …\n"),
            "science/technology ⏎ 解释: …",
        )

    def test_single_line_unchanged(self):
        self.assertEqual(inspect_failures._oneline("  travel  "), "travel")


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifySib200(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def _via(self, raw):
        return inspect_failures.classify_sib200(raw, self.ns)

    def test_exact(self):
        self.assertEqual(
            self._via("science/technology"), ("science/technology", "exact")
        )
        self.assertEqual(self._via("travel"), ("travel", "exact"))

    def test_normalized_case(self):
        # Capitalised — needs lowercasing
        self.assertEqual(
            self._via("Science/Technology"), ("science/technology", "normalized")
        )

    def test_normalized_punctuation(self):
        # Quote-wrapped — needs punctuation stripping
        self.assertEqual(self._via('"travel"'), ("travel", "normalized"))
        self.assertEqual(self._via("travel."), ("travel", "normalized"))

    def test_substring(self):
        # Canonical token embedded in a longer first line
        pred, via = self._via("The answer is travel")
        self.assertEqual(pred, "travel")
        self.assertEqual(via, "substring")

    def test_rule_a_invented_subcategory(self):
        self.assertEqual(self._via("science/AI"), ("science/technology", "rule_a"))
        self.assertEqual(self._via("science/physics"), ("science/technology", "rule_a"))

    def test_rule_b_native_scripts(self):
        self.assertEqual(self._via("سائنس/ٹکنالوجی"), ("science/technology", "rule_b"))
        self.assertEqual(self._via("科学/技术"), ("science/technology", "rule_b"))
        self.assertEqual(
            self._via("ciencia/tecnología"), ("science/technology", "rule_b")
        )

    def test_rule_c_bare_subcategory(self):
        self.assertEqual(self._via("physics"), ("science/technology", "rule_c"))
        self.assertEqual(self._via("ai"), ("science/technology", "rule_c"))

    def test_alias(self):
        # "sport" → "sports" via SIB200_ALIASES
        self.assertEqual(self._via("sport"), ("sports", "alias"))

    def test_none(self):
        self.assertEqual(self._via("absolutely unparseable"), (None, "none"))

    def test_rule_a_does_not_mask_canonical(self):
        # "science/technology" technically enters the Rule A branch (it starts
        # with "science/"), but the classifier sub-classifies it back to
        # `exact` / `normalized` so a perfect canonical answer is never
        # mislabelled as a lenient rescue. Only genuine invented sub-categories
        # get `rule_a`.
        self.assertEqual(
            self._via("science/technology"), ("science/technology", "exact")
        )
        self.assertEqual(
            self._via("Science/Technology"), ("science/technology", "normalized")
        )
        self.assertEqual(self._via("science/AI"), ("science/technology", "rule_a"))

    def test_multiline_first_line_only(self):
        # The classifier reads only the first line, like the extractor.
        pred, _ = self._via("travel\nThis passage is about going places.")
        self.assertEqual(pred, "travel")


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyXnli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def _via(self, raw):
        return inspect_failures.classify_xnli(raw, self.ns)

    def test_exact_english(self):
        self.assertEqual(self._via("entailment"), ("entailment", "exact"))
        self.assertEqual(self._via("contradiction"), ("contradiction", "exact"))

    def test_english_substring(self):
        pred, via = self._via("The relationship is entailment")
        self.assertEqual(pred, "entailment")
        self.assertEqual(via, "english_substring")

    def test_native_exact_chinese(self):
        self.assertEqual(self._via("矛盾"), ("contradiction", "native_exact"))

    def test_native_substring(self):
        pred, via = self._via("答案是 矛盾 因为…")
        self.assertEqual(pred, "contradiction")
        self.assertEqual(via, "native_substring")

    def test_none(self):
        self.assertEqual(self._via("???"), (None, "none"))


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyChoice(unittest.TestCase):
    def test_bare_letter(self):
        self.assertEqual(
            inspect_failures.classify_choice("A", "ABCDE"), ("A", "bare_letter")
        )

    def test_letter_in_text(self):
        pred, via = inspect_failures.classify_choice("The answer is C.", "ABCDE")
        self.assertEqual(pred, "C")
        self.assertEqual(via, "letter_in_text")

    def test_answer_with_standalone_letter_is_letter_in_text(self):
        # "ANSWER: D" — the bare-letter regex `\bD\b` matches the standalone D
        # FIRST, before the `ANSWER:` branch is ever reached. This is faithful
        # to the live extractor's ordering. The `answer_prefix` stage is
        # near-vestigial as a result (see next test).
        pred, via = inspect_failures.classify_choice("ANSWER: D", "ABCD")
        self.assertEqual(pred, "D")
        self.assertEqual(via, "letter_in_text")

    def test_answer_prefix_only_when_letter_not_standalone(self):
        # The `ANSWER:` branch is reachable only when the choice letter is
        # glued to other word chars so `\b[ABCD]\b` fails first — e.g. the
        # contrived "ANSWERD". Real model outputs almost never hit this; the
        # bare-letter regex catches them. Worth knowing the branch is
        # effectively dead code in the live extractor.
        pred, via = inspect_failures.classify_choice("ANSWERD", "ABCD")
        self.assertEqual(pred, "D")
        self.assertEqual(via, "answer_prefix")

    def test_none(self):
        self.assertEqual(
            inspect_failures.classify_choice("hello", "ABCDE"), (None, "none")
        )

    def test_belebele_excludes_e(self):
        # Belebele is 4-way (ABCD) — a bare "E" should not match.
        self.assertEqual(inspect_failures.classify_choice("E", "ABCD"), (None, "none"))


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_correct(self):
        row = inspect_failures.classify_row(
            "sib200", "science/technology", "science/technology", self.ns
        )
        self.assertEqual(row["outcome"], "correct")
        self.assertFalse(row["multiline"])

    def test_wrong_label(self):
        row = inspect_failures.classify_row(
            "sib200", "travel", "science/technology", self.ns
        )
        self.assertEqual(row["outcome"], "wrong_label")
        self.assertEqual(row["pred"], "travel")

    def test_parse_fail(self):
        row = inspect_failures.classify_row(
            "sib200", "completely unparseable text", "science/technology", self.ns
        )
        self.assertEqual(row["outcome"], "parse_fail")
        self.assertEqual(row["match_via"], "none")

    def test_multiline_flag(self):
        row = inspect_failures.classify_row(
            "sib200", "travel\nextra explanation line", "travel", self.ns
        )
        self.assertTrue(row["multiline"])
        self.assertEqual(row["outcome"], "correct")

    def test_correct_via_native_rule(self):
        # The headline case: a correct answer that only matched via Rule B.
        row = inspect_failures.classify_row(
            "sib200", "سائنس/ٹکنالوجی", "science/technology", self.ns
        )
        self.assertEqual(row["outcome"], "correct")
        self.assertEqual(row["match_via"], "rule_b")


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_aggregation(self):
        rows = [
            {"raw_output": "science/technology", "gold": "science/technology"},
            {"raw_output": "سائنس/ٹکنالوجی", "gold": "science/technology"},
            {"raw_output": "travel", "gold": "science/technology"},
            {"raw_output": "garbage", "gold": "science/technology"},
        ]
        report = inspect_failures.classify_cell(
            "template1_sib200_data=ur_instr=ur", rows, self.ns
        )
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["benchmark"], "sib200")
        # 2 correct (exact + rule_b), 1 wrong_label, 1 parse_fail
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["parse_fail_rate"], 0.25)
        self.assertEqual(report["bucket_counts"][("correct", "exact")], 1)
        self.assertEqual(report["bucket_counts"][("correct", "rule_b")], 1)
        self.assertEqual(report["bucket_counts"][("wrong_label", "exact")], 1)
        self.assertEqual(report["bucket_counts"][("parse_fail", "none")], 1)


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestInstrumentedMatchesLive(unittest.TestCase):
    """The strongest guarantee: the instrumented classifier's prediction must
    equal the live extractor's prediction on a battery of inputs spanning
    every match stage and every language."""

    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_sib200_agreement(self):
        live = self.ns["extract_sib200_category"]
        cases = [
            "science/technology",
            "Science/Technology.",
            '"travel"',
            "The answer is travel",
            "science/AI",
            "سائنس/ٹکنالوجی",
            "科学/技术",
            "ciencia/tecnología",
            "physics",
            "sport",
            "garbage text",
            "travel\nexplanation",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                instrumented, _ = inspect_failures.classify_sib200(raw, self.ns)
                self.assertEqual(instrumented, live(raw))

    def test_xnli_agreement(self):
        live = self.ns["extract_xnli_label"]
        cases = [
            "entailment",
            "The relationship is contradiction",
            "矛盾",
            "答案是 矛盾",
            "لازمی",
            "neutro",
            "???",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                instrumented, _ = inspect_failures.classify_xnli(raw, self.ns)
                self.assertEqual(instrumented, live(raw))

    def test_choice_agreement(self):
        live = self.ns["extract_choice"]
        cases = ["A", "ANSWER: C", "The answer is B.", "hello"]
        for raw in cases:
            with self.subTest(raw=raw):
                instrumented, _ = inspect_failures.classify_choice(raw, "ABCDE")
                self.assertEqual(instrumented, live(raw, choices="ABCDE"))


if __name__ == "__main__":
    unittest.main()

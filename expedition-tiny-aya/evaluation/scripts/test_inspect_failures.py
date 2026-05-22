"""Unit tests for inspect_failures.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_inspect_failures.py -v

The classifiers call the live extractors (from run_eval_single.py) for the
prediction, so classifier-dependent tests are skipped when that file is not
present next to this one. Pure-helper tests always run.
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
        cases = {
            "template1_sib200_data=ur_instr=ur": "sib200",
            "template2_xnli_data=en_instr=zh": "xnli",
            "template1_csqa_data=es_instr=en": "csqa",
            "template2_belebele_data=zh_instr=ur": "belebele",
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(inspect_failures.benchmark_from_key(key), expected)

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
    """match_via vocabulary: single | multi_category | fallback | none."""

    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def c(self, raw):
        return inspect_failures.classify_sib200(raw, self.ns)

    def test_single_english(self):
        self.assertEqual(self.c("politics"), ("politics", "single"))
        self.assertEqual(self.c("travel"), ("travel", "single"))

    def test_single_canonical_compound_one_category(self):
        # "science/technology" splits to two pieces, both → science/technology
        self.assertEqual(self.c("science/technology"), ("science/technology", "single"))
        self.assertEqual(self.c("science/AI"), ("science/technology", "single"))

    def test_single_native(self):
        self.assertEqual(self.c("سیاست"), ("politics", "single"))
        self.assertEqual(self.c("旅行"), ("travel", "single"))
        self.assertEqual(self.c("viajes"), ("travel", "single"))

    def test_multi_category_hedge(self):
        # Cross-category compounds → parse-failure (pred None), via multi_category
        self.assertEqual(self.c("سیاست/تکنالوجی"), (None, "multi_category"))
        self.assertEqual(self.c("science/health"), (None, "multi_category"))
        self.assertEqual(self.c("کھیل/تکنالوجی"), (None, "multi_category"))

    def test_same_category_compound_resolves(self):
        # کھیل/سپورٹس — both halves mean sports → single
        self.assertEqual(self.c("کھیل/سپورٹس"), ("sports", "single"))

    def test_fallback(self):
        # Canonical English name embedded in a sentence — no resolvable pieces
        self.assertEqual(self.c("the answer is travel"), ("travel", "fallback"))

    def test_none(self):
        self.assertEqual(self.c("absolutely unparseable"), (None, "none"))


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyXnli(unittest.TestCase):
    """match_via: tier1_english | tier1_native | tier2_cjk_glued |
    tier3_paraphrase | none."""

    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def c(self, raw):
        return inspect_failures.classify_xnli(raw, self.ns)

    def test_tier1_english(self):
        self.assertEqual(self.c("entailment"), ("entailment", "tier1_english"))
        self.assertEqual(self.c("Contradiction."), ("contradiction", "tier1_english"))

    def test_tier1_native(self):
        self.assertEqual(self.c("矛盾"), ("contradiction", "tier1_native"))
        self.assertEqual(self.c("لازمی"), ("entailment", "tier1_native"))

    def test_tier2_cjk_glued(self):
        self.assertEqual(
            self.c("假设是entailment。"), ("entailment", "tier2_cjk_glued")
        )

    def test_tier2_negated_is_rejected(self):
        # "没有entailment" — the model negates the label → parse-failure
        self.assertEqual(self.c("假设和前提之间没有entailment或"), (None, "none"))

    def test_tier3_paraphrase(self):
        self.assertEqual(
            self.c("假设是前提的直接结果。"), ("entailment", "tier3_paraphrase")
        )
        self.assertEqual(
            self.c("假设是前提的否定。"), ("contradiction", "tier3_paraphrase")
        )
        self.assertEqual(
            self.c("假设和前提之间没有关系。"), ("neutral", "tier3_paraphrase")
        )

    def test_none(self):
        self.assertEqual(self.c("假设。"), (None, "none"))
        self.assertEqual(self.c("1. 2. 3."), (None, "none"))


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_bare_letter(self):
        self.assertEqual(
            inspect_failures.classify_choice("A", self.ns, "ABCDE"),
            ("A", "bare_letter"),
        )

    def test_letter_in_text(self):
        self.assertEqual(
            inspect_failures.classify_choice("The answer is C.", self.ns, "ABCDE"),
            ("C", "letter_in_text"),
        )

    def test_answer_prefix(self):
        # "ANSWERD" — no standalone letter, the ANSWER: branch fires
        self.assertEqual(
            inspect_failures.classify_choice("ANSWERD", self.ns, "ABCD"),
            ("D", "answer_prefix"),
        )

    def test_none(self):
        self.assertEqual(
            inspect_failures.classify_choice("hello", self.ns, "ABCDE"), (None, "none")
        )

    def test_belebele_excludes_e(self):
        self.assertEqual(
            inspect_failures.classify_choice("E", self.ns, "ABCD"), (None, "none")
        )


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

    def test_parse_fail_multi_category(self):
        # A cross-category hedge is a parse-failure, tagged multi_category
        row = inspect_failures.classify_row(
            "sib200", "سیاست/تکنالوجی", "politics", self.ns
        )
        self.assertEqual(row["outcome"], "parse_fail")
        self.assertEqual(row["match_via"], "multi_category")

    def test_multiline_flag(self):
        row = inspect_failures.classify_row(
            "sib200", "travel\nextra explanation line", "travel", self.ns
        )
        self.assertTrue(row["multiline"])
        self.assertEqual(row["outcome"], "correct")

    def test_correct_via_native_term(self):
        row = inspect_failures.classify_row("sib200", "سیاست", "politics", self.ns)
        self.assertEqual(row["outcome"], "correct")
        self.assertEqual(row["match_via"], "single")


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestClassifyCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_aggregation(self):
        rows = [
            {"raw_output": "science/technology", "gold": "science/technology"},
            {"raw_output": "سیاست", "gold": "politics"},
            {"raw_output": "travel", "gold": "science/technology"},
            {"raw_output": "سیاست/تکنالوجی", "gold": "politics"},
        ]
        report = inspect_failures.classify_cell(
            "template1_sib200_data=ur_instr=ur", rows, self.ns
        )
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["benchmark"], "sib200")
        # 2 correct (science/technology, سیاست), 1 wrong (travel), 1 parse_fail
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["parse_fail_rate"], 0.25)
        self.assertEqual(report["bucket_counts"][("correct", "single")], 2)
        self.assertEqual(report["bucket_counts"][("wrong_label", "single")], 1)
        self.assertEqual(report["bucket_counts"][("parse_fail", "multi_category")], 1)


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestAggregateSurfaceForms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def _fake_dataset(self):
        return {
            "summary": {},
            "parse_failure_rates": {},
            "template1_sib200_data=ur_instr=ur": [
                {"raw_output": "سیاست", "gold": "politics"},
                {"raw_output": "سیاست", "gold": "travel"},
                {"raw_output": "سیاست", "gold": "sports"},
                {"raw_output": "travel", "gold": "travel"},
            ],
        }

    def test_groups_identical_first_lines(self):
        rows = inspect_failures.aggregate_surface_forms([self._fake_dataset()], self.ns)
        by_form = {r["first_line"]: r for r in rows}
        self.assertEqual(by_form["سیاست"]["total"], 3)
        self.assertEqual(by_form["travel"]["total"], 1)

    def test_outcome_split_within_a_form(self):
        rows = inspect_failures.aggregate_surface_forms([self._fake_dataset()], self.ns)
        siyasat = next(r for r in rows if r["first_line"] == "سیاست")
        # سیاست now resolves to politics → correct once, wrong twice
        self.assertEqual(siyasat["pred"], "politics")
        self.assertEqual(siyasat["correct"], 1)
        self.assertEqual(siyasat["wrong_label"], 2)
        self.assertEqual(siyasat["parse_fail"], 0)

    def test_sorted_by_total_descending(self):
        rows = inspect_failures.aggregate_surface_forms([self._fake_dataset()], self.ns)
        totals = [r["total"] for r in rows]
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_pools_across_multiple_datasets(self):
        rows = inspect_failures.aggregate_surface_forms(
            [self._fake_dataset(), self._fake_dataset()], self.ns
        )
        siyasat = next(r for r in rows if r["first_line"] == "سیاست")
        self.assertEqual(siyasat["total"], 6)

    def test_benchmark_filter(self):
        rows = inspect_failures.aggregate_surface_forms(
            [self._fake_dataset()], self.ns, only_benchmarks={"xnli"}
        )
        self.assertEqual(rows, [])


@unittest.skipUnless(HAS_SOURCE, "run_eval_single.py not next to test file")
class TestInstrumentedMatchesLive(unittest.TestCase):
    """The strongest guarantee: the instrumented classifier's prediction must
    equal the live extractor's prediction on a battery of inputs covering
    every tier/branch and every supported language.

    inspect_failures.py only earns trust if `classify_*(raw, ...)[0]` agrees
    with `extract_*(raw, ...)` on every input. Disagreement means the
    instrumented classifier has drifted from the extractor it's meant to
    mirror, and any per-row analysis it produces is suspect.

    Asserts AGREEMENT, not correctness — whatever the live extractor returns,
    the classifier must return the same. Correctness lives in the extractor.
    """

    @classmethod
    def setUpClass(cls):
        cls.ns = inspect_failures.load_extractor_namespace()

    def test_sib200_agreement(self):
        live = self.ns["extract_sib200_category"]
        cases = [
            # English canonical / normalized / embedded (fallback)
            "science/technology",
            "Science/Technology.",
            '"travel"',
            "The answer is travel",
            "POLITICS",
            # Sci/tech sub-topics (single piece → 1 distinct → science/technology)
            "science/AI",
            "physics",
            "interactive design",
            # Multi-term hedge across DIFFERENT categories → None
            "science/politics",
            "سیاست/تکنالوجی",
            "science/health",
            "کھیل/سائنس/تکنالوجی",
            # Same-category compound (both halves → sports) → sports
            "کھیل/سپورٹس",
            # Native single-word answers
            "سیاست",
            "سفر",
            "سیاحت",
            "صحت",
            "تفریح",
            "کھیل",
            "旅行",
            "政治",
            "娱乐",
            "地理",
            "体育",
            "viajes",
            "política",
            "Política",  # capitalised
            "deportes",
            "salud",
            "entretenimiento",
            # Native science/tech variants
            "سائنس/ٹکنالوجی",
            "سائنس/ٹیکنالوجی",
            "سائنس/تکنولوجی",
            "علم و ٹیکنالوجی",
            "科学/技术",
            "科学和技术",
            "科学与技术",
            "ciencia/tecnología",
            "ciencia y tecnología",
            "tecnología",
            # Arabic code-switches (Urdu-prompted model)
            "الرياضة",
            "السياسة",
            "التكنولوجيا",
            # Multiline outputs — first line wins
            "سفر\nاس متن میں سفر کا ذکر ہے۔",
            "travel\nThis passage is about going places.",
            # Off-scheme / no category at all
            "naturaleza",
            "教育",
            "garbage text with no category",
            "",
            "狮群",  # passage echo
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                instrumented, _ = inspect_failures.classify_sib200(raw, self.ns)
                self.assertEqual(instrumented, live(raw))

    def test_xnli_agreement(self):
        live = self.ns["extract_xnli_label"]
        cases = [
            # Tier 1a — verbatim English labels
            "entailment",
            "contradiction",
            "neutral",
            "Entailment.",
            "Neutral.",
            "Contradiction.",
            # Tier 1b — native label words
            "矛盾",
            "蕴含",
            "中立",
            "Contradicción",
            "Contradicción.",
            "implicación",
            "neutro",
            "لازمی",
            "لازم آتی ہے",
            "تردید",
            # Tier 2 — English label glued to a CJK frame
            "假设是entailment。",
            "假设entailment",
            "假设是entailment，因为它直接从",
            # Tier 2 — negated CJK frame, guard skips Tier 2
            "假设和前提之间没有entailment",
            "前提和假设之间没有entailment、",
            # Tier 3 — semantic paraphrase
            "假设是前提的直接结果。",
            "假设是前提的否定。",
            "假设是前提的推论",
            "假设是前提的等同",
            "假设是前提的自然结果",
            "premise اور hypothesis کے درمیان کوئی تعلق نہیں",
            # Tier 3 — documented negation-gap inputs
            # (instrumented MUST match live: both will currently return entailment
            # on the negated paraphrase — that's the documented zero-impact gap)
            "假设不是前提的直接结果",
            "没有否定前提",
            # None — role-tokens, empty, off-task
            "假设",
            "假设。",
            "Hypothesis",
            "???",
            "",
            "1. 2. 3.",
            "平衡",
            "无",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                instrumented, _ = inspect_failures.classify_xnli(raw, self.ns)
                self.assertEqual(instrumented, live(raw))

    def test_choice_agreement(self):
        live = self.ns["extract_choice"]
        cases = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "ANSWER: C",
            "ANSWER:D",
            "ANSWERA",  # vestigial answer-prefix path
            "B. office",
            "A.",
            "The answer is B.",
            "hello",
            "",
        ]
        for raw in cases:
            for choices in ("ABCDE", "ABCD"):
                with self.subTest(raw=raw, choices=choices):
                    instrumented, _ = inspect_failures.classify_choice(
                        raw, self.ns, choices
                    )
                    self.assertEqual(instrumented, live(raw, choices=choices))


if __name__ == "__main__":
    unittest.main()

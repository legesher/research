"""Unit tests for reparse_results.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_reparse_results.py -v

All tests run unconditionally — extractors are defined inline in
reparse_results.py, so the previous "skip when run_eval_single.py is missing"
gates no longer apply.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make the script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reparse_results  # noqa: E402


class TestPathMangling(unittest.TestCase):
    """Pure string substitution; no extractor source needed."""

    def test_local_path_round_trip(self):
        inp = Path("/tmp/condition-2-ur-5k_seed42_results_template1.json")
        out = reparse_results.reparsed_summary_path_local(inp)
        self.assertEqual(
            out.name, "condition-2-ur-5k_seed42_summary_reparsed_template1.json"
        )
        self.assertEqual(out.parent, inp.parent)

    def test_local_path_smoke_infix(self):
        inp = Path("baseline_seednone_smoke20_results_template2.json")
        out = reparse_results.reparsed_summary_path_local(inp)
        self.assertEqual(
            out.name, "baseline_seednone_smoke20_summary_reparsed_template2.json"
        )

    def test_local_path_rejects_non_results(self):
        with self.assertRaisesRegex(ValueError, "_results_"):
            reparse_results.reparsed_summary_path_local(
                Path("baseline_summary_template1.json")
            )

    def test_remote_path_keeps_parent(self):
        inp = "phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json"
        out = reparse_results.reparsed_summary_path_remote(inp)
        self.assertEqual(
            out,
            "phase3/conditions/baseline/seednone/"
            "baseline_seednone_summary_reparsed_template1.json",
        )

    def test_remote_path_bare_basename(self):
        # No slashes — just a filename
        out = reparse_results.reparsed_summary_path_remote("x_results_template1.json")
        self.assertEqual(out, "x_summary_reparsed_template1.json")

    def test_remote_path_rejects_non_results(self):
        with self.assertRaisesRegex(ValueError, "_results_"):
            reparse_results.reparsed_summary_path_remote("phase3/x/y/foo.json")


class TestBenchmarkFromKey(unittest.TestCase):
    """Pure string parsing; no extractor source needed."""

    def test_known_benchmarks(self):
        cases = {
            "template1_sib200_data=ur_instr=ur": "sib200",
            "template2_xnli_data=en_instr=zh": "xnli",
            "template1_csqa_data=es_instr=en": "csqa",
            "template2_belebele_data=zh_instr=ur": "belebele",
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(reparse_results.benchmark_from_key(key), expected)

    def test_unknown_benchmark_raises(self):
        with self.assertRaisesRegex(ValueError, "Couldn't infer benchmark"):
            reparse_results.benchmark_from_key(
                "template1_unknownbench_data=en_instr=en"
            )


class TestBuildReparsedSummary(unittest.TestCase):
    """Schema + delta semantics. Requires extractor source for provenance."""

    def _synthetic_rows(self):
        # Two cells: one changed, one unchanged.
        return [
            {
                "cell": "template1_sib200_data=ur_instr=ur",
                "n": 20,
                "new_correct": 17,
                "old_acc": 0.0,
                "new_acc": 0.85,
                "old_fail": 1.0,
                "new_fail": 0.15,
            },
            {
                "cell": "template1_xnli_data=en_instr=en",
                "n": 20,
                "new_correct": 10,
                "old_acc": 0.5,
                "new_acc": 0.5,
                "old_fail": 0.0,
                "new_fail": 0.0,
            },
        ]

    def test_top_level_schema(self):
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertIn("summary", body)
        self.assertIn("parse_failure_rates", body)
        self.assertIn("reparse_metadata", body)

    def test_summary_acc_keys_match_original_schema(self):
        # Original Kaggle summary uses `<cell>_acc` keys.
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertIn("template1_sib200_data=ur_instr=ur_acc", body["summary"])
        self.assertAlmostEqual(
            body["summary"]["template1_sib200_data=ur_instr=ur_acc"], 0.85
        )

    def test_parse_failure_rates_no_acc_suffix(self):
        # Original Kaggle parse_failure_rates uses the bare cell key.
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertIn("template1_sib200_data=ur_instr=ur", body["parse_failure_rates"])
        self.assertAlmostEqual(
            body["parse_failure_rates"]["template1_sib200_data=ur_instr=ur"], 0.15
        )

    def test_delta_only_includes_changed_cells(self):
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        deltas = body["reparse_metadata"]["delta_per_cell"]
        self.assertIn("template1_sib200_data=ur_instr=ur", deltas)
        self.assertNotIn("template1_xnli_data=en_instr=en", deltas)
        self.assertEqual(body["reparse_metadata"]["cells_changed"], 1)

    def test_delta_values(self):
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        d = body["reparse_metadata"]["delta_per_cell"][
            "template1_sib200_data=ur_instr=ur"
        ]
        self.assertAlmostEqual(d["delta_acc"], 0.85)
        self.assertAlmostEqual(d["delta_fail"], -0.85)

    def test_metadata_records_original_filename(self):
        body = reparse_results.build_reparsed_summary(
            Path("/tmp/baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertEqual(
            body["reparse_metadata"]["original_results_filename"],
            "baseline_seednone_results_template1.json",
        )

    def test_metadata_records_all_extractors(self):
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertEqual(
            sorted(body["reparse_metadata"]["extractors_applied"]),
            sorted(reparse_results.EXTRACTOR_NAMES),
        )

    def test_provenance_block_shape(self):
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        prov = body["reparse_metadata"]["extractor_provenance"]
        self.assertIn("source_path", prov)
        self.assertIn("content_sha256", prov)
        self.assertEqual(len(prov["content_sha256"]), 64)  # sha-256 hex digest

    def test_summary_includes_count_and_correct_per_cell(self):
        # Paper-grade reporting needs n + correct per cell directly in the
        # summary, not buried in the full results JSON. Additive keys; doesn't
        # break readers of the original `_acc` schema.
        body = reparse_results.build_reparsed_summary(
            Path("baseline_seednone_results_template1.json"),
            self._synthetic_rows(),
        )
        self.assertEqual(body["summary"]["template1_sib200_data=ur_instr=ur_count"], 20)
        self.assertEqual(body["summary"]["template1_sib200_data=ur_instr=ur_correct"], 17)
        self.assertEqual(body["summary"]["template1_xnli_data=en_instr=en_count"], 20)
        self.assertEqual(body["summary"]["template1_xnli_data=en_instr=en_correct"], 10)


class TestReparseFile(unittest.TestCase):
    """End-to-end against a synthetic results JSON. Only exercises one
    extractor (SIB-200) to keep the fixture small."""

    def _write_synthetic_results(self, tmpdir: Path) -> Path:
        # Minimal results JSON: one SIB-200 cell with 2 rows.
        # Row 1 is "science/technology" — extractor returns it directly.
        # Row 2 is "سائنس/ٹکنالوجی" (Urdu) — extractor v2 recognises it,
        # original extractor would have returned None.
        results = {
            "summary": {"template1_sib200_data=ur_instr=ur_acc": 0.0},
            "parse_failure_rates": {"template1_sib200_data=ur_instr=ur": 1.0},
            "template1_sib200_data=ur_instr=ur": [
                {
                    "raw_output": "science/technology",
                    "gold": "science/technology",
                    "pred": None,
                    "correct": False,
                },
                {
                    "raw_output": "سائنس/ٹکنالوجی",
                    "gold": "science/technology",
                    "pred": None,
                    "correct": False,
                },
            ],
        }
        path = tmpdir / "fake_seednone_results_template1.json"
        path.write_text(json.dumps(results))
        return path

    def test_reparse_recovers_native_script_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            fixture = self._write_synthetic_results(tmpdir)
            rows = reparse_results.reparse_file(fixture, only=None)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["cell"], "template1_sib200_data=ur_instr=ur")
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["new_correct"], 2)  # both rows pred == gold
            self.assertEqual(row["new_acc"], 1.0)  # both correct under v2 extractor
            self.assertEqual(row["new_fail"], 0.0)
            # old_acc / old_fail come from the synthetic summary block
            self.assertEqual(row["old_acc"], 0.0)
            self.assertEqual(row["old_fail"], 1.0)

    def test_reparse_only_filter(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            fixture = self._write_synthetic_results(tmpdir)
            # Filter to xnli only — should skip the sib200 cell.
            rows = reparse_results.reparse_file(fixture, only={"xnli"})
            self.assertEqual(rows, [])


class TestExtractorsLoadable(unittest.TestCase):
    """Smoke-tests every extractor with a representative input from each
    language. With inline extractors, these confirm the public extractor
    dict resolves to working callables for all four benchmarks."""

    def setUp(self):
        self.extractors = reparse_results._load_extractors()

    def test_extract_xnli_english(self):
        self.assertEqual(self.extractors["xnli"]("entailment"), "entailment")
        self.assertEqual(self.extractors["xnli"]("contradiction"), "contradiction")
        self.assertEqual(self.extractors["xnli"]("neutral"), "neutral")

    def test_extract_xnli_chinese(self):
        # Native Chinese labels from NATIVE_LABEL_MAP — catches the bug end-to-end
        self.assertEqual(self.extractors["xnli"]("蕴含"), "entailment")
        self.assertEqual(self.extractors["xnli"]("矛盾"), "contradiction")
        self.assertEqual(self.extractors["xnli"]("中立"), "neutral")

    def test_extract_xnli_spanish(self):
        self.assertEqual(self.extractors["xnli"]("contradicción"), "contradiction")
        self.assertEqual(self.extractors["xnli"]("implicación"), "entailment")
        self.assertEqual(self.extractors["xnli"]("neutro"), "neutral")

    def test_extract_xnli_urdu(self):
        self.assertEqual(self.extractors["xnli"]("لازمی"), "entailment")
        self.assertEqual(self.extractors["xnli"]("تردید"), "contradiction")
        self.assertEqual(self.extractors["xnli"]("غیرجانبدار"), "neutral")

    def test_extract_xnli_returns_none_for_unparseable(self):
        self.assertIsNone(self.extractors["xnli"]("???"))

    def test_extract_sib200_english_canonical(self):
        self.assertEqual(
            self.extractors["sib200"]("science/technology"), "science/technology"
        )
        self.assertEqual(self.extractors["sib200"]("travel"), "travel")

    def test_extract_sib200_native_urdu(self):
        self.assertEqual(
            self.extractors["sib200"]("سائنس/ٹکنالوجی"), "science/technology"
        )

    def test_extract_sib200_native_chinese(self):
        self.assertEqual(self.extractors["sib200"]("科学/技术"), "science/technology")

    def test_extract_sib200_native_spanish(self):
        self.assertEqual(
            self.extractors["sib200"]("ciencia/tecnología"), "science/technology"
        )
        self.assertEqual(
            self.extractors["sib200"]("ciencia y tecnología"), "science/technology"
        )

    def test_extract_sib200_invented_subcategory(self):
        # Rule A: science/<X> → science/technology
        self.assertEqual(self.extractors["sib200"]("science/AI"), "science/technology")
        self.assertEqual(
            self.extractors["sib200"]("science/physics"), "science/technology"
        )

    def test_extract_sib200_bare_subcategory(self):
        # Rule C: template2 sometimes strips the "science/" prefix
        self.assertEqual(self.extractors["sib200"]("physics"), "science/technology")
        self.assertEqual(self.extractors["sib200"]("ai"), "science/technology")

    def test_extract_choice_abcde(self):
        self.assertEqual(self.extractors["csqa"]("A"), "A")
        self.assertEqual(self.extractors["csqa"]("Answer: C"), "C")

    def test_extract_choice_abcd(self):
        self.assertEqual(self.extractors["belebele"]("B"), "B")
        self.assertEqual(self.extractors["belebele"]("Answer: D"), "D")


class TestSib200MultiTermHedge(unittest.TestCase):
    """The model sometimes lists two unrelated categories — that's a hedge,
    not an answer. Even with the refined extractor's broader scope on
    native-script and code-switched forms, a multi-category emission must
    not be credited as a single classification."""

    def setUp(self):
        self.sib = reparse_results.extract_sib200_category

    def test_urdu_politics_technology_compound(self):
        # The bad PR-#49 entry that motivated the multi-term rule:
        # سیاست/تکنالوجی = politics + technology, scored as None.
        self.assertIsNone(self.sib("سیاست/تکنالوجی"))

    def test_english_two_distinct_categories(self):
        self.assertIsNone(self.sib("science and politics"))
        self.assertIsNone(self.sib("politics, sports"))
        self.assertIsNone(self.sib("travel/health"))

    def test_many_pieces_one_category_resolves(self):
        # 4 pieces, all collapse to science/technology — not a hedge.
        self.assertEqual(
            self.sib("science / technology / AI / physics"), "science/technology"
        )


class TestSib200ConjunctionNormalization(unittest.TestCase):
    """Compounds joined by language-specific conjunctions split into pieces."""

    def setUp(self):
        self.sib = reparse_results.extract_sib200_category

    def test_english_and(self):
        self.assertEqual(self.sib("science and technology"), "science/technology")

    def test_spanish_y(self):
        self.assertEqual(self.sib("ciencia y tecnología"), "science/technology")
        self.assertEqual(self.sib("ciencia y tecnologia"), "science/technology")

    def test_chinese_he(self):
        self.assertEqual(self.sib("科学和技术"), "science/technology")

    def test_chinese_yu(self):
        self.assertEqual(self.sib("科学与技术"), "science/technology")


class TestSib200CjkGluedFallback(unittest.TestCase):
    """Phase-3 baseline emits forms like "答案是travel" (CJK frame + English
    answer glued together). Python's Unicode `\\b` refuses a boundary between
    a CJK char (`\\w` in unicode mode) and a Latin letter, so the fallback
    uses plain substring matching to catch these."""

    def setUp(self):
        self.sib = reparse_results.extract_sib200_category

    def test_cjk_glued_english_answer(self):
        self.assertEqual(self.sib("答案是travel"), "travel")

    def test_english_embedded_answer(self):
        self.assertEqual(self.sib("the answer is travel"), "travel")
        self.assertEqual(self.sib("The category is health."), "health")

    def test_fallback_returns_none_on_multi_match(self):
        # Fallback only returns when exactly one canonical category appears.
        # Note: avoid conjunctions like " and " — those are normalized by
        # _sib200_split into separators BEFORE the fallback runs.
        self.assertIsNone(self.sib("this is health travel related"))


class TestSib200ReviewerDecisions(unittest.TestCase):
    """Surface-form review decisions captured in the analysis ledger."""

    def setUp(self):
        self.sib = reparse_results.extract_sib200_category

    def test_chinese_public_transport_is_travel(self):
        # analysis/phase-3/chinese-surface-forms-review.md Section C:
        # 公共交通 ("public transportation") classified as travel, not sci/tech.
        self.assertEqual(self.sib("公共交通"), "travel")

    def test_spanish_accent_stripped(self):
        self.assertEqual(self.sib("politica"), "politics")
        self.assertEqual(self.sib("tecnologia"), "science/technology")

    def test_urdu_transliterations(self):
        self.assertEqual(self.sib("سپورٹس"), "sports")  # sports transliteration
        self.assertEqual(self.sib("سفر"), "travel")  # safar
        self.assertEqual(self.sib("سیاحت"), "travel")  # siyahat - tourism

    def test_arabic_code_switch(self):
        # Urdu-prompted model code-switches to Arabic — covered in
        # phase-3/urdu-surface-forms-review.md Section D.
        self.assertEqual(self.sib("السياسة"), "politics")
        self.assertEqual(self.sib("التكنولوجيا"), "science/technology")


class TestSib200EdgeCases(unittest.TestCase):
    def setUp(self):
        self.sib = reparse_results.extract_sib200_category

    def test_empty_input(self):
        self.assertIsNone(self.sib(""))

    def test_punctuation_only(self):
        self.assertIsNone(self.sib(".,;!"))

    def test_only_first_line_matters(self):
        # Subsequent lines are ignored, even if they'd otherwise change the result.
        self.assertEqual(self.sib("travel\npolitics"), "travel")

    def test_unmappable_returns_none(self):
        self.assertIsNone(self.sib("???"))
        self.assertIsNone(self.sib("a vague topic"))

    def test_trailing_punctuation_stripped(self):
        # SIB200_STRIP handles Chinese full-stop, Arabic comma, etc.
        self.assertEqual(self.sib("Politics."), "politics")
        self.assertEqual(self.sib("科学/技术。"), "science/technology")


class TestXnliTiers(unittest.TestCase):
    """Tier ordering: 1a English -> 1b native -> 2 CJK-glued -> 3 paraphrase."""

    def setUp(self):
        self.xnli = reparse_results.extract_xnli_label

    def test_tier1a_english_word_boundary(self):
        self.assertEqual(self.xnli("entailment"), "entailment")
        self.assertEqual(self.xnli("the answer is entailment"), "entailment")

    def test_tier1b_native_label(self):
        # Already covered by TestExtractorsLoadable; one assertion per language
        # here to lock the tier interleaving against tier 1a.
        self.assertEqual(self.xnli("蕴含"), "entailment")
        self.assertEqual(self.xnli("contradicción"), "contradiction")
        self.assertEqual(self.xnli("لازمی"), "entailment")

    def test_tier2_cjk_glued_english_label(self):
        # 答案是entailment — \b refuses boundary, substring catches it.
        self.assertEqual(self.xnli("答案是entailment"), "entailment")

    def test_tier2_negation_guard_blocks_glued_label(self):
        # 没有entailment = "there is no entailment". The negation marker
        # forces a Tier-2 skip; no Tier-3 paraphrase matches => parse failure.
        self.assertIsNone(self.xnli("没有entailment"))

    def test_tier3_paraphrase_neutral_chinese(self):
        self.assertEqual(self.xnli("两句话没有任何关系"), "neutral")
        self.assertEqual(self.xnli("没有关联"), "neutral")
        self.assertEqual(self.xnli("没有联系"), "neutral")

    def test_tier3_paraphrase_neutral_urdu(self):
        self.assertEqual(self.xnli("کوئی واضح تعلق نہیں ہے"), "neutral")

    def test_tier3_paraphrase_contradiction(self):
        self.assertEqual(self.xnli("第二句话是对第一句话的否定"), "contradiction")

    def test_tier3_paraphrase_entailment(self):
        self.assertEqual(self.xnli("这是一个直接结果"), "entailment")
        self.assertEqual(self.xnli("这是一种推论"), "entailment")
        self.assertEqual(self.xnli("两个句子等同"), "entailment")

    def test_unparseable_returns_none(self):
        self.assertIsNone(self.xnli("???"))


class TestProvenanceHashesThisFile(unittest.TestCase):
    """The provenance block must hash reparse_results.py itself — that's how
    paper reviewers verify they're scoring with the same extractor code."""

    def test_provenance_hashes_reparse_results_py(self):
        import hashlib

        rr_path = Path(reparse_results.__file__).resolve()
        expected_hash = hashlib.sha256(rr_path.read_bytes()).hexdigest()
        prov = reparse_results._extractor_provenance()
        self.assertEqual(prov["content_sha256"], expected_hash)

    def test_provenance_source_path_is_basename_only(self):
        # No absolute path leak — only the filename should appear.
        prov = reparse_results._extractor_provenance()
        self.assertEqual(prov["source_path"], "reparse_results.py")
        self.assertNotIn("/", prov["source_path"])


if __name__ == "__main__":
    unittest.main()

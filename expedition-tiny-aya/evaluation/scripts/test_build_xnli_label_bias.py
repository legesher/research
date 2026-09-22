"""Unit tests for build_xnli_label_bias.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_build_xnli_label_bias.py -v

Or the whole scripts suite from the repository root (the scripts import
huggingface_hub at module level, so it has to be installed):

    uv run --with huggingface_hub python -m unittest discover \\
        -s expedition-tiny-aya/evaluation/scripts -p 'test_*.py' -v

These pin the hardening landed in commit 8caafac: `_CJK_RE` starts at
U+3001 (U+3000 IDEOGRAPHIC SPACE is whitespace, not framing), the Tier-2
CJK-frame check reads the stripped first line, the tagged and canonical
extractors are held equal with a raise (not an assert that `python -O`
would strip), an unexpected gold label is skipped and reported on stderr
instead of raising KeyError, and per-cell accuracy is cross-checked
against the published reparsed summary. No network: the two Hub
touchpoints (`hf_hub_download`, `load_reparsed_summary`) are patched onto
a temp fixture.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make the script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_xnli_label_bias as bias  # noqa: E402
import reparse_results  # noqa: E402


class TestCjkRe(unittest.TestCase):
    """`_CJK_RE` decides whether a Tier-2 first line is CJK-framed. Its
    ranges are U+3001-303F (CJK Symbols and Punctuation minus IDEOGRAPHIC
    SPACE), U+3400-4DBF (Extension A), U+4E00-9FFF (Unified Ideographs)
    and U+F900-FAFF (Compatibility Ideographs). Every edge is asserted."""

    def _matches(self, text: str) -> bool:
        return bias._CJK_RE.search(text) is not None

    def test_ideographic_space_is_whitespace_not_framing(self):
        # Before 8caafac the class started at U+3000, so a first line
        # ending in an ideographic space counted as CJK-framed with no CJK
        # character present. It is whitespace to Python and stays out.
        self.assertTrue("　".isspace())
        self.assertFalse(self._matches("　"))
        self.assertFalse(self._matches("entailment　"))

    def test_cjk_punctuation_matches_from_u3001_to_u303f(self):
        for ch in ("、", "。", "「", "」", "〿"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertTrue(self._matches(ch))

    def test_unified_ideographs_match(self):
        for ch in ("一", "中", "鿿"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertTrue(self._matches(ch))

    def test_extension_a_matches(self):
        for ch in ("㐀", "䶿"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertTrue(self._matches(ch))

    def test_compatibility_ideographs_match(self):
        for ch in ("豈", "﫿"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertTrue(self._matches(ch))

    def test_one_past_each_range_edge_does_not_match(self):
        # Just outside each range: 3040, 33FF, 4DC0, 4DFF, A000, F8FF, FB00.
        for ch in ("぀", "㏿", "䷀", "䷿", "ꀀ", "", "ﬀ"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertFalse(self._matches(ch))

    def test_ascii_does_not_match(self):
        for text in ("entailment", "The answer is: contradiction.", "0123456789", ""):
            with self.subTest(text=text):
                self.assertFalse(self._matches(text))

    def test_kana_and_fullwidth_punctuation_are_outside_the_ranges(self):
        # Hiragana (U+3040-309F), Katakana (U+30A0-30FF) and the Halfwidth
        # and Fullwidth Forms block (U+FF00-FFEF) are not in the class: a
        # line framed only by a fullwidth comma or colon does not count as
        # CJK-framed. Pinned so that widening the class is a deliberate change.
        for ch in ("あ", "ア", "，", "：", "！"):
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertFalse(self._matches(ch))

    def test_glued_frame_matches_inside_a_line(self):
        self.assertTrue(self._matches("答案是entailment"))


# (raw_output, label expected from both extractors, tier expected from the
# tagged twin). Covers each tier, the negation guard, a no-match, trailing
# whitespace and multi-line outputs.
AGREEMENT_CASES = (
    ("entailment", "entailment", "tier1a_english"),
    ("The answer is contradiction.", "contradiction", "tier1a_english"),
    ("Neutral", "neutral", "tier1a_english"),
    ("蕴含", "entailment", "tier1b_native"),
    ("矛盾", "contradiction", "tier1b_native"),
    ("中立", "neutral", "tier1b_native"),
    ("implicación", "entailment", "tier1b_native"),
    ("contradicción", "contradiction", "tier1b_native"),
    ("neutro", "neutral", "tier1b_native"),
    ("لازمی", "entailment", "tier1b_native"),
    ("تردید", "contradiction", "tier1b_native"),
    ("غیرجانبدار", "neutral", "tier1b_native"),
    ("答案是entailment", "entailment", "tier2_glued"),
    ("答案是contradiction。", "contradiction", "tier2_glued"),
    ("两句话没有任何关系", "neutral", "tier3_paraphrase"),
    ("第二句话是对第一句话的否定", "contradiction", "tier3_paraphrase"),
    ("这是一个直接结果", "entailment", "tier3_paraphrase"),
    ("کوئی واضح تعلق نہیں ہے", "neutral", "tier3_paraphrase"),
    ("没有entailment", None, None),  # negation guard blocks Tier 2; no Tier 3
    ("???", None, None),
    ("", None, None),
    # Only the stripped first line is read.
    ("neutral   ", "neutral", "tier1a_english"),
    ("entailment   \nsecond line says contradiction", "entailment", "tier1a_english"),
    ("\n\n答案是entailment  \n", "entailment", "tier2_glued"),
    ("中立\nentailment", "neutral", "tier1b_native"),
)


class TestTaggedCanonicalAgreement(unittest.TestCase):
    """`extract_xnli_label_tagged` must agree with the canonical
    `reparse_results.extract_xnli_label` on every input and name the tier
    that matched."""

    def test_tagged_label_equals_canonical_and_tier_is_named(self):
        for raw, label, tier in AGREEMENT_CASES:
            with self.subTest(raw=raw):
                tagged = bias.extract_xnli_label_tagged(raw)
                self.assertEqual(tagged[0], reparse_results.extract_xnli_label(raw))
                self.assertEqual(tagged, (label, tier))

    def test_cases_cover_every_tier_name(self):
        seen = {tier for _, _, tier in AGREEMENT_CASES if tier is not None}
        self.assertEqual(seen, set(bias.TIER_NAMES))

    def test_every_tier_name_has_a_via_column(self):
        for tier in bias.TIER_NAMES:
            with self.subTest(tier=tier):
                self.assertIn(f"via_{tier}", bias.COLUMNS)


CELL = "template1_xnli_data=zh_instr=zh"
REMOTE = "phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json"

# 8 rows, 6 correct. Row 4 is the 8caafac regression: a Tier-2 glued label
# whose first line ends in U+3000 before a second line. What keeps it out
# of tier2_cjk_frame, as seen through process_file, is the U+3001 lower
# bound on _CJK_RE. The per-line strip is defensive: once the class starts
# at U+3001 no whitespace code point is inside it, so a revert of the strip
# alone is not observable here.
VALID_ROWS = [
    {"gold": "entailment", "raw_output": "entailment"},  # 1a, correct
    {"gold": "entailment", "raw_output": "答案是entailment"},  # 2 + CJK, correct
    {"gold": "contradiction", "raw_output": "答案是contradiction"},  # 2 + CJK, correct
    {"gold": "contradiction", "raw_output": "thecontradiction　\nsecond line"},  # 2, no CJK
    {"gold": "neutral", "raw_output": "entailment"},  # 1a, wrong
    {"gold": "neutral", "raw_output": "两句话没有任何关系"},  # 3, correct
    {"gold": "neutral", "raw_output": "???"},  # no match, wrong
    {"gold": "neutral", "raw_output": "中立"},  # 1b, correct
]
BOGUS_ROW = {"gold": "bogus", "raw_output": "entailment"}


class _ProcessFileHarness(unittest.TestCase):
    """Writes a one-cell results JSON to a temp dir and patches the two Hub
    touchpoints so `process_file` reads it. Returns the result and stderr."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmpdir = Path(td.name)

    def run_process_file(self, rows, published_acc, extra_keys=None):
        data = {"summary": {}, "parse_failure_rates": {}, CELL: rows}
        if extra_keys:
            data.update(extra_keys)
        fixture = self.tmpdir / "baseline_seednone_results_template1.json"
        fixture.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        summary = {"summary": {}}
        if published_acc is not None:
            summary["summary"][f"{CELL}_acc"] = published_acc
        stderr = io.StringIO()
        with mock.patch.object(
            bias, "hf_hub_download", return_value=str(fixture)
        ) as download, mock.patch.object(
            bias, "load_reparsed_summary", return_value=summary
        ), contextlib.redirect_stderr(stderr):
            result = bias.process_file(REMOTE, "baseline", "none")
        self.assertEqual(download.call_args.kwargs["filename"], REMOTE)
        return result, stderr.getvalue()


class TestUnexpectedGoldGuard(_ProcessFileHarness):
    def test_bogus_gold_is_skipped_reported_and_never_a_row(self):
        # 6 correct; the denominator is the cell size including the skipped
        # row (9), the same n = len(items) the canonical reparse_file uses.
        (rows, mismatches), err = self.run_process_file(
            VALID_ROWS + [BOGUS_ROW], published_acc=6 / 9
        )
        self.assertEqual(mismatches, 0)
        self.assertEqual([r["gold"] for r in rows], list(bias.XNLI_LABELS))
        self.assertEqual(sum(r["n_gold"] for r in rows), len(VALID_ROWS))
        self.assertIn("1 row(s) skipped", err)
        self.assertIn("bogus", err)
        self.assertIn(REMOTE, err)

    def test_confusion_and_tier_counts_per_gold(self):
        (rows, _), _ = self.run_process_file(VALID_ROWS, published_acc=6 / 8)
        by_gold = {r["gold"]: r for r in rows}
        self.assertEqual(set(rows[0]), set(bias.COLUMNS))

        ent = by_gold["entailment"]
        self.assertEqual(ent["n_gold"], 2)
        self.assertEqual(ent["pred_entailment"], 2)
        self.assertEqual(ent["via_tier1a_english"], 1)
        self.assertEqual(ent["via_tier2_glued"], 1)
        self.assertEqual(ent["tier2_pred_entailment"], 1)
        self.assertEqual(ent["tier2_cjk_frame"], 1)
        self.assertEqual(ent["acc_on_gold"], 1.0)

        con = by_gold["contradiction"]
        self.assertEqual(con["n_gold"], 2)
        self.assertEqual(con["pred_contradiction"], 2)
        self.assertEqual(con["via_tier2_glued"], 2)
        self.assertEqual(con["tier2_pred_entailment"], 0)
        self.assertEqual(con["acc_on_gold"], 1.0)

        neu = by_gold["neutral"]
        self.assertEqual(neu["n_gold"], 4)
        self.assertEqual(neu["pred_entailment"], 1)
        self.assertEqual(neu["pred_neutral"], 2)
        self.assertEqual(neu["pred_none"], 1)
        self.assertEqual(neu["via_tier1a_english"], 1)
        self.assertEqual(neu["via_tier1b_native"], 1)
        self.assertEqual(neu["via_tier3_paraphrase"], 1)
        self.assertEqual(neu["acc_on_gold"], 0.5)

    def test_tier2_cjk_frame_ignores_ideographic_space_on_a_stripped_line(self):
        # Two Tier-2 contradiction rows, only one of them CJK-framed: the
        # "thecontradiction　\nsecond line" row must not count.
        (rows, _), _ = self.run_process_file(VALID_ROWS, published_acc=6 / 8)
        con = {r["gold"]: r for r in rows}["contradiction"]
        self.assertEqual(con["via_tier2_glued"], 2)
        self.assertEqual(con["tier2_cjk_frame"], 1)

    def test_no_report_when_every_gold_is_expected(self):
        (_, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=6 / 8)
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")

    def test_skipped_row_stays_in_the_accuracy_denominator(self):
        # reparse_file scores an unexpected gold as incorrect rather than
        # dropping it from n, so 6/8 would be a mismatch here and 6/9 is
        # the number the cross-check expects.
        (_, mismatches), err = self.run_process_file(
            VALID_ROWS + [BOGUS_ROW], published_acc=6 / 8
        )
        self.assertEqual(mismatches, 1)
        self.assertIn(f"recomputed {6 / 9:.6f} vs published 0.75", err)

    def test_non_xnli_and_non_list_keys_are_ignored(self):
        (rows, mismatches), err = self.run_process_file(
            VALID_ROWS,
            published_acc=6 / 8,
            extra_keys={
                "template1_sib200_data=zh_instr=zh": [{"gold": "travel", "raw_output": "travel"}],
                "template2_xnli_data=zh_instr=zh": "not a list",
            },
        )
        self.assertEqual(len(rows), len(bias.XNLI_LABELS))
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")


class TestTaggedCanonicalDivergence(_ProcessFileHarness):
    def test_divergence_raises_runtime_error_naming_both_values(self):
        # A RuntimeError survives `python -O`; the old bare assert did not.
        with mock.patch.object(bias, "extract_xnli_label", lambda text: "neutral"):
            with self.assertRaises(RuntimeError) as cm:
                self.run_process_file(
                    [{"gold": "entailment", "raw_output": "entailment"}], published_acc=1.0
                )
        message = str(cm.exception)
        self.assertIn("tagged='entailment'", message)
        self.assertIn("canonical='neutral'", message)
        self.assertIn(CELL, message)


class TestAccuracyCrossCheck(_ProcessFileHarness):
    def test_published_within_tolerance_is_not_a_mismatch(self):
        (_, mismatches), err = self.run_process_file(
            VALID_ROWS, published_acc=6 / 8 + 5e-10
        )
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")

    def test_published_off_by_more_than_tolerance_counts_and_prints(self):
        (rows, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=0.5)
        self.assertEqual(mismatches, 1)
        self.assertIn(f"!! accuracy mismatch baseline/seednone/{CELL}", err)
        self.assertIn("recomputed 0.750000 vs published 0.5", err)
        # Reported, not fatal: the rows still come back and main() turns the
        # count into the exit code.
        self.assertEqual(len(rows), len(bias.XNLI_LABELS))

    def test_missing_published_key_is_a_mismatch(self):
        (_, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=None)
        self.assertEqual(mismatches, 1)
        self.assertIn("vs published None", err)


if __name__ == "__main__":
    unittest.main()

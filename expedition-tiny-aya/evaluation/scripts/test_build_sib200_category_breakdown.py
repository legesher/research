"""Unit tests for build_sib200_category_breakdown.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_build_sib200_category_breakdown.py -v

Or the whole scripts suite from the repository root (the scripts import
huggingface_hub at module level, so it has to be installed):

    uv run --with huggingface_hub python -m unittest discover \\
        -s expedition-tiny-aya/evaluation/scripts -p 'test_*.py' -v

These pin the hardening landed in commit 8caafac for this script: an
unexpected gold label (including a row with no `gold` key at all, since
the guard reads `row.get("gold")`) is skipped and reported on stderr
instead of raising KeyError, and per-cell accuracy is cross-checked
against the published reparsed summary. The script owns no extractor or
CJK logic of its own: native-script and CJK-glued answers are resolved by
`reparse_results.extract_sib200_category` (covered in
test_reparse_results.py), so the fixture only checks that those results
land in the right `pred_<category>` column. No network: the two Hub
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
import build_sib200_category_breakdown as sib  # noqa: E402


class TestSlug(unittest.TestCase):
    def test_slash_becomes_underscore(self):
        self.assertEqual(sib._slug("science/technology"), "science_technology")

    def test_plain_category_is_unchanged(self):
        self.assertEqual(sib._slug("travel"), "travel")

    def test_category_slugs_are_unique_and_slash_free(self):
        slugs = [sib._slug(cat) for cat in sib.SIB200_CATEGORIES]
        self.assertEqual(len(set(slugs)), len(slugs))
        for slug in slugs:
            with self.subTest(slug=slug):
                self.assertNotIn("/", slug)

    def test_columns_carry_one_pred_column_per_category_then_none(self):
        pred_columns = [c for c in sib.COLUMNS if c.startswith("pred_")]
        self.assertEqual(
            pred_columns,
            [f"pred_{sib._slug(cat)}" for cat in sib.SIB200_CATEGORIES] + ["pred_none"],
        )


CELL = "template1_sib200_data=ur_instr=ur"
REMOTE = "phase3/conditions/baseline/seednone/baseline_seednone_results_template1.json"

# 5 rows, 3 correct.
VALID_ROWS = [
    {"gold": "science/technology", "raw_output": "science/technology"},  # correct
    {"gold": "science/technology", "raw_output": "سائنس/ٹکنالوجی"},  # native Urdu, correct
    {"gold": "travel", "raw_output": "答案是travel"},  # CJK-glued fallback, correct
    {"gold": "travel", "raw_output": "science and politics"},  # hedge -> None, wrong
    {"gold": "health", "raw_output": "travel"},  # wrong
]
UNEXPECTED_ROWS = [
    {"gold": "bogus", "raw_output": "travel"},
    {"raw_output": "health"},  # no gold key: row.get("gold") is None
]


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
            sib, "hf_hub_download", return_value=str(fixture)
        ) as download, mock.patch.object(
            sib, "load_reparsed_summary", return_value=summary
        ), contextlib.redirect_stderr(stderr):
            result = sib.process_file(REMOTE, "baseline", "none")
        self.assertEqual(download.call_args.kwargs["filename"], REMOTE)
        return result, stderr.getvalue()


class TestUnexpectedGoldGuard(_ProcessFileHarness):
    def test_unexpected_golds_are_skipped_reported_and_never_rows(self):
        # 3 correct; the denominator is the cell size including the two
        # skipped rows (7), the same n = len(items) reparse_file uses.
        (rows, mismatches), err = self.run_process_file(
            VALID_ROWS + UNEXPECTED_ROWS, published_acc=3 / 7
        )
        self.assertEqual(mismatches, 0)
        self.assertEqual([r["gold_category"] for r in rows], list(sib.SIB200_CATEGORIES))
        self.assertEqual(sum(r["n_gold"] for r in rows), len(VALID_ROWS))
        self.assertIn("2 row(s) skipped", err)
        self.assertIn("{'bogus': 1, None: 1}", err)  # None is the row with no gold key
        self.assertIn(REMOTE, err)

    def test_confusion_counts_per_gold_category(self):
        (rows, _), _ = self.run_process_file(VALID_ROWS, published_acc=3 / 5)
        by_gold = {r["gold_category"]: r for r in rows}
        self.assertEqual(set(rows[0]), set(sib.COLUMNS))

        sci = by_gold["science/technology"]
        self.assertEqual(sci["n_gold"], 2)
        self.assertEqual(sci["correct"], 2)
        self.assertEqual(sci["pred_science_technology"], 2)
        self.assertEqual(sci["acc_on_gold"], 1.0)

        travel = by_gold["travel"]
        self.assertEqual(travel["n_gold"], 2)
        self.assertEqual(travel["correct"], 1)
        self.assertEqual(travel["pred_travel"], 1)
        self.assertEqual(travel["pred_none"], 1)
        self.assertEqual(travel["acc_on_gold"], 0.5)

        health = by_gold["health"]
        self.assertEqual(health["n_gold"], 1)
        self.assertEqual(health["correct"], 0)
        self.assertEqual(health["pred_travel"], 1)
        self.assertEqual(health["acc_on_gold"], 0.0)

        for cat in ("politics", "sports", "entertainment", "geography"):
            with self.subTest(category=cat):
                self.assertEqual(by_gold[cat]["n_gold"], 0)
                self.assertEqual(by_gold[cat]["acc_on_gold"], 0.0)

    def test_no_report_when_every_gold_is_expected(self):
        (_, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=3 / 5)
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")

    def test_skipped_rows_stay_in_the_accuracy_denominator(self):
        # reparse_file scores an unexpected gold as incorrect rather than
        # dropping it from n, so 3/5 would be a mismatch here and 3/7 is
        # the number the cross-check expects.
        (_, mismatches), err = self.run_process_file(
            VALID_ROWS + UNEXPECTED_ROWS, published_acc=3 / 5
        )
        self.assertEqual(mismatches, 1)
        self.assertIn(f"recomputed {3 / 7:.6f} vs published 0.6", err)

    def test_non_sib200_and_non_list_keys_are_ignored(self):
        (rows, mismatches), err = self.run_process_file(
            VALID_ROWS,
            published_acc=3 / 5,
            extra_keys={
                "template1_xnli_data=ur_instr=ur": [
                    {"gold": "entailment", "raw_output": "entailment"}
                ],
                "template2_sib200_data=ur_instr=ur": "not a list",
            },
        )
        self.assertEqual(len(rows), len(sib.SIB200_CATEGORIES))
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")


class TestAccuracyCrossCheck(_ProcessFileHarness):
    def test_published_within_tolerance_is_not_a_mismatch(self):
        (_, mismatches), err = self.run_process_file(
            VALID_ROWS, published_acc=3 / 5 + 5e-10
        )
        self.assertEqual(mismatches, 0)
        self.assertEqual(err, "")

    def test_published_off_by_more_than_tolerance_counts_and_prints(self):
        (rows, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=0.5)
        self.assertEqual(mismatches, 1)
        self.assertIn(f"!! accuracy mismatch baseline/seednone/{CELL}", err)
        self.assertIn("recomputed 0.600000 vs published 0.5", err)
        # Reported, not fatal: the rows still come back and main() turns the
        # count into the exit code.
        self.assertEqual(len(rows), len(sib.SIB200_CATEGORIES))

    def test_missing_published_key_is_a_mismatch(self):
        (_, mismatches), err = self.run_process_file(VALID_ROWS, published_acc=None)
        self.assertEqual(mismatches, 1)
        self.assertIn("vs published None", err)


if __name__ == "__main__":
    unittest.main()

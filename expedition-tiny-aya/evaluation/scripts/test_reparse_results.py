"""Unit tests for reparse_results.py.

Stdlib-only (unittest). Run from this directory:

    python -m unittest test_reparse_results.py -v

Pure-function tests (path mangling, benchmark_from_key) always run.
Extractor-dependent tests are skipped when `run_eval_single.py` is not
present next to this file — that's the expected state in a fresh
checkout where the file is extracted from evaluate.ipynb at runtime.
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

HAS_EXTRACTOR_SOURCE = reparse_results._source_path is not None


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


@unittest.skipUnless(HAS_EXTRACTOR_SOURCE, "run_eval_single.py not next to test file")
class TestBuildReparsedSummary(unittest.TestCase):
    """Schema + delta semantics. Requires extractor source for provenance."""

    def _synthetic_rows(self):
        # Two cells: one changed, one unchanged.
        return [
            {
                "cell": "template1_sib200_data=ur_instr=ur",
                "n": 20,
                "old_acc": 0.0,
                "new_acc": 0.85,
                "old_fail": 1.0,
                "new_fail": 0.15,
            },
            {
                "cell": "template1_xnli_data=en_instr=en",
                "n": 20,
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


@unittest.skipUnless(HAS_EXTRACTOR_SOURCE, "run_eval_single.py not next to test file")
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

    def test_reparse_recovers_lenient_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            fixture = self._write_synthetic_results(tmpdir)
            rows = reparse_results.reparse_file(fixture, only=None)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["cell"], "template1_sib200_data=ur_instr=ur")
            self.assertEqual(row["n"], 2)
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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for analysis/scripts/build_phase3_tables.py.

Stdlib unittest; no network. The script lives in analysis/scripts, so its
directory is added to sys.path here. These pin the data-driven header
accounting (sessions per condition and the template count come from the
cells frame, a condition absent from the hard-coded label list is still
listed), the per-session provisional note, the two-seed ladder cell (mean
followed by both per-seed deltas, never a two-seed standard deviation) and
the local --tables-dir loader.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "analysis" / "scripts"))
import build_phase3_tables as tables  # noqa: E402


def _cells(rows):
    """rows: (condition, seed, template, benchmark, data, instr, baseline_rep, cond_rep)."""
    df = pd.DataFrame(
        rows,
        columns=["condition", "seed", "template", "benchmark", "data", "instr", "baseline_rep", "cond_rep"],
    )
    df["delta_rep"] = df["cond_rep"] - df["baseline_rep"]
    return df


class AccountingHeaderTests(unittest.TestCase):
    def test_sessions_and_seeds_come_from_the_data(self):
        df = _cells([
            ("condition-2-es-20k", "42", 1, "sib200", "es", "es", 0.5, 0.6),
            ("condition-2-es-20k", "42", 2, "sib200", "es", "es", 0.5, 0.6),
            ("condition-2-es-20k", "123", 1, "sib200", "es", "es", 0.5, 0.6),
            ("condition-2-es-20k", "123", 2, "sib200", "es", "es", 0.5, 0.6),
            ("condition-5-ur-5k", "42", 1, "sib200", "es", "es", 0.5, 0.6),
            ("condition-5-ur-5k", "42", 2, "sib200", "es", "es", 0.5, 0.6),
        ])
        lines = tables.accounting_header(df, None)
        self.assertIn("% 4 sessions total = 3 fine-tuned (condition × seed) + 1 baseline", lines)
        self.assertIn("%   - condition-2-es-20k: 2 seeds (42, 123) = 2 multi-seed sessions", lines)
        self.assertIn("%   - condition-5-ur-5k: 1 seed (42) = 1 single-seed session", lines)
        self.assertIn("% 8 observations = 6 fine-tuned cells + 2 baseline cells", lines)
        self.assertFalse(any("PROVISIONAL" in l for l in lines))

    def test_unknown_condition_and_template_count_are_not_hard_coded(self):
        df = _cells([
            ("condition-9-xx-5k", "7", t, "xnli", "en", "en", 0.3, 0.4) for t in (1, 2, 3)
        ])
        lines = tables.accounting_header(df, None)
        self.assertIn("%   - condition-9-xx-5k: 1 seed (7) = 1 single-seed session", lines)
        self.assertTrue(any("× 3 templates" in l for l in lines), lines)
        self.assertTrue(any("3 baseline cells = 1 (benchmark × data × instr) cells" in l for l in lines), lines)

    def test_provisional_sessions_listed_per_session(self):
        df = _cells([
            ("condition-2-ur-5k", "123", 1, "xnli", "ur", "ur", 0.3, 0.4),
            ("condition-2-ur-20k", "123", 1, "xnli", "ur", "ur", 0.3, 0.4),
            ("condition-2-ur-20k", "42", 1, "xnli", "ur", "ur", 0.3, 0.4),
        ])
        lines = tables.accounting_header(df, {"*-20k/seed123"})
        self.assertIn("%   - condition-2-ur-20k/seed123", lines)
        self.assertFalse(any("condition-2-ur-5k/seed123" in l for l in lines))
        note = tables.provisional_note(df, {"*-20k/seed123"})
        self.assertEqual(
            note,
            "% provisional: includes condition-2-ur-20k/seed123; not final until the evaluation drift check is resolved",
        )
        self.assertEqual(tables.provisional_note(df, None), "")
        self.assertEqual(tables.provisional_note(df, {"condition-3-*/seed123"}), "")


class LadderCellTests(unittest.TestCase):
    def test_two_seeds_show_mean_and_both_values_not_sd(self):
        cell = tables.fmt_delta_with_std(-0.0175, 0.0106, [-0.025, -0.010])
        self.assertEqual(cell, "$-$1.8 ($-$2.5/$-$1.0)")
        self.assertNotIn("pm", cell)

    def test_three_seeds_show_mean_plus_minus_sd(self):
        self.assertEqual(tables.fmt_delta_with_std(0.05, 0.01, [0.06, 0.05, 0.04]), "+5.0\\,$\\pm$\\,1.0")

    def test_one_seed_is_a_point_estimate(self):
        self.assertEqual(tables.fmt_delta_with_std(0.012, float("nan"), [0.012]), "+1.2")
        self.assertEqual(tables.fmt_delta_with_std(float("nan"), float("nan"), []), "--")


class LoadCellsTests(unittest.TestCase):
    def test_tables_dir_reads_local_file_with_string_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vs_baseline_cells.tsv"
            path.write_text(
                "condition\tseed\ttemplate\tbenchmark\tdata\tinstr\tbaseline_orig\tcond_orig\tdelta_orig\tbaseline_rep\tcond_rep\tdelta_rep\n"
                "condition-2-ur-5k\t42\t1\tsib200\tur\tur\t0.5\t0.6\t+0.1000\t0.5\t0.6\t+0.1000\n",
                encoding="utf-8",
            )
            df = tables.load_cells(Path(d))
            self.assertEqual(df.seed.tolist(), ["42"])
            self.assertAlmostEqual(df.delta_rep.iloc[0], 0.1)
            with self.assertRaises(FileNotFoundError):
                tables.load_cells(Path(d) / "missing")


if __name__ == "__main__":
    unittest.main()

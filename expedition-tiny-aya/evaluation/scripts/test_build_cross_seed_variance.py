"""Unit tests for build_cross_seed_variance.py.

Stdlib unittest; no network. The cells frame is built in memory with the
columns of vs_baseline_cells.tsv. These pin the method documented in
analysis/phase-3/cross-seed-variance.md: templates (and data languages, for
the aggregate row) collapse within each seed first, cross-seed statistics
are taken over the per-seed aggregates, std is NA for one seed, and nothing
is imputed. They also pin the three generated columns (seeds, per_seed,
provisional) and the two-seed wording in the Markdown companion.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cross_seed_variance as xsv  # noqa: E402


def _cells(rows: list[tuple[str, str, str, str, str, str, float]]) -> pd.DataFrame:
    """rows: (condition, seed, template, benchmark, data, instr, delta_rep)."""
    return pd.DataFrame(
        rows,
        columns=["condition", "seed", "template", "benchmark", "data", "instr", "delta_rep"],
    )


class PerSeedAggregateTests(unittest.TestCase):
    def test_templates_collapse_within_seed_before_cross_seed_stats(self):
        # Seed 42: templates 0.10 and 0.30 -> 0.20 (20 pp). Seed 123: 0.00 and 0.10 -> 5 pp.
        df = _cells([
            ("condition-2-ur-5k", "42", "1", "sib200", "ur", "ur", 0.10),
            ("condition-2-ur-5k", "42", "2", "sib200", "ur", "ur", 0.30),
            ("condition-2-ur-5k", "123", "1", "sib200", "ur", "ur", 0.00),
            ("condition-2-ur-5k", "123", "2", "sib200", "ur", "ur", 0.10),
        ])
        ps = xsv.per_seed_aggregate(df)
        self.assertEqual(list(ps.index), ["42", "123"])
        self.assertAlmostEqual(ps["42"], 20.0)
        self.assertAlmostEqual(ps["123"], 5.0)

    def test_seeds_sort_numerically_not_lexically(self):
        df = _cells([
            ("c", "456", "1", "xnli", "en", "en", 0.01),
            ("c", "42", "1", "xnli", "en", "en", 0.02),
            ("c", "123", "1", "xnli", "en", "en", 0.03),
        ])
        self.assertEqual(list(xsv.per_seed_aggregate(df).index), ["42", "123", "456"])


class BuildRowsTests(unittest.TestCase):
    def setUp(self):
        self.df = _cells([
            # two-seed 20k condition, two data langs, one instr lang
            ("condition-2-es-20k", "42", "1", "sib200", "en", "es", 0.02),
            ("condition-2-es-20k", "42", "2", "sib200", "en", "es", 0.04),
            ("condition-2-es-20k", "42", "1", "sib200", "es", "es", -0.10),
            ("condition-2-es-20k", "42", "2", "sib200", "es", "es", -0.06),
            ("condition-2-es-20k", "123", "1", "sib200", "en", "es", 0.00),
            ("condition-2-es-20k", "123", "2", "sib200", "en", "es", 0.02),
            ("condition-2-es-20k", "123", "1", "sib200", "es", "es", -0.02),
            ("condition-2-es-20k", "123", "2", "sib200", "es", "es", -0.04),
            # single-seed condition
            ("condition-5-ur-5k", "42", "1", "xnli", "ur", "ur", 0.015),
            ("condition-5-ur-5k", "42", "2", "xnli", "ur", "ur", 0.025),
            # baseline rows must be ignored
            ("baseline", "none", "1", "xnli", "ur", "ur", 0.0),
        ])
        self.rows = {
            (r["condition"], r["benchmark"], r["data_lang"], r["instr_lang"]): r
            for r in xsv.build_rows(self.df, provisional={"*-20k/seed123"})
        }

    def test_all_row_is_mean_over_data_and_templates_per_seed(self):
        r = self.rows[("condition-2-es-20k", "sib200", "ALL", "es")]
        # seed 42: mean(0.02, 0.04, -0.10, -0.06) = -0.025 -> -2.5 pp
        # seed 123: mean(0.00, 0.02, -0.02, -0.04) = -0.01 -> -1.0 pp
        self.assertEqual(r["n_seeds"], 2)
        self.assertEqual(r["per_seed"], "42=-2.5000;123=-1.0000")
        self.assertEqual(r["mean"], "-1.7500")
        self.assertEqual(r["min"], "-2.5000")
        self.assertEqual(r["max"], "-1.0000")
        self.assertEqual(r["seeds"], "42,123")

    def test_two_seed_std_is_sample_std(self):
        r = self.rows[("condition-2-es-20k", "sib200", "ALL", "es")]
        # sample std of (-2.5, -1.0) = |diff| / sqrt(2) = 1.0607
        self.assertEqual(r["std"], "1.0607")

    def test_single_seed_std_is_na_and_not_imputed(self):
        r = self.rows[("condition-5-ur-5k", "xnli", "ALL", "ur")]
        self.assertEqual(r["n_seeds"], 1)
        self.assertEqual(r["std"], "NA")
        self.assertEqual(r["mean"], "+2.0000")
        self.assertEqual(r["min"], r["max"])
        self.assertEqual(r["per_seed"], "42=+2.0000")

    def test_provisional_column_names_only_matching_sessions(self):
        self.assertEqual(self.rows[("condition-2-es-20k", "sib200", "ALL", "es")]["provisional"], "123")
        self.assertEqual(self.rows[("condition-5-ur-5k", "xnli", "ALL", "ur")]["provisional"], "no")

    def test_provisional_is_per_session_not_per_seed(self):
        # seed 123 of a 5k condition is a Phase-3 result; the 20k pattern must not touch it
        df = _cells([
            ("condition-2-ur-5k", "42", "1", "xnli", "ur", "ur", 0.01),
            ("condition-2-ur-5k", "123", "1", "xnli", "ur", "ur", 0.02),
            ("condition-2-ur-20k", "42", "1", "xnli", "ur", "ur", 0.01),
            ("condition-2-ur-20k", "123", "1", "xnli", "ur", "ur", 0.02),
        ])
        rows = {r["condition"]: r for r in xsv.build_rows(df, {"*-20k/seed123"}) if r["data_lang"] == "ALL"}
        self.assertEqual(rows["condition-2-ur-5k"]["provisional"], "no")
        self.assertEqual(rows["condition-2-ur-20k"]["provisional"], "123")

    def test_baseline_is_excluded_and_row_order_puts_all_first(self):
        conds = {k[0] for k in self.rows}
        self.assertNotIn("baseline", conds)
        ordered = xsv.build_rows(self.df, set())
        es = [r["data_lang"] for r in ordered if r["condition"] == "condition-2-es-20k"]
        self.assertEqual(es, ["ALL", "en", "es"])

    def test_negative_zero_prints_as_positive_zero(self):
        df = _cells([
            ("c", "42", "1", "xnli", "en", "en", 0.0),
            ("c", "42", "2", "xnli", "en", "en", -0.0),
        ])
        r = xsv.build_rows(df, set())[0]
        self.assertEqual(r["mean"], "+0.0000")

    def test_tsv_round_trips_with_declared_columns(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.tsv"
            xsv.write_tsv(xsv.build_rows(self.df, {"*-20k/seed123"}), out)
            back = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
            self.assertEqual(list(back.columns), xsv.TSV_COLUMNS)
            self.assertEqual(len(back), 5)  # es-20k: ALL, en, es; cond-5: ALL, ur


class MarkdownTests(unittest.TestCase):
    def test_two_seed_rows_report_range_not_sd_and_mark_provisional(self):
        df = _cells([
            ("condition-2-es-20k", "42", "1", "sib200", "es", "es", -0.10),
            ("condition-2-es-20k", "42", "2", "sib200", "es", "es", -0.06),
            ("condition-2-es-20k", "123", "1", "sib200", "es", "es", -0.02),
            ("condition-2-es-20k", "123", "2", "sib200", "es", "es", -0.04),
        ])
        md = xsv.build_markdown(df, {"condition-2-es-20k/seed123"}, source="test")
        self.assertIn("| SIB-200 es x es diagonal | 2 (42, 123) | -8.00 / -3.00* | mean -5.50, range -8.00 to -3.00 |", md)
        self.assertIn("**Provisional:** 1 session(s) (condition-2-es-20k/seed123)", md)
        self.assertNotIn("+/-", md.split("## condition-2-es-20k")[1])

    def test_three_seed_rows_report_mean_plus_minus_sd(self):
        df = _cells([
            ("condition-2-ur-5k", s, "1", "belebele", "ur", "ur", v)
            for s, v in (("42", 0.06), ("123", 0.05), ("456", 0.04))
        ])
        md = xsv.build_markdown(df, set(), source="test")
        self.assertIn("| Belebele ur x ur diagonal | 3 (42, 123, 456) | +6.00 / +5.00 / +4.00 | +5.00 +/- 1.00 |", md)
        self.assertNotIn("Provisional", md)


if __name__ == "__main__":
    unittest.main()

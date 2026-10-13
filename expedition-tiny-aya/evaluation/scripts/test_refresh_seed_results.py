"""Unit tests for refresh_seed_results.py.

Stdlib unittest; no network. The Hugging Face inventory is a dict built in
the test, the eval notebook is a small JSON fixture with the same shape as
evaluate.ipynb's EVAL_MATRIX cell, and the results files are in-memory
dicts carrying Urdu and Chinese inputs. These pin: the ladder is read from
the notebook (not a frozen list) and unregistered seeds on the Hub are
reported, each session state is classified from what is actually present,
the status diff names what changed, and the seed-vs-reference check fails
on a missing cell, a row-count change, a changed input and a changed gold
label.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refresh_seed_results as rsr  # noqa: E402

NOTEBOOK_CELL = '''
LORA_REPO = "legesher/language-decoded-lora"
BASE_NS   = "tiny-aya-base"

EVAL_MATRIX = {
    "baseline": {
        "repo": "CohereLabs/tiny-aya-base",
        "subfolder_template": None,
        "seeds": [None],
        "dataset_langs": ["en", "zh", "es", "ur"],
        "instruction_langs": ["en", "zh", "es", "ur"],
    },
    "condition-2-ur-20k": {
        "repo": LORA_REPO,
        "subfolder_template": f"{BASE_NS}/condition-2-ur-20k-seed{{seed}}",
        "seeds": [42, 123],
        "dataset_langs": ["en", "zh", "es", "ur"],
        "instruction_langs": ["en", "ur"],
    },
    "condition-3-zh-5k": {
        "repo": LORA_REPO,
        "subfolder_template": f"{BASE_NS}/condition-3-zh-5k-native-code-seed{{seed}}",
        "seeds": [42],
        "dataset_langs": ["en", "zh", "es", "ur"],
        "instruction_langs": ["en", "zh"],
    },
}
'''


def _notebook(tmp: Path) -> Path:
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# EVAL_MATRIX = { in prose must be ignored"]},
        {"cell_type": "code", "source": NOTEBOOK_CELL.splitlines(keepends=True)},
    ]}
    p = tmp / "evaluate.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def _session_files(cond: str, seed: str, *, results=True, summary=True, reparsed=False) -> set[str]:
    sp = rsr.session_paths(cond, seed)
    files: set[str] = set()
    if results:
        files |= set(sp["results"])
    if summary:
        files |= set(sp["summary"])
    if reparsed:
        files |= set(sp["reparsed"])
    return files


class LadderTests(unittest.TestCase):
    def test_eval_matrix_is_read_from_the_notebook(self):
        with tempfile.TemporaryDirectory() as d:
            m = rsr.load_eval_matrix(_notebook(Path(d)))
        self.assertEqual(m["baseline"]["seeds"], ["none"])
        self.assertEqual(m["condition-2-ur-20k"]["seeds"], ["42", "123"])
        self.assertEqual(
            m["condition-3-zh-5k"]["subfolder_template"],
            "tiny-aya-base/condition-3-zh-5k-native-code-seed{seed}",
        )

    def test_expected_cells_follow_the_registered_languages(self):
        entry = {"dataset_langs": ["en", "ur"], "instruction_langs": ["ur"]}
        cells = rsr.expected_cells(entry)
        self.assertEqual(len(cells), 2 * 4 * 2 * 1)
        self.assertIn("template2_sib200_data=ur_instr=ur", cells)
        self.assertNotIn("template1_sib200_data=ur_instr=en", cells)


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as d:
            self.matrix = rsr.load_eval_matrix(_notebook(Path(d)))

    def _inv(self, exp: set[str], lora: set[str], prs: dict | None = None) -> dict:
        return {"experiments_main": exp, "lora_main": lora, "open_prs": prs or {}}

    def test_complete_needs_reparse_awaiting_eval_awaiting_adapter(self):
        lora = {"tiny-aya-base/condition-2-ur-20k-seed42/adapter_config.json",
                "tiny-aya-base/condition-2-ur-20k-seed123/adapter_config.json"}
        exp = _session_files("condition-2-ur-20k", "42", reparsed=True)
        inv = self._inv(exp, lora)
        e = self.matrix["condition-2-ur-20k"]
        self.assertEqual(rsr.classify_session("condition-2-ur-20k", "42", e, inv, True, set())["state"], "complete")
        self.assertEqual(rsr.classify_session("condition-2-ur-20k", "123", e, inv, True, set())["state"], "awaiting eval")
        inv2 = self._inv(exp | _session_files("condition-2-ur-20k", "123"), lora)
        r = rsr.classify_session("condition-2-ur-20k", "123", e, inv2, True, {"*-20k/seed123"})
        self.assertEqual(r["state"], "needs reparse")
        self.assertTrue(r["provisional"])
        r42 = rsr.classify_session("condition-2-ur-20k", "42", e, inv2, True, {"*-20k/seed123"})
        self.assertFalse(r42["provisional"])
        inv3 = self._inv(exp, set())
        self.assertEqual(rsr.classify_session("condition-2-ur-20k", "123", e, inv3, True, set())["state"], "awaiting adapter")

    def test_open_pr_carrying_the_files_is_reported(self):
        e = self.matrix["condition-2-ur-20k"]
        lora = {"tiny-aya-base/condition-2-ur-20k-seed123/adapter_config.json"}
        prs = {57: {"title": "reparsed", "added": _session_files("condition-2-ur-20k", "123", results=False, summary=False, reparsed=True), "note": ""}}
        inv = self._inv(_session_files("condition-2-ur-20k", "123"), lora, prs)
        r = rsr.classify_session("condition-2-ur-20k", "123", e, inv, True, set())
        self.assertEqual(r["state"], "reparse PR #57 open")
        self.assertEqual(r["open_prs"], [57])

    def test_misnamed_cond3_adapter_is_detected(self):
        e = self.matrix["condition-3-zh-5k"]
        lora = {"tiny-aya-base/phase-2-the-stack-v1-condition-3-zh-5k-seed123/adapter_config.json"}
        inv = self._inv(set(), lora)
        r = rsr.classify_session("condition-3-zh-5k", "123", e, inv, False, set())
        self.assertEqual(r["adapter"], "misnamed (phase-2-the-stack-v1-condition-3-zh-5k-seed123)")
        self.assertEqual(r["state"], "adapter misnamed (unregistered)")

    def test_unregistered_seed_on_hub_appears_in_status(self):
        lora = {"tiny-aya-base/condition-3-zh-5k-native-code-seed123/adapter_config.json"}
        inv = self._inv(set(), lora)
        rows = rsr.build_status(self.matrix, inv, set())
        states = {(r["condition"], r["seed"]): r for r in rows}
        self.assertIn(("condition-3-zh-5k", "123"), states)
        self.assertFalse(states[("condition-3-zh-5k", "123")]["registered"])
        self.assertTrue(states[("condition-3-zh-5k", "123")]["state"].endswith("(unregistered)"))


class DiffStatusTests(unittest.TestCase):
    def test_diff_names_changed_fields_and_new_rows(self):
        prev = [{"condition": "c", "seed": "123", "adapter": "present", "results": "0/2", "summary": "0/2", "reparsed": "0/2", "state": "awaiting eval"}]
        cur = [
            {"condition": "c", "seed": "123", "adapter": "present", "results": "2/2", "summary": "2/2", "reparsed": "0/2", "state": "needs reparse"},
            {"condition": "c", "seed": "456", "adapter": "present", "results": "0/2", "summary": "0/2", "reparsed": "0/2", "state": "awaiting eval"},
        ]
        out = rsr.diff_status(prev, cur)
        self.assertEqual(out[0], "c seed 123: results 0/2 -> 2/2; summary 0/2 -> 2/2; state awaiting eval -> needs reparse")
        self.assertEqual(out[1], "c seed 456: new row, awaiting eval")
        self.assertEqual(rsr.diff_status(cur, cur), ["no change since the last run"])


class VerifySessionTests(unittest.TestCase):
    CELL1 = "template1_sib200_data=ur_instr=ur"
    CELL2 = "template2_sib200_data=ur_instr=ur"

    def _rows(self, gold="سائنس/ٹیکنالوجی", text="یہ ایک جملہ ہے۔", n=2):
        return [{"text": text, "gold": gold, "raw_output": "科学/技术", "pred": "science/technology", "correct": True} for _ in range(n)]

    def _session(self, **kw):
        return {"template1": {"summary": {}, self.CELL1: self._rows(**kw)},
                "template2": {"summary": {}, self.CELL2: self._rows(**kw)}}

    def _reparsed(self, cells=(CELL1, CELL2)):
        return {t: {"summary": {f"{c}_acc": 1.0 for c in cells if c.startswith(t)}} for t in ("template1", "template2")}

    def test_identical_sessions_pass(self):
        exp = {self.CELL1, self.CELL2}
        self.assertEqual(rsr.verify_session(self._session(), self._session(), exp, self._reparsed()), [])

    def test_changed_gold_is_a_problem(self):
        exp = {self.CELL1, self.CELL2}
        new = self._session()
        new["template1"][self.CELL1][1]["gold"] = "سیاست"
        problems = rsr.verify_session(new, self._session(), exp)
        self.assertEqual(problems, [f"{self.CELL1}: gold labels differ from reference"])

    def test_changed_input_text_is_a_problem(self):
        exp = {self.CELL1, self.CELL2}
        new = self._session()
        new["template2"][self.CELL2][0]["text"] = "这是另一个句子。"
        self.assertEqual(rsr.verify_session(new, self._session(), exp), [f"{self.CELL2}: input text differs from reference"])

    def test_row_count_and_missing_cell_are_problems(self):
        exp = {self.CELL1, self.CELL2, "template1_xnli_data=zh_instr=ur"}
        new = self._session(n=3)
        problems = rsr.verify_session(new, self._session(), exp)
        self.assertIn(f"{self.CELL1}: 3 rows vs 2 in reference", problems)
        self.assertIn("template1_xnli_data=zh_instr=ur: missing in new session", problems)

    def test_reparsed_summary_missing_a_cell_is_a_problem(self):
        exp = {self.CELL1, self.CELL2}
        problems = rsr.verify_session(self._session(), self._session(), exp, self._reparsed(cells=(self.CELL1,)))
        self.assertEqual(problems, [f"{self.CELL2}: no accuracy in reparsed summary"])


if __name__ == "__main__":
    unittest.main()

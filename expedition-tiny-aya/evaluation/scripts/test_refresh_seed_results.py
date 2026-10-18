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
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        "subfolder_template": f"{BASE_NS}/condition-3-zh-5k-seed{{seed}}",
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
            "tiny-aya-base/condition-3-zh-5k-seed{seed}",
        )

    def test_real_notebook_ladder_matches_the_tables_condition_list(self):
        """The ladder the refresh reads and the condition list the LaTeX tables
        render must name the same conditions; a condition added to one and not
        the other would be silently dropped from status or from the tables."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "analysis" / "scripts"))
        import build_phase3_tables as tables  # noqa: E402

        matrix = rsr.load_eval_matrix()
        self.assertEqual(set(matrix), {c for c, _ in tables.CONDITIONS})
        for cond, entry in matrix.items():
            self.assertTrue(entry["seeds"], cond)
            self.assertEqual(len(rsr.expected_cells(entry)) % 8, 0, cond)

    def test_real_notebook_adapter_folders_follow_the_naming_rule(self):
        """Every adapter folder is ``{data config}-seed{seed}``, the name the
        training notebook uploads under, so no seed needs a rename on upload.
        The data config is the condition key, plus the generator-model slug for
        condition 5. A folder carrying any other suffix (cond-3 carried
        ``-native-code`` until October 2026) breaks the eval load."""
        analysis_scripts = str(Path(__file__).resolve().parent.parent.parent / "analysis" / "scripts")
        with mock.patch.object(sys, "path", [analysis_scripts, *sys.path]):
            import build_cond5_idx_ledger as cond5  # noqa: E402

        matrix = rsr.load_eval_matrix()
        adapters = {c: e for c, e in matrix.items() if e["subfolder_template"] is not None}
        # Guard against a vacuous pass: both rule branches must be exercised.
        self.assertTrue(any(c.startswith("condition-5-") for c in adapters), sorted(matrix))
        self.assertTrue(any(not c.startswith("condition-5-") for c in adapters), sorted(matrix))
        for cond, entry in adapters.items():
            tmpl = entry["subfolder_template"]
            with self.subTest(condition=cond):
                base_ns, folder = tmpl.split("/")
                self.assertEqual(base_ns, "tiny-aya-base")
                self.assertTrue(folder.endswith("-seed{seed}"), folder)
                data_config = folder.removesuffix("-seed{seed}")
                allowed = {cond}
                if cond.startswith("condition-5-"):
                    allowed = {f"{cond}-{cond5.MODEL_SLUG}"}
                self.assertIn(data_config, allowed)

    def test_picker_lists_the_registered_seeds_and_the_readme_never_runs_ahead(self):
        """The eval notebook's picker comment is what the person at Kaggle
        reads, so it lists exactly the seeds EVAL_MATRIX registers. The
        evaluation README's matrix describes seeds with results and is updated
        when they land, so it may trail EVAL_MATRIX (a seed registered ahead of
        its runs) but never list a seed the notebook does not register, and its
        session total matches its own table."""
        matrix = rsr.load_eval_matrix()
        nb = json.loads(rsr.NOTEBOOK.read_text(encoding="utf-8"))
        picker_src = next(
            "".join(c["source"]) for c in nb["cells"]
            if "# Available conditions and their registered seeds" in "".join(c["source"])
        )
        picker_rows = [
            (m.group(1), [s.strip() for s in m.group(2).split(",")])
            for m in re.finditer(r"^#\s+(baseline|condition-\S+)\s+seeds?: (.+)$", picker_src, re.M)
        ]
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
        table_rows = [
            (m.group(1), [s.strip() for s in m.group(2).split(",")])
            for m in re.finditer(r"^\|\s*(condition-\S+)\s*\|\s*([0-9, ]+?)\s*\|", readme, re.M)
        ]
        # A second row for one condition would otherwise win silently.
        self.assertEqual(len(picker_rows), len({c for c, _ in picker_rows}), picker_rows)
        self.assertEqual(len(table_rows), len({c for c, _ in table_rows}), table_rows)
        picker, table = dict(picker_rows), dict(table_rows)
        self.assertEqual(set(picker), set(matrix))
        self.assertEqual(set(table), set(matrix) - {"baseline"})
        for cond, entry in matrix.items():
            with self.subTest(condition=cond):
                self.assertEqual(picker[cond], entry["seeds"])
                if cond != "baseline":
                    self.assertTrue(set(table[cond]) <= set(entry["seeds"]), f"README lists {table[cond]}, EVAL_MATRIX {entry['seeds']}")
        sweep = re.search(r"the full sweep is \*\*(\d+) Kaggle sessions\*\*", readme)
        self.assertIsNotNone(sweep, "evaluation README states no session count")
        assert sweep is not None
        self.assertEqual(int(sweep.group(1)), 1 + sum(len(s) for s in table.values()), "README session total vs its own table")

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

    def test_misnamed_unregistered_cond3_adapter_surfaces_in_status(self):
        lora = {"tiny-aya-base/phase-2-the-stack-v1-condition-3-zh-5k-seed123/adapter_config.json"}
        inv = self._inv(set(), lora)
        rows = {(r["condition"], r["seed"]): r for r in rsr.build_status(self.matrix, inv, set())}
        self.assertIn(("condition-3-zh-5k", "123"), rows)
        r = rows[("condition-3-zh-5k", "123")]
        self.assertEqual(r["adapter"], "misnamed (phase-2-the-stack-v1-condition-3-zh-5k-seed123)")
        self.assertEqual(r["state"], "adapter misnamed (unregistered)")
        # the properly named seed-42 folder of the same condition is not mistaken for seed 123
        self.assertEqual(rows[("condition-3-zh-5k", "42")]["adapter"], "missing")

    def test_results_only_in_a_pr_are_not_reparsed_from_main(self):
        e = self.matrix["condition-2-ur-20k"]
        lora = {"tiny-aya-base/condition-2-ur-20k-seed123/adapter_config.json"}
        prs = {9: {"title": "eval", "added": _session_files("condition-2-ur-20k", "123", reparsed=True), "note": ""}}
        inv = self._inv(set(), lora, prs)
        r = rsr.classify_session("condition-2-ur-20k", "123", e, inv, True, set())
        self.assertEqual(r["state"], "results PR #9 open")

    def test_reference_seed_prefers_42_then_any_complete_other_seed(self):
        e = {"seeds": ["42", "123", "456"]}
        rows = [
            {"condition": "c", "seed": "42", "results": "2/2"},
            {"condition": "c", "seed": "123", "results": "2/2"},
            {"condition": "c", "seed": "456", "results": "0/2"},
        ]
        self.assertEqual(rsr.reference_seed("c", "123", rows, e), "42")
        self.assertEqual(rsr.reference_seed("c", "42", rows, e), "123")
        self.assertIsNone(rsr.reference_seed("c", "42", [rows[0]], e))

    def test_unregistered_seed_on_hub_appears_in_status(self):
        lora = {"tiny-aya-base/condition-3-zh-5k-seed123/adapter_config.json"}
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
        self.assertEqual(rsr.verify_session(new, self._session(), exp), [f"{self.CELL2}: input fields differ from reference"])

    def test_row_count_and_missing_cell_are_problems(self):
        exp = {self.CELL1, self.CELL2, "template1_xnli_data=zh_instr=ur"}
        new = self._session(n=3)
        problems = rsr.verify_session(new, self._session(), exp)
        self.assertIn(f"{self.CELL1}: 3 rows vs 2 in reference", problems)
        self.assertIn("template1_xnli_data=zh_instr=ur: missing in new session", problems)

    def test_input_fields_are_compared_for_every_benchmark_not_only_sib200_text(self):
        # XNLI rows carry premise/hypothesis, Belebele rows passage/question: no "text" key.
        xnli = "template1_xnli_data=ur_instr=ur"
        beleb = "template2_belebele_data=zh_instr=zh"
        def rows_x(premise):
            return [{"premise": premise, "hypothesis": "वह घर पर है।", "gold": "entailment", "raw_output": "x", "pred": "x", "correct": False}]
        def rows_b(passage):
            return [{"passage": passage, "question": "作者的观点是什么？", "gold": "B", "raw_output": "B", "pred": "B", "correct": True}]
        ref = {"template1": {xnli: rows_x("वह घर गया।")}, "template2": {beleb: rows_b("这是一段文字。")}}
        same = {"template1": {xnli: rows_x("वह घर गया।")}, "template2": {beleb: rows_b("这是一段文字。")}}
        self.assertEqual(rsr.verify_session(same, ref, {xnli, beleb}), [])
        changed = {"template1": {xnli: rows_x("वह बाज़ार गया।")}, "template2": {beleb: rows_b("这是另一段文字。")}}
        self.assertEqual(
            rsr.verify_session(changed, ref, {xnli, beleb}),
            [f"{xnli}: input fields differ from reference", f"{beleb}: input fields differ from reference"],
        )

    def test_without_a_reference_only_the_inventory_is_checked(self):
        exp = {self.CELL1, self.CELL2, "template1_xnli_data=zh_instr=ur"}
        problems = rsr.verify_session(self._session(), None, exp, self._reparsed())
        self.assertEqual(problems, ["template1_xnli_data=zh_instr=ur: missing in new session", "template1_xnli_data=zh_instr=ur: no accuracy in reparsed summary"])

    def test_reparsed_summary_missing_a_cell_is_a_problem(self):
        exp = {self.CELL1, self.CELL2}
        problems = rsr.verify_session(self._session(), self._session(), exp, self._reparsed(cells=(self.CELL1,)))
        self.assertEqual(problems, [f"{self.CELL2}: no accuracy in reparsed summary"])


class AuditCheckTests(unittest.TestCase):
    """The pure checks behind ``refresh_seed_results.py audit``."""

    CELL = "template1_sib200_data=ur_instr=zh"

    def _results(self, n=3):
        return {"template1": {self.CELL: [{"text": "वह घर गया।", "gold": "عِلْمٌ", "raw_output": "科学"}] * n}}

    def _reparsed(self, correct=2, count=3, acc=None, fail=0.0, sha="a" * 64):
        return {"template1": {
            "summary": {f"{self.CELL}_count": count, f"{self.CELL}_correct": correct, f"{self.CELL}_acc": correct / count if acc is None else acc},
            "parse_failure_rates": {self.CELL: fail},
            "reparse_metadata": {"extractor_provenance": {"content_sha256": sha}},
        }}

    def test_counts_pass_when_consistent(self):
        self.assertEqual(rsr.audit_counts(self._results(), self._reparsed()), [])

    def test_count_against_rows_is_checked(self):
        self.assertEqual(
            rsr.audit_counts(self._results(n=4), self._reparsed()),
            [f"{self.CELL}: reparsed count 3 vs 4 rows in the results file"],
        )

    def test_accuracy_against_correct_over_count_is_checked(self):
        problems = rsr.audit_counts(self._results(), self._reparsed(acc=0.9))
        self.assertEqual(problems, [f"{self.CELL}: accuracy 0.9 does not equal correct/count 2/3"])

    def _adapter(self, **over):
        cfg = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none", "base_model_name_or_path": "CohereLabs/tiny-aya-base",
               "target_modules": ["q_proj", "v_proj"], "peft_version": "0.18.1"}
        return {**cfg, **over}

    def _metrics(self, step=1125, loss=1.0861, seed=None):
        m = {"train_result": {"train_loss": loss}, "log_history": [{"loss": 1.2}, {"step": step} if step is not None else {}]}
        if seed is not None:
            m["seed"] = seed
        return m

    def test_adapter_passes_with_only_the_stack_changed(self):
        problems, info = rsr.audit_adapter(
            self._adapter(), self._adapter(peft_version="0.20.0", target_modules=["v_proj", "q_proj"]),
            self._metrics(), self._metrics(loss=1.0864, seed=123), new_seed="123",
        )
        self.assertEqual(problems, [])
        self.assertEqual(info, {"loss": (1.0861, 1.0864), "peft": ("0.18.1", "0.20.0")})

    def test_adapter_setting_and_step_differences_are_problems(self):
        problems, _ = rsr.audit_adapter(self._adapter(), self._adapter(r=8), self._metrics(), self._metrics(step=1124))
        self.assertEqual(problems, ["adapter r: 8 vs 16 in the reference adapter", "training steps: 1124 vs 1125 in the reference adapter"])

    def test_missing_steps_and_a_wrong_seed_are_problems(self):
        problems, _ = rsr.audit_adapter(self._adapter(), self._adapter(), self._metrics(step=None), self._metrics(step=None, seed=42), new_seed="123")
        self.assertEqual(problems, ["training steps: missing from training_metrics.json", "training_metrics.json records seed 42, not 123"])

    def test_deltas_spread_and_parse_failure_rises(self):
        new = self._reparsed(correct=3, fail=0.40)
        ref = self._reparsed(correct=2, fail=0.10)
        d = rsr.refined_deltas(new, ref)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d[0], 100 / 3)
        self.assertEqual(rsr.spread([1.0, -3.0])["sd"], 2.0)
        self.assertEqual(rsr.spread([1.0, -3.0])["max_abs"], 3.0)
        self.assertEqual(rsr.spread([])["n"], 0)
        rises = rsr.parse_failure_rises(new, ref, 0.05)
        self.assertEqual([c for c, _ in rises], [self.CELL])
        self.assertAlmostEqual(rises[0][1], 0.30)
        self.assertEqual(rsr.parse_failure_rises(new, ref, 0.35), [])

    def test_extractor_sha_and_rescore_differences(self):
        self.assertEqual(rsr.extractor_shas(self._reparsed(sha="b" * 64)), {"b" * 64})
        no_meta = {"template1": {"summary": {}}}
        self.assertEqual(rsr.extractor_shas(no_meta), {"missing"})
        pub = self._reparsed()["template1"]
        self.assertEqual(rsr.rescore_differences(pub, json.loads(json.dumps(pub))), [])
        changed = json.loads(json.dumps(pub))
        changed["summary"][f"{self.CELL}_correct"] = 1
        changed["parse_failure_rates"][self.CELL] = 0.5
        diffs = rsr.rescore_differences(pub, changed)
        self.assertEqual(len(diffs), 2)
        self.assertTrue(diffs[0].startswith(f"{self.CELL}_correct: published 2"))
        self.assertTrue(diffs[1].startswith(f"{self.CELL} parse-failure rate"))


class AuditSelectionTests(unittest.TestCase):
    def test_only_conditions_registering_both_seeds_are_audited(self):
        """On the real ladder every adapter condition registering both seeds is
        selected; on a copy with the reference seed removed from one
        condition, that condition is dropped and naming it is an error."""
        import copy

        matrix = rsr.load_eval_matrix()
        both = {c for c, e in matrix.items() if e["subfolder_template"] and {"42", "123"} <= set(e["seeds"])}
        self.assertTrue(both)
        self.assertEqual(set(rsr.audit_conditions(matrix, "123", "42")), both)
        victim = sorted(both)[0]
        trimmed = copy.deepcopy(matrix)
        trimmed[victim]["seeds"] = [s for s in trimmed[victim]["seeds"] if s != "42"]
        self.assertEqual(set(rsr.audit_conditions(trimmed, "123", "42")), both - {victim})
        with self.assertRaises(SystemExit):
            rsr.audit_conditions(trimmed, "123", "42", [victim])


def _two_seed_condition() -> str:
    """The first real ladder condition registering seeds 42 and 123."""
    return next(c for c, e in rsr.load_eval_matrix().items() if e["subfolder_template"] and {"42", "123"} <= set(e["seeds"]))


class AuditWiringTests(unittest.TestCase):
    """``Refresh.audit`` end to end against a stubbed Hub: one real ladder
    condition, its expected cells built from the real eval matrix. Inputs are
    Devanagari and gold labels vocalised Arabic, so every JSON round trip
    carries combining marks."""

    TEXT = "वह घर गया।"
    GOLD = "عِلْمٌ"

    def setUp(self):
        import reparse_results

        self.reparse_results = reparse_results
        self.sha = reparse_results._extractor_provenance()["content_sha256"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.matrix = rsr.load_eval_matrix()
        self.cond = _two_seed_condition()
        self.cells = sorted(rsr.expected_cells(self.matrix[self.cond]))
        self.files: dict = {}
        for seed in ("42", "123"):
            sp = rsr.session_paths(self.cond, seed)
            for path, t in zip(sp["results"], rsr.TEMPLATES):
                self.files[path] = {c: [{"text": self.TEXT, "gold": self.GOLD, "raw_output": "عِلْم"}] * 2
                                    for c in self.cells if c.startswith(t + "_")}
            for path in sp["summary"]:
                self.files[path] = {"summary": {}, "parse_failure_rates": {}}
            for path, t in zip(sp["reparsed"], rsr.TEMPLATES):
                cells = [c for c in self.cells if c.startswith(t + "_")]
                summary = {}
                for c in cells:
                    summary.update({f"{c}_acc": 0.5, f"{c}_count": 2, f"{c}_correct": 1})
                self.files[path] = {"summary": summary, "parse_failure_rates": {c: 0.0 for c in cells},
                                    "reparse_metadata": {"extractor_provenance": {"content_sha256": self.sha}}}
        self.lora: dict = {}
        for seed in ("42", "123"):
            folder = self.matrix[self.cond]["subfolder_template"].format(seed=seed)
            self.lora[f"{folder}/adapter_config.json"] = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"], "peft_version": "0.18.1"}
            self.lora[f"{folder}/training_metrics.json"] = {"seed": int(seed), "train_result": {"train_loss": 1.07}, "log_history": [{"step": 1125}]}
        self.cells_rows = len(self.cells)
        self.xsv: str | None = None

    def _write(self, base: Path, path: str, body) -> Path:
        p = base / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if isinstance(body, str) else json.dumps(body, ensure_ascii=False), encoding="utf-8")
        return p

    def _run(self, drop: tuple[str, ...] = (), drop_lora: tuple[str, ...] = (), provisional=(), conditions=None, seed="123") -> tuple[int, str]:
        import argparse

        args = argparse.Namespace(refresh_dir=str(self.root / "refresh"), provisional=list(provisional), seed=seed, reference_seed="42",
                                  conditions=[self.cond] if conditions is None else conditions, parse_failure_threshold=0.05)
        r = rsr.Refresh(args)
        exp_main = (set(self.files) | {f"{rsr.TABLES_PREFIX}/cells.tsv"}) - set(drop)
        r._inv = {"experiments_main": exp_main, "lora_main": set(self.lora) - set(drop_lora), "open_prs": {}}
        # seed before condition, plus rows that must not be counted: the seed-42
        # session and another condition's seed-123 session.
        cells_tsv = "seed\tcell\tcondition\n" + "".join(
            [f"123\t{i}\t{self.cond}\n" for i in range(self.cells_rows)]
            + [f"42\t{i}\t{self.cond}\n" for i in range(5)]
            + [f"123\t{i}\tcondition-x\n" for i in range(5)]
        )
        files = {**self.files, f"{rsr.TABLES_PREFIX}/cells.tsv": cells_tsv}
        r._download = lambda p, revision=None: self._write(self.root / "exp", p, files[p])  # type: ignore[method-assign]
        r._download_lora = lambda p: self._write(self.root / "lora", p, self.lora[p])  # type: ignore[method-assign]
        analysis = self.root / "analysis"
        analysis.mkdir(exist_ok=True)
        if self.xsv is not None:
            (analysis / "cross-seed-variance.tsv").write_text(self.xsv, encoding="utf-8")
        with mock.patch("sys.stdout"), mock.patch.object(rsr, "PHASE3_ANALYSIS", analysis):
            n = r.audit()
        return n, (self.root / "refresh" / f"audit-seed{seed}.md").read_text(encoding="utf-8")

    def _published(self, results_path: Path, seed: str) -> dict:
        """The reparsed summary of ``seed`` matching a downloaded results file."""
        template = results_path.name.rsplit("_results_", 1)[1].removesuffix(".json")
        path = next(p for p in rsr.session_paths(self.cond, seed)["reparsed"] if p.endswith(f"_{template}.json"))
        return json.loads(json.dumps(self.files[path]))

    def test_clean_batch_has_no_problems(self):
        n, report = self._run()
        self.assertEqual(n, 0, report)
        self.assertIn(f"| {self.cond} | no | {len(self.cells)} |", report)
        self.assertIn("0 problem(s) across 1 condition(s).", report)

    def test_every_missing_session_file_is_named(self):
        dropped = (rsr.session_paths(self.cond, "42")["reparsed"][1], rsr.session_paths(self.cond, "123")["summary"][0])
        n, report = self._run(drop=dropped)
        self.assertEqual(n, 2)
        for p in dropped:
            self.assertIn(f"missing on the experiments repo: {p}", report)

    def test_missing_adapter_file_fails(self):
        folder = self.matrix[self.cond]["subfolder_template"].format(seed="123")
        n, report = self._run(drop_lora=(f"{folder}/training_metrics.json",))
        self.assertEqual(n, 1)
        self.assertIn(f"missing on the LoRA repo: {folder}/training_metrics.json", report)

    def test_cells_table_counts_only_the_audited_session(self):
        self.cells_rows -= 1
        n, report = self._run()
        self.assertEqual(n, 1)
        self.assertIn(f"refined cells.tsv has {len(self.cells) - 1} rows for seed 123", report)

    def test_reparsed_count_disagreeing_with_rows_fails(self):
        path = rsr.session_paths(self.cond, "123")["reparsed"][0]
        key = next(k for k in self.files[path]["summary"] if k.endswith("_count"))
        self.files[path]["summary"][key] = 3
        n, report = self._run()
        self.assertEqual(n, 2, report)
        self.assertIn(f"{key[: -len('_count')]}: reparsed count 3 vs 2 rows in the results file", report)

    def test_changed_gold_and_adapter_settings_fail(self):
        first = rsr.session_paths(self.cond, "123")["results"][0]
        cell = next(iter(self.files[first]))
        self.files[first][cell] = [{"text": self.TEXT, "gold": "سِيَاسَةٌ", "raw_output": "عِلْم"}] * 2
        folder = self.matrix[self.cond]["subfolder_template"].format(seed="123")
        self.lora[f"{folder}/adapter_config.json"] = {"r": 8, "lora_alpha": 32, "target_modules": ["q_proj"], "peft_version": "0.18.1"}
        n, report = self._run()
        self.assertEqual(n, 2, report)
        self.assertIn(f"{cell}: gold labels differ from reference", report)
        self.assertIn("adapter r: 8 vs 16 in the reference adapter", report)

    def _rescore(self, seed: str, change: bool):
        for path in rsr.session_paths(self.cond, seed)["reparsed"]:
            self.files[path]["reparse_metadata"]["extractor_provenance"]["content_sha256"] = "0" * 64

        def build(path, rows):
            body = self._published(path, seed)
            if change:
                key = next(k for k in body["summary"] if k.endswith("_correct"))
                body["summary"][key] = 2
            return body

        with mock.patch.object(self.reparse_results, "reparse_file", lambda p: []), \
             mock.patch.object(self.reparse_results, "build_reparsed_summary", build):
            return self._run()

    def test_old_extractor_on_either_seed_is_rescored(self):
        for seed in ("42", "123"):
            with self.subTest(seed=seed):
                self.setUp()
                n, report = self._rescore(seed, change=False)
                self.assertEqual(n, 0, report)
                self.assertIn(f"seed {seed} was reparsed by extractor 000000000000", report)
                self.setUp()
                n, report = self._rescore(seed, change=True)
                self.assertEqual(n, 2, report)
                self.assertIn(f"seed {seed} rescored with the current extractor:", report)

    def test_provisional_sessions_leave_the_reference_and_are_cross_checked(self):
        self.xsv = "condition\tdata_lang\tn_seeds\tprovisional\n" + f"{self.cond}\tALL\t2\tno\n"
        n, report = self._run(provisional=[f"{self.cond}/seed123"])
        self.assertEqual(n, 0, report)
        self.assertIn(f"| {self.cond} | yes |", report)
        self.assertIn("Same-stack reference (conditions where neither session is provisional): 0 cells", report)
        self.assertIn("seed 123 matches --provisional but cross-seed-variance.tsv does not flag it", report)

    def test_unlabelled_provisional_session_is_noted(self):
        self.xsv = "condition\tdata_lang\tn_seeds\tprovisional\n" + f"{self.cond}\tALL\t2\t123\n"
        folder = self.matrix[self.cond]["subfolder_template"].format(seed="123")
        self.lora[f"{folder}/adapter_config.json"]["peft_version"] = "0.20.0"
        n, report = self._run()
        self.assertEqual(n, 0, report)
        self.assertIn("flags seed 123 provisional but it matches no --provisional pattern", report)
        self.assertIn("0 cells", report)

    def test_peft_change_without_provisional_patterns_is_noted(self):
        folder = self.matrix[self.cond]["subfolder_template"].format(seed="123")
        self.lora[f"{folder}/adapter_config.json"]["peft_version"] = "0.20.0"
        n, report = self._run()
        self.assertEqual(n, 0, report)
        self.assertIn("peft differs (0.18.1 vs 0.20.0) although neither session is provisional", report)

    def test_unusable_selections_exit_with_an_error(self):
        with self.assertRaises(SystemExit) as cm:
            self._run(conditions=["condition-9-xx-1k"])
        self.assertIn("not registered with seeds 42 and 123", str(cm.exception.code))
        with self.assertRaises(SystemExit) as cm:
            self._run(conditions=[], seed="999")
        self.assertIn("no condition in EVAL_MATRIX registers both seed 42 and seed 999", str(cm.exception.code))
        with self.assertRaises(SystemExit) as cm:
            self._run(seed="42")
        self.assertIn("--seed and --reference-seed are both 42", str(cm.exception.code))


class AuditExitCodeTests(unittest.TestCase):
    def _main(self, argv_tail: list[str], problems: int = 0):
        with tempfile.TemporaryDirectory() as d:
            argv = ["refresh_seed_results.py", "audit", "--refresh-dir", d, *argv_tail]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(rsr.Refresh, "audit", return_value=problems), \
                 mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
                rsr.main()
        return cm.exception.code

    def test_exit_code_is_non_zero_only_when_problems_are_found(self):
        self.assertEqual(self._main(["--seed", "123"], 0), 0)
        self.assertEqual(self._main(["--seed", "123"], 3), 1)

    def test_audit_without_a_seed_is_a_usage_error(self):
        self.assertEqual(self._main([]), 2)


if __name__ == "__main__":
    unittest.main()

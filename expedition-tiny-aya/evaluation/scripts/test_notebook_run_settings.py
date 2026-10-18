"""Unit tests for the per-run settings in the training and evaluation notebooks.

Stdlib unittest; no network, no GPU. Each notebook has two values that must
change for every run (training: ``CONDITION_NAME`` and ``TRAIN_SEEDS``;
evaluation: ``CONDITION`` and ``SEED``). They ship empty, with examples in
comments, and the cell that holds them is the first code cell, so a notebook
run as it is stops within seconds instead of installing packages, training for
hours and uploading a run nobody asked for. These pin: the settings cell comes
before the install cell, it raises while either value is unset, and it runs
once both are filled in.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING = ROOT / "training" / "scripts" / "qlora.ipynb"
EVALUATION = ROOT / "evaluation" / "scripts" / "evaluate.ipynb"


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]


def _run(src: str) -> dict:
    ns: dict = {}
    with mock.patch.dict(os.environ, {}, clear=False):
        exec(compile(src, "<settings cell>", "exec"), ns)
    return ns


class RunSettingsTests(unittest.TestCase):
    def _settings(self, path: Path, marker: str) -> tuple[int, int, str]:
        cells = _code_cells(path)
        settings = next(i for i, s in enumerate(cells) if marker in s)
        install = next(i for i, s in enumerate(cells) if s.lstrip().startswith("!pip install"))
        return settings, install, cells[settings]

    def test_training_settings_come_first_and_stop_an_unedited_run(self):
        settings, install, src = self._settings(TRAINING, "CONDITION_NAME = ")
        self.assertEqual(settings, 0, "the training settings cell must be the first code cell")
        self.assertLess(settings, install)
        with self.assertRaisesRegex(ValueError, "CONDITION_NAME is not set"):
            _run(src)
        only_condition = src.replace("CONDITION_NAME = None\n", 'CONDITION_NAME = "condition-2-ur-20k"\n', 1)
        self.assertNotEqual(only_condition, src)
        with self.assertRaisesRegex(ValueError, "TRAIN_SEEDS is not set"):
            _run(only_condition)
        filled = only_condition.replace("TRAIN_SEEDS = []\n", "TRAIN_SEEDS = [456]\n", 1)
        self.assertNotEqual(filled, only_condition)
        with mock.patch("builtins.print"):
            ns = _run(filled)
        self.assertEqual(ns["CONDITION_NAME"], "condition-2-ur-20k")
        self.assertEqual(ns["TRAIN_SEED"], 456)

    def test_training_seeds_must_be_integers(self):
        _, _, src = self._settings(TRAINING, "CONDITION_NAME = ")
        named = src.replace("CONDITION_NAME = None\n", 'CONDITION_NAME = "condition-3-zh-5k"\n', 1)
        for bad in ('["456"]', "[True]", "[]"):
            with self.subTest(seeds=bad), self.assertRaisesRegex(ValueError, "TRAIN_SEEDS is not set"):
                _run(named.replace("TRAIN_SEEDS = []\n", f"TRAIN_SEEDS = {bad}\n", 1))

    def test_evaluation_settings_come_first_and_stop_an_unedited_run(self):
        settings, install, src = self._settings(EVALUATION, "CONDITION = ")
        self.assertEqual(settings, 0, "the evaluation picker cell must be the first code cell")
        self.assertLess(settings, install)
        with self.assertRaisesRegex(ValueError, "CONDITION is not set"):
            _run(src)
        only_condition = src.replace("CONDITION = None\n", 'CONDITION = "condition-3-zh-5k"\n', 1)
        self.assertNotEqual(only_condition, src)
        with self.assertRaisesRegex(ValueError, "SEED is not set"):
            _run(only_condition)
        for seed in ("123", '"none"'):
            with self.subTest(seed=seed):
                ns = _run(only_condition.replace("SEED = None\n", f"SEED = {seed}\n", 1))
                self.assertEqual(ns["CONDITION"], "condition-3-zh-5k")


if __name__ == "__main__":
    unittest.main()

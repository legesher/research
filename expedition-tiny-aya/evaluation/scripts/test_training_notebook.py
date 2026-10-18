"""Unit tests for the training notebook's upload step (training/scripts/qlora.ipynb).

Stdlib unittest; no network, no GPU. The notebook writes ``train.py`` from a
``%%writefile`` cell; these tests parse that cell and the config cell with
``ast``, and run two of its small functions against a stub Hub client. From
commit 37f5a29 (April 2026) until October 2026 the ``api.upload_folder(...)``
call sat inside a bare string, so a run trained an adapter and never
published it, and the evaluation notebook could not load it. These pin: the
upload is a live call on the main process, on by default and skipped only by
the switch, it keeps the folder contract the evaluation template reads, the
config cell's booleans survive the trip through the environment, a run stops
before training rather than overwrite a published adapter, and each run
records the package versions it trained with.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

NOTEBOOK = Path(__file__).resolve().parent.parent.parent / "training" / "scripts" / "qlora.ipynb"


def _cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]


def _writefile(name: str) -> str:
    for src in _cells():
        first, _, rest = src.partition("\n")
        if first.strip() == f"%%writefile {name}":
            return rest
    raise AssertionError(f"no %%writefile {name} cell in {NOTEBOOK.name}")


def _config_cell() -> str:
    for src in _cells():
        if not src.startswith("%%") and "CONDITION_NAME = " in src and "os.environ[" in src:
            return src
    raise AssertionError(f"no config cell in {NOTEBOOK.name}")


def _assignments(tree: ast.AST) -> dict[str, ast.expr]:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
    return out


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing_if_tests(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    tests = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            tests.append(ast.unparse(node.test))
    return tests


class TrainingUploadTests(unittest.TestCase):
    def setUp(self):
        self.train_src = _writefile("train.py")
        self.train = ast.parse(self.train_src)
        self.names = _assignments(self.train)
        self.parents = _parents(self.train)
        self.funcs = {n.name: n for n in self.train.body if isinstance(n, ast.FunctionDef)}

    def _calls(self, attr: str, root: ast.AST | None = None) -> list[ast.Call]:
        return [
            n for n in ast.walk(root or self.train)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
            and (n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id) == attr
        ]

    def _upload_call(self) -> ast.Call:
        calls = self._calls("upload_folder")
        self.assertEqual(len(calls), 1, "api.upload_folder(...) must be executable code in train.py")
        return calls[0]

    def _run_function(self, name: str, ns: dict) -> dict:
        segment = ast.get_source_segment(self.train_src, self.funcs[name])
        self.assertIsNotNone(segment, name)
        exec(segment or "", ns)
        return ns

    def test_upload_is_a_live_call_not_a_string(self):
        self._upload_call()
        strings = [n.value for n in ast.walk(self.train) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertFalse(any("upload_folder(" in s for s in strings), "upload_folder call found inside a string literal")

    def test_upload_runs_on_the_main_process_and_only_the_switch_skips_it(self):
        upload = self._upload_call()
        self.assertIn("is_main_process()", _enclosing_if_tests(upload, self.parents))
        main = self.funcs["main"]
        returns = [n for n in ast.walk(main) if isinstance(n, ast.Return) and n.lineno < upload.lineno]
        self.assertTrue(returns, "the UPLOAD_TO_HF = False path must return before the upload")
        for ret in returns:
            self.assertEqual(_enclosing_if_tests(ret, self.parents)[0], "not UPLOAD_TO_HF", ast.unparse(ret))

    def test_upload_keeps_the_folder_contract(self):
        kw = {k.arg: k.value for k in self._upload_call().keywords}
        self.assertEqual(ast.literal_eval(self.names["LORA_REPO"]), "legesher/language-decoded-lora")
        self.assertEqual(ast.unparse(kw["repo_id"]), "LORA_REPO")
        self.assertEqual(ast.literal_eval(kw["repo_type"]), "model")
        self.assertEqual(ast.literal_eval(kw["allow_patterns"]), ["adapter_*", "training_metrics.json"])
        self.assertEqual(ast.unparse(kw["path_in_repo"]), "UPLOAD_SUBFOLDER")
        self.assertEqual(ast.unparse(self.names["UPLOAD_SUBFOLDER"]), "f'{MODEL_CONFIG_NAME}/{OUTPUT_NAME}'")
        self.assertEqual(ast.unparse(self.names["OUTPUT_NAME"]), "f'{CONDITION_NAME}-seed{TRAIN_SEED}'")

    def test_switches_default_safely_and_are_exported(self):
        self.assertEqual(ast.unparse(self.names["UPLOAD_TO_HF"]), "read_bool_env('UPLOAD_TO_HF', True)")
        self.assertEqual(ast.unparse(self.names["OVERWRITE_HF_ADAPTER"]), "read_bool_env('OVERWRITE_HF_ADAPTER', False)")
        config_tree = ast.parse(_config_cell())
        config = _assignments(config_tree)
        self.assertIs(ast.literal_eval(config["UPLOAD_TO_HF"]), True)
        self.assertIs(ast.literal_eval(config["OVERWRITE_HF_ADAPTER"]), False)
        exports = {
            ast.unparse(n.targets[0]): ast.unparse(n.value)
            for n in ast.walk(config_tree)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
        }
        for name in ("UPLOAD_TO_HF", "OVERWRITE_HF_ADAPTER"):
            self.assertEqual(exports.get(f"os.environ['{name}']"), f"str({name})", name)

    def test_config_boolean_round_trips_through_the_environment(self):
        """The config cell exports ``str(True)`` ("True"); train.py must read it back as True."""
        ns = self._run_function("read_bool_env", {"os": os})
        for value in (True, False):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"UPLOAD_TO_HF": str(value)}):
                self.assertIs(ns["read_bool_env"]("UPLOAD_TO_HF", not value), value)

    def test_upload_target_is_checked_before_training(self):
        main = self.funcs["main"]
        checks = self._calls("check_upload_target", main)
        self.assertEqual(len(checks), 1)
        self.assertIn("UPLOAD_TO_HF", _enclosing_if_tests(checks[0], self.parents))
        trains = self._calls("train", main)
        self.assertEqual(len(trains), 1)
        self.assertLess(checks[0].lineno, trains[0].lineno)

    def test_existing_adapter_stops_the_run_unless_overwrite_is_set(self):
        class StubApi:
            exists = True
            seen: list = []

            def whoami(self, token=None):
                self.seen.append(("whoami", token))

            def file_exists(self, repo_id, filename, token=None):
                self.seen.append((repo_id, filename))
                return StubApi.exists

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = StubApi  # type: ignore[attr-defined]
        base = {"LORA_REPO": "legesher/language-decoded-lora", "UPLOAD_SUBFOLDER": "tiny-aya-base/condition-2-ur-5k-seed42"}
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            for exists, overwrite, should_raise in ((True, False, True), (True, True, False), (False, False, False)):
                with self.subTest(exists=exists, overwrite=overwrite):
                    StubApi.exists = exists
                    ns = self._run_function("check_upload_target", {**base, "OVERWRITE_HF_ADAPTER": overwrite})
                    if should_raise:
                        with self.assertRaisesRegex(RuntimeError, "already exists"):
                            ns["check_upload_target"]("tok")
                    else:
                        ns["check_upload_target"]("tok")
        self.assertIn(("whoami", "tok"), StubApi.seen)
        self.assertIn(("legesher/language-decoded-lora", "tiny-aya-base/condition-2-ur-5k-seed42/adapter_config.json"), StubApi.seen)

    def test_training_metrics_record_the_environment(self):
        payload = next(
            n.value for n in ast.walk(self.train)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "metrics_payload" for t in n.targets)
        )
        self.assertIsInstance(payload, ast.Dict)
        assert isinstance(payload, ast.Dict)
        keys = [ast.literal_eval(k) for k in payload.keys if k is not None]
        self.assertIn("environment", keys)
        self.assertEqual(ast.unparse(payload.values[keys.index("environment")]), "environment()")
        env_dicts = [n for n in ast.walk(self.funcs["environment"]) if isinstance(n, ast.Dict)]
        self.assertEqual(
            {ast.literal_eval(k) for k in env_dicts[0].keys if k is not None}, {"python", "cuda", "gpu", "packages"}
        )
        packages = ast.literal_eval(self.names["ENV_PACKAGES"])
        for pkg in ("unsloth", "transformers", "trl", "peft", "torch"):
            self.assertIn(pkg, packages)


if __name__ == "__main__":
    unittest.main()

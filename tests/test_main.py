"""Tests for src/__main__.py.

Note: get_vocabulary and write_output are imported directly from the module.
      Since src/__main__.py guards main() with `if __name__ == "__main__"`,
      importing it is safe and will not trigger execution.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.__main__ import (  # noqa: E402
    _generate_parallel,
    get_vocabulary,
    write_output,
)
from src.parsing import FunctionDef  # noqa: E402
from src.process import PrecomputedVocab  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(ids: list[int]) -> MagicMock:
    t = MagicMock()
    t.tolist.return_value = ids
    return t


def _func(
    name: str = "fn_test",
    desc: str = "Test.",
    params: dict[str, Any] | None = None,
    returns: str | None = "string",
) -> FunctionDef:
    return FunctionDef.model_validate(
        {
            "name": name,
            "description": desc,
            "parameters": params,
            "returns": returns,
        }
    )


def _small_pv() -> PrecomputedVocab:
    """Minimal PrecomputedVocab sufficient for test stubs."""
    decoded = {i: str(i) for i in range(10)}
    decoded[10] = '"'
    decoded[11] = ","
    return PrecomputedVocab.build(decoded)


# ---------------------------------------------------------------------------
# get_vocabulary
# ---------------------------------------------------------------------------


class TestGetVocabulary(unittest.TestCase):
    def _write_vocab(self, vocab: dict[str, int]) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(vocab, f)
        f.close()
        return f.name

    def test_loads_vocab_correctly(self) -> None:
        vocab = {"hello": 0, "world": 1, "<unk>": 2}
        path = self._write_vocab(vocab)
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = path
            result = get_vocabulary(model)
            self.assertEqual(result, vocab)
        finally:
            os.unlink(path)

    def test_returns_dict(self) -> None:
        path = self._write_vocab({"a": 0})
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = path
            self.assertIsInstance(get_vocabulary(model), dict)
        finally:
            os.unlink(path)

    def test_calls_get_path_to_vocab_file(self) -> None:
        path = self._write_vocab({})
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = path
            get_vocabulary(model)
            model.get_path_to_vocab_file.assert_called_once()
        finally:
            os.unlink(path)

    def test_large_vocab_loaded_completely(self) -> None:
        vocab = {str(i): i for i in range(1000)}
        path = self._write_vocab(vocab)
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = path
            result = get_vocabulary(model)
            self.assertEqual(len(result), 1000)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------


class TestWriteOutput(unittest.TestCase):
    def _sample_results(self) -> list[dict[str, Any]]:
        return [
            {
                "prompt": "Add 2 and 3",
                "name": "fn_add",
                "parameters": {"a": 2.0, "b": 3.0},
            },
            {
                "prompt": "Greet john",
                "name": "fn_greet",
                "parameters": {"name": "john"},
            },
        ]

    def test_creates_valid_json_file(self) -> None:
        results = self._sample_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, results)

    def test_creates_nested_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "output.json")
            with patch("builtins.print"):
                write_output([], path)
            self.assertTrue(os.path.exists(path))

    def test_empty_results_writes_empty_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output([], path)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])

    def test_preserves_float_type(self) -> None:
        results = [
            {"prompt": "p", "name": "fn", "parameters": {"a": 2.5}}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsInstance(
                loaded[0]["parameters"]["a"], float
            )

    def test_preserves_int_type(self) -> None:
        results = [
            {"prompt": "p", "name": "fn", "parameters": {"n": 3}}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsInstance(loaded[0]["parameters"]["n"], int)

    def test_preserves_string_type(self) -> None:
        results = [
            {
                "prompt": "p",
                "name": "fn",
                "parameters": {"s": "hello"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsInstance(loaded[0]["parameters"]["s"], str)

    def test_non_ascii_characters_preserved(self) -> None:
        results = [
            {
                "prompt": "héllo wörld",
                "name": "fn",
                "parameters": {"s": "café"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded[0]["prompt"], "héllo wörld")
            self.assertEqual(loaded[0]["parameters"]["s"], "café")

    def test_os_error_does_not_raise(self) -> None:
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            patch("os.makedirs"),
            patch("builtins.print"),
        ):
            write_output([], "/fake/path/output.json")

    def test_output_is_indented_json(self) -> None:
        results = [
            {"prompt": "p", "name": "fn", "parameters": {}}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            self.assertIn("\n", raw)

    def test_multiple_entries_all_written(self) -> None:
        results = self._sample_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            with patch("builtins.print"):
                write_output(results, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["name"], "fn_add")
            self.assertEqual(loaded[1]["name"], "fn_greet")


# ---------------------------------------------------------------------------
# _generate_parallel
# ---------------------------------------------------------------------------


class TestGenerateParallel(unittest.TestCase):
    """Test the parallel orchestrator.

    We mock the entire Pool so no real child processes are spawned
    and no model loading occurs.
    """

    def test_preserves_prompt_order(self) -> None:
        """Results must be reassembled in input order regardless of
        which worker finishes first."""
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]
        prompts = ["Greet alice", "Greet bob", "Greet charlie"]

        def fake_imap(fn: Any, batches: Any) -> list[Any]:
            """Simulate imap_unordered without calling the real
            worker (which would try to load the LLM model)."""
            all_results = []
            for batch_args in batches:
                indexed_prompts = batch_args[0]
                batch_results = [
                    (idx, {
                        "prompt": p,
                        "name": "fn_greet",
                        "parameters": {"name": p.split()[-1]},
                    })
                    for idx, p in indexed_prompts
                ]
                all_results.append(batch_results)
            return all_results

        with patch("builtins.print"):
            with patch("src.__main__.multiprocessing") as mock_mp:
                mock_ctx = MagicMock()
                mock_mp.get_context.return_value = mock_ctx
                mock_pool = MagicMock()
                mock_ctx.Pool.return_value.__enter__ = (
                    lambda self: mock_pool
                )
                mock_ctx.Pool.return_value.__exit__ = (
                    lambda self, *a: None
                )
                mock_pool.imap_unordered.side_effect = fake_imap

                results = _generate_parallel(
                    prompts, functions, "fake-model", True, 2,
                )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["prompt"], "Greet alice")
        self.assertEqual(results[1]["prompt"], "Greet bob")
        self.assertEqual(results[2]["prompt"], "Greet charlie")

    def test_single_prompt_uses_one_worker(self) -> None:
        """num_workers is capped at len(prompts)."""
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]

        def fake_imap(fn: Any, batches: Any) -> list[Any]:
            return [
                [(0, {
                    "prompt": "p",
                    "name": "fn_greet",
                    "parameters": {},
                })]
            ]

        with patch("builtins.print"):
            with patch("src.__main__.multiprocessing") as mock_mp:
                mock_ctx = MagicMock()
                mock_mp.get_context.return_value = mock_ctx
                mock_pool = MagicMock()
                mock_ctx.Pool.return_value.__enter__ = (
                    lambda self: mock_pool
                )
                mock_ctx.Pool.return_value.__exit__ = (
                    lambda self, *a: None
                )
                mock_pool.imap_unordered.side_effect = fake_imap

                results = _generate_parallel(
                    ["p"], functions, "fake-model", True, 4,
                )
                mock_ctx.Pool.assert_called_once_with(processes=1)

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()

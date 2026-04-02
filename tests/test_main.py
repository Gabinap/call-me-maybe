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

from src.__main__ import get_vocabulary, write_output  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

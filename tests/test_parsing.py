"""Tests for src/parsing.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.parsing import (
    Args,
    FunctionDef,
    command_parsing,
    functions_def_parsing,
    parse,
    prompt_parsing,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data) -> str:
    """Write data to a temp JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, f)
    f.close()
    return f.name


def _valid_func_data() -> dict:
    return {
        "name": "fn_add",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    }


# ---------------------------------------------------------------------------
# FunctionDef model
# ---------------------------------------------------------------------------


class TestFunctionDef(unittest.TestCase):
    def test_valid(self):
        fd = FunctionDef.model_validate(
            {
                "name": "fn_add",
                "description": "Add.",
                "parameters": None,
                "returns": "number",
            }
        )
        self.assertEqual(fd.name, "fn_add")
        self.assertEqual(fd.returns_, "number")
        self.assertIsNone(fd.parameters)

    def test_with_parameters(self):
        fd = FunctionDef.model_validate(
            {
                "name": "fn_add",
                "description": "Add.",
                "parameters": {"a": {"type": "number"}},
                "returns": "number",
            }
        )
        self.assertIn("a", fd.parameters)

    def test_empty_name_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            FunctionDef.model_validate(
                {
                    "name": "  ",
                    "description": "D.",
                    "parameters": None,
                    "returns": "number",
                }
            )

    def test_empty_description_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            FunctionDef.model_validate(
                {
                    "name": "fn_x",
                    "description": "",
                    "parameters": None,
                    "returns": "number",
                }
            )

    def test_none_returns_allowed(self):
        # returns_ is str | None — None should not raise
        fd = FunctionDef.model_validate(
            {
                "name": "fn_x",
                "description": "D.",
                "parameters": None,
                "returns": None,
            }
        )
        self.assertIsNone(fd.returns_)


# ---------------------------------------------------------------------------
# Args model
# ---------------------------------------------------------------------------


class TestArgs(unittest.TestCase):
    def test_defaults(self):
        args = Args()
        self.assertEqual(args.mode, "thinking")
        self.assertEqual(args.prompts, [])
        self.assertEqual(args.functions, [])

    def test_fast_mode(self):
        self.assertEqual(Args(mode="fast").mode, "fast")

    def test_thinking_mode(self):
        self.assertEqual(Args(mode="thinking").mode, "thinking")

    def test_invalid_mode_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            Args(mode="turbo")

    def test_default_paths(self):
        args = Args()
        self.assertIn("input", args.input)
        self.assertIn("output", args.output)
        self.assertIn("functions", args.functions_definition)

    def test_custom_paths(self):
        args = Args(input="my/input.json", output="my/output.json")
        self.assertEqual(args.input, "my/input.json")
        self.assertEqual(args.output, "my/output.json")


# ---------------------------------------------------------------------------
# prompt_parsing
# ---------------------------------------------------------------------------


class TestPromptParsing(unittest.TestCase):
    def test_valid_prompts(self):
        path = _write_json([{"prompt": "Greet john"}, {"prompt": "Add 2 and 3"}])
        try:
            self.assertEqual(prompt_parsing(path), ["Greet john", "Add 2 and 3"])
        finally:
            os.unlink(path)

    def test_empty_list(self):
        path = _write_json([])
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        self.assertEqual(prompt_parsing("does_not_exist_xyz.json"), [])

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not valid json")
            path = f.name
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_not_a_list(self):
        path = _write_json({"prompt": "hello"})
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_prompt_too_long(self):
        path = _write_json([{"prompt": "x" * 301}, {"prompt": "short"}])
        try:
            self.assertEqual(prompt_parsing(path), ["short"])
        finally:
            os.unlink(path)

    def test_prompt_at_boundary_accepted(self):
        path = _write_json([{"prompt": "x" * 299}])
        try:
            self.assertEqual(len(prompt_parsing(path)), 1)
        finally:
            os.unlink(path)

    def test_skips_non_dict_items(self):
        path = _write_json(["not a dict", {"prompt": "valid"}])
        try:
            self.assertEqual(prompt_parsing(path), ["valid"])
        finally:
            os.unlink(path)

    def test_missing_prompt_key_skipped(self):
        path = _write_json([{"text": "no prompt key"}, {"prompt": "ok"}])
        try:
            # item with no "prompt" key returns "" which has len < 300 — accepted
            result = prompt_parsing(path)
            self.assertIn("ok", result)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# functions_def_parsing
# ---------------------------------------------------------------------------


class TestFunctionsDefParsing(unittest.TestCase):
    def test_valid_single_function(self):
        path = _write_json([_valid_func_data()])
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "fn_add")
            self.assertEqual(result[0].returns_, "number")
        finally:
            os.unlink(path)

    def test_valid_multiple_functions(self):
        data = [
            {
                "name": "fn_a",
                "description": "A.",
                "parameters": None,
                "returns": {"type": "string"},
            },
            {
                "name": "fn_b",
                "description": "B.",
                "parameters": None,
                "returns": {"type": "integer"},
            },
        ]
        path = _write_json(data)
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].name, "fn_a")
            self.assertEqual(result[1].name, "fn_b")
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        self.assertEqual(functions_def_parsing("no_such_file_xyz.json"), [])

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{bad json")
            path = f.name
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_not_a_list(self):
        path = _write_json({"name": "fn_a"})
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_missing_returns_key(self):
        path = _write_json([{"name": "fn_a", "description": "A.", "parameters": None}])
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_returns_not_a_dict(self):
        path = _write_json(
            [
                {
                    "name": "fn_a",
                    "description": "A.",
                    "parameters": None,
                    "returns": "number",
                }
            ]
        )
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_returns_type_empty_string(self):
        path = _write_json(
            [
                {
                    "name": "fn_a",
                    "description": "A.",
                    "parameters": None,
                    "returns": {"type": "  "},
                }
            ]
        )
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_returns_type_missing(self):
        path = _write_json(
            [
                {
                    "name": "fn_a",
                    "description": "A.",
                    "parameters": None,
                    "returns": {},
                }
            ]
        )
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_non_dict_items_keeps_valid(self):
        data = [
            "not a dict",
            {
                "name": "fn_ok",
                "description": "OK.",
                "parameters": None,
                "returns": {"type": "string"},
            },
        ]
        path = _write_json(data)
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "fn_ok")
        finally:
            os.unlink(path)

    def test_empty_list(self):
        path = _write_json([])
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_empty_name(self):
        path = _write_json(
            [
                {
                    "name": "",
                    "description": "A.",
                    "parameters": None,
                    "returns": {"type": "number"},
                }
            ]
        )
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# command_parsing
# ---------------------------------------------------------------------------


class TestCommandParsing(unittest.TestCase):
    def test_defaults(self):
        with patch("sys.argv", ["prog"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "thinking")
        self.assertIn("functions_definition", result)
        self.assertIn("input", result)
        self.assertIn("output", result)

    def test_fast_mode_arg(self):
        with patch("sys.argv", ["prog", "--mode", "fast"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "fast")

    def test_thinking_mode_arg(self):
        with patch("sys.argv", ["prog", "--mode", "thinking"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "thinking")

    def test_invalid_mode_raises_system_exit(self):
        with patch("sys.argv", ["prog", "--mode", "turbo"]):
            with self.assertRaises(SystemExit):
                command_parsing()

    def test_custom_input(self):
        with patch("sys.argv", ["prog", "--input", "my/input.json"]):
            result = command_parsing()
        self.assertEqual(result["input"], "my/input.json")

    def test_custom_output(self):
        with patch("sys.argv", ["prog", "--output", "my/output.json"]):
            result = command_parsing()
        self.assertEqual(result["output"], "my/output.json")

    def test_custom_functions_definition(self):
        with patch("sys.argv", ["prog", "--functions_definition", "my/funcs.json"]):
            result = command_parsing()
        self.assertEqual(result["functions_definition"], "my/funcs.json")


# ---------------------------------------------------------------------------
# parse (integration)
# ---------------------------------------------------------------------------


class TestParse(unittest.TestCase):
    def test_parse_returns_args_instance(self):
        prompts_path = _write_json([{"prompt": "hello"}])
        funcs_path = _write_json(
            [
                {
                    "name": "fn_a",
                    "description": "A.",
                    "parameters": None,
                    "returns": {"type": "string"},
                }
            ]
        )
        try:
            with patch(
                "sys.argv",
                [
                    "prog",
                    "--input",
                    prompts_path,
                    "--functions_definition",
                    funcs_path,
                    "--output",
                    "data/output/out.json",
                ],
            ):
                result = parse()
            self.assertIsInstance(result, Args)
            self.assertEqual(result.prompts, ["hello"])
            self.assertEqual(len(result.functions), 1)
            self.assertEqual(result.functions[0].name, "fn_a")
        finally:
            os.unlink(prompts_path)
            os.unlink(funcs_path)

    def test_parse_with_missing_files_returns_empty_lists(self):
        with patch(
            "sys.argv",
            [
                "prog",
                "--input",
                "no_such_prompts.json",
                "--functions_definition",
                "no_such_funcs.json",
            ],
        ):
            result = parse()
        self.assertIsInstance(result, Args)
        self.assertEqual(result.prompts, [])
        self.assertEqual(result.functions, [])


if __name__ == "__main__":
    unittest.main()

"""Tests for src/parsing.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsing import (  # noqa: E402
    Args,
    FunctionDef,
    _validate_param_schema,
    _validate_parameters,
    command_parsing,
    functions_def_parsing,
    parse,
    prompt_parsing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data: Any) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, f)
    f.close()
    return f.name


def _valid_func_data(**kwargs: Any) -> dict[str, Any]:
    base = {
        "name": "fn_add",
        "description": "Add two numbers.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# FunctionDef model
# ---------------------------------------------------------------------------


class TestFunctionDef(unittest.TestCase):
    def test_valid(self) -> None:
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

    def test_with_parameters(self) -> None:
        fd = FunctionDef.model_validate(
            {
                "name": "fn_add",
                "description": "Add.",
                "parameters": {"a": {"type": "number"}},
                "returns": "number",
            }
        )
        assert fd.parameters is not None
        self.assertIn("a", fd.parameters)

    def test_empty_name_raises(self) -> None:
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

    def test_empty_description_raises(self) -> None:
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

    def test_none_returns_allowed(self) -> None:
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
    def test_default_mode_is_fast(self) -> None:
        """Default mode must now be 'fast'."""
        self.assertEqual(Args().mode, "fast")

    def test_thinking_mode(self) -> None:
        self.assertEqual(Args(mode="thinking").mode, "thinking")

    def test_fast_mode(self) -> None:
        self.assertEqual(Args(mode="fast").mode, "fast")

    def test_invalid_mode_raises(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            Args(mode=cast(Any, "turbo"))

    def test_defaults(self) -> None:
        args = Args()
        self.assertEqual(args.prompts, [])
        self.assertEqual(args.functions, [])

    def test_custom_paths(self) -> None:
        args = Args(input="my/input.json", output="my/output.json")
        self.assertEqual(args.input, "my/input.json")
        self.assertEqual(args.output, "my/output.json")

    def test_default_model(self) -> None:
        args = Args()
        self.assertEqual(args.model, "Qwen/Qwen3-0.6B")

    def test_custom_model(self) -> None:
        args = Args(model="meta-llama/Llama-3-8B")
        self.assertEqual(args.model, "meta-llama/Llama-3-8B")


# ---------------------------------------------------------------------------
# _validate_param_schema
# ---------------------------------------------------------------------------


class TestValidateParamSchema(unittest.TestCase):
    def test_valid_string(self) -> None:
        self.assertTrue(
            _validate_param_schema({"type": "string"}, "p")
        )

    def test_valid_number(self) -> None:
        self.assertTrue(
            _validate_param_schema({"type": "number"}, "p")
        )

    def test_valid_integer(self) -> None:
        self.assertTrue(
            _validate_param_schema({"type": "integer"}, "p")
        )

    def test_valid_boolean(self) -> None:
        self.assertTrue(
            _validate_param_schema({"type": "boolean"}, "p")
        )

    def test_valid_array(self) -> None:
        self.assertTrue(
            _validate_param_schema({"type": "array"}, "p")
        )

    def test_valid_flat_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        self.assertTrue(_validate_param_schema(schema, "user"))

    def test_valid_nested_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                }
            },
        }
        self.assertTrue(_validate_param_schema(schema, "user"))

    def test_invalid_not_a_dict(self) -> None:
        self.assertFalse(_validate_param_schema("string", "p"))
        self.assertFalse(_validate_param_schema(42, "p"))
        self.assertFalse(_validate_param_schema(None, "p"))

    def test_invalid_unknown_type(self) -> None:
        self.assertFalse(
            _validate_param_schema({"type": "uuid"}, "p")
        )

    def test_invalid_missing_type(self) -> None:
        self.assertFalse(_validate_param_schema({}, "p"))

    def test_object_missing_properties(self) -> None:
        self.assertFalse(
            _validate_param_schema({"type": "object"}, "p")
        )

    def test_object_empty_properties(self) -> None:
        self.assertFalse(
            _validate_param_schema(
                {"type": "object", "properties": {}}, "p"
            )
        )

    def test_object_properties_not_dict(self) -> None:
        self.assertFalse(
            _validate_param_schema(
                {"type": "object", "properties": "bad"}, "p"
            )
        )

    def test_object_invalid_nested_property(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "bad": {"type": "unknown_type"},
            },
        }
        self.assertFalse(_validate_param_schema(schema, "p"))


# ---------------------------------------------------------------------------
# _validate_parameters
# ---------------------------------------------------------------------------


class TestValidateParameters(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(_validate_parameters(None, "fn"))

    def test_not_a_dict_returns_none(self) -> None:
        self.assertIsNone(
            _validate_parameters("bad", "fn")  # type: ignore
        )

    def test_valid_flat_params(self) -> None:
        params = {"a": {"type": "number"}, "b": {"type": "string"}}
        result = _validate_parameters(params, "fn")
        self.assertEqual(result, params)

    def test_invalid_param_dropped(self) -> None:
        params = {
            "good": {"type": "string"},
            "bad": {"type": "unknown"},
        }
        result = _validate_parameters(params, "fn")
        assert result is not None
        self.assertIn("good", result)
        self.assertNotIn("bad", result)

    def test_all_invalid_returns_none(self) -> None:
        params = {"a": {"type": "bad1"}, "b": {"type": "bad2"}}
        result = _validate_parameters(params, "fn")
        self.assertIsNone(result)

    def test_valid_nested_object_kept(self) -> None:
        params = {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        }
        result = _validate_parameters(params, "fn")
        assert result is not None
        self.assertIn("user", result)

    def test_empty_dict_returns_none(self) -> None:
        result = _validate_parameters({}, "fn")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# prompt_parsing
# ---------------------------------------------------------------------------


class TestPromptParsing(unittest.TestCase):
    def test_valid_prompts(self) -> None:
        path = _write_json(
            [{"prompt": "Greet john"}, {"prompt": "Add 2 and 3"}]
        )
        try:
            self.assertEqual(
                prompt_parsing(path), ["Greet john", "Add 2 and 3"]
            )
        finally:
            os.unlink(path)

    def test_empty_list(self) -> None:
        path = _write_json([])
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        self.assertEqual(
            prompt_parsing("does_not_exist_xyz.json"), []
        )

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not valid json")
            path = f.name
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_not_a_list(self) -> None:
        path = _write_json({"prompt": "hello"})
        try:
            self.assertEqual(prompt_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_prompt_too_long(self) -> None:
        path = _write_json(
            [{"prompt": "x" * 301}, {"prompt": "short"}]
        )
        try:
            self.assertEqual(prompt_parsing(path), ["short"])
        finally:
            os.unlink(path)

    def test_prompt_at_boundary_accepted(self) -> None:
        path = _write_json([{"prompt": "x" * 299}])
        try:
            self.assertEqual(len(prompt_parsing(path)), 1)
        finally:
            os.unlink(path)

    def test_skips_non_dict_items(self) -> None:
        path = _write_json(["not a dict", {"prompt": "valid"}])
        try:
            self.assertEqual(prompt_parsing(path), ["valid"])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# functions_def_parsing
# ---------------------------------------------------------------------------


class TestFunctionsDefParsing(unittest.TestCase):
    def test_valid_single_function(self) -> None:
        path = _write_json([_valid_func_data()])
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "fn_add")
            self.assertEqual(result[0].returns_, "number")
        finally:
            os.unlink(path)

    def test_valid_nested_object_parameter(self) -> None:
        data = [
            {
                "name": "fn_create",
                "description": "Create a user.",
                "parameters": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    }
                },
                "returns": {"type": "string"},
            }
        ]
        path = _write_json(data)
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 1)
            params = result[0].parameters
            assert params is not None
            self.assertIn("user", params)
            self.assertEqual(params["user"]["type"], "object")
        finally:
            os.unlink(path)

    def test_invalid_nested_param_dropped(self) -> None:
        data = [
            {
                "name": "fn_x",
                "description": "X.",
                "parameters": {
                    "good": {"type": "string"},
                    "bad": {"type": "object"},
                },
                "returns": {"type": "string"},
            }
        ]
        path = _write_json(data)
        try:
            result = functions_def_parsing(path)
            self.assertEqual(len(result), 1)
            params = result[0].parameters
            assert params is not None
            self.assertIn("good", params)
            self.assertNotIn("bad", params)
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        self.assertEqual(
            functions_def_parsing("no_such_file_xyz.json"), []
        )

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{bad json")
            path = f.name
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_not_a_list(self) -> None:
        path = _write_json({"name": "fn_a"})
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_missing_returns_key(self) -> None:
        path = _write_json(
            [
                {
                    "name": "fn_a",
                    "description": "A.",
                    "parameters": None,
                }
            ]
        )
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_returns_not_a_dict(self) -> None:
        path = _write_json([_valid_func_data(returns="number")])
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_skips_empty_name(self) -> None:
        path = _write_json([_valid_func_data(name="")])
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)

    def test_empty_list(self) -> None:
        path = _write_json([])
        try:
            self.assertEqual(functions_def_parsing(path), [])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# command_parsing
# ---------------------------------------------------------------------------


class TestCommandParsing(unittest.TestCase):
    def test_default_mode_is_fast(self) -> None:
        with patch("sys.argv", ["prog"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "fast")

    def test_thinking_mode_arg(self) -> None:
        with patch("sys.argv", ["prog", "--mode", "thinking"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "thinking")

    def test_fast_mode_arg(self) -> None:
        with patch("sys.argv", ["prog", "--mode", "fast"]):
            result = command_parsing()
        self.assertEqual(result["mode"], "fast")

    def test_invalid_mode_raises_system_exit(self) -> None:
        with patch("sys.argv", ["prog", "--mode", "turbo"]):
            with self.assertRaises(SystemExit):
                command_parsing()

    def test_custom_input(self) -> None:
        with patch("sys.argv", ["prog", "--input", "my/input.json"]):
            self.assertEqual(
                command_parsing()["input"], "my/input.json"
            )

    def test_custom_output(self) -> None:
        with patch(
            "sys.argv", ["prog", "--output", "my/output.json"]
        ):
            self.assertEqual(
                command_parsing()["output"], "my/output.json"
            )

    def test_custom_functions_definition(self) -> None:
        with patch(
            "sys.argv",
            ["prog", "--functions_definition", "my/funcs.json"],
        ):
            self.assertEqual(
                command_parsing()["functions_definition"],
                "my/funcs.json",
            )

    def test_default_model(self) -> None:
        with patch("sys.argv", ["prog"]):
            result = command_parsing()
        self.assertEqual(result["model"], "Qwen/Qwen3-0.6B")

    def test_custom_model(self) -> None:
        with patch(
            "sys.argv", ["prog", "--model", "gpt2"]
        ):
            result = command_parsing()
        self.assertEqual(result["model"], "gpt2")


# ---------------------------------------------------------------------------
# parse (integration)
# ---------------------------------------------------------------------------


class TestParse(unittest.TestCase):
    def test_parse_returns_args_with_fast_default(self) -> None:
        prompts_path = _write_json([{"prompt": "hello"}])
        funcs_path = _write_json([_valid_func_data()])
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
            self.assertEqual(result.mode, "fast")
            self.assertEqual(result.model, "Qwen/Qwen3-0.6B")
            self.assertEqual(result.prompts, ["hello"])
            self.assertEqual(len(result.functions), 1)
        finally:
            os.unlink(prompts_path)
            os.unlink(funcs_path)

    def test_parse_thinking_mode_via_cli(self) -> None:
        prompts_path = _write_json([])
        funcs_path = _write_json([])
        try:
            with patch(
                "sys.argv",
                [
                    "prog",
                    "--input",
                    prompts_path,
                    "--functions_definition",
                    funcs_path,
                    "--mode",
                    "thinking",
                ],
            ):
                result = parse()
            self.assertEqual(result.mode, "thinking")
        finally:
            os.unlink(prompts_path)
            os.unlink(funcs_path)

    def test_parse_with_missing_files_returns_empty_lists(
        self,
    ) -> None:
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
        self.assertEqual(result.prompts, [])
        self.assertEqual(result.functions, [])


if __name__ == "__main__":
    unittest.main()

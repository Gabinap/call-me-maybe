"""Tests for src/get_from_llm.py."""

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.get_from_llm import (
    _cast,
    _generate_object,
    get_function_parameters,
    get_valid_function_name,
)
from src.parsing import FunctionDef
from src.process import PrecomputedVocab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def _base_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = [_tensor([1])]
    model.decode.return_value = "x"
    model.get_logits_from_input_ids.return_value = [0.1] * 200
    return model


def _small_pv() -> PrecomputedVocab:
    """Minimal PrecomputedVocab sufficient for test stubs."""
    decoded = {i: str(i) for i in range(10)}
    decoded[10] = '"'
    decoded[11] = ","
    return PrecomputedVocab.build(decoded)


# ---------------------------------------------------------------------------
# _cast
# ---------------------------------------------------------------------------


class TestCast(unittest.TestCase):
    def test_number_float(self) -> None:
        self.assertEqual(_cast("3", "number"), 3.0)
        self.assertIsInstance(_cast("3", "number"), float)

    def test_number_scientific(self) -> None:
        self.assertAlmostEqual(_cast("2.5e34", "number"), 2.5e34)

    def test_number_negative(self) -> None:
        self.assertAlmostEqual(_cast("-7.5", "number"), -7.5)

    def test_number_zero(self) -> None:
        self.assertEqual(_cast("0", "number"), 0.0)

    def test_number_invalid_returns_raw(self) -> None:
        self.assertEqual(_cast("not_a_number", "number"), "not_a_number")

    def test_integer_positive(self) -> None:
        result = _cast("42", "integer")
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_integer_negative(self) -> None:
        self.assertEqual(_cast("-5", "integer"), -5)

    def test_integer_float_string_invalid(self) -> None:
        self.assertEqual(_cast("3.14", "integer"), "3.14")

    def test_boolean_true(self) -> None:
        self.assertIs(_cast("true", "boolean"), True)
        self.assertIs(_cast("True", "boolean"), True)

    def test_boolean_false(self) -> None:
        self.assertIs(_cast("false", "boolean"), False)

    def test_string_strips_quotes(self) -> None:
        self.assertEqual(_cast('"hello"', "string"), "hello")

    def test_string_empty(self) -> None:
        self.assertEqual(_cast('""', "string"), "")

    def test_string_no_quotes_raw(self) -> None:
        self.assertEqual(_cast("hello", "string"), "hello")

    def test_none_type_raw(self) -> None:
        self.assertEqual(_cast("x", None), "x")

    def test_unknown_type_raw(self) -> None:
        self.assertEqual(_cast("x", "array"), "x")


# ---------------------------------------------------------------------------
# _generate_object
# ---------------------------------------------------------------------------


class TestGenerateObject(unittest.TestCase):
    def test_flat_object_returns_dict(self) -> None:
        properties = {"name": {"type": "string"}, "age": {"type": "integer"}}
        model = _base_model()
        pv = _small_pv()

        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_string",
                return_value=(
                    [0],
                    '"alice"',
                ),
            ),
            patch(
                "src.get_from_llm.process_integer", return_value=([0], "30")
            ),
        ):
            encoded, result = _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties=properties,
            )

        self.assertIsInstance(result, dict)
        self.assertIn("name", result)
        self.assertIn("age", result)
        self.assertEqual(result["name"], "alice")
        self.assertEqual(result["age"], 30)

    def test_nested_object_recurses(self) -> None:
        properties = {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
            }
        }
        model = _base_model()
        pv = _small_pv()

        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_string",
                return_value=(
                    [0],
                    '"Paris"',
                ),
            ),
        ):
            encoded, result = _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties=properties,
            )

        self.assertIsInstance(result["address"], dict)
        self.assertEqual(result["address"]["city"], "Paris")

    def test_number_property_cast_to_float(self) -> None:
        properties = {"score": {"type": "number"}}
        model = _base_model()
        pv = _small_pv()

        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number", return_value=([0], "9.5")
            ),
        ):
            encoded, result = _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties=properties,
            )

        self.assertEqual(result["score"], 9.5)
        self.assertIsInstance(result["score"], float)

    def test_empty_properties_returns_empty_dict(self) -> None:
        model = _base_model()
        pv = _small_pv()

        with patch("builtins.print"):
            encoded, result = _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties={},
            )

        self.assertEqual(result, {})

    def test_prints_braces(self) -> None:
        properties = {"x": {"type": "integer"}}
        model = _base_model()
        pv = _small_pv()

        with (
            patch("builtins.print") as mock_print,
            patch("src.get_from_llm.process_integer", return_value=([0], "1")),
        ):
            _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties=properties,
            )

        printed = "".join(
            str(c.args[0]) for c in mock_print.call_args_list if c.args
        )
        self.assertIn("{", printed)
        self.assertIn("}", printed)

    def test_indent_increases_for_nested(self) -> None:
        """Nested object call should receive a deeper indent."""
        properties = {
            "inner": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            }
        }
        model = _base_model()
        pv = _small_pv()

        with (
            patch("builtins.print") as mock_print,
            patch("src.get_from_llm.process_integer", return_value=([0], "1")),
        ):
            _generate_object(
                model,
                [],
                pv,
                fast=True,
                context_parts=[],
                properties=properties,
                indent="      ",
            )

        # Deeper indent (8+ spaces) should appear for the nested property
        self.assertTrue(
            any(
                "        " in str(c.args[0])
                for c in mock_print.call_args_list
                if c.args
            ),
            "Expected deeper indentation for nested object",
        )


# ---------------------------------------------------------------------------
# get_valid_function_name
# ---------------------------------------------------------------------------


class TestGetValidFunctionName(unittest.TestCase):
    def _functions(self) -> list[FunctionDef]:
        return [
            _func("fn_add", "Add.", returns="number"),
            _func("fn_greet", "Greet.", returns="string"),
        ]

    def test_returns_tuple_of_list_and_str(self) -> None:
        model = _base_model()
        model.decode.return_value = '"'
        reverse_vocab = {0: "fn", 1: "_", 2: "a", 3: '"'}
        pv = PrecomputedVocab.build(reverse_vocab)

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, self._functions(), "prompt", [], pv
            )

        self.assertIsInstance(tokens, list)
        self.assertIsInstance(name, str)

    def test_shortcut_single_match(self) -> None:
        functions = self._functions()
        reverse_vocab = {0: "fn_a", 1: '"'}
        pv = PrecomputedVocab.build({0: "fn_a", 1: '"'})
        model = _base_model()
        model.get_logits_from_input_ids.return_value = [1.0, 0.0]
        model.decode.return_value = "fn_a"

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, functions, "prompt", [], pv
            )

        self.assertEqual(name, "fn_add")

    def test_encoded_func_names_prepended(self) -> None:
        model = _base_model()
        model.decode.return_value = '"'
        reverse_vocab = {0: '"'}
        pv = PrecomputedVocab.build({0: '"'})
        encoded_func_names = [10, 20, 30]

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab,
                model,
                self._functions(),
                "prompt",
                encoded_func_names,
                pv,
            )

        self.assertEqual(tokens[:3], [10, 20, 30])

    def test_closing_quote_comma_appended(self) -> None:
        model = _base_model()
        model.decode.return_value = '"'
        model.encode.side_effect = [
            [_tensor([5])],
            [_tensor([6, 7])],
        ]
        reverse_vocab = {0: '"'}
        pv = PrecomputedVocab.build({0: '"'})
        model.get_logits_from_input_ids.return_value = [1.0]

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, self._functions(), "prompt", [], pv
            )

        self.assertIn(6, tokens)
        self.assertIn(7, tokens)


# ---------------------------------------------------------------------------
# get_function_parameters
# ---------------------------------------------------------------------------


class TestGetFunctionParameters(unittest.TestCase):
    def test_none_params_returns_empty(self) -> None:
        fd = _func(params=None)
        pv = _small_pv()
        with patch("builtins.print"):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertEqual(result, {})

    def test_number_cast_to_float(self) -> None:
        fd = _func(params={"a": {"type": "number"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number",
                return_value=([0], "2.5"),
            ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertEqual(result["a"], 2.5)
        self.assertIsInstance(result["a"], float)

    def test_integer_cast_to_int(self) -> None:
        fd = _func(params={"n": {"type": "integer"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_integer", return_value=([0], "7")),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertEqual(result["n"], 7)
        self.assertIsInstance(result["n"], int)

    def test_string_quotes_stripped(self) -> None:
        fd = _func(params={"s": {"type": "string"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_string",
                return_value=([0], '"hello"'),
            ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertEqual(result["s"], "hello")

    def test_nested_object_returns_dict(self) -> None:
        fd = _func(
            params={
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        )
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm._generate_object",
                return_value=([0], {"name": "alice"}),
            ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertIsInstance(result["user"], dict)
        self.assertEqual(result["user"]["name"], "alice")

    def test_multiple_params_all_present(self) -> None:
        fd = _func(
            params={
                "x": {"type": "number"},
                "y": {"type": "integer"},
                "label": {"type": "string"},
            }
        )
        pv = _small_pv()
        with (
            patch("builtins.print"),
                patch(
                    "src.get_from_llm.process_number",
                    return_value=([0], "1.5"),
                ),
                patch(
                    "src.get_from_llm.process_integer",
                    return_value=([0], "2"),
                ),
                patch(
                    "src.get_from_llm.process_string",
                    return_value=([0], '"lbl"'),
                ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        self.assertEqual(len(result), 3)
        self.assertEqual(result["x"], 1.5)
        self.assertEqual(result["y"], 2)
        self.assertEqual(result["label"], "lbl")

    def test_fast_mode_forwarded(self) -> None:
        fd = _func(params={"a": {"type": "number"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number", return_value=([0], "3.0")
            ) as mock_num,
        ):
            get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=True
            )
        _, kwargs = mock_num.call_args
        self.assertTrue(kwargs.get("fast", True))

    def test_thinking_mode_forwarded(self) -> None:
        fd = _func(params={"a": {"type": "number"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number", return_value=([0], "3.0")
            ) as mock_num,
        ):
            get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=False
            )
        _, kwargs = mock_num.call_args
        self.assertFalse(kwargs.get("fast", True))

    def test_negative_number_preserved(self) -> None:
        fd = _func(params={"a": {"type": "number"}})
        pv = _small_pv()
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number",
                return_value=([0], "-3.5"),
            ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", pv, fast=False
            )
        self.assertEqual(result["a"], -3.5)


if __name__ == "__main__":
    unittest.main()

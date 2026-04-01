"""Tests for src/get_from_llm.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.get_from_llm import _cast, get_function_parameters, get_valid_function_name
from src.parsing import FunctionDef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(ids: list[int]) -> MagicMock:
    """Return a mock tensor whose .tolist() returns ids."""
    t = MagicMock()
    t.tolist.return_value = ids
    return t


def _func(name="fn_test", desc="Test.", params=None, returns="string") -> FunctionDef:
    return FunctionDef.model_validate(
        {
            "name": name,
            "description": desc,
            "parameters": params,
            "returns": returns,
        }
    )


def _base_model() -> MagicMock:
    """Mock model with encode returning a proper tensor-like object."""
    model = MagicMock()
    model.encode.return_value = [_tensor([1])]
    model.decode.return_value = "x"
    model.get_logits_from_input_ids.return_value = [0.1] * 200
    return model


# ---------------------------------------------------------------------------
# _cast
# ---------------------------------------------------------------------------


class TestCast(unittest.TestCase):
    # number
    def test_number_integer_string(self):
        result = _cast("3", "number")
        self.assertEqual(result, 3.0)
        self.assertIsInstance(result, float)

    def test_number_float_string(self):
        self.assertAlmostEqual(_cast("3.14", "number"), 3.14)

    def test_number_scientific_notation(self):
        self.assertAlmostEqual(_cast("2.5e34", "number"), 2.5e34)

    def test_number_negative(self):
        self.assertAlmostEqual(_cast("-7.5", "number"), -7.5)

    def test_number_zero(self):
        self.assertEqual(_cast("0", "number"), 0.0)

    def test_number_invalid_returns_raw(self):
        self.assertEqual(_cast("not_a_number", "number"), "not_a_number")

    # integer
    def test_integer_positive(self):
        result = _cast("42", "integer")
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_integer_negative(self):
        self.assertEqual(_cast("-5", "integer"), -5)

    def test_integer_zero(self):
        self.assertEqual(_cast("0", "integer"), 0)

    def test_integer_float_string_invalid(self):
        self.assertEqual(_cast("3.14", "integer"), "3.14")

    # boolean
    def test_boolean_true_lowercase(self):
        self.assertIs(_cast("true", "boolean"), True)

    def test_boolean_false_lowercase(self):
        self.assertIs(_cast("false", "boolean"), False)

    def test_boolean_true_mixed_case(self):
        self.assertIs(_cast("True", "boolean"), True)

    def test_boolean_false_mixed_case(self):
        self.assertIs(_cast("False", "boolean"), False)

    # string
    def test_string_strips_outer_quotes(self):
        self.assertEqual(_cast('"hello"', "string"), "hello")

    def test_string_empty(self):
        self.assertEqual(_cast('""', "string"), "")

    def test_string_with_spaces(self):
        self.assertEqual(_cast('"hello world"', "string"), "hello world")

    def test_string_no_quotes_returns_raw(self):
        self.assertEqual(_cast("hello", "string"), "hello")

    def test_string_single_char_not_stripped(self):
        # Only one char — not a valid pair of quotes — returns raw
        self.assertEqual(_cast('"', "string"), '"')

    # unknown / None
    def test_none_type_returns_raw(self):
        self.assertEqual(_cast("whatever", None), "whatever")

    def test_unknown_type_returns_raw(self):
        self.assertEqual(_cast("foo", "array"), "foo")


# ---------------------------------------------------------------------------
# get_valid_function_name
# ---------------------------------------------------------------------------


class TestGetValidFunctionName(unittest.TestCase):
    def _functions(self):
        return [
            _func("fn_add", "Add.", returns="number"),
            _func("fn_greet", "Greet.", returns="string"),
        ]

    def test_returns_tuple_of_list_and_str(self):
        model = _base_model()
        # decode returns '"' → loop exits after one step
        model.decode.return_value = '"'
        reverse_vocab = {0: "fn", 1: "_", 2: "a", 3: '"'}

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, self._functions(), "Add 2 and 3", []
            )

        self.assertIsInstance(tokens, list)
        self.assertIsInstance(name, str)

    def test_shortcut_single_match_resolves_fn_add(self):
        """Partial name 'fn_a' matches only fn_add → remaining 'dd' is appended."""
        functions = self._functions()
        reverse_vocab = {0: "fn_a", 1: '"'}
        model = _base_model()
        model.get_logits_from_input_ids.return_value = [1.0, 0.0]
        model.decode.return_value = "fn_a"

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, functions, "Add 2 and 3", []
            )

        self.assertEqual(name, "fn_add")

    def test_encoded_func_names_prepended_to_context(self):
        """encoded_func_names must be at the start of the returned token list."""
        model = _base_model()
        model.decode.return_value = '"'
        reverse_vocab = {0: '"'}
        encoded_func_names = [10, 20, 30]

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, self._functions(), "prompt", encoded_func_names
            )

        self.assertEqual(tokens[:3], [10, 20, 30])

    def test_closing_quote_comma_appended(self):
        """'\",' tokens must be appended after the function name."""
        model = _base_model()
        model.decode.return_value = '"'
        # encode returns different tensors for each call
        model.encode.side_effect = [
            [_tensor([5])],  # encode(prompt_prefix)
            [_tensor([6, 7])],  # encode('",')
        ]
        reverse_vocab = {0: '"'}
        model.get_logits_from_input_ids.return_value = [1.0]

        with patch("builtins.print"):
            tokens, name = get_valid_function_name(
                reverse_vocab, model, self._functions(), "prompt", []
            )

        self.assertIn(6, tokens)
        self.assertIn(7, tokens)


# ---------------------------------------------------------------------------
# get_function_parameters
# ---------------------------------------------------------------------------


class TestGetFunctionParameters(unittest.TestCase):
    def test_none_parameters_returns_empty_dict(self):
        fd = _func(params=None)
        with patch("builtins.print"):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertEqual(result, {})

    def test_number_param_cast_to_float(self):
        fd = _func(params={"a": {"type": "number"}})
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_number", return_value=([0], "2.5")),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertEqual(result["a"], 2.5)
        self.assertIsInstance(result["a"], float)

    def test_integer_param_cast_to_int(self):
        fd = _func(params={"n": {"type": "integer"}})
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_integer", return_value=([0], "7")),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertEqual(result["n"], 7)
        self.assertIsInstance(result["n"], int)

    def test_string_param_quotes_stripped(self):
        fd = _func(params={"name": {"type": "string"}})
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_string", return_value=([0], '"hello"')),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertEqual(result["name"], "hello")
        self.assertIsInstance(result["name"], str)

    def test_boolean_param_cast(self):
        fd = _func(params={"flag": {"type": "boolean"}})
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_anything", return_value=([0], "true")),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertIs(result["flag"], True)

    def test_multiple_params_all_present(self):
        fd = _func(
            params={
                "x": {"type": "number"},
                "y": {"type": "integer"},
                "label": {"type": "string"},
            }
        )
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_number", return_value=([0], "1.5")),
            patch("src.get_from_llm.process_integer", return_value=([0], "2")),
            patch("src.get_from_llm.process_string", return_value=([0], '"lbl"')),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertEqual(len(result), 3)
        self.assertEqual(result["x"], 1.5)
        self.assertEqual(result["y"], 2)
        self.assertEqual(result["label"], "lbl")

    def test_fast_mode_passed_to_process_number(self):
        fd = _func(params={"a": {"type": "number"}})
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number", return_value=([0], "3.0")
            ) as mock_num,
        ):
            get_function_parameters({}, _base_model(), [], fd, "prompt", {}, fast=True)
        _, kwargs = mock_num.call_args
        self.assertTrue(kwargs.get("fast", True))

    def test_thinking_mode_passed_to_process_number(self):
        fd = _func(params={"a": {"type": "number"}})
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_number", return_value=([0], "3.0")
            ) as mock_num,
        ):
            get_function_parameters({}, _base_model(), [], fd, "prompt", {}, fast=False)
        _, kwargs = mock_num.call_args
        self.assertFalse(kwargs.get("fast", True))

    def test_negative_number_preserved(self):
        fd = _func(params={"a": {"type": "number"}})
        with (
            patch("builtins.print"),
            patch("src.get_from_llm.process_number", return_value=([0], "-3.5")),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=False
            )
        self.assertEqual(result["a"], -3.5)

    def test_result_keys_match_param_names(self):
        fd = _func(params={"source": {"type": "string"}, "target": {"type": "string"}})
        with (
            patch("builtins.print"),
            patch(
                "src.get_from_llm.process_string",
                side_effect=[([0], '"src"'), ([0], '"tgt"')],
            ),
        ):
            result = get_function_parameters(
                {}, _base_model(), [], fd, "prompt", {}, fast=True
            )
        self.assertIn("source", result)
        self.assertIn("target", result)


if __name__ == "__main__":
    unittest.main()

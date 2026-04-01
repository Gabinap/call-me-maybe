"""Tests for src/process.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.process import (
    _ANYTHING_COMPLETE,
    _ANYTHING_PREFIX,
    _INTEGER_COMPLETE,
    _INTEGER_PREFIX,
    _NUMBER_COMPLETE,
    _NUMBER_PREFIX,
    _STRING_COMPLETE,
    _STRING_PREFIX,
    _get_minus_token,
    _score_with_negative,
    process_anything,
    process_integer,
    process_number,
    process_string,
    score_candidates,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(ids: list[int]) -> MagicMock:
    """Return a mock tensor whose .tolist() returns ids."""
    t = MagicMock()
    t.tolist.return_value = ids
    return t


def _mock(logits_sequence: list[list[float]]) -> MagicMock:
    """Mock model with sequential logit responses and a working encode."""
    model = MagicMock()
    model.get_logits_from_input_ids.side_effect = logits_sequence
    model.encode.return_value = [_tensor([10, 11])]
    model.decode.return_value = "x"
    return model


def _mock_static(logits: list[float]) -> MagicMock:
    """Mock model that always returns the same logits."""
    model = MagicMock()
    model.get_logits_from_input_ids.return_value = logits
    model.encode.return_value = [_tensor([10])]
    model.decode.return_value = "x"
    return model


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


class TestPatterns(unittest.TestCase):
    # --- number prefix ---
    def test_number_prefix_empty_string(self):
        # The entire group is optional (?), so "" matches
        self.assertIsNotNone(_NUMBER_PREFIX.match(""))

    def test_number_prefix_minus_only(self):
        # "-" is a valid partial number start
        self.assertIsNotNone(_NUMBER_PREFIX.match("-"))

    def test_number_prefix_digits(self):
        for s in ("3", "3.", "3.1", "-2.5", "2e", "2e+", "-2.5e"):
            with self.subTest(s=s):
                self.assertIsNotNone(_NUMBER_PREFIX.match(s))

    def test_number_prefix_rejects_terminal(self):
        self.assertIsNone(_NUMBER_PREFIX.match("3,"))
        self.assertIsNone(_NUMBER_PREFIX.match("3}"))

    # --- number complete ---
    def test_number_complete_accepts(self):
        for s in ("3,", "-2.5,", "1e10}", "2.5e+34,", "0}"):
            with self.subTest(s=s):
                self.assertIsNotNone(_NUMBER_COMPLETE.match(s))

    def test_number_complete_rejects_partial(self):
        self.assertIsNone(_NUMBER_COMPLETE.match("3"))
        self.assertIsNone(_NUMBER_COMPLETE.match("-"))

    # --- integer prefix ---
    def test_integer_prefix_accepts_digits(self):
        # \d+? requires at least one digit — "" and "-" do NOT match
        for s in ("0", "42", "-7", "100"):
            with self.subTest(s=s):
                self.assertIsNotNone(_INTEGER_PREFIX.match(s))

    def test_integer_prefix_rejects_empty(self):
        self.assertIsNone(_INTEGER_PREFIX.match(""))

    def test_integer_prefix_rejects_minus_only(self):
        # "-" alone has no digit → no match
        self.assertIsNone(_INTEGER_PREFIX.match("-"))

    def test_integer_prefix_rejects_float(self):
        self.assertIsNone(_INTEGER_PREFIX.match("3.14"))

    # --- integer complete ---
    def test_integer_complete_accepts(self):
        for s in ("0,", "42}", "-7,", "100}"):
            with self.subTest(s=s):
                self.assertIsNotNone(_INTEGER_COMPLETE.match(s))

    def test_integer_complete_rejects_partial(self):
        self.assertIsNone(_INTEGER_COMPLETE.match("42"))
        self.assertIsNone(_INTEGER_COMPLETE.match("-"))

    # --- string prefix ---
    def test_string_prefix_accepts_content(self):
        for s in ("", "hello", "world", "abc123"):
            with self.subTest(s=s):
                self.assertIsNotNone(_STRING_PREFIX.match(s))

    def test_string_prefix_accepts_escaped_quote(self):
        self.assertIsNotNone(_STRING_PREFIX.match('\\"'))

    def test_string_prefix_rejects_unescaped_quote(self):
        self.assertIsNone(_STRING_PREFIX.match('"'))
        self.assertIsNone(_STRING_PREFIX.match('say "hi"'))

    # --- string complete ---
    def test_string_complete_accepts(self):
        for s in ('"', 'hello"', 'world"', 'abc123"'):
            with self.subTest(s=s):
                self.assertIsNotNone(_STRING_COMPLETE.match(s))

    def test_string_complete_rejects_no_closing_quote(self):
        self.assertIsNone(_STRING_COMPLETE.match("hello"))

    # --- anything ---
    def test_anything_prefix_accepts(self):
        for s in ("true", "false", "null", ""):
            with self.subTest(s=s):
                self.assertIsNotNone(_ANYTHING_PREFIX.match(s))

    def test_anything_prefix_rejects_terminal(self):
        self.assertIsNone(_ANYTHING_PREFIX.match("x,"))

    def test_anything_complete_accepts(self):
        for s in ("true,", "false}", "null,"):
            with self.subTest(s=s):
                self.assertIsNotNone(_ANYTHING_COMPLETE.match(s))


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------


class TestScoreCandidates(unittest.TestCase):
    def test_single_beam_greedy(self):
        vocab = {0: "4", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value, score = score_candidates(
            model,
            [99],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=1,
        )
        self.assertEqual(value, "4")
        self.assertIsInstance(score, float)
        self.assertIn(0, tokens)
        self.assertIn(1, tokens)

    def test_multi_token_value(self):
        vocab = {0: "4", 1: "2", 2: ","}
        model = _mock([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=1,
        )
        self.assertEqual(value, "42")

    def test_terminal_stripped_from_value(self):
        vocab = {0: "7", 1: "}"}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=1,
        )
        self.assertNotIn(",", value)
        self.assertNotIn("}", value)

    def test_max_tokens_stops_generation(self):
        vocab = {0: "a", 1: "b"}
        model = _mock_static([1.0, 0.0])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _STRING_PREFIX,
            _STRING_COMPLETE,
            num_beams=1,
            max_tokens=3,
        )
        self.assertIsInstance(value, str)

    def test_fallback_when_no_continuation_tokens(self):
        # "3" starts valid; only '"' can complete → fallback picks '"'
        vocab = {0: "3", 1: '"'}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _STRING_PREFIX,
            _STRING_COMPLETE,
            num_beams=1,
        )
        self.assertEqual(value, "3")

    def test_forced_starts_overrides_num_beams(self):
        vocab = {0: "5", 1: "-", 2: ","}
        model = _mock_static([0.5, 0.1, 1.0])

        tokens, value, score = score_candidates(
            model,
            [99],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            forced_starts=[None, 1],
        )
        self.assertIsInstance(value, str)
        self.assertIsInstance(score, float)

    def test_four_beams_returns_best_score(self):
        vocab = {0: "1", 1: "2", 2: "3", 3: "4", 4: ","}
        model = _mock_static([0.1, 0.2, 0.3, 2.0, 0.0])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=4,
        )
        self.assertIsInstance(value, str)
        self.assertIsInstance(score, float)

    def test_stream_prints_non_terminal_tokens(self):
        vocab = {0: "4", 1: "2", 2: ","}
        model = _mock([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with patch("builtins.print") as mock_print:
            score_candidates(
                model,
                [0],
                vocab,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                num_beams=1,
            )

        printed = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("4", printed)
        self.assertIn("2", printed)
        self.assertNotIn(",", printed)

    def test_no_stream_with_multiple_beams(self):
        vocab = {0: "5", 1: "3", 2: ","}
        model = _mock_static([1.0, 0.5, 0.0])

        with patch("builtins.print") as mock_print:
            score_candidates(
                model,
                [0],
                vocab,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                forced_starts=[None, 1],
            )

        printed = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertEqual(printed, "")

    def test_avg_score_is_mean_of_token_logits(self):
        vocab = {0: "7", 1: ","}
        model = _mock([[0.8, 0.0], [0.0, 0.4]])

        tokens, value, score = score_candidates(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=1,
        )
        self.assertAlmostEqual(score, (0.8 + 0.4) / 2, places=5)


# ---------------------------------------------------------------------------
# _get_minus_token
# ---------------------------------------------------------------------------


class TestGetMinusToken(unittest.TestCase):
    def test_returns_none_when_no_minus(self):
        vocab = {0: "3", 1: ".", 2: ","}
        self.assertIsNone(_get_minus_token(vocab, [0], _mock_static([1.0, 0.5, 0.0])))

    def test_returns_id_when_unique_minus(self):
        vocab = {0: "3", 1: "-", 2: ","}
        self.assertEqual(_get_minus_token(vocab, [0], _mock_static([1.0, 0.5, 0.0])), 1)

    def test_returns_highest_logit_when_multiple_minus(self):
        vocab = {0: "-", 1: "-", 2: "3"}
        model = _mock_static([0.1, 0.9, 0.5])
        self.assertEqual(_get_minus_token(vocab, [0], model), 1)


# ---------------------------------------------------------------------------
# _score_with_negative
# ---------------------------------------------------------------------------


class TestScoreWithNegative(unittest.TestCase):
    def test_falls_back_gracefully_when_no_minus(self):
        vocab = {0: "5", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value = _score_with_negative(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            strip_chars=",}",
        )
        self.assertEqual(value, "5")

    def test_strips_terminal_characters(self):
        vocab = {0: "9", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value = _score_with_negative(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            strip_chars=",}",
        )
        self.assertNotIn(",", value)
        self.assertNotIn("}", value)

    def test_no_terminal_in_value_with_minus(self):
        vocab = {0: "5", 1: "-", 2: ","}
        model = _mock_static([0.1, 2.0, 0.5])

        tokens, value = _score_with_negative(
            model,
            [0],
            vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            strip_chars=",}",
        )
        self.assertNotIn(",", value)
        self.assertNotIn("}", value)


# ---------------------------------------------------------------------------
# process_number
# ---------------------------------------------------------------------------


class TestProcessNumber(unittest.TestCase):
    def test_fast_returns_correct_value(self):
        vocab = {0: "7", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_number(model, [0], vocab, fast=True)

        self.assertEqual(value, "7")
        self.assertIsInstance(tokens, list)

    def test_fast_no_terminal_in_value(self):
        vocab = {0: "3", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_number(model, [0], vocab, fast=True)

        self.assertNotIn(",", value)
        self.assertNotIn("}", value)

    def test_thinking_returns_string(self):
        vocab = {0: "4", 1: "-", 2: ","}
        model = _mock_static([1.0, 0.1, 0.5])

        with patch("builtins.print"):
            tokens, value = process_number(model, [0], vocab, fast=False)

        self.assertIsInstance(value, str)

    def test_fast_streams_tokens(self):
        vocab = {0: "4", 1: "2", 2: ","}
        model = _mock([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with patch("builtins.print") as mock_print:
            process_number(model, [0], vocab, fast=True)

        printed = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("4", printed)
        self.assertIn("2", printed)


# ---------------------------------------------------------------------------
# process_integer
# ---------------------------------------------------------------------------


class TestProcessInteger(unittest.TestCase):
    def test_fast_positive(self):
        vocab = {0: "4", 1: "2", 2: ","}
        model = _mock([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], vocab, fast=True)

        self.assertEqual(value, "42")

    def test_thinking_runs_without_error(self):
        vocab = {0: "9", 1: "-", 2: ","}
        model = _mock_static([1.0, 0.1, 0.5])

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], vocab, fast=False)

        self.assertIsInstance(value, str)

    def test_value_castable_to_int(self):
        vocab = {0: "3", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], vocab, fast=True)

        self.assertEqual(int(value), 3)


# ---------------------------------------------------------------------------
# process_string
# ---------------------------------------------------------------------------


class TestProcessString(unittest.TestCase):
    def _vocab_and_model(self):
        """Generates: h → i → closing quote."""
        vocab = {0: "h", 1: "i", 2: '"'}
        model = _mock([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return vocab, model

    def test_fast_value_wrapped_in_quotes(self):
        vocab, model = self._vocab_and_model()

        with patch("builtins.print"):
            tokens, value = process_string(model, "ctx", vocab, fast=True)

        self.assertTrue(value.startswith('"'), f"Got {value!r}")
        self.assertTrue(value.endswith('"'), f"Got {value!r}")

    def test_thinking_value_wrapped_in_quotes(self):
        vocab = {0: "h", 1: "i", 2: '"'}
        model = _mock_static([1.0, 0.5, 0.0])

        with patch("builtins.print"):
            tokens, value = process_string(model, "ctx", vocab, fast=False)

        self.assertTrue(value.startswith('"'))
        self.assertTrue(value.endswith('"'))

    def test_inner_content_has_no_extra_quotes(self):
        vocab, model = self._vocab_and_model()

        with patch("builtins.print"):
            tokens, value = process_string(model, "ctx", vocab, fast=True)

        inner = value[1:-1]
        self.assertFalse(inner.startswith('"'), f"Inner: {inner!r}")
        self.assertFalse(inner.endswith('"'), f"Inner: {inner!r}")

    def test_fast_streams_content_tokens(self):
        vocab, model = self._vocab_and_model()

        with patch("builtins.print") as mock_print:
            process_string(model, "ctx", vocab, fast=True)

        printed = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn('"', printed)
        self.assertIn("h", printed)
        self.assertIn("i", printed)


# ---------------------------------------------------------------------------
# process_anything
# ---------------------------------------------------------------------------


class TestProcessAnything(unittest.TestCase):
    def test_returns_value_without_terminal(self):
        vocab = {0: "t", 1: "r", 2: "u", 3: "e", 4: ","}
        model = _mock(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        with patch("builtins.print"):
            tokens, value = process_anything(model, [0], vocab, fast=True)

        self.assertEqual(value, "true")
        self.assertNotIn(",", value)
        self.assertNotIn("}", value)

    def test_always_single_beam(self):
        vocab = {0: "x", 1: ","}
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_anything(model, [0], vocab, fast=False)

        self.assertEqual(value, "x")


if __name__ == "__main__":
    unittest.main()

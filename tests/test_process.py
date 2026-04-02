"""Tests for src/process.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.process import (  # noqa: E402
    _ANYTHING_COMPLETE,
    _ANYTHING_PREFIX,
    _INTEGER_COMPLETE,
    _INTEGER_PREFIX,
    _NUMBER_COMPLETE,
    _NUMBER_PREFIX,
    _STRING_COMPLETE,
    _STRING_PREFIX,
    PrecomputedVocab,
    _ensure_float_dot,
    _get_minus_token,
    _score_with_negative,
    process_anything,
    process_integer,
    process_number,
    process_string,
    score_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(ids: list[int]) -> MagicMock:
    t = MagicMock()
    t.tolist.return_value = ids
    return t


def _mock(logits_sequence: list[list[float]]) -> MagicMock:
    model = MagicMock()
    model.get_logits_from_input_ids.side_effect = logits_sequence
    model.encode.return_value = [_tensor([10, 11])]
    model.decode.return_value = "x"
    return model


def _mock_static(logits: list[float]) -> MagicMock:
    model = MagicMock()
    model.get_logits_from_input_ids.return_value = logits
    model.encode.return_value = [_tensor([10])]
    model.decode.return_value = "x"
    return model


def _pv(decoded_vocab: dict[int, str]) -> PrecomputedVocab:
    """Build a PrecomputedVocab from a small decoded_vocab dict."""
    return PrecomputedVocab.build(decoded_vocab)


# ---------------------------------------------------------------------------
# PrecomputedVocab
# ---------------------------------------------------------------------------


class TestPrecomputedVocab(unittest.TestCase):
    def _full_vocab(self) -> dict[int, str]:
        return {
            0: "4", 1: "2", 2: ",", 3: "-", 4: "h", 5: '"', 6: "true"
        }

    def test_build_returns_instance(self) -> None:
        pv = _pv(self._full_vocab())
        self.assertIsInstance(pv, PrecomputedVocab)

    def test_decoded_preserved(self) -> None:
        vocab = self._full_vocab()
        pv = _pv(vocab)
        self.assertEqual(pv.decoded, vocab)

    def test_minus_ids_extracted(self) -> None:
        pv = _pv({0: "3", 1: "-", 2: ","})
        self.assertIn(1, pv.minus_ids)
        self.assertNotIn(0, pv.minus_ids)

    def test_no_minus_ids_when_absent(self) -> None:
        pv = _pv({0: "3", 1: ","})
        self.assertEqual(pv.minus_ids, [])

    def test_integer_starts_only_digits_and_minus(self) -> None:
        pv = _pv({0: "4", 1: ",", 2: "h", 3: "-"})
        self.assertIn(0, pv.integer_starts)
        self.assertIn(3, pv.integer_starts)
        self.assertNotIn(1, pv.integer_starts)
        self.assertNotIn(2, pv.integer_starts)

    def test_integer_starts_includes_minus(self) -> None:
        pv = _pv({0: "-", 1: "3"})
        self.assertIn(0, pv.integer_starts)
        self.assertIn(1, pv.integer_starts)

    def test_string_starts_excludes_quote(self) -> None:
        pv = _pv({0: "h", 1: '"'})
        self.assertIn(0, pv.string_starts)
        self.assertNotIn(1, pv.string_starts)

    def test_number_starts_includes_minus(self) -> None:
        # "-" alone matches _NUMBER_PREFIX (optional digit group)
        pv = _pv({0: "-", 1: "3"})
        self.assertIn(0, pv.number_starts)
        self.assertIn(1, pv.number_starts)

    def test_empty_vocab(self) -> None:
        pv = _pv({})
        self.assertEqual(pv.integer_starts, [])
        self.assertEqual(pv.minus_ids, [])


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


class TestPatterns(unittest.TestCase):
    def test_number_prefix_empty_and_minus(self) -> None:
        for s in ("", "-", "3", "3.", "-2.5", "2e+"):
            with self.subTest(s=s):
                self.assertIsNotNone(_NUMBER_PREFIX.match(s))

    def test_number_prefix_rejects_terminal(self) -> None:
        self.assertIsNone(_NUMBER_PREFIX.match("3,"))

    def test_number_complete_accepts(self) -> None:
        for s in ("3,", "-2.5,", "1e10}", "2.5e+34,"):
            with self.subTest(s=s):
                self.assertIsNotNone(_NUMBER_COMPLETE.match(s))

    def test_integer_prefix_accepts_digits(self) -> None:
        for s in ("0", "42", "-7"):
            with self.subTest(s=s):
                self.assertIsNotNone(_INTEGER_PREFIX.match(s))

    def test_integer_prefix_accepts_minus_only(self) -> None:
        self.assertIsNotNone(_INTEGER_PREFIX.match("-"))

    def test_integer_prefix_rejects_empty(self) -> None:
        self.assertIsNone(_INTEGER_PREFIX.match(""))

    def test_integer_prefix_rejects_float(self) -> None:
        self.assertIsNone(_INTEGER_PREFIX.match("3.14"))

    def test_integer_complete_accepts(self) -> None:
        for s in ("0,", "42}", "-7,"):
            with self.subTest(s=s):
                self.assertIsNotNone(_INTEGER_COMPLETE.match(s))

    def test_string_prefix_accepts_content(self) -> None:
        for s in ("", "hello", "abc123"):
            with self.subTest(s=s):
                self.assertIsNotNone(_STRING_PREFIX.match(s))

    def test_string_prefix_rejects_unescaped_quote(self) -> None:
        self.assertIsNone(_STRING_PREFIX.match('"'))

    def test_string_complete_accepts(self) -> None:
        for s in ('"', 'hello"', 'world"'):
            with self.subTest(s=s):
                self.assertIsNotNone(_STRING_COMPLETE.match(s))

    def test_anything_prefix_accepts(self) -> None:
        for s in ("true", "false", ""):
            with self.subTest(s=s):
                self.assertIsNotNone(_ANYTHING_PREFIX.match(s))

    def test_anything_complete_accepts(self) -> None:
        for s in ("true,", "false}"):
            with self.subTest(s=s):
                self.assertIsNotNone(_ANYTHING_COMPLETE.match(s))


# ---------------------------------------------------------------------------
# _ensure_float_dot
# ---------------------------------------------------------------------------


class TestEnsureFloatDot(unittest.TestCase):
    def test_plain_integer_gets_dot(self) -> None:
        self.assertEqual(_ensure_float_dot("42"), "42.0")

    def test_negative_integer_gets_dot(self) -> None:
        self.assertEqual(_ensure_float_dot("-7"), "-7.0")

    def test_float_unchanged(self) -> None:
        self.assertEqual(_ensure_float_dot("3.14"), "3.14")

    def test_scientific_unchanged(self) -> None:
        self.assertEqual(_ensure_float_dot("2e34"), "2e34")
        self.assertEqual(_ensure_float_dot("2.5E+34"), "2.5E+34")

    def test_zero_gets_dot(self) -> None:
        self.assertEqual(_ensure_float_dot("0"), "0.0")

    def test_result_parseable_as_float(self) -> None:
        for raw in ("42", "-7", "3.14", "2e34", "0"):
            with self.subTest(raw=raw):
                self.assertIsInstance(
                    float(_ensure_float_dot(raw)), float
                )


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------


class TestScoreCandidates(unittest.TestCase):
    def _int_vocab(self) -> tuple[dict[int, str], PrecomputedVocab]:
        vocab = {0: "4", 1: "2", 2: ","}
        return vocab, _pv(vocab)

    def test_single_beam_greedy(self) -> None:
        vocab, pv = self._int_vocab()
        model = _mock(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        tokens, value, score = score_candidates(
            model,
            [99],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=1,
        )
        # score_candidates no longer strips — caller must strip
        value = value.rstrip(",} \n\r\t")
        self.assertEqual(value, "42")
        self.assertIsInstance(score, float)

    def test_terminal_stripped_from_value(self) -> None:
        vocab, pv = self._int_vocab()
        model = _mock([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        tokens, value, score = score_candidates(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=1,
        )
        # The terminal char (,) is removed by score_candidates [:-1]
        self.assertNotIn(",", value)

    def test_max_tokens_stops_generation(self) -> None:
        vocab = {0: "a", 1: "b"}
        pv = _pv(vocab)
        model = _mock_static([1.0, 0.0])

        tokens, value, score = score_candidates(
            model,
            [0],
            pv,
            _STRING_PREFIX,
            _STRING_COMPLETE,
            pv.string_starts,
            pv.string_fallback,
            num_beams=1,
            max_tokens=3,
        )
        self.assertIsInstance(value, str)

    def test_four_beams_returns_best_score(self) -> None:
        vocab = {0: "1", 1: "2", 2: "3", 3: "4", 4: ","}
        pv = _pv(vocab)
        model = _mock_static([0.1, 0.2, 0.3, 2.0, 0.0])

        tokens, value, score = score_candidates(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=4,
        )
        self.assertIsInstance(value, str)
        self.assertIsInstance(score, float)

    def test_stream_true_prints_tokens(self) -> None:
        vocab, pv = self._int_vocab()
        model = _mock(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        with patch("builtins.print") as mock_print:
            score_candidates(
                model,
                [0],
                pv,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                pv.integer_starts,
                pv.integer_fallback,
                num_beams=1,
            )
            printed = "".join(
                str(c.args[0])
                for c in mock_print.call_args_list
                if c.args
            )
        self.assertIn("4", printed)
        self.assertIn("2", printed)
        self.assertNotIn(",", printed)

    def test_stream_false_prints_nothing(self) -> None:
        vocab, pv = self._int_vocab()
        model = _mock_static([1.0, 0.5, 0.0])

        with patch("builtins.print") as mock_print:
            score_candidates(
                model,
                [0],
                pv,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                pv.integer_starts,
                pv.integer_fallback,
                num_beams=1,
                stream=False,
            )
            printed = "".join(
                str(c.args[0])
                for c in mock_print.call_args_list
                if c.args
            )
        self.assertEqual(printed, "")

    def test_multi_beam_auto_no_stream(self) -> None:
        """With 2 beams, stream=None -> auto resolves to False."""
        vocab, pv = self._int_vocab()
        model = _mock_static([1.0, 0.5, 0.0])

        with patch("builtins.print") as mock_print:
            score_candidates(
                model,
                [0],
                pv,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                pv.integer_starts,
                pv.integer_fallback,
                num_beams=2,
            )
            printed = "".join(
                str(c.args[0])
                for c in mock_print.call_args_list
                if c.args
            )
        self.assertEqual(printed, "")

    def test_avg_score_is_mean_of_logits(self) -> None:
        vocab, pv = self._int_vocab()
        model = _mock([[0.8, 0.0, 0.0], [0.0, 0.0, 0.4]])

        tokens, value, score = score_candidates(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=1,
        )
        self.assertAlmostEqual(score, (0.8 + 0.4) / 2, places=5)

    def test_step_cache_avoids_redundant_calls(self) -> None:
        """Both beams hitting same current_str should share cache."""
        vocab = {0: "4", 1: "4", 2: ","}
        pv = _pv(vocab)
        model = _mock_static([1.0, 1.0, 0.0])

        tokens, value, score = score_candidates(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=2,
        )
        self.assertIsInstance(value, str)


# ---------------------------------------------------------------------------
# _get_minus_token
# ---------------------------------------------------------------------------


class TestGetMinusToken(unittest.TestCase):
    def test_returns_none_when_no_minus(self) -> None:
        pv = _pv({0: "3", 1: ","})
        self.assertIsNone(
            _get_minus_token(pv, [0], _mock_static([1.0, 0.0]))
        )

    def test_returns_id_when_unique(self) -> None:
        pv = _pv({0: "3", 1: "-", 2: ","})
        self.assertEqual(
            _get_minus_token(pv, [0], _mock_static([1.0, 0.5, 0.0])),
            1,
        )

    def test_returns_highest_logit_when_multiple(self) -> None:
        pv = _pv({0: "-", 1: "-", 2: "3"})
        model = _mock_static([0.1, 0.9, 0.5])
        self.assertEqual(_get_minus_token(pv, [0], model), 1)


# ---------------------------------------------------------------------------
# _score_with_negative
# ---------------------------------------------------------------------------


class TestScoreWithNegative(unittest.TestCase):
    def test_positive_value_when_no_minus(self) -> None:
        pv = _pv({0: "5", 1: ","})
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value = _score_with_negative(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            strip_chars=",}",
        )
        self.assertEqual(value, "5")
        self.assertNotIn(",", value)

    def test_no_terminal_in_value(self) -> None:
        pv = _pv({0: "9", 1: ","})
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        tokens, value = _score_with_negative(
            model,
            [0],
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            strip_chars=",}",
        )
        self.assertNotIn(",", value)
        self.assertNotIn("}", value)

    def test_silent_during_comparison(self) -> None:
        """Nothing must be printed during positive/negative comparison."""
        pv = _pv({0: "5", 1: "-", 2: ","})
        model = _mock_static([0.1, 2.0, 0.5])

        with patch("builtins.print") as mock_print:
            _score_with_negative(
                model,
                [0],
                pv,
                _INTEGER_PREFIX,
                _INTEGER_COMPLETE,
                pv.integer_starts,
                pv.integer_fallback,
                strip_chars=",}",
            )
        printed = "".join(
            str(c.args[0])
            for c in mock_print.call_args_list
            if c.args
        )
        self.assertEqual(printed, "")


# ---------------------------------------------------------------------------
# process_number
# ---------------------------------------------------------------------------


class TestProcessNumber(unittest.TestCase):
    def test_fast_returns_value_with_dot(self) -> None:
        pv = _pv({0: "7", 1: ","})
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_number(model, [0], pv, fast=True)

        self.assertIn(".", value)
        self.assertIsInstance(float(value), float)

    def test_fast_streams_tokens(self) -> None:
        pv = _pv({0: "4", 1: "2", 2: ","})
        model = _mock(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        with patch("builtins.print") as mock_print:
            process_number(model, [0], pv, fast=True)

        printed = "".join(
            str(c.args[0])
            for c in mock_print.call_args_list
            if c.args
        )
        self.assertIn("4", printed)

    def test_thinking_prints_once(self) -> None:
        pv = _pv({0: "4", 1: "-", 2: ","})
        model = _mock_static([1.0, 0.1, 0.5])

        with patch("builtins.print") as mock_print:
            tokens, value = process_number(model, [0], pv, fast=False)

        printed_values = [
            c.args[0]
            for c in mock_print.call_args_list
            if c.args and c.args[0] == value
        ]
        self.assertEqual(len(printed_values), 1)

    def test_no_terminal_in_value(self) -> None:
        pv = _pv({0: "3", 1: ","})
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_number(model, [0], pv, fast=True)

        self.assertNotIn(",", value)
        self.assertNotIn("}", value)


# ---------------------------------------------------------------------------
# process_integer
# ---------------------------------------------------------------------------


class TestProcessInteger(unittest.TestCase):
    def test_fast_positive(self) -> None:
        pv = _pv({0: "4", 1: "2", 2: ","})
        model = _mock(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], pv, fast=True)

        self.assertEqual(value, "42")

    def test_value_castable_to_int(self) -> None:
        pv = _pv({0: "3", 1: ","})
        model = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], pv, fast=True)

        self.assertEqual(int(value), 3)

    def test_thinking_runs_without_error(self) -> None:
        pv = _pv({0: "9", 1: "-", 2: ","})
        model = _mock_static([1.0, 0.1, 0.5])

        with patch("builtins.print"):
            tokens, value = process_integer(model, [0], pv, fast=False)

        self.assertIsInstance(value, str)


# ---------------------------------------------------------------------------
# process_string
# ---------------------------------------------------------------------------


class TestProcessString(unittest.TestCase):
    def _str_vocab_model(
        self,
    ) -> tuple[PrecomputedVocab, MagicMock]:
        vocab = {0: "h", 1: "i", 2: '"'}
        pv = _pv(vocab)
        model = _mock(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        return pv, model

    def test_fast_wrapped_in_quotes(self) -> None:
        pv, model = self._str_vocab_model()

        with patch("builtins.print"):
            tokens, value = process_string(model, "ctx", pv, fast=True)

        self.assertTrue(value.startswith('"'))
        self.assertTrue(value.endswith('"'))

    def test_thinking_wrapped_in_quotes(self) -> None:
        vocab = {0: "h", 1: "i", 2: '"'}
        pv = _pv(vocab)
        model = _mock_static([1.0, 0.5, 0.0])

        with patch("builtins.print"):
            tokens, value = process_string(
                model, "ctx", pv, fast=False
            )

        self.assertTrue(value.startswith('"'))
        self.assertTrue(value.endswith('"'))

    def test_inner_content_has_no_extra_quotes(self) -> None:
        pv, model = self._str_vocab_model()

        with patch("builtins.print"):
            tokens, value = process_string(model, "ctx", pv, fast=True)

        inner = value[1:-1]
        self.assertFalse(inner.startswith('"'))
        self.assertFalse(inner.endswith('"'))

    def test_fast_streams_content(self) -> None:
        pv, model = self._str_vocab_model()

        with patch("builtins.print") as mock_print:
            process_string(model, "ctx", pv, fast=True)

        printed = "".join(
            str(c.args[0])
            for c in mock_print.call_args_list
            if c.args
        )
        self.assertIn("h", printed)
        self.assertIn("i", printed)


# ---------------------------------------------------------------------------
# process_anything
# ---------------------------------------------------------------------------


class TestProcessAnything(unittest.TestCase):
    def test_returns_value_without_terminal(self) -> None:
        pv = _pv({0: "t", 1: "r", 2: "u", 3: "e", 4: ","})
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
            tokens, value = process_anything(model, [0], pv, fast=True)

        self.assertEqual(value, "true")

    def test_same_behaviour_fast_and_thinking(self) -> None:
        """process_anything always uses 1 beam regardless of fast."""
        pv = _pv({0: "x", 1: ","})
        model_fast = _mock([[1.0, 0.0], [0.0, 1.0]])
        model_slow = _mock([[1.0, 0.0], [0.0, 1.0]])

        with patch("builtins.print"):
            _, v_fast = process_anything(
                model_fast, [0], pv, fast=True
            )
            _, v_slow = process_anything(
                model_slow, [0], pv, fast=False
            )

        self.assertEqual(v_fast, v_slow)


if __name__ == "__main__":
    unittest.main()

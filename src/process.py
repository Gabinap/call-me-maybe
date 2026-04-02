import re
from dataclasses import dataclass

import numpy as np

from llm_sdk import Small_LLM_Model

# ---------------------------------------------------------------------------
# Patterns used by each process_* function
# ---------------------------------------------------------------------------

_NUMBER_PREFIX = re.compile(r"^-?(\d+(\.\d*)?([eE][+-]?\d*)?)?$")
_NUMBER_COMPLETE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[,}]$")

_INTEGER_PREFIX = re.compile(r"^-?\d+?$")
_INTEGER_COMPLETE = re.compile(r"^-?\d+?[,}]$")

_STRING_PREFIX = re.compile(r'^([^"\\]|\\.)*$')
_STRING_COMPLETE = re.compile(r'^([^"\\]|\\.)*"$')

_ANYTHING_PREFIX = re.compile(r"^[^,}]*$")
_ANYTHING_COMPLETE = re.compile(r"^[^,}]*[,}]$")

# Characters stripped from the raw end of any generated value.
_TERMINAL_CHARS = ",} \n\r\t"


# ---------------------------------------------------------------------------
# Precomputed vocabulary — built once at startup
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedVocab:
    """All per-pattern token lists derived from the vocabulary.

    Built once in ``__main__.py`` via :py:meth:`build` and passed through
    the call chain so that no regex scan of the full vocabulary ever happens
    at generation time.
    """

    decoded: dict[int, str]
    number_starts: list[int]
    number_fallback: list[int]
    integer_starts: list[int]
    integer_fallback: list[int]
    string_starts: list[int]
    string_fallback: list[int]
    anything_starts: list[int]
    anything_fallback: list[int]
    minus_ids: list[int]

    @classmethod
    def build(cls, decoded_vocab: dict[int, str]) -> "PrecomputedVocab":
        """Scan the decoded vocabulary once and store all derived lists.

        Args:
            decoded_vocab: Mapping from token ID to clean decoded string.

        Returns:
            Fully populated PrecomputedVocab instance.
        """
        return cls(
            decoded=decoded_vocab,
            number_starts=[
                t
                for t, s in decoded_vocab.items()
                if _NUMBER_PREFIX.match(s)
            ],
            number_fallback=[
                t
                for t, s in decoded_vocab.items()
                if _NUMBER_COMPLETE.match(s)
            ],
            integer_starts=[
                t
                for t, s in decoded_vocab.items()
                if _INTEGER_PREFIX.match(s)
            ],
            integer_fallback=[
                t
                for t, s in decoded_vocab.items()
                if _INTEGER_COMPLETE.match(s)
            ],
            string_starts=[
                t
                for t, s in decoded_vocab.items()
                if _STRING_PREFIX.match(s)
            ],
            string_fallback=[
                t
                for t, s in decoded_vocab.items()
                if _STRING_COMPLETE.match(s)
            ],
            anything_starts=[
                t
                for t, s in decoded_vocab.items()
                if _ANYTHING_PREFIX.match(s)
            ],
            anything_fallback=[
                t
                for t, s in decoded_vocab.items()
                if _ANYTHING_COMPLETE.match(s)
            ],
            minus_ids=[t for t, s in decoded_vocab.items() if s == "-"],
        )


# ---------------------------------------------------------------------------
# Core generation primitive
# ---------------------------------------------------------------------------


def _best_tokens_numpy(
    logits: list[float],
    valid_ids: list[int],
    n: int,
) -> list[int]:
    """Return the n token IDs from valid_ids with the highest logits.

    Uses numpy fancy indexing and argsort for speed over Python's heapq
    when valid_ids is large (e.g. the full vocabulary for string generation).

    Args:
        logits: Raw logit vector for the current position.
        valid_ids: Candidate token IDs to consider.
        n: Number of top tokens to return.

    Returns:
        List of up to n token IDs, highest logit first.
    """
    logits_np = np.asarray(logits)
    valid_arr = np.asarray(valid_ids)
    if n == 1:
        return [int(valid_arr[np.argmax(logits_np[valid_arr])])]
    top = np.argsort(logits_np[valid_arr])[-n:][::-1]
    return [int(token_id) for token_id in valid_arr[top].tolist()]


def score_candidates(
    model_instance: Small_LLM_Model,
    encoded_context: list[int],
    pv: PrecomputedVocab,
    prefix_pattern: re.Pattern[str],
    complete_pattern: re.Pattern[str],
    valid_start_ids: list[int],
    fallback_ids: list[int],
    num_beams: int = 1,
    max_tokens: int = 50,
    stream: bool | None = None,
) -> tuple[list[int], str, float]:
    """Run beam search constrained by prefix/complete patterns.

    Optimisations applied here:
    - ``valid_start_ids`` and ``fallback_ids`` are precomputed (no vocab scan).
    - ``_best_tokens_numpy`` uses numpy fancy indexing instead of Python
      ``heapq`` / ``max`` for token selection.
    - ``_step_cache`` caches per-``current_str`` valid token lists so that
      beams sharing a prefix pay the regex cost only once per unique prefix.

    Streaming behaviour:
    - ``stream=None`` (default): auto — stream iff exactly one beam runs.
    - ``stream=False``: always silent (used by ``_score_with_negative``).

    Args:
        model_instance: The language model instance.
        encoded_context: Token IDs representing the current context.
        pv: Precomputed vocabulary data (decoded strings, pattern lists).
        prefix_pattern: Regex that a valid partial value must satisfy.
        complete_pattern: Regex that a complete value must satisfy.
        valid_start_ids: Precomputed token IDs valid as first token.
        fallback_ids: Precomputed token IDs that force termination.
        num_beams: Number of independent beams to run.
        max_tokens: Maximum tokens per beam before forcing termination.
        stream: Streaming override. ``None`` = auto, ``False`` = silent.

    Returns:
        Tuple of (token sequence including terminal, value without terminal
        character, average log-score of the winning beam).
    """
    logits: list[float] = model_instance.get_logits_from_input_ids(
        encoded_context
    )
    best_start_ids: list[int] = _best_tokens_numpy(
        logits, valid_start_ids, num_beams
    )

    do_stream: bool = (len(best_start_ids) == 1) if stream is None else stream

    beams: list[tuple[list[int], str, list[float]]] = [
        (encoded_context + [t], pv.decoded[t], [logits[t]])
        for t in best_start_ids
    ]

    if do_stream and best_start_ids:
        first_str = pv.decoded[best_start_ids[0]]
        if not complete_pattern.match(first_str):
            print(first_str, end="", flush=True)

    # Optimisation 4: cache valid token IDs per unique current_str prefix.
    # Beams that reach the same partial value share the cached result.
    _step_cache: dict[str, list[int]] = {}

    completed: list[tuple[list[int], str, float]] = []

    for beam_tokens, beam_str, beam_scores in beams:
        current_tokens = beam_tokens[:]
        current_str = beam_str
        scores = beam_scores[:]

        while not complete_pattern.match(current_str):
            if len(scores) >= max_tokens:
                break

            if current_str not in _step_cache:
                _step_cache[current_str] = [
                    t
                    for t, s in pv.decoded.items()
                    if prefix_pattern.match(current_str + s)
                    or complete_pattern.match(current_str + s)
                ] or fallback_ids

            valid_token_ids = _step_cache[current_str]
            if not valid_token_ids:
                break

            step_logits: list[float] = (
                model_instance.get_logits_from_input_ids(current_tokens)
            )
            # Optimisation 3: numpy argmax over valid subset
            best: int = _best_tokens_numpy(
                step_logits, valid_token_ids, 1
            )[0]
            scores.append(step_logits[best])

            next_str = current_str + pv.decoded[best]
            if do_stream and not complete_pattern.match(next_str):
                print(pv.decoded[best], end="", flush=True)

            current_tokens.append(best)
            current_str = next_str

        avg_score: float = (
            sum(scores) / len(scores) if scores else float("-inf")
        )
        value: str = (
            current_str[:-1]
            if complete_pattern.match(current_str)
            else current_str
        )
        value = value.rstrip(_TERMINAL_CHARS)
        completed.append((current_tokens, value, avg_score))

    best_tokens, best_value, best_score = max(completed, key=lambda x: x[2])
    return best_tokens, best_value, best_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_minus_token(
    pv: PrecomputedVocab,
    encoded: list[int],
    model_instance: Small_LLM_Model,
) -> int | None:
    """Return the best '-' token ID, or None if no such token exists.

    Args:
        pv: Precomputed vocabulary (minus_ids already extracted).
        encoded: Current token sequence (used to get logits if ambiguous).
        model_instance: The language model instance.

    Returns:
        Token ID for '-', or None.
    """
    if not pv.minus_ids:
        return None
    if len(pv.minus_ids) == 1:
        return pv.minus_ids[0]
    logits = model_instance.get_logits_from_input_ids(encoded)
    return _best_tokens_numpy(logits, pv.minus_ids, 1)[0]


def _score_with_negative(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    pv: PrecomputedVocab,
    prefix_pattern: re.Pattern[str],
    complete_pattern: re.Pattern[str],
    valid_start_ids: list[int],
    fallback_ids: list[int],
    strip_chars: str,
) -> tuple[list[int], str]:
    """Run a free beam and a forced-negative beam silently; return the best.

    Both beams use ``stream=False``. The '-' token is injected into the
    context of the negative beam (not scored) so both beams are compared
    fairly on digit logits only. The caller prints the winning value.

    Args:
        model_instance: The language model instance.
        encoded: Current token sequence.
        pv: Precomputed vocabulary data.
        prefix_pattern: Pattern for valid partial values.
        complete_pattern: Pattern for complete values.
        valid_start_ids: Precomputed valid start token IDs.
        fallback_ids: Precomputed fallback token IDs.
        strip_chars: Characters to strip from the end of generated values.

    Returns:
        Tuple of (best token sequence, best value string).
    """
    tokens_pos, value_pos, score_pos = score_candidates(
        model_instance,
        encoded,
        pv,
        prefix_pattern,
        complete_pattern,
        valid_start_ids,
        fallback_ids,
        num_beams=1,
        stream=False,
    )
    value_pos = value_pos.rstrip(strip_chars)

    minus_id = _get_minus_token(pv, encoded, model_instance)
    if minus_id is None:
        return tokens_pos, value_pos

    tokens_neg, value_neg, score_neg = score_candidates(
        model_instance,
        encoded + [minus_id],
        pv,
        prefix_pattern,
        complete_pattern,
        valid_start_ids,
        fallback_ids,
        num_beams=1,
        stream=False,
    )
    value_neg = value_neg.rstrip(strip_chars)
    if not value_neg.startswith("-"):
        value_neg = "-" + value_neg

    if score_neg > score_pos:
        return tokens_neg, value_neg
    return tokens_pos, value_pos


def _ensure_float_dot(value: str) -> str:
    """Ensure a number string has '.' or 'e' so JSON treats it as float.

    Args:
        value: Raw number string without terminal character.

    Returns:
        Number string guaranteed to contain '.' or 'e'/'E'.
    """
    if "." not in value and "e" not in value.lower():
        value += ".0"
    return value


# ---------------------------------------------------------------------------
# Public process_* helpers
# ---------------------------------------------------------------------------


def process_number(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    pv: PrecomputedVocab,
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a number parameter.

    The returned value always contains a decimal point (e.g. '42' → '42.0').

    fast=True : 1 beam, stream=auto (True) → each token printed live.
    fast=False: two silent beams (positive vs forced negative), winner
                printed once at the end.
    """
    if fast:
        result_tokens, value, _ = score_candidates(
            model_instance,
            encoded,
            pv,
            _NUMBER_PREFIX,
            _NUMBER_COMPLETE,
            pv.number_starts,
            pv.number_fallback,
            num_beams=1,
        )
        value = _ensure_float_dot(value)
    else:
        result_tokens, value = _score_with_negative(
            model_instance,
            encoded,
            pv,
            _NUMBER_PREFIX,
            _NUMBER_COMPLETE,
            pv.number_starts,
            pv.number_fallback,
            strip_chars=_TERMINAL_CHARS,
        )
        value = _ensure_float_dot(value)
        print(value, end="", flush=True)
    return result_tokens, value


def process_integer(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    pv: PrecomputedVocab,
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate an integer parameter.

    fast=True : 1 beam, stream=auto (True) → each token printed live.
    fast=False: two silent beams (positive vs forced negative), winner
                printed once at the end.
    """
    if fast:
        result_tokens, value, _ = score_candidates(
            model_instance,
            encoded,
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            num_beams=1,
        )
    else:
        result_tokens, value = _score_with_negative(
            model_instance,
            encoded,
            pv,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            pv.integer_starts,
            pv.integer_fallback,
            strip_chars=_TERMINAL_CHARS,
        )
        print(value, end="", flush=True)
    return result_tokens, value


def process_string(
    model_instance: Small_LLM_Model,
    context: str,
    pv: PrecomputedVocab,
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a string parameter.

    fast=True : 1 beam, stream=auto (True) → opening quote, tokens live,
                closing quote.
    fast=False: 4 beams, stream=auto (False) → silent search, full value
                printed once at the end.
    """
    encoded_context: list[int] = (
        model_instance.encode(context + '"')[0].tolist()
    )
    print('"', end="", flush=True)

    result_tokens, value, _ = score_candidates(
        model_instance,
        encoded_context,
        pv,
        _STRING_PREFIX,
        _STRING_COMPLETE,
        pv.string_starts,
        pv.string_fallback,
        num_beams=1 if fast else 4,
    )
    value = value.strip('"')

    if fast:
        print('"', end="", flush=True)
    else:
        print(value + '"', end="", flush=True)

    return result_tokens, f'"{value}"'


def process_anything(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    pv: PrecomputedVocab,
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a parameter of unknown type (always 1 beam, streamed)."""
    result_tokens, value, _ = score_candidates(
        model_instance,
        encoded,
        pv,
        _ANYTHING_PREFIX,
        _ANYTHING_COMPLETE,
        pv.anything_starts,
        pv.anything_fallback,
        num_beams=1,
    )
    value = value.rstrip(_TERMINAL_CHARS)
    return result_tokens, value

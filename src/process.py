import heapq
import re

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

# Characters that may appear at the end of a raw generated value and must
# be stripped before the value is returned to the caller.
_TERMINAL_CHARS = ",} \n\r\t"


# ---------------------------------------------------------------------------
# Core generation primitive
# ---------------------------------------------------------------------------


def score_candidates(
    model_instance: Small_LLM_Model,
    encoded_context: list[int],
    decoded_vocab: dict[int, str],
    prefix_pattern: re.Pattern,
    complete_pattern: re.Pattern,
    num_beams: int = 1,
    max_tokens: int = 50,
    stream: bool | None = None,
) -> tuple[list[int], str, float]:
    """Run beam search constrained by prefix/complete patterns.

    Streaming behaviour is controlled by ``stream``:
    - ``None`` (default): auto-derived — stream if and only if exactly one
      beam is running (``len(start_ids) == 1``).
    - ``False``: always silent, regardless of beam count. Used internally by
      ``_score_with_negative`` so that comparison beams never print.

    The terminal token IS included in the returned token sequence so the
    model retains full context. The terminal character is stripped from the
    returned value string.

    Args:
        model_instance: The language model instance.
        encoded_context: Token IDs representing the current context.
        decoded_vocab: Precomputed mapping (token ID -> clean decoded string).
        prefix_pattern: Regex that a valid partial value must satisfy.
        complete_pattern: Regex that a complete value must satisfy.
        num_beams: Number of independent beams; best average-score one returned.
        max_tokens: Maximum tokens per beam before forcing termination.
        stream: Streaming override. ``None`` = auto, ``False`` = always silent.

    Returns:
        Tuple of (full token sequence including terminal, value without
        terminal character, average log-score of the winning beam).
    """
    valid_start_ids: list[int] = [
        t for t, s in decoded_vocab.items() if prefix_pattern.match(s)
    ]
    fallback_ids: list[int] = [
        t for t, s in decoded_vocab.items() if complete_pattern.match(s)
    ]

    logits: list[float] = model_instance.get_logits_from_input_ids(encoded_context)
    best_start_ids: list[int] = heapq.nlargest(
        num_beams, valid_start_ids, key=lambda t: logits[t]
    )

    # Resolve streaming: auto means stream iff exactly one beam
    do_stream: bool = (len(best_start_ids) == 1) if stream is None else stream

    beams: list[tuple[list[int], str, list[float]]] = [
        (encoded_context + [t], decoded_vocab[t], [logits[t]]) for t in best_start_ids
    ]

    # Stream the first token of the single beam if it is not already terminal
    if do_stream and best_start_ids:
        first_str = decoded_vocab[best_start_ids[0]]
        if not complete_pattern.match(first_str):
            print(first_str, end="", flush=True)

    completed: list[tuple[list[int], str, float]] = []

    for beam_tokens, beam_str, beam_scores in beams:
        current_tokens = beam_tokens[:]
        current_str = beam_str
        scores = beam_scores[:]

        while not complete_pattern.match(current_str):
            if len(scores) >= max_tokens:
                break

            valid_token_ids: list[int] = [
                t
                for t, s in decoded_vocab.items()
                if prefix_pattern.match(current_str + s)
                or complete_pattern.match(current_str + s)
            ]

            if not valid_token_ids:
                valid_token_ids = fallback_ids
            if not valid_token_ids:
                break

            step_logits: list[float] = model_instance.get_logits_from_input_ids(
                current_tokens
            )
            best: int = max(valid_token_ids, key=lambda t: step_logits[t])
            scores.append(step_logits[best])

            next_str = current_str + decoded_vocab[best]
            if do_stream and not complete_pattern.match(next_str):
                print(decoded_vocab[best], end="", flush=True)

            current_tokens.append(best)
            current_str = next_str

        avg_score: float = sum(scores) / len(scores) if scores else float("-inf")
        # Strip the terminal character AND any trailing whitespace that BPE
        # tokens may have embedded (e.g. a "}\n" token would leave a stray \n)
        value: str = (
            current_str[:-1] if complete_pattern.match(current_str) else current_str
        )
        value = value.rstrip(_TERMINAL_CHARS)
        completed.append((current_tokens, value, avg_score))

    best_tokens, best_value, best_score = max(completed, key=lambda x: x[2])
    return best_tokens, best_value, best_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_minus_token(
    decoded_vocab: dict[int, str],
    encoded: list[int],
    model_instance: Small_LLM_Model,
) -> int | None:
    """Return the token ID whose decoded form is exactly '-', or None."""
    minus_ids: list[int] = [t for t, s in decoded_vocab.items() if s == "-"]
    if not minus_ids:
        return None
    if len(minus_ids) == 1:
        return minus_ids[0]
    logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
    return max(minus_ids, key=lambda t: logits[t])


def _score_with_negative(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    decoded_vocab: dict[int, str],
    prefix_pattern: re.Pattern,
    complete_pattern: re.Pattern,
    strip_chars: str,
) -> tuple[list[int], str]:
    """Run a free beam and a forced-negative beam silently; return the best.

    Both beams use ``stream=False`` so nothing is printed during comparison.
    The '-' token is injected into the *context* of the negative beam (not
    scored as a beam token) so both beams are compared fairly on digit logits.
    The caller is responsible for printing the winning value.
    """
    tokens_pos, value_pos, score_pos = score_candidates(
        model_instance,
        encoded,
        decoded_vocab,
        prefix_pattern,
        complete_pattern,
        num_beams=1,
        stream=False,
    )
    value_pos = value_pos.rstrip(strip_chars)

    minus_id = _get_minus_token(decoded_vocab, encoded, model_instance)
    if minus_id is None:
        return tokens_pos, value_pos

    tokens_neg, value_neg, score_neg = score_candidates(
        model_instance,
        encoded + [minus_id],
        decoded_vocab,
        prefix_pattern,
        complete_pattern,
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
    """Ensure a number string has a decimal point so JSON parses it as float.

    Scientific notation (e.g. '2e34') already implies float and is left
    unchanged. Plain integers like '42' or '-7' become '42.0' and '-7.0'.

    Args:
        value: Raw number string (no terminal character).

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
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a number parameter.

    The returned value always contains a decimal point so it serialises as a
    JSON float (e.g. '42' → '42.0').

    fast=True : 1 beam, stream=auto (True) → each token printed live.
    fast=False: two silent beams (positive vs forced negative), winner
                printed once at the end.
    """
    if fast:
        result_tokens, value, _ = score_candidates(
            model_instance,
            encoded,
            decoded_vocab,
            _NUMBER_PREFIX,
            _NUMBER_COMPLETE,
            num_beams=1,
        )
        value = _ensure_float_dot(value)
    else:
        result_tokens, value = _score_with_negative(
            model_instance,
            encoded,
            decoded_vocab,
            _NUMBER_PREFIX,
            _NUMBER_COMPLETE,
            strip_chars=_TERMINAL_CHARS,
        )
        value = _ensure_float_dot(value)
        print(value, end="", flush=True)
    return result_tokens, value


def process_integer(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    decoded_vocab: dict[int, str],
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
            decoded_vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            num_beams=1,
        )
    else:
        result_tokens, value = _score_with_negative(
            model_instance,
            encoded,
            decoded_vocab,
            _INTEGER_PREFIX,
            _INTEGER_COMPLETE,
            strip_chars=_TERMINAL_CHARS,
        )
        print(value, end="", flush=True)
    return result_tokens, value


def process_string(
    model_instance: Small_LLM_Model,
    context: str,
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a string parameter.

    fast=True : 1 beam, stream=auto (True) → opening quote printed, then
                each content token live, then closing quote.
    fast=False: 4 beams, stream=auto (False since >1 beam) → nothing
                printed during search, full value printed once at the end.
    """
    encoded_context: list[int] = model_instance.encode(context + '"')[0].tolist()
    print('"', end="", flush=True)

    result_tokens, value, _ = score_candidates(
        model_instance,
        encoded_context,
        decoded_vocab,
        _STRING_PREFIX,
        _STRING_COMPLETE,
        num_beams=1 if fast else 4,
    )
    value = value.strip('"')

    if fast:
        # Content already streamed token by token; only closing quote remains
        print('"', end="", flush=True)
    else:
        # Nothing was printed during 4-beam search; print full value now
        print(value + '"', end="", flush=True)

    return result_tokens, f'"{value}"'


def process_anything(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a parameter of unknown type.

    Always uses 1 beam → stream=auto (True). Both modes behave identically
    since there is no multi-beam strategy for unknown types.
    """
    result_tokens, value, _ = score_candidates(
        model_instance,
        encoded,
        decoded_vocab,
        _ANYTHING_PREFIX,
        _ANYTHING_COMPLETE,
        num_beams=1,
    )
    value = value.rstrip(_TERMINAL_CHARS)
    return result_tokens, value

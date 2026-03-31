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
    forced_starts: list[int | None] | None = None,
) -> tuple[list[int], str, float]:
    """Run beam search constrained by prefix/complete patterns.

    Streaming is derived automatically: tokens are printed immediately when
    exactly one beam is running (``effective_count == 1``). Terminal tokens
    are never printed — callers handle them.

    Args:
        model_instance: The language model instance.
        encoded_context: Token IDs representing the current context.
        decoded_vocab: Precomputed mapping (token ID -> clean decoded string).
        prefix_pattern: Regex that a valid partial value must satisfy.
        complete_pattern: Regex that a complete value must satisfy.
        num_beams: Number of beams when ``forced_starts`` is not provided.
        max_tokens: Maximum tokens per beam before forcing termination.
        forced_starts: If provided, overrides ``num_beams``. Each entry is
                       either a token ID (forced first token for that beam) or
                       ``None`` (pick the best token freely from the vocab).
                       Two entries → 2 beams → no streaming; one entry → 1
                       beam → streaming.

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

    if forced_starts is not None:
        best_free: int = max(valid_start_ids, key=lambda t: logits[t])
        start_ids: list[int] = [
            best_free if fs is None else fs for fs in forced_starts
        ]
    else:
        start_ids = heapq.nlargest(num_beams, valid_start_ids, key=lambda t: logits[t])

    # Stream only when a single beam is running
    stream: bool = len(start_ids) == 1

    beams: list[tuple[list[int], str, list[float]]] = [
        (encoded_context + [t], decoded_vocab[t], [logits[t]])
        for t in start_ids
    ]

    if stream and start_ids:
        first_str = decoded_vocab[start_ids[0]]
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
                t for t, s in decoded_vocab.items()
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
            if stream and not complete_pattern.match(next_str):
                print(decoded_vocab[best], end="", flush=True)

            current_tokens.append(best)
            current_str = next_str

        avg_score: float = sum(scores) / len(scores) if scores else float("-inf")
        value: str = current_str[:-1] if complete_pattern.match(current_str) else current_str
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
    """Run a free beam and a forced-negative beam; return the best result.

    Uses ``forced_starts=[None, minus_id]`` → 2 beams → no streaming.
    Both beams are compared silently; the caller prints the winner.
    """
    minus_id = _get_minus_token(decoded_vocab, encoded, model_instance)

    if minus_id is None:
        result_tokens, value, _ = score_candidates(
            model_instance, encoded, decoded_vocab,
            prefix_pattern, complete_pattern, num_beams=1,
        )
        return result_tokens, value.rstrip(strip_chars)

    result_tokens, value, _ = score_candidates(
        model_instance, encoded, decoded_vocab,
        prefix_pattern, complete_pattern,
        forced_starts=[None, minus_id],
    )
    return result_tokens, value.rstrip(strip_chars)


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

    fast=True : 1 beam → stream=True automatically, each token printed live.
    fast=False: forced_starts=[None, minus_id] → 2 beams → stream=False,
                best of positive and negative printed once at the end.
    """
    if fast:
        result_tokens, value, _ = score_candidates(
            model_instance, encoded, decoded_vocab,
            _NUMBER_PREFIX, _NUMBER_COMPLETE, num_beams=1,
        )
        value = value.rstrip(",}")
    else:
        result_tokens, value = _score_with_negative(
            model_instance, encoded, decoded_vocab,
            _NUMBER_PREFIX, _NUMBER_COMPLETE, strip_chars=",}",
        )
        print(value, end="", flush=True)
    return result_tokens, value


def process_integer(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate an integer parameter.

    fast=True : 1 beam → stream=True automatically, each token printed live.
    fast=False: forced_starts=[None, minus_id] → 2 beams → stream=False,
                best of positive and negative printed once at the end.
    """
    if fast:
        result_tokens, value, _ = score_candidates(
            model_instance, encoded, decoded_vocab,
            _INTEGER_PREFIX, _INTEGER_COMPLETE, num_beams=1,
        )
        value = value.rstrip(",}")
    else:
        result_tokens, value = _score_with_negative(
            model_instance, encoded, decoded_vocab,
            _INTEGER_PREFIX, _INTEGER_COMPLETE, strip_chars=",}",
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

    fast=True : 1 beam → stream=True, tokens printed live; only `"` printed
                at the end by this function.
    fast=False: 4 beams → stream=False, full value printed once at the end.
    """
    encoded_context: list[int] = model_instance.encode(context + '"')[0].tolist()
    print('"', end="", flush=True)

    result_tokens, value, _ = score_candidates(
        model_instance, encoded_context, decoded_vocab,
        _STRING_PREFIX, _STRING_COMPLETE,
        num_beams=1 if fast else 4,
    )
    value = value.strip('"')

    if fast:
        print('"', end="", flush=True)          # content already streamed
    else:
        print(value + '"', end="", flush=True)  # print full value at once

    return result_tokens, f'"{value}"'


def process_anything(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> tuple[list[int], str]:
    """Generate a parameter of unknown type.

    Always uses 1 beam (stream=True); no multi-beam strategy for unknown types.
    In thinking mode the single beam still streams — the distinction is only
    meaningful for numbers and strings.
    """
    result_tokens, value, _ = score_candidates(
        model_instance, encoded, decoded_vocab,
        _ANYTHING_PREFIX, _ANYTHING_COMPLETE, num_beams=1,
    )
    value = value.rstrip(",}")
    return result_tokens, value
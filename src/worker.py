"""Worker function for multiprocessing-based parallel generation.

This module exists separately from ``__main__`` because
``multiprocessing`` with the ``"spawn"`` start method pickles the
target function by reference (e.g. ``src.worker.worker_process_batch``).
If the function lived in ``__main__``, the child process would try
to look it up in Python's built-in ``__main__`` module and fail with
``AttributeError: Can't get attribute ... on <module '__main__'>``.

By placing it here, the pickle reference resolves correctly in the
child process via a normal ``import src.worker``.
"""

from typing import Any

import json
import os
import sys

from .get_from_llm import get_function_parameters, get_valid_function_name
from .parsing import FunctionDef
from .process import PrecomputedVocab

from llm_sdk import Small_LLM_Model


def _get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary from the model's vocabulary file.

    Args:
        model_instance: The language model instance.

    Returns:
        A dictionary mapping token strings to their token IDs.
    """
    with open(
        model_instance.get_path_to_vocab_file(), encoding="utf-8"
    ) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    return {
        token: token_id
        for token, token_id in raw.items()
        if isinstance(token, str) and isinstance(token_id, int)
    }


def _init_model(
    model_name: str,
    functions: list[FunctionDef],
) -> tuple[
    Small_LLM_Model,
    dict[int, str],
    PrecomputedVocab,
    list[int],
]:
    """Load the model, build the vocabulary, and precompute token lists.

    Called once per worker process.

    Args:
        model_name: HuggingFace model identifier.
        functions: Function definitions (used to encode function names).

    Returns:
        Tuple of (model_instance, reverse_vocab, pv, encoded_func_names).
    """
    model_instance = Small_LLM_Model(model_name=model_name)

    vocab: dict[str, int] = _get_vocabulary(model_instance)
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}

    decoded_vocab: dict[int, str] = {
        t: model_instance.decode([t]) for t in reverse_vocab
    }
    pv = PrecomputedVocab.build(decoded_vocab)

    func_names: str = ", ".join(f.name for f in functions)
    encoded_func_names: list[int] = (
        model_instance.encode(func_names)[0].tolist()
    )

    return model_instance, reverse_vocab, pv, encoded_func_names


def _process_one_prompt(
    prompt: str,
    model_instance: Small_LLM_Model,
    reverse_vocab: dict[int, str],
    functions: list[FunctionDef],
    encoded_func_names: list[int],
    pv: PrecomputedVocab,
    fast: bool,
) -> dict[str, Any]:
    """Process a single prompt and return the result dict.

    Args:
        prompt: Natural language prompt to process.
        model_instance: The language model.
        reverse_vocab: Token-ID-to-BPE-string mapping.
        functions: Available function definitions.
        encoded_func_names: Pre-encoded function name context tokens.
        pv: Precomputed vocabulary data.
        fast: If True, use single-beam streaming mode.

    Returns:
        Dict with keys ``prompt``, ``name``, ``parameters``.
    """
    encoded, func_name = get_valid_function_name(
        reverse_vocab,
        model_instance,
        functions,
        prompt,
        encoded_func_names,
        pv,
    )
    encoded = encoded[len(encoded_func_names):]
    function_def = next(
        f for f in functions if f.name == func_name
    )

    parameters: dict[str, Any] = get_function_parameters(
        reverse_vocab,
        model_instance,
        encoded,
        function_def,
        prompt,
        pv,
        fast=fast,
    )

    return {
        "prompt": prompt,
        "name": func_name,
        "parameters": parameters,
    }


def worker_process_batch(
    batch_args: tuple[
        list[tuple[int, str]],
        str,
        list[FunctionDef],
        bool,
    ],
) -> list[tuple[int, dict[str, Any]]]:
    """Worker function executed in a child process.

    Each child process loads its own copy of the model and processes
    its assigned batch of prompts.  Because each process has its own
    Python interpreter and GIL, all workers run with true CPU
    parallelism.

    Args:
        batch_args: Tuple of:
            - indexed_prompts: list of ``(original_index, prompt_text)``
            - model_name: HuggingFace model identifier
            - functions: Function definitions
            - fast: Streaming mode flag

    Returns:
        List of ``(original_index, result_dict)`` pairs.
    """
    indexed_prompts, model_name, functions, fast = batch_args

    # Silence stdout in the child process.  Every process inherits
    # the parent's stdout, so the token-by-token streaming prints
    # from score_candidates / process_string / etc. would interleave
    # across workers and produce garbled output.  Redirecting to
    # devnull suppresses all streaming; the parent process prints
    # a clean progress counter instead.
    sys.stdout = open(os.devnull, "w")

    model_instance, reverse_vocab, pv, encoded_func_names = (
        _init_model(model_name, functions)
    )

    results: list[tuple[int, dict[str, Any]]] = []
    for idx, prompt in indexed_prompts:
        try:
            result = _process_one_prompt(
                prompt, model_instance, reverse_vocab,
                functions, encoded_func_names, pv, fast,
            )
            results.append((idx, result))
        except Exception as exc:
            results.append((idx, {
                "prompt": prompt,
                "name": "error",
                "parameters": {"error": str(exc)},
            }))
    return results

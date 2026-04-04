import json
import math
import multiprocessing
import os
from typing import Any

from llm_sdk import Small_LLM_Model

from .parsing import Args, FunctionDef, parse
from .process import PrecomputedVocab
from .worker import (
    _init_model,
    _process_one_prompt,
    worker_process_batch,
)


def get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary from the model's vocabulary file.

    Args:
        model_instance: The language model instance to extract vocabulary
                        from.

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


def write_output(results: list[dict[str, Any]], output_path: str) -> None:
    """Write the results list to a JSON file, creating parent dirs.

    Args:
        results: List of result dicts, each with prompt/name/parameters
                 keys.
        output_path: Destination file path.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nOutput written to {output_path}", flush=True)
    except OSError as e:
        print(f"\nError: could not write output file — {e}", flush=True)


# ---------------------------------------------------------------------------
# Sequential generation (default, workers=1)
# ---------------------------------------------------------------------------


def _generate_sequential(
    prompts: list[str],
    model_instance: Small_LLM_Model,
    reverse_vocab: dict[int, str],
    functions: list[FunctionDef],
    encoded_func_names: list[int],
    pv: PrecomputedVocab,
    fast: bool,
) -> list[dict[str, Any]]:
    """Process prompts one by one with live streaming output.

    Args:
        prompts: List of natural language prompts.
        model_instance: The language model.
        reverse_vocab: Token-ID-to-BPE-string mapping.
        functions: Available function definitions.
        encoded_func_names: Pre-encoded function name context tokens.
        pv: Precomputed vocabulary data.
        fast: Single-beam streaming mode flag.

    Returns:
        Ordered list of result dicts.
    """
    results: list[dict[str, Any]] = []
    print("[", end="", flush=True)

    for i, prompt in enumerate(prompts):
        result = _process_one_prompt(
            prompt, model_instance, reverse_vocab,
            functions, encoded_func_names, pv, fast,
        )
        results.append(result)

        if i < len(prompts) - 1:
            print("\n  },", end="", flush=True)
        else:
            print("\n  }")

    print("]")
    return results


# ---------------------------------------------------------------------------
# Parallel generation (workers > 1) — one process per worker
# ---------------------------------------------------------------------------


def _generate_parallel(
    prompts: list[str],
    functions: list[FunctionDef],
    model_name: str,
    fast: bool,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Process prompts across multiple OS processes.

    The prompts are split into ``num_workers`` contiguous batches.
    Each batch is sent to a worker process which loads its own model
    and processes its prompts independently.  Results are reassembled
    in input order using the original indices.

    The worker function ``worker_process_batch`` lives in ``src.worker``
    (not in ``__main__``) so that ``multiprocessing`` with ``"spawn"``
    can pickle it by reference correctly.

    Args:
        prompts: List of natural language prompts.
        functions: Available function definitions.
        model_name: HuggingFace model identifier (each worker loads it).
        fast: Single-beam mode flag.
        num_workers: Number of worker processes.

    Returns:
        Ordered list of result dicts (same order as input prompts).
    """
    total = len(prompts)
    num_workers = min(num_workers, total)
    batch_size = math.ceil(total / num_workers)

    # Split prompts into batches, preserving original indices.
    batches: list[
        tuple[list[tuple[int, str]], str, list[FunctionDef], bool]
    ] = []
    for w in range(num_workers):
        start = w * batch_size
        end = min(start + batch_size, total)
        if start >= total:
            break
        indexed = [(i, prompts[i]) for i in range(start, end)]
        batches.append((indexed, model_name, functions, fast))

    print(
        f"Processing {total} prompts with {len(batches)} "
        f"worker processes (each loads its own model)...",
        flush=True,
    )

    # Use "spawn" context for portability (avoids fork + PyTorch issues).
    ctx = multiprocessing.get_context("spawn")
    results: list[dict[str, Any] | None] = [None] * total

    with ctx.Pool(processes=len(batches)) as pool:
        completed = 0
        for batch_results in pool.imap_unordered(
            worker_process_batch, batches
        ):
            for idx, result in batch_results:
                results[idx] = result
                completed += 1
            print(
                f"  [{completed}/{total}] prompts completed",
                flush=True,
            )

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def start_generation(args: Args) -> None:
    """Generate function call completions for the given prompts.

    When ``args.workers > 1`` and there are multiple prompts, the
    work is distributed across separate OS processes.  Each process
    loads its own model instance, giving true parallelism (no GIL
    contention).  Otherwise, sequential mode is used with live
    streaming output.

    Args:
        args: Parsed arguments containing prompts, function definitions,
              mode, model name, and worker count.
    """
    if args.workers > 1 and len(args.prompts) > 1:
        results = _generate_parallel(
            args.prompts,
            args.functions,
            args.model,
            args.mode == "fast",
            args.workers,
        )
    else:
        fast: bool = args.mode == "fast"
        model_instance, reverse_vocab, pv, encoded_func_names = (
            _init_model(args.model, args.functions)
        )
        results = _generate_sequential(
            args.prompts, model_instance, reverse_vocab,
            args.functions, encoded_func_names, pv, fast,
        )

    write_output(results, args.output)


def main() -> None:
    """Main entry point for the function calling generation task."""
    start_generation(parse())


if __name__ == "__main__":
    main()

import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .get_from_llm import get_function_parameters, get_valid_function_name
from .parsing import Args, parse
from .process import PrecomputedVocab


def get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary from the model's vocabulary file.

    Args:
        model_instance: The language model instance to extract vocabulary from.

    Returns:
        A dictionary mapping token strings to their token IDs.
    """
    with open(model_instance.get_path_to_vocab_file(), encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    return {
        token: token_id
        for token, token_id in raw.items()
        if isinstance(token, str) and isinstance(token_id, int)
    }


def write_output(results: list[dict[str, Any]], output_path: str) -> None:
    """Write the results list to a JSON file, creating parent dirs if needed.

    Args:
        results: List of result dicts, each with prompt/name/parameters keys.
        output_path: Destination file path.
    """
    import os

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nOutput written to {output_path}", flush=True)
    except OSError as e:
        print(f"\nError: could not write output file — {e}", flush=True)


def start_generation(args: Args, model: str) -> None:
    """Generate function call completions for the given prompts.

    Optimisation 1: ``PrecomputedVocab.build`` scans the full vocabulary
    once and stores all per-pattern token ID lists. Every call to
    ``score_candidates`` receives these precomputed lists directly, avoiding
    O(vocab × regex) scans at generation time.

    Args:
        args: Parsed arguments containing prompts, function definitions, mode.
        model: The name/path of the language model to use.
    """
    fast: bool = args.mode == "fast"

    model_instance = Small_LLM_Model(model_name=model)

    vocab: dict[str, int] = get_vocabulary(model_instance)
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}

    # Decode every token once; pass the result everywhere instead of calling
    # model_instance.decode() repeatedly at generation time.
    decoded_vocab: dict[int, str] = {
        t: model_instance.decode([t]) for t in reverse_vocab
    }

    # Optimisation 1: single full-vocabulary scan at startup.
    pv = PrecomputedVocab.build(decoded_vocab)

    func_names: str = ", ".join(f.name for f in args.functions)
    encoded_func_names: list[int] = (
        model_instance.encode(func_names)[0].tolist()
    )

    results: list[dict[str, Any]] = []

    print("[", end="", flush=True)

    for i, prompt in enumerate(args.prompts):
        encoded, func_name = get_valid_function_name(
            reverse_vocab,
            model_instance,
            args.functions,
            prompt,
            encoded_func_names,
            pv,
        )
        encoded = encoded[len(encoded_func_names):]
        function_def = next(f for f in args.functions if f.name == func_name)

        parameters: dict[str, Any] = get_function_parameters(
            reverse_vocab,
            model_instance,
            encoded,
            function_def,
            prompt,
            pv,
            fast=fast,
        )

        results.append(
            {
                "prompt": prompt,
                "name": func_name,
                "parameters": parameters,
            }
        )

        if i < len(args.prompts) - 1:
            print("\n  },", end="", flush=True)
        else:
            print("\n  }")

    print("]")
    write_output(results, args.output)


def main(model: str = "Qwen/Qwen3-0.6B") -> None:
    """Main entry point for the function calling generation task.

    Args:
        model: The language model to use (default: Qwen/Qwen3-0.6B).
    """
    start_generation(parse(), model)


if __name__ == "__main__":
    main()

import json
import os
from typing import Any

from llm_sdk import Small_LLM_Model

from .get_from_llm import get_function_parameters, get_valid_function_name
from .parsing import Args, parse


def get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary from the model's vocabulary file.

    Args:
        model_instance: The language model instance to extract vocabulary from.

    Returns:
        A dictionary mapping token strings to their token IDs.
    """
    with open(model_instance.get_path_to_vocab_file()) as f:
        return json.load(f)


def write_output(results: list[dict[str, Any]], output_path: str) -> None:
    """Write the results list to a JSON file, creating parent dirs if needed.

    Args:
        results: List of result dicts, each with prompt/name/parameters keys.
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


def start_generation(args: Args, model: str) -> None:
    """Generate function call completions for the given prompts.

    For each prompt, uses the language model to select a function and generate
    its arguments using constrained decoding. Results are printed incrementally
    to stdout and written as a JSON array to ``args.output``.

    Args:
        args: Parsed arguments containing prompts, function definitions, and mode.
        model: The name/path of the language model to use.
    """
    fast: bool = args.mode == "fast"

    model_instance = Small_LLM_Model(model_name=model)

    vocab: dict[str, int] = get_vocabulary(model_instance)
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
    decoded_vocab: dict[int, str] = {
        t: model_instance.decode([t]) for t in reverse_vocab
    }

    func_names: str = ", ".join(f.name for f in args.functions)
    encoded_func_names: list[int] = model_instance.encode(func_names)[0].tolist()

    results: list[dict[str, Any]] = []

    # `[` has no trailing newline: the leading `\n` in each prompt_prefix acts as
    # the line separator between entries and as the single newline after `[`.
    print("[", end="", flush=True)

    for i, prompt in enumerate(args.prompts):
        encoded, func_name = get_valid_function_name(
            reverse_vocab,
            model_instance,
            args.functions,
            prompt,
            encoded_func_names,
        )
        encoded = encoded[len(encoded_func_names):]
        function_def = next(f for f in args.functions if f.name == func_name)

        parameters: dict[str, Any] = get_function_parameters(
            reverse_vocab,
            model_instance,
            encoded,
            function_def,
            prompt,
            decoded_vocab,
            fast=fast,
        )

        results.append(
            {
                "prompt": prompt,
                "name": func_name,
                "parameters": parameters,
            }
        )

        # No trailing newline: next entry's leading `\n` (prompt_prefix) provides it.
        if i < len(args.prompts) - 1:
            print("\n  },", end="", flush=True)
        else:
            print("\n  }")  # last entry: print's own \n separates from `]`

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

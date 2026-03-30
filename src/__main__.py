import json

from llm_sdk import Small_LLM_Model

from .get_from_llm import get_function_parameters, get_valid_function_name
from .parsing import Args, parse


def get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    """Load vocabulary from the model's vocabulary file.

    Args:
        model_instance: The language model instance to extract vocabulary from.

    Returns:
        A dictionary mapping vocabulary words to their token IDs.
    """
    vocab_file_path = model_instance.get_path_to_vocab_file()
    with open(vocab_file_path) as f:
        vocab = json.load(f)
    return vocab


def start_generation(args: Args, model: str) -> None:
    """Generate function call completions for the given prompts.

    For each prompt, constructs a JSON structure and uses the language model to
    generate valid function names from the available functions.

    Args:
        args: Command-line arguments containing prompts and functions.
        model: The name/path of the language model to use.
    """
    func_names: str = ", ".join(name.name for name in args.functions)
    model_instance = Small_LLM_Model(model_name=model)
    vocab: dict[str, int] = get_vocabulary(model_instance)
    encoded_func_names: list[int] = model_instance.encode(func_names)[0].tolist()
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
    print("[")
    for prompt in args.prompts[8:12]:  # TODO enlever le slicing
        encoded, func_name = get_valid_function_name(
            reverse_vocab,
            model_instance,
            args.functions,
            prompt,
            encoded_func_names,
        )
        encoded = encoded[len(encoded_func_names) :]
        get_function_parameters(
            reverse_vocab,
            model_instance,
            encoded,
            next(function for function in args.functions if function.name == func_name),
            prompt,
        )
    print("]")


def main(model: str = "Qwen/Qwen3-0.6B") -> None:
    """Main entry point for the function calling generation task.

    Parses command-line arguments and initiates the generation process.

    Args:
        model: The language model to use (default: Qwen/Qwen3-0.6B).
    """
    start_generation(parse(), model)


if __name__ == "__main__":
    main()

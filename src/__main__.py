import json

from llm_sdk import Small_LLM_Model

from .parsing import Args, FunctionDef, parse


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


def get_valid_function_name(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    encoded: list[int],
    functions: list[FunctionDef],
) -> tuple[list[int], str]:
    """Generate and validate a function name using the language model.

    Iteratively generates tokens to form a function name, validating that each
    partial name matches at least one available function. If a single function
    matches, its remaining characters are printed directly.

    Args:
        name: The base name prefix for validation.
        reverse_vocab: Mapping from token IDs to vocabulary tokens.
        model_instance: The language model instance for generating logits.
        encoded: List of encoded token IDs for the model input.
        functions: List of available function definitions to validate against.
    """
    valid_function_names: list[str] = [fname.name for fname in functions]
    decoded: list[str] = [""]
    clean_name: str = ""
    while '"' not in decoded[-1]:
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        # Filter logits to keep only valid function name characters
        current_name: str = "".join(c for c in decoded)
        clean_name = current_name.split('"')[0]

        # Check if clean_name matches exactly one function
        matching_functions: list[str] = [
            fname
            for fname in valid_function_names
            if fname.startswith(clean_name)
        ]
        if len(matching_functions) == 1:
            # Only one function matches, write the rest directly
            remaining: str = matching_functions[0][len(clean_name):]
            encoded.extend(model_instance.encode(remaining)[0].tolist())
            print(remaining, end="", flush=True)
            clean_name = matching_functions[0]
            break

        # Get valid next character tokens from function names
        valid_next_chars: set[str] = set()
        for func_name in valid_function_names:
            starts_ok: bool = func_name.startswith(clean_name)
            longer: bool = len(func_name) > len(clean_name)
            if starts_ok and longer:
                valid_next_chars.add(
                    "".join(
                        c
                        for c in func_name
                        if func_name.index(c) > len(clean_name)
                    )
                )

        # Filter logits to valid character tokens
        valid_token_ids: list[int] = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if any(
                func_name.startswith(clean_name + token_str)
                for func_name in valid_function_names
            )
        ]

        if valid_token_ids:
            # Get the best logit among valid tokens
            best_token: int = max(valid_token_ids, key=lambda t: logits[t])
            next_token_id: int = best_token
        else:
            break

        decoded.append(model_instance.decode([next_token_id]))
        encoded.append(next_token_id)
        print(decoded[-1], end="", flush=True)
    encoded.append(model_instance.encode('",')[0].tolist()[0])
    print('",')
    return encoded, clean_name


def get_function_parameters(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    encoded: list[int],
    functions: list[FunctionDef],
    func_name: str,
) -> None:
    """Generate function parameters for the given function name.

    Args:
        reverse_vocab: Mapping from token IDs to vocabulary tokens.
        model_instance: The language model instance for generating logits.
        encoded: List of encoded token IDs for the model input.
        functions: List of available function definitions.
    """
    pass


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
    encoded_func_names: list[int] = model_instance.encode(func_names)[
        0
    ].tolist()
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
    for prompt in args.prompts:
        prompt_prefix = f'\n[\n  {{\n    "prompt": "{prompt},\n    "name": "'
        print(prompt_prefix, end="", flush=True)
        encoded: list[int] = (
            encoded_func_names
            + model_instance.encode(prompt_prefix)[0].tolist()
        )
        encoded, func_name = get_valid_function_name(
            reverse_vocab, model_instance, encoded, args.functions
        )
        encoded = encoded[len(encoded_func_names):]
        # TODO rajouter les args de la fonction en context, peut etre quy a pas besoin
        # print in red, all the context
        print("\033[91m" + model_instance.decode(encoded) + "\033[0m")
        get_function_parameters(
            reverse_vocab, model_instance, encoded, args.functions, func_name
        )


def main(model: str = "Qwen/Qwen3-0.6B") -> None:
    """Main entry point for the function calling generation task.

    Parses command-line arguments and initiates the generation process.

    Args:
        model: The language model to use (default: Qwen/Qwen3-0.6B).
    """
    start_generation(parse(), model)


if __name__ == "__main__":
    main()

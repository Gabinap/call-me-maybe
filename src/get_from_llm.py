from typing import Any

from llm_sdk import Small_LLM_Model

from .parsing import FunctionDef
from .process import (
    process_anything,
    process_integer,
    process_number,
    process_string,
)


def get_valid_function_name(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    functions: list[FunctionDef],
    prompt: str,
    encoded_func_names: list[int],
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
    prompt_prefix = f'\n[\n  {{\n    "prompt": "{prompt},\n    "name": "'
    print(prompt_prefix, end="", flush=True)
    encoded: list[int] = (
        encoded_func_names + model_instance.encode(prompt_prefix)[0].tolist()
    )
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
            remaining: str = matching_functions[0][len(clean_name) :]
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
    print('",', end="", flush=True)
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
    parameters_prefix: str = '\n    "parameters": {'
    print(parameters_prefix, end="", flush=True)
    encoded.extend(model_instance.encode(parameters_prefix)[0].tolist())
    params: dict[str, Any] | None = next(
        (f.parameters for f in functions if f.name == func_name), {}
    )
    encoded = model_instance.encode(str(params))[0].tolist() + encoded
    if params is None:
        print("\n" + " " * 19 + "}\n    }\n]")
        return
    for param_name, param_dict in params.items():
        param_value = param_dict.get("type")
        next_param_prefix: str = (
            "\n" + " " * 19 + f'"{param_name}": '
        )
        print(next_param_prefix, end="", flush=True)
        encoded.extend(model_instance.encode(next_param_prefix)[0].tolist())
        if param_value == "number":
            encoded = process_number(model_instance, encoded, reverse_vocab)
        elif param_value == "integer":
            encoded = process_integer(model_instance, encoded, reverse_vocab)
        elif param_value == "string":
            encoded = process_string(model_instance, encoded, reverse_vocab)
        else:
            encoded = process_anything(model_instance, encoded, reverse_vocab)
        # verifier que les elements actuels ne soient pas les derniers, sinon ecrire "'"
        if param_name != list(params.keys())[-1]:
            print(",", end="", flush=True)
            encoded.extend(model_instance.encode(",")[0].tolist())

    print("\n" + " " * 19 + "}\n    }\n]")
    print("\033[91m" + model_instance.decode(encoded) + "\033[0m")

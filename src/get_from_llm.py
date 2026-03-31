from typing import Any

from llm_sdk import Small_LLM_Model

from .parsing import FunctionDef
from .process import (
    process_anything,
    process_integer,
    process_number,
    process_string,
)

# Mapping from JSON schema type strings to Python callables used for casting.
_TYPE_CASTERS: dict[str, Any] = {
    "number": float,
    "integer": int,
    "boolean": lambda v: v.lower() == "true",
}


def _cast(value: str, param_type: str | None) -> Any:
    """Cast a raw string value to the appropriate Python type.

    Args:
        value: The raw string produced by a process_* function.
               Strings include surrounding quotes (e.g. ``'"hello"'``);
               all other types are bare values (e.g. ``'3.14'``).
        param_type: The JSON schema type string from the function definition.

    Returns:
        The value cast to the correct Python type, or the raw string on error.
    """
    if param_type == "string":
        # Strip the surrounding quotes that process_string adds
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]
        return value

    caster = _TYPE_CASTERS.get(param_type or "")
    if caster is None:
        return value

    try:
        return caster(value)
    except (ValueError, TypeError):
        return value


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
    matches, its remaining characters are appended directly.

    Args:
        reverse_vocab: Mapping from token IDs to raw BPE strings.
        model_instance: The language model instance for generating logits.
        functions: List of available function definitions to validate against.
        prompt: The original natural language prompt.
        encoded_func_names: Pre-encoded token IDs for function name context.

    Returns:
        Tuple of (updated token sequence, selected function name).
    """
    prompt_prefix = f'\n  {{\n    "prompt": "{prompt}",\n    "name": "'
    print(prompt_prefix, end="", flush=True)
    encoded: list[int] = (
        encoded_func_names + model_instance.encode(prompt_prefix)[0].tolist()
    )
    valid_function_names: list[str] = [f.name for f in functions]
    decoded_tokens: list[str] = [""]
    clean_name = ""

    while '"' not in decoded_tokens[-1]:
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        clean_name = "".join(decoded_tokens).split('"')[0]

        matching: list[str] = [
            n for n in valid_function_names if n.startswith(clean_name)
        ]
        if len(matching) == 1:
            remaining: str = matching[0][len(clean_name):]
            encoded.extend(model_instance.encode(remaining)[0].tolist())
            print(remaining, end="", flush=True)
            clean_name = matching[0]
            break

        valid_token_ids: list[int] = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if any(n.startswith(clean_name + token_str) for n in valid_function_names)
        ]

        if not valid_token_ids:
            break

        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded_tokens.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        print(decoded_tokens[-1], end="", flush=True)

    encoded.extend(model_instance.encode('",')[0].tolist())
    print('",', end="", flush=True)
    return encoded, clean_name


def get_function_parameters(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    encoded: list[int],
    function_def: FunctionDef,
    prompt: str,
    decoded_vocab: dict[int, str],
    fast: bool = True,
) -> dict[str, Any]:
    """Generate function parameters using the language model.

    Each parameter value is cast to the correct Python type as defined in the
    function definition (number → float, integer → int, string → str, etc.)
    so the returned dict can be serialised directly to JSON.

    Args:
        reverse_vocab: Mapping from token IDs to raw BPE strings.
        model_instance: The language model instance for generating logits.
        encoded: Current token ID sequence.
        function_def: The function definition containing parameter specs.
        prompt: The original natural language prompt.
        decoded_vocab: Precomputed decoded vocabulary (token ID -> clean string).
        fast: If True, stream each token immediately (1 beam). If False, use
              multi-beam strategies for numbers/integers and strings.

    Returns:
        Dictionary mapping parameter names to their typed Python values.
    """
    parameters_prefix = '\n    "parameters": {'
    print(parameters_prefix, end="", flush=True)
    encoded.extend(model_instance.encode(parameters_prefix)[0].tolist())

    params: dict[str, Any] | None = (
        function_def.parameters if function_def and function_def.parameters else None
    )
    if params is None:
        print('\n    }', end="", flush=True)
        return {}

    context: str = (
        f'User prompt: "{prompt}"\n'
        f"Function description: {function_def.description}\n"
        f"Function name: {function_def.name}\n"
        f"Parameters: {', '.join(params.keys())}\n"
    )

    param_names: list[str] = list(params.keys())
    result: dict[str, Any] = {}

    for param_name, param_dict in params.items():
        context += f"{param_name}: "
        param_type: str | None = param_dict.get("type")
        param_prefix: str = f'\n      "{param_name}": '
        print(param_prefix, end="", flush=True)
        encoded.extend(model_instance.encode(param_prefix)[0].tolist())

        if param_type == "number":
            encoded, printable = process_number(
                model_instance, encoded, decoded_vocab, fast=fast
            )
        elif param_type == "integer":
            encoded, printable = process_integer(
                model_instance, encoded, decoded_vocab, fast=fast
            )
        elif param_type == "string":
            encoded, printable = process_string(
                model_instance, context, decoded_vocab, fast=fast
            )
        else:
            encoded, printable = process_anything(
                model_instance, encoded, decoded_vocab, fast=fast
            )

        result[param_name] = _cast(printable, param_type)
        context += f"{printable}\n"

        if param_name != param_names[-1]:
            print(",", end="", flush=True)
            # number/integer: terminal token (`,`) already in encoded via result_tokens
            # string/anything: terminal is `"` or `}`, so `,` separator must be added
            if param_type not in ("number", "integer"):
                encoded.extend(model_instance.encode(",")[0].tolist())

    print('\n    }', end="", flush=True)
    return result
import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .parsing import FunctionDef
from .process import (
    PrecomputedVocab,
    process_anything,
    process_integer,
    process_number,
    process_string,
)

# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------


def _cast(value: str, param_type: str | None) -> Any:
    """Cast a raw string value to the appropriate Python type.

    Args:
        value: Raw string produced by a process_* function.
               Strings include surrounding quotes (e.g. ``'"hello"'``);
               all other types are bare values (e.g. ``'3.14'``).
        param_type: JSON schema type string from the function definition.

    Returns:
        Value cast to the correct Python type, or the raw string on error.
    """
    try:
        if param_type == "number":
            return float(value)
        if param_type == "integer":
            return int(value)
        if param_type == "boolean":
            return value.lower() == "true"
        if param_type == "string":
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                return value[1:-1]
    except (ValueError, TypeError):
        pass
    return value


# ---------------------------------------------------------------------------
# Recursive object generation
# ---------------------------------------------------------------------------


def _generate_object(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    pv: PrecomputedVocab,
    fast: bool,
    context_parts: list[str],
    properties: dict[str, Any],
    indent: str = "      ",
) -> tuple[list[int], dict[str, Any]]:
    """Recursively generate a JSON object by iterating over its properties.

    For each property, the appropriate ``process_*`` function is called
    based on the property's type. If the type is ``"object"``, this
    function calls itself recursively with the nested ``properties``.

    The opening ``{`` is printed before the first property; the closing ``}``
    is printed after the last one. Each property is printed on its own line
    indented by ``indent``.

    Args:
        model_instance: The language model instance.
        encoded: Current token ID sequence.
        pv: Precomputed vocabulary data.
        fast: Streaming mode flag.
        context_parts: Accumulated context parts (joined when passed to
                       ``process_string``).
        properties: Dict mapping property names to their schema dicts
                    (e.g. ``{"name": {"type": "string"}, ...}``).
        indent: Indentation string for pretty-printing nested levels.

    Returns:
        Tuple of (updated token sequence, dict of generated property values).
    """
    print("{", end="", flush=True)
    encoded.extend(model_instance.encode("{")[0].tolist())

    result: dict[str, Any] = {}
    prop_names = list(properties.keys())

    for prop_name, prop_dict in properties.items():
        prop_type: str | None = prop_dict.get("type")
        prop_prefix: str = f'\n{indent}"{prop_name}": '
        print(prop_prefix, end="", flush=True)
        encoded.extend(model_instance.encode(prop_prefix)[0].tolist())
        context_parts.append(f"{prop_name}: ")

        if prop_type == "object":
            sub_properties: dict[str, Any] = prop_dict.get("properties", {})
            encoded, nested_value = _generate_object(
                model_instance,
                encoded,
                pv,
                fast,
                context_parts,
                sub_properties,
                indent=indent + "  ",
            )
            result[prop_name] = nested_value
            printable = json.dumps(nested_value)

        elif prop_type == "number":
            encoded, printable = process_number(model_instance, encoded, pv,
                                                fast=fast)
            result[prop_name] = _cast(printable, "number")

        elif prop_type == "integer":
            encoded, printable = process_integer(model_instance, encoded,
                                                 pv, fast=fast)
            result[prop_name] = _cast(printable, "integer")

        elif prop_type == "string":
            encoded, printable = process_string(
                model_instance, "".join(context_parts), pv, fast=fast
            )
            result[prop_name] = _cast(printable, "string")

        else:
            encoded, printable = process_anything(
                model_instance, encoded, pv, fast=fast
            )
            result[prop_name] = _cast(printable, prop_type)

        context_parts.append(f"{printable}\n")

        if prop_name != prop_names[-1]:
            print(",", end="", flush=True)
            if prop_type not in ("number", "integer"):
                encoded.extend(model_instance.encode(",")[0].tolist())

    closing = f"\n{indent[:-2]}}}"
    print(closing, end="", flush=True)
    encoded.extend(model_instance.encode(closing)[0].tolist())
    return encoded, result


# ---------------------------------------------------------------------------
# Function name generation
# ---------------------------------------------------------------------------


def get_valid_function_name(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    functions: list[FunctionDef],
    prompt: str,
    encoded_func_names: list[int],
    pv: PrecomputedVocab,
) -> tuple[list[int], str]:
    """Generate and validate a function name using the language model.

    Optimisations applied:
    - ``valid_next`` set: token membership check is O(1) per token instead
      of O(functions) with repeated ``startswith`` calls.
    - ``pv.decoded`` lookup replaces ``model_instance.decode([id])``.

    Args:
        reverse_vocab: Mapping from token IDs to raw BPE strings.
        model_instance: The language model instance for generating logits.
        functions: List of available function definitions to validate against.
        prompt: The original natural language prompt.
        encoded_func_names: Pre-encoded token IDs for function name context.
        pv: Precomputed vocabulary data.

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

        # Build set of valid next-token decoded strings — O(1) membership check
        valid_next: set[str] = {
            pv.decoded[t]
            for t in reverse_vocab
            if any(
                n.startswith(
                    clean_name + pv.decoded[t]
                    ) for n in valid_function_names
            )
        }
        valid_token_ids: list[int] = [
            t for t in reverse_vocab if pv.decoded.get(t, "") in valid_next
        ]

        if not valid_token_ids:
            break

        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        next_token_str: str = pv.decoded[best_token_id]
        decoded_tokens.append(next_token_str)
        encoded.append(best_token_id)
        print(next_token_str, end="", flush=True)

    encoded.extend(model_instance.encode('",')[0].tolist())
    print('",', end="", flush=True)
    return encoded, clean_name


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------


def get_function_parameters(
    reverse_vocab: dict[int, str],
    model_instance: Small_LLM_Model,
    encoded: list[int],
    function_def: FunctionDef,
    prompt: str,
    pv: PrecomputedVocab,
    fast: bool = True,
) -> dict[str, Any]:
    """Generate function parameters using the language model.

    Handles flat types (number, integer, string, boolean) and nested
    objects recursively via ``_generate_object``.

    Args:
        reverse_vocab: Mapping from token IDs to raw BPE strings.
        model_instance: The language model instance for generating logits.
        encoded: Current token ID sequence.
        function_def: The function definition containing parameter specs.
        prompt: The original natural language prompt.
        pv: Precomputed vocabulary data passed to process_* functions.
        fast: If True, stream each token immediately (1 beam). If False, use
              multi-beam strategies for numbers/integers and strings.

    Returns:
        Dictionary mapping parameter names to their typed Python values.
    """
    parameters_prefix = '\n    "parameters": {'
    print(parameters_prefix, end="", flush=True)
    encoded.extend(model_instance.encode(parameters_prefix)[0].tolist())

    params: dict[str, Any] | None = (
        function_def.parameters if function_def
        and function_def.parameters else None
    )
    if params is None:
        print("\n    }", end="", flush=True)
        return {}

    context_parts: list[str] = [
        f'User prompt: "{prompt}"\n',
        f"Function description: {function_def.description}\n",
        f"Function name: {function_def.name}\n",
        f"Parameters: {', '.join(params.keys())}\n",
    ]

    param_names: list[str] = list(params.keys())
    result: dict[str, Any] = {}

    for param_name, param_dict in params.items():
        context_parts.append(f"{param_name}: ")
        param_type: str | None = param_dict.get("type")
        param_prefix: str = f'\n      "{param_name}": '
        print(param_prefix, end="", flush=True)
        encoded.extend(model_instance.encode(param_prefix)[0].tolist())

        if param_type == "object":
            # Recursively generate nested object; indent starts at 8 spaces
            sub_properties: dict[str, Any] = param_dict.get("properties", {})
            encoded, obj_value = _generate_object(
                model_instance,
                encoded,
                pv,
                fast,
                context_parts,
                sub_properties,
                indent="        ",
            )
            result[param_name] = obj_value
            printable = json.dumps(obj_value)

        elif param_type == "number":
            encoded, printable = process_number(model_instance, encoded,
                                                pv, fast=fast)
            result[param_name] = _cast(printable, "number")

        elif param_type == "integer":
            encoded, printable = process_integer(model_instance, encoded,
                                                 pv, fast=fast)
            result[param_name] = _cast(printable, "integer")

        elif param_type == "string":
            encoded, printable = process_string(
                model_instance, "".join(context_parts), pv, fast=fast
            )
            result[param_name] = _cast(printable, "string")

        else:
            encoded, printable = process_anything(
                model_instance, encoded, pv, fast=fast
            )
            result[param_name] = _cast(printable, param_type)

        context_parts.append(f"{printable}\n")

        if param_name != param_names[-1]:
            print(",", end="", flush=True)
            if param_type not in ("number", "integer", "object"):
                encoded.extend(model_instance.encode(",")[0].tolist())

    print("\n    }", end="", flush=True)
    return result

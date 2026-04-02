from __future__ import annotations

import argparse
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FunctionDef(BaseModel):
    """Schema for a single callable function definition."""

    name: str
    description: str
    parameters: dict[str, Any] | None
    returns_: str | None = Field(..., alias="returns")

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Ensure required string fields are not blank."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class Args(BaseModel):
    """Validated CLI arguments and loaded input data."""

    functions_definition: str = Field(
        default="data/input/functions_definition.json",
        description="Functions definition file.",
    )
    input: str = Field(
        default="data/input/function_calling_tests.json",
        description="Prompts file.",
    )
    output: str = Field(
        default="data/output/function_calls.json",
        description="Functions called by LLM file.",
    )
    mode: Literal["fast", "thinking"] = Field(
        default="fast",
        description=(
            "fast: single beam, tokens streamed immediately. "
            "thinking: positive+negative beams for numbers/integers, "
            "4 beams for strings."
        ),
    )
    model: str = Field(
        default="Qwen/Qwen3-0.6B",
        description="HuggingFace model name or path.",
    )
    prompts: list[str] = Field(
        default_factory=list,
        description="LLM prompts.",
    )
    functions: list[FunctionDef] = Field(
        default_factory=list,
        description="Function definitions.",
    )


# ---------------------------------------------------------------------------
# Parameter schema validation helpers
# ---------------------------------------------------------------------------

_VALID_TYPES = {"string", "number", "integer", "boolean", "object", "array"}


def _validate_param_schema(schema: Any, path: str = "") -> bool:
    """Recursively validate a parameter schema dict.

    Accepted types: string, number, integer, boolean, object, array.
    For ``"object"`` types, ``"properties"`` must be a non-empty dict and
    each property is validated recursively.

    Args:
        schema: The parameter schema value to validate (should be a dict).
        path: Dotted path string for error messages.

    Returns:
        True if valid, False otherwise (with a warning printed).
    """
    if not isinstance(schema, dict):
        print(f"Warning: parameter schema at '{path}' must be an object")
        return False

    param_type = schema.get("type")
    if not isinstance(param_type, str) or param_type not in _VALID_TYPES:
        print(
            f"Warning: parameter '{path}' has invalid or missing type "
            f"(got {param_type!r}). Valid types: {sorted(_VALID_TYPES)}"
        )
        return False

    if param_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            print(
                f"Warning: parameter '{path}' is type 'object' but "
                f"'properties' is missing or empty — will be skipped"
            )
            return False
        for prop_name, prop_schema in properties.items():
            if not _validate_param_schema(
                prop_schema, path=f"{path}.{prop_name}"
            ):
                return False

    return True


def _validate_parameters(
    parameters: dict[str, Any] | None, func_name: str
) -> dict[str, Any] | None:
    """Validate all parameter schemas for a function definition.

    Invalid parameters are removed with a warning rather than rejecting
    the whole function definition.

    Args:
        parameters: Raw parameters dict from the JSON input.
        func_name: Function name used in warning messages.

    Returns:
        Filtered parameters dict with only valid entries, or None if
        empty.
    """
    if parameters is None:
        return None
    if not isinstance(parameters, dict):
        print(
            f"Warning: 'parameters' for '{func_name}' "
            "must be an object — ignored"
        )
        return None

    valid: dict[str, Any] = {}
    for param_name, param_schema in parameters.items():
        if _validate_param_schema(
            param_schema, path=f"{func_name}.{param_name}"
        ):
            valid[param_name] = param_schema
        else:
            print(
                f"Warning: dropping invalid parameter "
                f"'{param_name}' from '{func_name}'"
            )

    return valid if valid else None


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def functions_def_parsing(f_d_file: str) -> list[FunctionDef]:
    """Parse and validate the functions definition file.

    Supports flat parameter types (string, number, integer, boolean) and
    nested object types (``"type": "object"`` with ``"properties"``).

    Args:
        f_d_file: Path to the JSON file containing function definitions.

    Returns:
        List of validated FunctionDef instances. Empty on error.
    """
    try:
        with open(f_d_file, encoding="utf-8") as f:
            json_f_d = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {f_d_file} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: {f_d_file} is not valid JSON")
        return []

    if not isinstance(json_f_d, list):
        print("Error: JSON content must be a list of objects")
        return []

    valid_functions: list[FunctionDef] = []
    for item in json_f_d:
        if not isinstance(item, dict):
            print("Skipping invalid function: item must be an object")
            continue
        try:
            returns_field = item["returns"]
            if not isinstance(returns_field, dict):
                print(
                    "Skipping invalid function: "
                    "'returns' must be an object"
                )
                continue
            returns_type = returns_field.get("type")
            if (
                not isinstance(returns_type, str)
                or not returns_type.strip()
            ):
                print(
                    "Skipping invalid function: 'returns.type' must be"
                    " a non-empty string"
                )
                continue

            validated_params = _validate_parameters(
                item.get("parameters"), item.get("name", "<unknown>")
            )

            func = FunctionDef.model_validate(
                {
                    **item,
                    "returns": returns_type,
                    "parameters": validated_params,
                }
            )
            valid_functions.append(func)
        except ValidationError as e:
            print(f"Skipping invalid function: {e}")
        except KeyError as e:
            print(f"Skipping function with missing key: {e}")

    return valid_functions


def prompt_parsing(input_file: str) -> list[str]:
    """Parse and validate the prompts file.

    Args:
        input_file: Path to the JSON file containing prompt objects.

    Returns:
        List of prompt strings. Empty on error.
    """
    try:
        with open(input_file, encoding="utf-8") as f:
            json_prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not valid JSON")
        return []

    if not isinstance(json_prompts, list):
        print("Error: JSON content must be a list of objects")
        return []

    prompts: list[str] = []
    for item in json_prompts:
        if not isinstance(item, dict):
            print("Warning: Skipping non-dict item")
            continue
        prompt = item.get("prompt", "")
        if isinstance(prompt, str) and len(prompt) < 300:
            prompts.append(prompt)
        else:
            print("Warning: Skipping invalid prompt")

    return prompts


def command_parsing() -> dict[str, Any]:
    """Parse arguments from the CLI.

    Returns:
        Dictionary of provided argument names to their values.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into structured "
        "function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "thinking"],
        default="fast",
        help=(
            "fast: single beam, tokens streamed immediately (default). "
            "thinking: positive+negative beams for numbers/integers, "
            "4 beams for strings."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or path (default: Qwen/Qwen3-0.6B).",
    )

    return {
        k: v for k, v in vars(parser.parse_args()).items()
        if v is not None
    }


def parse() -> Args:
    """Parse all CLI arguments and input files.

    Returns:
        Fully populated Args instance.
    """
    cli: dict[str, Any] = command_parsing()
    cli["functions"] = functions_def_parsing(cli["functions_definition"])
    cli["prompts"] = prompt_parsing(cli["input"])
    return Args(**cli)

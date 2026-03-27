from __future__ import annotations

import argparse
import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class Args(BaseModel):
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
    prompts: list[str] = Field(
        default_factory=list,
        description="LLM's prompts",
    )
    functions: list[FunctionDef] = Field(
        default_factory=list,
        description="Functions definitions",
    )


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] | None
    returns_: str | None = Field(..., alias="returns")

    @field_validator("name", "description", "returns_")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


def functions_def_parsing(f_d_file: str) -> list[FunctionDef]:
    """Parse functions definition with validation."""
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
                print("Skipping invalid function: 'returns' must be an object")
                continue

            returns_type = returns_field["type"]
            if not isinstance(returns_type, str) or not returns_type.strip():
                print(
                    "Skipping invalid function: 'returns.type' must be a"
                    " non-empty string"
                )
                continue

            normalized_item = {**item, "returns": returns_type}
            func = FunctionDef.model_validate(normalized_item)
            valid_functions.append(func)
        except ValidationError as e:
            print(f"Skipping invalid function: {e}")
            continue
        except KeyError as e:
            print(f"Skipping function with missing key: {e}")
            continue
    return valid_functions


def command_parsing() -> dict[str, str]:
    """Parse arguments from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Functions definition file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Prompts file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Functions called by LLM file.",
    )

    args = parser.parse_args()

    config_dict = {k: v for k, v in vars(args).items() if v is not None}

    return config_dict


def prompt_parsing(input_file: str) -> list[str]:
    """Parse prompt file with validation."""
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

    prompts = []
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


def parse() -> Args:
    """Parse arguments from CLI and return an Args instance."""
    args: dict[str, Any] = command_parsing()
    args["functions"] = functions_def_parsing(args["functions_definition"])
    args["prompts"] = prompt_parsing(args["input"])
    return Args(**args)

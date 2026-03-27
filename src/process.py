import re

from llm_sdk import Small_LLM_Model


def process_number(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> list[int]:
    number_prefix: re.Pattern = re.compile(
        r"^-?(\d+(\.\d*)?([eE][+-]?\d*)?)?$"
    )
    current_number: str = ""
    number_complete: re.Pattern = re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$')
    decoded: list[str] = [""]
    valid_token_ids = [
        token_id
        for token_id, token_str in reverse_vocab.items()
        if number_prefix.match(current_number + token_str)
    ]
    # pour linstant il ecrit que le premier token et sort
    # mon regex qui sera a changer car invalide en json
    while True:
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        current_number += decoded[-1]
        print(decoded[-1], end="", flush=True)
        if number_complete.match(current_number):
            break
    return encoded


def process_integer(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> list[int]:
    integer_prefix: re.Pattern = re.compile(r"^-?\d+$")
    current_integer: str = ""

    decoded: list[str] = [""]

    while '"' not in decoded[-1]:
        valid_token_ids = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if integer_prefix.match(current_integer + token_str)
        ]
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        print(decoded[-1], end="", flush=True)
        current_integer += decoded[-1]
    return encoded


def process_string(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> list[int]:
    string_prefix: re.Pattern = re.compile(r'^".*?"$')
    current_string: str = ""

    decoded: list[str] = [""]

    while '"' not in decoded[-1]:
        valid_token_ids = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if string_prefix.match(current_string + token_str)
        ]
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        print(decoded[-1], end="", flush=True)
        current_string += decoded[-1]
    return encoded


def process_anything(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> list[int]:
    # Placeholder for handling any other types of parameters
    return encoded
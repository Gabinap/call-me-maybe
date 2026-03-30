import re

from llm_sdk import Small_LLM_Model


def process_number(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[list[int], str]:
    number_prefix: re.Pattern = re.compile(r"^-?(\d+(\.\d*)?([eE][+-]?\d*)?)?$")
    current_number: str = ""
    number_complete: re.Pattern = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[,}]$")
    decoded: list[str] = [""]
    printable: str = ""
    while not number_complete.match(current_number):
        valid_token_ids = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if number_prefix.match(current_number + token_str)
            or number_complete.match(current_number + token_str)
        ]
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        current_number += decoded[-1]
        printable = "".join(c for c in decoded[-1] if c not in ",}")
        print(printable, end="", flush=True)
    return encoded, printable


def process_integer(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[list[int], str]:
    integer_prefix: re.Pattern = re.compile(r"^-?\d+?$")
    current_integer: str = ""
    integer_complete: re.Pattern = re.compile(r"^-?\d+?[,}]$")
    decoded: list[str] = [""]
    printable: str = ""

    while not integer_complete.match(current_integer):
        valid_token_ids = [
            token_id
            for token_id, token_str in reverse_vocab.items()
            if integer_prefix.match(current_integer + token_str)
            or integer_complete.match(current_integer + token_str)
        ]
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        current_integer += decoded[-1]
        printable = "".join(c for c in decoded[-1] if c not in ",}")
        print(printable, end="", flush=True)
    return encoded, printable


def extract_string_candidates(prompt: str) -> list[str]:
    """Extrait les valeurs candidates depuis le prompt (quoted + mots)."""
    candidates = []
    # Strings entre guillemets doubles
    candidates += re.findall(r'"([^"]*)"', prompt)
    # Strings entre guillemets simples
    candidates += re.findall(r"'([^']*)'", prompt)
    # Mots individuels
    candidates += re.findall(r"\b\w+\b", prompt)
    return candidates


# calculate with the 4 best tokens
# finir leurs 4 generations et garder celui qui a la meilleure moyenne
def score_candidates(
    model_instance: Small_LLM_Model,
    encoded_context: list[int],
    reverse_vocab: dict[int, str],
    max_tokens: int = 50,
) -> str:
    """Génère 4 beams depuis les 4 meilleurs premiers tokens, retourne le meilleur."""
    import heapq

    string_prefix: re.Pattern = re.compile(r'^([^"\\]|\\.)*$')
    string_complete: re.Pattern = re.compile(r'^([^"\\]|\\.)*"$')

    logits: list[float] = model_instance.get_logits_from_input_ids(encoded_context)

    # On filtre avec decode() pour éviter les Ġ dans la comparaison regex
    valid_start_ids = [
        t for t in reverse_vocab
        if string_prefix.match(model_instance.decode([t]))
    ]
    best_start_ids: list[int] = heapq.nlargest(
        4, valid_start_ids, key=lambda t: logits[t]
    )

    beams: list[tuple[list[int], str, list[float]]] = []
    for token_id in best_start_ids:
        beams.append((
            encoded_context + [token_id],
            model_instance.decode([token_id]),  # decode() au lieu de reverse_vocab
            [logits[token_id]],
        ))

    completed: list[tuple[str, float]] = []
    for beam_tokens, beam_str, beam_scores in beams:
        current_tokens = beam_tokens[:]
        current_str = beam_str
        scores = beam_scores[:]

        while not string_complete.match(current_str):
            if len(scores) >= max_tokens:
                current_str += '"'
                break

            valid_token_ids = [
                t for t in reverse_vocab
                if (
                    string_prefix.match(current_str + model_instance.decode([t]))
                    or string_complete.match(current_str + model_instance.decode([t]))
                )
            ]
            if not valid_token_ids:
                current_str += '"'
                break

            step_logits = model_instance.get_logits_from_input_ids(current_tokens)
            best = max(valid_token_ids, key=lambda t: step_logits[t])
            scores.append(step_logits[best])
            current_tokens.append(best)
            current_str += model_instance.decode([best])  # decode() ici aussi

        avg_score = sum(scores) / len(scores) if scores else float("-inf")
        completed.append((current_str.rstrip('"'), avg_score))

    best_str, _ = max(completed, key=lambda x: x[1])
    return best_str


def process_string(
    model_instance: Small_LLM_Model,
    reverse_vocab: dict[int, str],
    prompt: str,
    context: str,
) -> tuple[list[int], str]:
    # candidates = extract_string_candidates(prompt)
    encoded_context = model_instance.encode(context + '"')[0].tolist()
    print('"', end="")
    best = score_candidates(model_instance, encoded_context, reverse_vocab)
    print(best, end="")
    return encoded_context, f'"{best}"'


def process_anything(
    model_instance: Small_LLM_Model,
    encoded: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[list[int], str]:
    # Placeholder for handling any other types of parameters
    decoded: list[str] = [""]
    valid_token_ids = list(reverse_vocab.keys())
    printable: str = ""
    while not any(c in [",", "}"] for c in decoded[-1]):
        logits: list[float] = model_instance.get_logits_from_input_ids(encoded)
        best_token_id: int = max(valid_token_ids, key=lambda t: logits[t])
        decoded.append(model_instance.decode([best_token_id]))
        encoded.append(best_token_id)
        printable = "".join(c for c in decoded[-1] if c not in ",}")
        print(printable, end="", flush=True)
    return encoded, printable

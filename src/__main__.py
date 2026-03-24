from llm_sdk import Small_LLM_Model

from .parsing import parse, Args


def get_vocabulary(model_instance: Small_LLM_Model) -> dict[str, int]:
    vocab_file_path = model_instance.get_path_to_vocab_file()
    with open(vocab_file_path) as f:
        vocab = {line.strip(): i for i, line in enumerate(f)}
    return vocab


def gingembre(args: Args, model: str) -> None:
    model_instance = Small_LLM_Model(model_name=model)
    vocab: dict[str, int] = get_vocabulary(model_instance)
    reverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
    decoded: str = ""
    for prompt in args.prompts[:1]:
        encoded = model_instance.encode(prompt)
        while '\'' or '\"' not in decoded:
            logits = model_instance.get_logits_from_input_ids(encoded[0].tolist())
            best_token = logits.index(max(logits))
            token_id: int = best_token if best_token in reverse_vocab else 0
            decoded = model_instance.decode([token_id])
            print(decoded)
            encoded = encoded + [token_id]


def main(model: str = "Qwen/Qwen3-0.6B") -> None:
    gingembre(parse(), model)


if __name__ == "__main__":
    main()

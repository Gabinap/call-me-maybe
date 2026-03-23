from .parsing import parse, Args
from llm_sdk import Small_LLM_Model


def gingembre(args: Args, model: str) -> None:
    model_instance = Small_LLM_Model(model_name=model)
    for prompt in args.prompts:
        encoded = model_instance.encode(prompt)
        logits = model_instance.get_logits_from_input_ids(encoded[0].tolist())
        answer = max(logits)
        decoded = model_instance.decode([int(answer)])
        print(decoded)


def main(model: str = "Qwen/Qwen3-0.6B") -> None:
    gingembre(parse(), model)


if __name__ == "__main__":
    main()

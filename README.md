*This project has been created as part of the 42 curriculum by gagulhon.*

---

# call me maybe — Introduction to Function Calling in LLMs

## Description

**call me maybe** is a function calling system that translates natural language prompts into structured, schema-validated JSON function calls using a small language model (Qwen3-0.6B by default).

Given a prompt like `"What is the sum of 40 and 2?"`, the system does not answer the question directly. Instead it produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

The core challenge is reliability: small language models produce valid JSON only ~30% of the time when prompted naïvely. This project achieves near-perfect reliability through **constrained decoding** — a technique that restricts which tokens the model may produce at each generation step, guaranteeing syntactically and semantically valid output without relying on the model's instruction-following abilities.

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

Clone the repository, then install dependencies:

```bash
uv sync
```

The `llm_sdk/` directory must be present at the project root alongside `src/`. It is included in the repository.

### Running the program

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json \
  --mode fast
```

All arguments are optional and fall back to the paths shown above.

| Argument | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | JSON file containing function schemas |
| `--input` | `data/input/function_calling_tests.json` | JSON file containing natural language prompts |
| `--output` | `data/output/function_calls.json` | Destination for the generated JSON output |
| `--mode` | `fast` | `fast` or `thinking` (see below) |

### Makefile targets

```bash
make install     # install dependencies
make run         # run with default arguments
make debug       # run under pdb debugger
make clean       # remove __pycache__, .mypy_cache
make lint        # flake8 + mypy
make lint-strict # flake8 + mypy --strict
```

### Running the tests

```bash
uv run python -m pytest tests/ -v
```

---

## Algorithm Explanation

### Constrained Decoding

Language models generate text one token at a time. At each step the model produces a probability distribution (logits) over every token in its vocabulary (~150 000 tokens for Qwen3). Normally you would pick the highest-probability token — but nothing stops the model from generating `"hello"` when you expected a number.

Constrained decoding intercepts this process:

1. The model produces logits for all tokens.
2. A **prefix pattern** (regex) filters out every token that would make the partial value structurally invalid.
3. A **complete pattern** (regex) identifies which tokens would finish the value correctly.
4. Only tokens satisfying one of these two conditions are eligible; the rest are ignored.
5. The highest-logit eligible token is selected.
6. The chosen token is appended to the context and the loop repeats.

This guarantees that every generated value is valid by construction — no post-processing, no retries.

### Per-type constraints

| Type | Prefix pattern | Complete pattern | Example |
|---|---|---|---|
| `number` | `^-?(\d+(\.\d*)?([eE][+-]?\d*)?)?$` | `^-?\d+(\.\d+)?([eE][+-]?\d+)?[,}]$` | `3.14`, `-2.5e+34` |
| `integer` | `^-?\d+?$` | `^-?\d+?[,}]$` | `42`, `-7` |
| `string` | `^([^"\\]|\\.)*$` | `^([^"\\]|\\.)*"$` | `"hello"` |
| `anything` | `^[^,}]*$` | `^[^,}]*[,}]$` | `true`, `null` |

### Function name selection

The function name is generated token-by-token. At each step only tokens that are a valid continuation of at least one available function name are allowed. As soon as only one function name matches the partial prefix, the remaining characters are appended directly without querying the model.

### Beam search for strings (thinking mode)

In `thinking` mode, string parameters are generated with 4 parallel beams. All beams run silently and the one with the highest average log-score wins. This improves extraction accuracy at the cost of 4× more LLM calls per string.

### Negative beam for numbers (thinking mode)

Numbers are generated twice: once freely, once with a forced `-` prefix injected into the context. Both runs are silent and scored; the higher-scoring result is kept. The `-` token is injected into the **context**, not counted as a scored beam token, so both runs are compared fairly on digit-logit quality alone.

### Nested object support

Parameters of type `"object"` are handled recursively. `_generate_object` descends into the `"properties"` field and calls the appropriate `process_*` function for each leaf, building a nested Python dict that is later serialised by `json.dump`.

---

## Design Decisions

### `PrecomputedVocab` dataclass

The vocabulary (~150 000 tokens) is scanned once at startup to extract per-pattern token ID lists (`integer_starts`, `string_fallback`, `minus_ids`, etc.). These lists are stored in a `PrecomputedVocab` dataclass and passed through the call chain. This avoids rescanning the vocabulary on every generation step, which was the dominant non-LLM bottleneck.

### Numpy for token selection

`max(valid_token_ids, key=lambda t: logits[t])` is O(n) in pure Python. Replacing it with numpy fancy indexing (`logits_np[valid_arr].argmax()`) moves the inner loop to C and gives a measurable speedup when `valid_token_ids` is large (e.g. 80 000+ tokens for string generation).

### Step cache

Inside the generation loop, `valid_token_ids` for a given `current_str` prefix is identical across beams and across repeated visits to the same prefix. A per-call `dict[str, list[int]]` cache avoids redundant regex scans.

### `context_parts` list + join

The context string passed to `process_string` is built by appending to a list and calling `"".join(context_parts)` only when needed. This avoids the O(n²) cost of repeated string concatenation in a loop.

### `fast` vs `thinking` mode

| | `fast` | `thinking` |
|---|---|---|
| Numbers / integers | 1 beam, streamed live | 2 silent beams (positive + negative), printed once |
| Strings | 1 beam, streamed live | 4 silent beams, printed once |
| `process_anything` | 1 beam, streamed | 1 beam, streamed |

`fast` is the default. It gives immediate visual feedback (tokens appear as they are generated) and is significantly faster. `thinking` trades speed for higher accuracy on ambiguous or negative numeric values and on string extraction tasks.

### Output format

The output is a valid JSON array written to the path specified by `--output`. Types are cast to their schema-declared Python equivalents before serialisation (`float` for `number`, `int` for `integer`, stripped string for `string`) so `json.dump` produces the correct JSON types without extra encoding logic. Numbers of type `number` always include a decimal point (e.g. `42.0`) so JSON parsers treat them as floats.

---

## Performance Analysis

### Accuracy

On the provided test set (simple arithmetic, greetings, string reversal, regex substitution):

- **Function name selection**: 100% — constrained to valid names only.
- **Number / integer extraction**: ~95% in `fast` mode, ~98% in `thinking` mode (negative-beam improvement).
- **String extraction**: ~80–90% in `fast` mode, ~85–95% in `thinking` mode (beam search improvement). Accuracy degrades on prompts where the target string is not directly present in the prompt text (e.g. computed regex patterns).

### Speed

All timings on CPU (no GPU):

| Prompt type | `fast` | `thinking` |
|---|---|---|
| Number parameter | ~1–2 s | ~3–4 s |
| String parameter | ~3–5 s | ~10–20 s |
| Full test suite (11 prompts) | ~2–3 min | ~8–12 min |

The LLM forward pass dominates. The non-LLM optimisations (`PrecomputedVocab`, numpy, step cache) reduce the Python overhead between passes to negligible levels.

### Reliability

100% valid JSON output is guaranteed by construction: constrained decoding never produces a token that would make the value structurally invalid, and the output is assembled from typed Python values before being serialised by `json.dump`.

---

## Challenges Faced

### BPE token artefacts (Ġ characters)

Hugging Face tokenisers encode a leading space as a special character (`Ġ`) in the raw vocabulary. Using `reverse_vocab[t]` directly for string accumulation produced outputs like `"Ġworld"` instead of `"world"`. The fix was to always use `model_instance.decode([t])` (or the precomputed `decoded_vocab`) which strips these artefacts, and to pass `decoded_vocab` — not `reverse_vocab` — to all generation functions.

### Double commas and stray newlines

Early versions appended terminal tokens (`,`, `}`) to `encoded` and then added another `,` separator in the caller, producing `42,,`. The fix was to clarify ownership: for `number` and `integer`, the terminal token is already in `encoded` so the caller must not add another one; for `string` and `anything`, the terminal is `"` or `}`, so the caller does add `,` but also extends `encoded`. Additionally, BPE tokens sometimes decode to `}\n` — the `_TERMINAL_CHARS = ",} \n\r\t"` strip in `score_candidates` handles this.

### Negative number selection in thinking mode

The first implementation of the negative beam used `forced_starts=[None, minus_id]` inside a single `score_candidates` call. This caused the minus token's low logit to be included in the beam's average score, systematically penalising the negative beam even when the number should be negative. The fix was two separate silent `score_candidates` calls — the `-` token is injected into the *context* of the second call, not counted in its score.

### String generation loops and max_tokens

Without a hard token limit, string generation could loop indefinitely when no token in the vocabulary could close the current partial string. The `max_tokens=50` parameter and the `fallback_ids` list (tokens whose appended form satisfies `complete_pattern`) together guarantee termination.

---

## Testing Strategy

Tests are located in `tests/` and use the standard `unittest` framework (compatible with `pytest`).

All tests run without loading the LLM. The model is replaced by `MagicMock` objects with deterministic `get_logits_from_input_ids.side_effect` sequences, and `encode` returns mock tensors with a `.tolist()` method. This makes the full test suite run in under 5 seconds.

### Coverage by file

| Test file | What is tested |
|---|---|
| `test_parsing.py` | `FunctionDef`, `Args`, `_validate_param_schema`, `_validate_parameters`, `prompt_parsing`, `functions_def_parsing`, `command_parsing`, `parse` |
| `test_process.py` | `PrecomputedVocab.build`, all regex patterns, `_ensure_float_dot`, `score_candidates` (streaming, caching, beams, scoring), `_get_minus_token`, `_score_with_negative`, all `process_*` functions |
| `test_get_from_llm.py` | `_cast`, `_generate_object` (flat, nested, recursion, indentation), `get_valid_function_name`, `get_function_parameters` |
| `test_main.py` | `get_vocabulary`, `write_output` (paths, types, non-ASCII, error handling) |

### Edge cases covered

- Empty vocabulary, empty parameter lists, empty prompts
- Numbers: zero, negative, scientific notation, values that require `.0` appended
- Strings: empty, containing spaces, with/without surrounding quotes
- Integers: boundary between prefix and complete patterns
- JSON error handling: missing files, malformed JSON, wrong structure
- Nested objects: valid recursion, invalid schemas dropped with warnings
- `OSError` on output write does not crash the program

---

## Example Usage

### Minimal example

```bash
uv run python -m src
```

Uses all default paths and `--mode fast`.

### Custom files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/results.json \
  --mode thinking
```

### Input: `functions_definition.json`

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

### Input: `function_calling_tests.json`

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet john"}
]
```

### Output: `function_calls.json`

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": "john"
    }
  }
]
```

---

## Resources

### Constrained decoding and structured generation

- [Outlines — Structured Text Generation](https://github.com/outlines-dev/outlines) — reference implementation of grammar-constrained decoding
- [Guidance](https://github.com/guidance-ai/guidance) — alternative approach to constrained LLM generation
- [Lark parser + LLM integration](https://lark-parser.readthedocs.io/) — context-free grammar approach
- [JSON Schema specification](https://json-schema.org/) — formal definition of the schema format used for function parameters

### Language model internals

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers) — tokenisation, model loading, logits
- [Byte-Pair Encoding (BPE) — Sennrich et al. 2016](https://arxiv.org/abs/1508.07909) — original BPE paper
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the model used by default

### Function calling in production systems

- [OpenAI Function Calling documentation](https://platform.openai.com/docs/guides/function-calling) — production reference implementation
- [Anthropic Tool Use documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude's structured output approach

### Python tooling

- [Pydantic documentation](https://docs.pydantic.dev/) — used for model and parameter validation
- [mypy documentation](https://mypy.readthedocs.io/) — static type checking
- [pytest documentation](https://docs.pytest.org/) — test framework
- [numpy documentation](https://numpy.org/doc/) — used for vectorised token selection

### How AI was used

AI assistance (Claude) was used throughout this project for the following tasks:

- **Initial architecture design**: discussing the overall pipeline structure (tokenisation → logit filtering → constrained beam search).
- **Debugging**: identifying root causes of issues such as BPE artefacts, double commas in output, and negative beam scoring bugs.
- **Code review and refactoring**: cleaning up function signatures, removing redundant variables, unifying `process_*` functions around a shared `score_candidates` primitive.
- **Optimisation analysis**: identifying bottlenecks (vocabulary scanning, Python `max` vs numpy, string concatenation) and implementing fixes.
- **Test generation**: generating unittest skeletons that were then reviewed, corrected, and extended manually.
- **Documentation**: drafting docstrings and README sections that were reviewed and edited for accuracy.
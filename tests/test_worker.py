"""Tests for src/worker.py.

Tests the worker functions used by multiprocessing-based parallel
generation.  All tests mock the LLM model and sys.stdout to avoid
loading real weights and to prevent stdout redirection from
affecting the test runner.
"""

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.worker import (  # noqa: E402
    _get_vocabulary,
    _init_model,
    _process_one_prompt,
    worker_process_batch,
)
from src.parsing import FunctionDef  # noqa: E402
from src.process import PrecomputedVocab  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(ids: list[int]) -> MagicMock:
    t = MagicMock()
    t.tolist.return_value = ids
    return t


def _func(
    name: str,
    desc: str,
    params: dict[str, Any] | None,
) -> FunctionDef:
    return FunctionDef.model_validate(
        {
            "name": name,
            "description": desc,
            "parameters": params,
            "returns": "string",
        }
    )


def _small_pv() -> PrecomputedVocab:
    decoded = {i: str(i) for i in range(10)}
    decoded[10] = '"'
    decoded[11] = ","
    return PrecomputedVocab.build(decoded)


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = [_tensor([1])]
    model.decode.return_value = "x"
    model.get_logits_from_input_ids.return_value = [0.1] * 200
    return model


# ---------------------------------------------------------------------------
# _get_vocabulary
# ---------------------------------------------------------------------------


class TestGetVocabulary(unittest.TestCase):
    def test_loads_vocab(self) -> None:
        import json
        import tempfile
        import os

        vocab = {"hello": 0, "world": 1}
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(vocab, f)
        f.close()
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = f.name
            result = _get_vocabulary(model)
            self.assertEqual(result, vocab)
        finally:
            os.unlink(f.name)

    def test_non_dict_returns_empty(self) -> None:
        import json
        import tempfile
        import os

        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump([1, 2, 3], f)
        f.close()
        try:
            model = MagicMock()
            model.get_path_to_vocab_file.return_value = f.name
            result = _get_vocabulary(model)
            self.assertEqual(result, {})
        finally:
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# _init_model
# ---------------------------------------------------------------------------


class TestInitModel(unittest.TestCase):
    def test_returns_four_element_tuple(self) -> None:
        functions = [
            _func("fn_add", "Add.", {"a": {"type": "number"}})
        ]

        with (
            patch(
                "src.worker.Small_LLM_Model",
                return_value=_mock_model(),
            ),
            patch(
                "src.worker._get_vocabulary",
                return_value={"a": 0, "b": 1, ",": 2},
            ),
        ):
            result = _init_model("fake-model", functions)

        self.assertEqual(len(result), 4)
        model, reverse_vocab, pv, enc_func_names = result
        self.assertIsInstance(reverse_vocab, dict)
        self.assertIsInstance(pv, PrecomputedVocab)
        self.assertIsInstance(enc_func_names, list)


# ---------------------------------------------------------------------------
# _process_one_prompt
# ---------------------------------------------------------------------------


class TestProcessOnePrompt(unittest.TestCase):
    def test_returns_dict_with_required_keys(self) -> None:
        pv = _small_pv()
        model = _mock_model()
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]

        with (
            patch("builtins.print"),
            patch(
                "src.worker.get_valid_function_name",
                return_value=([1, 2, 3], "fn_greet"),
            ),
            patch(
                "src.worker.get_function_parameters",
                return_value={"name": "john"},
            ),
        ):
            result = _process_one_prompt(
                "Greet john", model, {}, functions, [1], pv, True,
            )

        self.assertEqual(result["prompt"], "Greet john")
        self.assertEqual(result["name"], "fn_greet")
        self.assertEqual(result["parameters"], {"name": "john"})

    def test_selects_correct_function_def(self) -> None:
        """Must pick the FunctionDef matching the generated name."""
        pv = _small_pv()
        model = _mock_model()
        functions = [
            _func("fn_add", "Add.", {"a": {"type": "number"}}),
            _func("fn_greet", "Greet.", {"name": {"type": "string"}}),
        ]

        with (
            patch("builtins.print"),
            patch(
                "src.worker.get_valid_function_name",
                return_value=([1], "fn_add"),
            ),
            patch(
                "src.worker.get_function_parameters",
                return_value={"a": 5.0},
            ) as mock_params,
        ):
            _process_one_prompt(
                "Add 2 and 3", model, {}, functions, [1], pv, True,
            )

        # Verify get_function_parameters was called with fn_add's def
        call_args = mock_params.call_args
        fd = call_args[0][3]  # 4th positional arg = function_def
        self.assertEqual(fd.name, "fn_add")


# ---------------------------------------------------------------------------
# worker_process_batch
# ---------------------------------------------------------------------------


class TestWorkerProcessBatch(unittest.TestCase):
    """Test the entry point that runs inside each child process."""

    def test_returns_indexed_results(self) -> None:
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]
        indexed_prompts = [(0, "Greet alice"), (1, "Greet bob")]

        mock_sys = MagicMock()

        with (
            patch("builtins.print"),
            patch("src.worker.sys", mock_sys),
            patch(
                "src.worker._process_one_prompt",
                side_effect=lambda prompt, *a, **kw: {
                    "prompt": prompt,
                    "name": "fn_greet",
                    "parameters": {"name": prompt.split()[-1]},
                },
            ),
            patch(
                "src.worker._init_model",
                return_value=(MagicMock(), {}, _small_pv(), [1]),
            ),
        ):
            results = worker_process_batch(
                (indexed_prompts, "fake-model", functions, True)
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(
            results[0][1]["parameters"]["name"], "alice"
        )
        self.assertEqual(results[1][0], 1)
        self.assertEqual(
            results[1][1]["parameters"]["name"], "bob"
        )

    def test_error_produces_error_entry(self) -> None:
        """A failing prompt must not crash the batch."""
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]
        indexed_prompts = [(5, "Greet crash")]

        mock_sys = MagicMock()

        with (
            patch("builtins.print"),
            patch("src.worker.sys", mock_sys),
            patch(
                "src.worker._process_one_prompt",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "src.worker._init_model",
                return_value=(MagicMock(), {}, _small_pv(), [1]),
            ),
        ):
            results = worker_process_batch(
                (indexed_prompts, "fake-model", functions, True)
            )

        self.assertEqual(len(results), 1)
        idx, result = results[0]
        self.assertEqual(idx, 5)
        self.assertEqual(result["name"], "error")

    def test_silences_stdout(self) -> None:
        """Worker must redirect sys.stdout to suppress streaming."""
        functions = [
            _func("fn_greet", "Greet.", {"name": {"type": "string"}})
        ]
        indexed_prompts = [(0, "Greet test")]

        import src.worker as worker_mod
        original_stdout = worker_mod.sys.stdout

        with (
            patch(
                "src.worker._process_one_prompt",
                return_value={
                    "prompt": "Greet test",
                    "name": "fn_greet",
                    "parameters": {"name": "test"},
                },
            ),
            patch(
                "src.worker._init_model",
                return_value=(MagicMock(), {}, _small_pv(), [1]),
            ),
        ):
            worker_process_batch(
                (indexed_prompts, "fake-model", functions, True)
            )
            self.assertIsNot(
                worker_mod.sys.stdout, original_stdout,
            )

        # Restore for other tests
        worker_mod.sys.stdout = original_stdout

    def test_processes_multiple_prompts_in_batch(self) -> None:
        """A batch with 3 prompts must produce 3 results."""
        functions = [
            _func("fn_add", "Add.", {"a": {"type": "number"}})
        ]
        indexed_prompts = [
            (2, "Add 1 and 2"),
            (5, "Add 3 and 4"),
            (8, "Add 5 and 6"),
        ]

        call_count = [0]

        def counting_prompt(prompt: str, *a: Any, **kw: Any) -> dict:
            call_count[0] += 1
            return {
                "prompt": prompt,
                "name": "fn_add",
                "parameters": {"a": float(call_count[0])},
            }

        mock_sys = MagicMock()

        with (
            patch("src.worker.sys", mock_sys),
            patch(
                "src.worker._process_one_prompt",
                side_effect=counting_prompt,
            ),
            patch(
                "src.worker._init_model",
                return_value=(MagicMock(), {}, _small_pv(), [1]),
            ),
        ):
            results = worker_process_batch(
                (indexed_prompts, "fake-model", functions, True)
            )

        self.assertEqual(len(results), 3)
        # Indices must be preserved
        self.assertEqual([r[0] for r in results], [2, 5, 8])
        # All 3 prompts were processed
        self.assertEqual(call_count[0], 3)


if __name__ == "__main__":
    unittest.main()

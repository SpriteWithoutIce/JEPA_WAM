"""Exercise padding utilities without importing the training dependency stack."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_selector():
    source = REPO_ROOT / "prismatic/models/vlms/prismatic.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PrismaticVLM")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_select_action_memory")
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


def _load_collator():
    import runpy

    return runpy.run_path(str(REPO_ROOT / "prismatic/util/data_utils.py"))["PaddedCollatorForActionPrediction"]


def test_mixed_instruction_lengths_preserve_all_64_placeholders_and_gradients():
    # The short sample has eight fewer prompt tokens, followed by the same 64 placeholders.
    instances = [
        {
            "input_ids": torch.tensor([1] * prompt_length + [151386] * 64),
            "pixel_values": torch.zeros(3, 2, 2),
            "actions": np.zeros((20, 7), dtype=np.float32),
        }
        for prompt_length in (16, 8)
    ]
    batch = _load_collator()(model_max_length=100, pad_token_id=0)(instances)
    # Include visual tokens inserted after the first prompt token by forward().
    mask = batch["attention_mask"]
    mask = torch.cat([mask[:, :1], torch.ones(2, 4, dtype=torch.bool), mask[:, 1:]], dim=1)
    hidden = torch.arange(2 * 84 * 3, dtype=torch.float32).reshape(2, 84, 3).requires_grad_()
    selected = _load_selector()(hidden, mask, 64)
    assert selected.shape == (2, 64, 3)
    torch.testing.assert_close(selected[0], hidden[0, 20:84])
    torch.testing.assert_close(selected[1], hidden[1, 12:76])
    selected.sum().backward()
    expected_grad = torch.zeros_like(hidden)
    expected_grad[0, 20:84] = 1
    expected_grad[1, 12:76] = 1
    torch.testing.assert_close(hidden.grad, expected_grad)


@pytest.mark.parametrize("mask,count", [([1, 1, 0, 0], 3), ([1, 0, 1, 0], 1), ([1, 1, 1, 1], 0)])
def test_invalid_action_spans_are_rejected(mask, count):
    with pytest.raises(ValueError):
        _load_selector()(torch.zeros(1, 4, 2), torch.tensor([mask]), count)


def test_collator_rejects_truncation_of_action_placeholders():
    with pytest.raises(ValueError, match="truncation would remove action placeholders"):
        _load_collator()(model_max_length=70, pad_token_id=0)([
            {"input_ids": torch.ones(80, dtype=torch.long)}
        ])

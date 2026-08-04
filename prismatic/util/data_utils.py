"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


def _stack_or_concat_pixel_values(values, wrist_values=None):
    example = values[0]
    if isinstance(example, torch.Tensor):
        stacked = torch.stack(values)
        if wrist_values is not None:
            stacked_wrist = torch.stack(wrist_values)
            return torch.cat((stacked.unsqueeze(1), stacked_wrist), dim=1)
        return stacked

    if isinstance(example, dict):
        return {
            key: _stack_or_concat_pixel_values(
                [value[key] for value in values],
                None if wrist_values is None else [value[key] for value in wrist_values],
            )
            for key in example
        }

    raise ValueError(f"Unsupported `pixel_values` type = {type(example)}")


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    target_action_dim: int | None = None
    target_proprio_dim: int | None = None

    @staticmethod
    def _right_pad_last_dim(tensor: torch.Tensor, target_dim: int | None, name: str) -> tuple[torch.Tensor, int]:
        valid_dim = tensor.shape[-1]
        if target_dim is None:
            return tensor, valid_dim
        if valid_dim > target_dim:
            raise ValueError(f"Cannot pad `{name}` with dim {valid_dim} down to target dim {target_dim}.")
        if valid_dim == target_dim:
            return tensor, valid_dim
        pad_shape = (*tensor.shape[:-1], target_dim - valid_dim)
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, padding), dim=-1), valid_dim

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        future_pixel_values = [instance.get("future_pixel_values") for instance in instances]
        future_pixel_values_wrist = [instance.get("future_pixel_values_wrist") for instance in instances]
        pair_pixel_values = [instance.get("pair_pixel_values") for instance in instances]
        pair_pixel_values_wrist = [instance.get("pair_pixel_values_wrist") for instance in instances]
        action_valid_masks = [instance.get("action_valid_mask") for instance in instances]
        context_proprios = [instance.get("context_proprio") for instance in instances]
        context_actions = [instance.get("context_actions") for instance in instances]
        context_delta_ts = [instance.get("context_delta_t") for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        if self.padding_side == "left":
            def left_pad_sequence(sequences, padding_value):
                max_len = max(seq.size(0) for seq in sequences)
                padded = []
                for seq in sequences:
                    pad_len = max_len - seq.size(0)
                    pad = torch.full((pad_len,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([pad, seq], dim=0)
                    padded.append(padded_seq)
                return torch.stack(padded)

            input_ids = left_pad_sequence(input_ids, self.pad_token_id)
            labels = left_pad_sequence(labels, IGNORE_INDEX)
        else:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances] if "pixel_values_wrist" in instances[0] else None
        pixel_values = _stack_or_concat_pixel_values(pixel_values, pixel_values_wrist)

        # Stack future_pixel_values (already tensors from batch transform)
        if future_pixel_values[0] is not None:
            future_pixel_values = _stack_or_concat_pixel_values(
                future_pixel_values,
                future_pixel_values_wrist if future_pixel_values_wrist[0] is not None else None,
            )
        else:
            future_pixel_values = None

        if pair_pixel_values[0] is not None:
            pair_pixel_values = _stack_or_concat_pixel_values(
                pair_pixel_values,
                pair_pixel_values_wrist if pair_pixel_values_wrist[0] is not None else None,
            )
        else:
            pair_pixel_values = None

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)
        actions, action_valid_dim = self._right_pad_last_dim(actions, self.target_action_dim, "actions")
        if action_valid_masks[0] is not None:
            if any(mask is None for mask in action_valid_masks):
                raise ValueError("Batch mixes examples with and without action_valid_mask.")
            action_valid_mask = torch.stack(
                [torch.from_numpy(np.copy(mask)).to(dtype=torch.bool) for mask in action_valid_masks]
            )
        else:
            action_valid_mask = None

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.stack(proprio))
            if proprio.dim() == 3 and proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)
            proprio, _ = self._right_pad_last_dim(proprio, self.target_proprio_dim, "proprio")
        else:
            proprio = None

        context = None
        if context_proprios[0] is not None:
            context = dict(proprio=torch.as_tensor(np.stack(context_proprios), dtype=torch.float32), actions=torch.as_tensor(np.stack(context_actions), dtype=torch.float32), delta_t=torch.as_tensor(np.stack(context_delta_ts), dtype=torch.long))
        output = dict(
            pixel_values=pixel_values,
            future_pixel_values=future_pixel_values,
            pair_pixel_values=pair_pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_valid_mask=action_valid_mask,
            context=context,
        )
        if self.target_action_dim is not None and action_valid_dim != actions.shape[-1]:
            output["action_valid_dim"] = action_valid_dim
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

"""Fixed LIBERO RLDS dataset adapter for JEPA-WAM."""

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.constants import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTIONS_CHUNK,
    NUM_TOKENS,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def _stack_pixel_values(values):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    if isinstance(values[0], dict):
        return {key: _stack_pixel_values([value[key] for value in values]) for key in values[0]}
    raise ValueError(f"Unsupported pixel value type `{type(values[0])}`")


def _frame_to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, (bytes, bytearray, np.bytes_)):
        return Image.open(BytesIO(frame)).convert("RGB")
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame).convert("RGB")
    raise ValueError(f"Unsupported frame type `{type(frame)}`")


def _transform_sequence(image_transform: ImageTransform, frames) -> Any:
    return _stack_pixel_values([image_transform(_frame_to_pil(frame)) for frame in frames])


@dataclass
class VLABatchTransform:
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    visual_token_pair_offset: int = 31

    @staticmethod
    def _wrist_keys(observation: Dict[str, Any], paired: bool = False) -> list[str]:
        prefix = "pair_image_" if paired else "image_"
        return sorted(key for key in observation if key.startswith(prefix) and "wrist" in key)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        observation = rlds_batch["observation"]
        instruction = rlds_batch["task"]["language_instruction"].decode().lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction}?")
        prompt_builder.add_turn("gpt", "")
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids.extend([ACTION_TOKEN_BEGIN_IDX] * self.flow_gr00t_placeholder_tokens)

        output = {
            "pixel_values": self.image_transform(_frame_to_pil(observation["image_primary"][0])),
            "input_ids": torch.tensor(input_ids),
            "dataset_name": rlds_batch["dataset_name"],
            "actions": rlds_batch["action"],
        }

        wrist_keys = self._wrist_keys(observation)
        if not wrist_keys:
            raise ValueError("The public recipe requires a wrist-camera observation.")
        output["pixel_values_wrist"] = _stack_pixel_values(
            [self.image_transform(_frame_to_pil(observation[key][0])) for key in wrist_keys]
        )

        if "proprio" in observation:
            output["proprio"] = observation["proprio"]
        if "action_valid_mask" in rlds_batch:
            output["action_valid_mask"] = rlds_batch["action_valid_mask"]

        if self.visual_token_pair_offset > 0:
            if "pair_image_primary" not in observation:
                raise ValueError("Paired primary frames are required for visual-token cosine supervision.")
            output["pair_pixel_values"] = _transform_sequence(
                self.image_transform, observation["pair_image_primary"]
            )

            pair_wrist_keys = self._wrist_keys(observation, paired=True)
            if not pair_wrist_keys:
                raise ValueError("Paired wrist frames are required for visual-token cosine supervision.")
            output["pair_pixel_values_wrist"] = _stack_pixel_values(
                [_transform_sequence(self.image_transform, observation[key]) for key in pair_wrist_keys]
            )

        return output


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: VLABatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 20_000,
        visual_token_pair_offset: int = 31,
    ) -> None:
        if data_mix != "libero_4_task_suites_no_noops":
            raise ValueError("The public recipe only supports `libero_4_task_suites_no_noops`.")

        self.batch_transform = batch_transform
        frame_transform_threads = int(os.getenv("VLA_RLDS_FRAME_TRANSFORM_THREADS", "16"))
        if frame_transform_threads <= 0:
            raise ValueError("VLA_RLDS_FRAME_TRANSFORM_THREADS must be positive.")

        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
        )
        frame_transform_kwargs = {
            "resize_size": resize_resolution,
            "num_parallel_calls": frame_transform_threads,
        }
        self.dataset, self.global_dataset_length, self.dataset_statistics = make_interleaved_dataset(
            traj_transform_kwargs={
                "window_size": 1,
                "future_action_window_size": NUM_ACTIONS_CHUNK - 1,
                "pair_target_offset": visual_token_pair_offset,
                "skip_unlabeled": True,
                "goal_relabeling_strategy": "uniform",
            },
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
        )
        self.dataset_length = self.global_dataset_length

    def __iter__(self):
        datasets = self.dataset if isinstance(self.dataset, list) else [self.dataset]
        for dataset in datasets:
            for batch in dataset.as_numpy_iterator():
                yield self.batch_transform(batch)

    def __len__(self) -> int:
        return self.dataset_length

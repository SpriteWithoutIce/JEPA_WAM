"""Build the fixed LIBERO RLDS dataset and collator."""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_TOKENS
from prismatic.vla.datasets import RLDSDataset, VLABatchTransform


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int = 20_000,
    visual_token_pair_offset: int = 31,
    target_action_dim: int = 7,
    target_proprio_dim: int = 8,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    batch_transform = VLABatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        flow_gr00t_placeholder_tokens=NUM_TOKENS,
        visual_token_pair_offset=visual_token_pair_offset,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
        target_action_dim=target_action_dim,
        target_proprio_dim=target_proprio_dim,
    )
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        visual_token_pair_offset=visual_token_pair_offset,
    )
    return dataset, collator

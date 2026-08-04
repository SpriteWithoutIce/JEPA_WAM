"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Type

from torch.utils.data import Dataset
from transformers import AutoProcessor, PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import NUM_TOKENS
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSDataset, VLABatchTransform


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    use_proprio: bool = False,
    use_wrist_image: bool = False,
    action_head_type: str = "flow_gr00t",
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS,
    use_llm_ce_loss: bool = False,
    future_obs_window_size: int = 8,
    future_obs_downsample_stride: int = 1,
    context: bool = False,
    strict_epoch_mode: bool = False,
    shared_dataset_statistics: bool = False,
    rank_shard_dataset_sources: bool = False,
    visual_token_pair_offset: int = 0,
    stitch_primary_and_wrist_images: bool = False,
    robotwin_aloha_mosaic: bool = False,
    rotation_representation: str = "axis_angle",
    fast_tokenizer_path: Optional[Path] = None,
    dataset_format: str = "rlds",
    lerobot_primary_image_key: Optional[str] = None,
    lerobot_wrist_image_keys: Tuple[str, ...] = (),
    lerobot_state_key: str = "observation.state",
    lerobot_action_key: str = "action",
    lerobot_use_quantile_normalization: bool = True,
    lerobot_normalization_clip_value: float = 15.0,
    lerobot_num_workers: int = 8,
    lerobot_prefetch_factor: int = 3,
    target_action_dim: int | None = None,
    target_proprio_dim: int | None = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize a VLA dataset backend, action tokenizer, batch transform, and collator."""
    action_tokenizer = ActionTokenizer(tokenizer)
    fast_tokenizer: Optional[Any] = None
    if use_llm_ce_loss:
        if fast_tokenizer_path is None:
            raise ValueError("`fast_tokenizer_path` is required when `use_llm_ce_loss=True`.")
        fast_tokenizer = AutoProcessor.from_pretrained(str(fast_tokenizer_path), trust_remote_code=True)
    batch_transform = VLABatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn,
        predict_stop_token=predict_stop_token, use_proprio=use_proprio,
        use_wrist_image=use_wrist_image, action_head_type=action_head_type,
        flow_gr00t_placeholder_tokens=flow_gr00t_placeholder_tokens,
        use_llm_ce_loss=use_llm_ce_loss,
        future_obs_window_size=future_obs_window_size,
        future_obs_downsample_stride=future_obs_downsample_stride,
        context=context,
        visual_token_pair_offset=visual_token_pair_offset,
        stitch_primary_and_wrist_images=stitch_primary_and_wrist_images,
        robotwin_aloha_mosaic=robotwin_aloha_mosaic,
        fast_tokenizer=fast_tokenizer,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=padding_side,
        target_action_dim=target_action_dim,
        target_proprio_dim=target_proprio_dim,
    )

    if dataset_format == "rlds":
        cls = RLDSDataset if not episodic else EpisodicRLDSDataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=(256, 256) if robotwin_aloha_mosaic else default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
            future_obs_window_size=future_obs_window_size,
            future_obs_downsample_stride=future_obs_downsample_stride,
            context=context,
            strict_epoch_mode=strict_epoch_mode,
            shared_dataset_statistics=shared_dataset_statistics,
            rank_shard_dataset_sources=rank_shard_dataset_sources,
            visual_token_pair_offset=visual_token_pair_offset,
            rotation_representation=rotation_representation,
        )
    elif dataset_format == "lerobot":
        if episodic:
            raise ValueError("Episodic visualization mode is not implemented for LeRobot datasets.")
        from prismatic.vla.datasets.lerobot import LeRobotVLADataSet

        dataset = LeRobotVLADataSet(
            data_root_dir,
            data_mix,
            batch_transform,
            primary_image_key=lerobot_primary_image_key,
            wrist_image_keys=lerobot_wrist_image_keys,
            state_key=lerobot_state_key,
            action_key=lerobot_action_key,
            future_obs_window_size=future_obs_window_size,
            visual_token_pair_offset=visual_token_pair_offset,
            use_quantile_normalization=lerobot_use_quantile_normalization,
            normalization_clip_value=lerobot_normalization_clip_value,
            dataloader_num_workers=lerobot_num_workers,
            dataloader_prefetch_factor=lerobot_prefetch_factor,
            strict_epoch_mode=strict_epoch_mode,
            shuffle=train,
        )
    else:
        raise ValueError(f"Unsupported dataset_format `{dataset_format}`; expected `rlds` or `lerobot`.")

    return dataset, action_tokenizer, collator

"""LeRobot v2.1/v3 adapter for the existing JEPA-WAM VLA batch pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_sequence_to_uint8_hwc(value: Any) -> np.ndarray:
    images = _to_numpy(value)
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4:
        raise ValueError(f"Expected image sequence [T,C,H,W] or [T,H,W,C], got {images.shape}.")
    if images.shape[1] in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))
    if np.issubdtype(images.dtype, np.floating):
        images = np.clip(images * 255.0, 0.0, 255.0)
    return images.astype(np.uint8)


def _normalize(
    values: np.ndarray,
    stats: Dict[str, np.ndarray],
    *,
    use_quantiles: bool,
    clip_value: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    low_key, high_key = ("q01", "q99") if use_quantiles else ("min", "max")
    if low_key not in stats or high_key not in stats:
        raise KeyError(
            f"LeRobot statistics are missing `{low_key}`/`{high_key}`. "
            "Generate `<dataset>/meta/stats.json` with reasoningVLA's `calculate_global_stats.py`."
        )
    low = np.asarray(stats[low_key], dtype=np.float32)
    high = np.asarray(stats[high_key], dtype=np.float32)
    normalized = 2.0 * (values.astype(np.float32) - low) / (high - low + 1e-8) - 1.0
    normalized = np.clip(normalized, -clip_value, clip_value).astype(np.float32)
    if mask is not None:
        normalized = np.where(mask, normalized, values).astype(np.float32)
    return normalized


class LeRobotVLADataSet(IterableDataset):
    """Expose reasoningVLA's LeRobotDataset through JEPA-WAM's RLDS-like contract."""

    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        batch_transform,
        *,
        primary_image_key: Optional[str] = None,
        wrist_image_keys: Sequence[str] = (),
        state_key: str = "observation.state",
        action_key: str = "action",
        action_horizon: int = NUM_ACTIONS_CHUNK,
        future_obs_window_size: int = 0,
        visual_token_pair_offset: int = 0,
        use_quantile_normalization: bool = True,
        normalization_clip_value: float = 15.0,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        strict_epoch_mode: bool = False,
        dataloader_num_workers: int = 8,
        dataloader_prefetch_factor: int = 3,
        shuffle: bool = True,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.action_horizon = action_horizon
        self.future_obs_window_size = future_obs_window_size
        self.visual_token_pair_offset = visual_token_pair_offset
        self.use_quantile_normalization = use_quantile_normalization
        self.normalization_clip_value = normalization_clip_value
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.strict_epoch_mode = strict_epoch_mode
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_pin_memory = True
        self.shuffle = shuffle
        self.seed = seed
        self._iteration = 0

        self.dataset = LeRobotDataset(repo_id=self.data_root_dir.name, root=self.data_root_dir)
        if len(self.dataset) != self.dataset.meta.total_frames:
            raise ValueError(
                f"LeRobot dataset is incomplete: loaded {len(self.dataset)} frames, "
                f"but metadata declares {self.dataset.meta.total_frames}."
            )
        camera_keys = list(self.dataset.meta.camera_keys)
        if not camera_keys:
            raise ValueError(f"LeRobot dataset `{self.data_root_dir}` has no image or video features.")

        self.primary_image_key = primary_image_key or camera_keys[0]
        self.wrist_image_keys = tuple(wrist_image_keys) if wrist_image_keys else tuple(
            key for key in camera_keys if key != self.primary_image_key
        )
        self.state_key = state_key
        self.action_key = action_key

        requested_keys = {
            self.primary_image_key,
            *self.wrist_image_keys,
            self.state_key,
            self.action_key,
        }
        missing_keys = sorted(requested_keys - set(self.dataset.meta.features))
        if missing_keys:
            raise KeyError(
                f"LeRobot dataset `{self.data_root_dir}` is missing features {missing_keys}. "
                f"Available features: {sorted(self.dataset.meta.features)}"
            )

        self._image_offsets = list(range(1 + self.future_obs_window_size))
        pair_offsets = [0, self.visual_token_pair_offset] if self.visual_token_pair_offset > 0 else [0]
        self._camera_offsets = sorted(set(self._image_offsets + pair_offsets))
        self._offset_to_position = {offset: i for i, offset in enumerate(self._camera_offsets)}

        fps = self.dataset.fps
        self.delta_timestamps = {
            key: [offset / fps for offset in self._camera_offsets]
            for key in (self.primary_image_key, *self.wrist_image_keys)
        }
        self.delta_timestamps[self.state_key] = [0.0]
        self.delta_timestamps[self.action_key] = [
            offset / fps for offset in range(self.action_horizon)
        ]

        raw_stats = self.dataset.meta.stats
        action_stats = raw_stats[self.action_key]
        proprio_stats = raw_stats[self.state_key]
        action_dim = len(np.asarray(action_stats["min"]))
        self.action_normalization_mask = np.ones(action_dim, dtype=bool)
        if "libero" in self.dataset_name.lower() and action_dim == 7:
            self.action_normalization_mask[-1] = False
        self.dataset_statistics = {
            self.dataset_name: {
                "action": {
                    **{key: np.asarray(value) for key, value in action_stats.items()},
                    "mask": self.action_normalization_mask,
                },
                "proprio": {
                    **{key: np.asarray(value) for key, value in proprio_stats.items()},
                    "mask": np.ones_like(np.asarray(proprio_stats["min"]), dtype=bool),
                },
                "num_transitions": len(self.dataset),
                "num_trajectories": self.dataset.meta.total_episodes,
            }
        }
        self.global_dataset_length = len(self.dataset)
        self.dataset_length = self.global_dataset_length

    @staticmethod
    def _rank_world_size() -> Tuple[int, int]:
        if os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None:
            return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _local_indices(self, iteration: int) -> np.ndarray:
        indices = np.arange(self.global_dataset_length)
        if self.shuffle:
            np.random.default_rng(self.seed + iteration).shuffle(indices)
        rank, world_size = self._rank_world_size()
        worker = get_worker_info()
        if worker is None:
            return indices[rank::world_size]
        shard_id = rank * worker.num_workers + worker.id
        num_shards = world_size * worker.num_workers
        return indices[shard_id::num_shards]

    def _select_camera_offsets(self, images: np.ndarray, offsets: Iterable[int]) -> np.ndarray:
        return np.stack([images[self._offset_to_position[offset]] for offset in offsets])

    def _adapt_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        primary = _image_sequence_to_uint8_hwc(sample[self.primary_image_key])
        observation: Dict[str, Any] = {
            "image_primary": self._select_camera_offsets(primary, self._image_offsets),
        }
        if self.visual_token_pair_offset > 0:
            observation["pair_image_primary"] = self._select_camera_offsets(
                primary, (0, self.visual_token_pair_offset)
            )

        for wrist_index, key in enumerate(self.wrist_image_keys):
            wrist = _image_sequence_to_uint8_hwc(sample[key])
            suffix = "wrist" if wrist_index == 0 else f"wrist_{wrist_index}"
            observation[f"image_{suffix}"] = self._select_camera_offsets(wrist, (0,))
            if self.visual_token_pair_offset > 0:
                observation[f"pair_image_{suffix}"] = self._select_camera_offsets(
                    wrist, (0, self.visual_token_pair_offset)
                )

        proprio = _to_numpy(sample[self.state_key]).astype(np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None]
        if self.normalize_proprio:
            proprio = _normalize(
                proprio,
                self.dataset.meta.stats[self.state_key],
                use_quantiles=self.use_quantile_normalization,
                clip_value=self.normalization_clip_value,
            )
        observation["proprio"] = proprio

        actions = _to_numpy(sample[self.action_key]).astype(np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if self.normalize_actions:
            actions = _normalize(
                actions,
                self.dataset.meta.stats[self.action_key],
                use_quantiles=self.use_quantile_normalization,
                clip_value=self.normalization_clip_value,
                mask=self.action_normalization_mask,
            )

        action_is_pad = _to_numpy(sample[f"{self.action_key}_is_pad"]).astype(bool)
        task = sample["task"] if isinstance(sample["task"], str) else str(sample["task"])
        return {
            "dataset_name": self.dataset_name,
            "observation": observation,
            "task": {"language_instruction": task.encode("utf-8")},
            "action": actions,
            "action_valid_mask": np.logical_not(action_is_pad),
        }

    def __iter__(self):
        iteration = self._iteration
        self._iteration += 1
        while True:
            for index in self._local_indices(iteration):
                sample = self.dataset.getitem_with_delta_timestamps(int(index), self.delta_timestamps)
                yield self.batch_transform(self._adapt_sample(sample))
            if self.strict_epoch_mode:
                return
            iteration += 1

    def __len__(self) -> int:
        rank, world_size = self._rank_world_size()
        return (self.global_dataset_length + world_size - rank - 1) // world_size

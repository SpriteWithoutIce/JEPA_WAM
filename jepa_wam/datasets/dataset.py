"""JEPA-WAM RLDS dataset wrapper."""

from pathlib import Path
from typing import Any, Dict

from torch.utils.data import IterableDataset

from jepa_wam.conf.config import DataConfig
from jepa_wam.datasets.batch_transform import JEPAWBatchTransform
from jepa_wam.datasets.rlds.dataset_pipeline import make_interleaved_dataset_jepa
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE


class JEPAWRLDSDataset(IterableDataset):
    """Lightweight wrapper around RLDS TFDS pipeline for JEPA-WAM."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: JEPAWBatchTransform,
        data_cfg: DataConfig,
        train: bool = True,
    ) -> None:
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform
        self.data_cfg = data_cfg
        self.train = train

        # Resolve mixture spec
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=data_cfg.window_size,
                future_action_window_size=data_cfg.future_action_window_size,
                future_obs_window_size=data_cfg.future_obs_window_size,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=data_cfg.resize_resolution,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=data_cfg.shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        if data_cfg.image_aug:
            rlds_config["frame_transform_kwargs"].update({
                "image_augment_kwargs": dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            })

        self.dataset, self.dataset_length, self.dataset_statistics = make_interleaved_dataset_jepa(
            **rlds_config
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__!")

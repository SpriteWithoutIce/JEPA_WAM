"""Extended RLDS dataset pipeline with future-frame chunking for JEPA-WAM.

Mirrors prismatic.vla.datasets.rlds.dataset but uses chunk_act_obs_with_future.
"""

import copy
from functools import partial
from typing import Dict, List, Optional

import dlimp as dl
import numpy as np
import tensorflow as tf
from absl import logging as absl_logging

from jepa_wam.datasets.rlds import traj_transforms as jepa_traj_transforms
from prismatic.vla.datasets.rlds.dataset import (
    apply_frame_transforms,
    allocate_threads,
    make_dataset_from_rlds,
    pprint_data_mixture,
)
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def apply_trajectory_transforms_jepa(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    future_obs_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = -1,  # tf.data.AUTOTUNE == -1
) -> dl.DLataset:
    """Same as apply_trajectory_transforms but chunks future observations too."""
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action))

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["observation"]["proprio"]) <= max_proprio))

    dataset = dataset.traj_map(jepa_traj_transforms.add_pad_mask_dict, num_parallel_calls)

    if goal_relabeling_strategy is not None:
        import prismatic.vla.datasets.rlds.utils.goal_relabeling as goal_relabeling
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    if train and task_augment_strategy is not None:
        import prismatic.vla.datasets.rlds.utils.task_augmentation as task_augmentation
        dataset = dataset.traj_map(
            partial(getattr(task_augmentation, task_augment_strategy), **task_augment_kwargs),
            num_parallel_calls,
        )

    # Core difference: use chunk_act_obs_with_future
    dataset = dataset.traj_map(
        partial(
            jepa_traj_transforms.chunk_act_obs_with_future,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
            future_obs_window_size=future_obs_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(jepa_traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset


def make_single_dataset_jepa(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
):
    """Same as make_single_dataset but uses JEPA trajectory transforms."""
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms_jepa(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)
    dataset = dataset.with_ram_budget(1)
    return dataset, dataset_statistics["num_trajectories"], dataset_statistics


def make_interleaved_dataset_jepa(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
):
    """Same as make_interleaved_dataset but uses JEPA trajectory transforms."""
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])

    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info("Threads per Dataset: %s", threads_per_dataset)
    overwatch.info("Reads per Dataset: %s", reads_per_dataset)

    overwatch.info("Constructing datasets...")
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs", {}) or {}
        )

        dataset_kwargs["num_parallel_reads"] = reads
        dataset_kwargs["num_parallel_calls"] = threads

        dataset, _, dataset_statistics = make_single_dataset_jepa(
            dataset_kwargs,
            train=train,
            traj_transform_kwargs={
                **traj_transform_kwargs,
                "num_parallel_calls": threads,
            },
            frame_transform_kwargs={
                **frame_transform_kwargs,
                **dataset_frame_transform_kwargs,
            },
        )
        datasets.append(dataset)

    overwatch.info("Interleaving datasets...")
    dataset = dl.DLataset.interleave(datasets, sample_weights)

    overwatch.info("Shuffling dataset with buffer size %d...", shuffle_buffer_size)
    dataset = dataset.shuffle(shuffle_buffer_size)

    if batch_size is not None:
        overwatch.info("Batching dataset with batch size %d...", batch_size)
        dataset = dataset.batch(batch_size)

    return dataset, dataset_len, all_dataset_statistics

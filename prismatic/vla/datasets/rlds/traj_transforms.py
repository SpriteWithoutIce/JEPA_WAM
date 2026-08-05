"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def chunk_act_obs(
    traj: Dict,
    window_size: int,
    future_action_window_size: int = 0,
    pair_target_offset: int = 0,
) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis at index 1.
    "action" is given a new axis (at index 1) of size `window_size + future_action_window_size`
    containing `window_size - 1` actions from the past, the current action, and
    `future_action_window_size` actions from the future. Out-of-range actions are
    replaced by the nearest endpoint and marked False in `action_valid_mask`.
    "pad_mask" is added to "observation".
    If `pair_target_offset > 0`, image observations additionally expose a two-frame
    pair `[current, current + pair_target_offset]`, where the second frame is clamped
    to the last valid frame near episode boundaries.
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    # Keep tail states for action supervision. Future action indices are padded
    # with the final action and masked out of the loss downstream.
    effective_traj_len = traj_len

    # Observation chunk indices
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1),
        [effective_traj_len, window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size],
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    # Clamp observation indices to valid range [0, traj_len - 1]
    floored_chunk_indices = tf.minimum(tf.maximum(chunk_indices, 0), traj_len - 1)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)
    action_valid_mask = tf.logical_and(
        action_chunk_indices >= 0,
        action_chunk_indices <= goal_timestep[:, None],
    )
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    old_obs = traj["observation"]
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), old_obs)
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    traj["action_valid_mask"] = action_valid_mask

    if pair_target_offset > 0:
        pair_indices = tf.broadcast_to(
            tf.constant([0, pair_target_offset], dtype=tf.int32),
            [effective_traj_len, 2],
        ) + tf.broadcast_to(
            tf.range(effective_traj_len, dtype=tf.int32)[:, None],
            [effective_traj_len, 2],
        )
        floored_pair_indices = tf.minimum(tf.maximum(pair_indices, 0), traj_len - 1)
        for key, value in old_obs.items():
            if key.startswith("image_") and not key.startswith("depth_"):
                traj["observation"][f"pair_{key}"] = tf.gather(value, floored_pair_indices)

    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj

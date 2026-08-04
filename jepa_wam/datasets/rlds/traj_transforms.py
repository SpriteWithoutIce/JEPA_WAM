"""Extended trajectory transforms that also chunk future observations."""

from functools import partial
from typing import Dict

import tensorflow as tf


def chunk_act_obs_with_future(
    traj: Dict,
    window_size: int = 1,
    future_action_window_size: int = 0,
    future_obs_window_size: int = 0,
) -> Dict:
    """Extended version of chunk_act_obs that also extracts future frames.

    In addition to the standard observation/action chunking, this adds
    `future_image_{name}` keys under observation for each image key.

    Args:
        window_size: length of the observation snippet (past + current).
        future_action_window_size: number of future actions beyond window_size.
        future_obs_window_size: number of future frames to extract (e.g. 8).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    max_future = tf.maximum(future_action_window_size, future_obs_window_size)
    effective_traj_len = traj_len - max_future

    # Observation chunk indices (past + current)
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [effective_traj_len, window_size]
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    # Action chunk indices (past + current + future)
    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    # Future observation chunk indices (current+1 .. current+future_obs_window_size)
    if future_obs_window_size > 0:
        future_obs_indices = tf.broadcast_to(
            tf.range(1, 1 + future_obs_window_size),
            [effective_traj_len, future_obs_window_size],
        ) + tf.broadcast_to(
            tf.range(effective_traj_len)[:, None],
            [effective_traj_len, future_obs_window_size],
        )
    else:
        future_obs_indices = None

    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)
    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None]
    )
    if future_obs_indices is not None:
        floored_future_obs_indices = tf.minimum(
            tf.maximum(future_obs_indices, 0), goal_timestep[:, None]
        )

    # Apply chunking to observations
    old_obs = traj["observation"]
    new_obs = {}
    for key, val in old_obs.items():
        if key.startswith("image_") or key.startswith("depth_"):
            # Standard chunk
            new_obs[key] = tf.gather(val, floored_chunk_indices)
            # Future chunk
            if future_obs_window_size > 0 and not key.startswith("depth_"):
                future_key = f"future_{key}"
                new_obs[future_key] = tf.gather(val, floored_future_obs_indices)
        else:
            new_obs[key] = tf.gather(val, floored_chunk_indices)

    traj["observation"] = new_obs
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # Padding mask for standard observations
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other trajectory elements
    traj["task"] = tf.nest.map_structure(
        lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"]
    )
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    if "absolute_action_mask" in traj:
        traj["absolute_action_mask"] = tf.gather(
            traj["absolute_action_mask"], tf.range(effective_traj_len)
        )

    return traj

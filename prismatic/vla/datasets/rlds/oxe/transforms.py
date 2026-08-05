"""LIBERO trajectory standardization for the public RLDS pipeline."""

from typing import Any, Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.oxe.configs import LIBERO_DATASETS
from prismatic.vla.datasets.rlds.utils.data_utils import invert_gripper_actions


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))
    trajectory["action"] = tf.concat([trajectory["action"][:, :6], gripper_action], axis=1)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {name: libero_dataset_transform for name in LIBERO_DATASETS}

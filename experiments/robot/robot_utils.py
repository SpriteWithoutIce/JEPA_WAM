"""Small evaluation utilities for the local JEPA-WAM policy."""

import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch

from experiments.robot.openvla_utils import get_vla, get_vla_action

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any) -> torch.nn.Module:
    return get_vla(cfg)


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    observation: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    return get_vla_action(cfg, model, observation, task_label)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    action = action.copy()
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] *= -1
    return action

"""Local JEPA-WAM checkpoint loading and inference helpers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from prismatic.models import load_vla

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _is_native_prismatic_checkpoint_path(model_path: Union[str, Path]) -> bool:
    path = Path(os.path.expanduser(str(model_path)))
    return path.is_file() and path.suffix == ".pt" and path.parent.name == "checkpoints"


def get_vla(cfg: Any) -> torch.nn.Module:
    checkpoint = Path(os.path.expanduser(str(cfg.pretrained_checkpoint)))
    if not _is_native_prismatic_checkpoint_path(checkpoint):
        raise ValueError("Evaluation requires a local `runs/.../checkpoints/*.pt` JEPA-WAM checkpoint.")
    model = load_vla(
        checkpoint,
        load_for_training=False,
        base_vlm=cfg.base_vlm,
        llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,
        load_visual_token_cosine_head=False,
    )
    return model.eval().to(DEVICE)


def get_vla_action(cfg: Any, vla: torch.nn.Module, obs: Dict[str, Any], task_label: str) -> List[np.ndarray]:
    images = [Image.fromarray(obs["full_image"]), Image.fromarray(obs["wrist_image"])]
    action_chunk = vla.predict_action(
        image=images,
        instruction=task_label,
        unnorm_key=cfg.unnorm_key,
        proprio=obs["state"],
    )
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None]
    return [action_chunk[index] for index in range(min(len(action_chunk), cfg.num_open_loop_steps))]

"""Batch transform: converts raw RLDS batches to JEPA-WAM format."""

from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from jepa_wam.conf.config import DataConfig


# Qwen2.5 system prompt (matches QwenPromptBuilder in Prismatic)
_QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class JEPAWBatchTransform:
    """Transform RLDS batches into dicts expected by JepaVLA."""

    base_tokenizer: PreTrainedTokenizerBase
    data_cfg: DataConfig
    prompt_template: str = "What action should the robot take to {instruction}?"

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]

        # ---- Images ----
        # Current frame: take the last entry of the observation window (index -1)
        # RLDS chunking puts [past..., current] at axis 1
        obs = rlds_batch["observation"]
        img_primary = obs["image_primary"][-1]  # [H, W, 3]  uint8
        current_img = Image.fromarray(np.array(img_primary))

        # Future frames (if available)
        future_frames = None
        if "future_image_primary" in obs:
            fut = obs["future_image_primary"]  # [8, H, W, 3] uint8
            future_frames = [Image.fromarray(np.array(f)) for f in fut]

        # Secondary / wrist views
        view_images = [current_img]
        future_views = []
        if "future_image_primary" in obs:
            future_views.append(future_frames)

        for key in obs:
            if key.startswith("image_") and key != "image_primary":
                view_images.append(Image.fromarray(np.array(obs[key][-1])))
                fut_key = f"future_{key}"
                if fut_key in obs:
                    future_views.append([Image.fromarray(np.array(f)) for f in obs[fut_key]])

        # ---- Language ----
        lang = rlds_batch["task"]["language_instruction"]
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8").lower()
        user_content = self.prompt_template.format(instruction=lang)

        # Use Qwen chat template so that <|im_start|>/<|im_end|> special tokens are
        # present — without them Qwen2.5 cannot activate its instruction-following
        # capabilities and the language conditioning has no effect on the action.
        messages = [
            {"role": "system", "content": _QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        tokenized = self.base_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # ---- Proprio ----
        proprio = None
        if "proprio" in obs:
            proprio = np.array(obs["proprio"][-1], dtype=np.float32)

        # ---- Action ----
        # RLDS action is chunked: [window_size + future_action_window_size, action_dim]
        # We want the future action chunk (current + future) for training.
        actions = np.array(rlds_batch["action"], dtype=np.float32)
        # actions shape should be [action_horizon, action_dim]

        return {
            "dataset_name": dataset_name,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "view_images": view_images,              # List[PIL.Image]  length V
            "future_view_images": future_views,      # List[List[PIL.Image]]  length V, each length 8
            "proprio": proprio,
            "action": actions,
        }

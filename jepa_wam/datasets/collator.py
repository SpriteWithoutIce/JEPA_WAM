"""Collator for JEPA-WAM: pads text tokens and stacks images/actions."""

from typing import Dict, List

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class JEPAWCollator:
    """Pads language sequences and stacks continuous tensors."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        img_size: int = 224,
        action_horizon: int = 8,
        action_dim: int = 7,
        proprio_dim: int = 7,
    ) -> None:
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        # V-JEPA expects ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _pil_to_tensor(self, img) -> torch.Tensor:
        """Convert PIL Image -> [3, H, W] float32 tensor, ImageNet normalized."""
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0  # [H, W, 3]
        arr = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        # Resize if needed
        if arr.shape[1] != self.img_size or arr.shape[2] != self.img_size:
            arr = torch.nn.functional.interpolate(
                arr.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        arr = (arr - self.mean) / self.std
        return arr

    def __call__(self, raw_batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # raw_batch: list of dicts from JEPAWBatchTransform

        # 1. Pad text tokens
        input_ids_list = [item["input_ids"] for item in raw_batch]
        attention_mask_list = [item["attention_mask"] for item in raw_batch]

        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)

        text_tokens = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(padded_mask, dtype=torch.long)

        # 2. Stack images per view
        # Determine number of views from first sample
        num_views = len(raw_batch[0]["view_images"])

        current_frames = []  # [V, B, 3, H, W]
        for v in range(num_views):
            view_stack = torch.stack([
                self._pil_to_tensor(item["view_images"][v])
                for item in raw_batch
            ])  # [B, 3, H, W]
            current_frames.append(view_stack)

        # [B, V, 3, H, W]
        images = torch.stack(current_frames, dim=1)

        # 3. Stack future frames
        future_frames = None
        if raw_batch[0]["future_view_images"]:
            future_views = []
            for v in range(num_views):
                # each item: List[8 PIL Images]
                fut_stack = torch.stack([
                    torch.stack([
                        self._pil_to_tensor(fimg)
                        for fimg in item["future_view_images"][v]
                    ])  # [8, 3, H, W]
                    for item in raw_batch
                ])  # [B, 8, 3, H, W]
                future_views.append(fut_stack)
            # [B, V, 8, 3, H, W]
            future_frames = torch.stack(future_views, dim=1)

        # 4. Stack proprio
        proprio_list = []
        for item in raw_batch:
            p = item["proprio"]
            if p is None:
                p = np.zeros(self.proprio_dim, dtype=np.float32)
            proprio_list.append(p)
        proprio = torch.tensor(np.stack(proprio_list), dtype=torch.float32)

        # 5. Stack actions
        actions = torch.tensor(np.stack([item["action"] for item in raw_batch]), dtype=torch.float32)
        # Ensure shape [B, H_a, D_action]
        if actions.dim() == 2:
            # [B, action_dim]  -> single step, pad or repeat?
            # For safety, if action_horizon > 1 but we got 1, we need to handle this.
            # RLDS chunking should already give [action_horizon, action_dim]
            pass

        batch_dict = {
            "images": images,
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "proprio": proprio,
            "action": actions,
        }
        if future_frames is not None:
            batch_dict["future_frames"] = future_frames

        return batch_dict

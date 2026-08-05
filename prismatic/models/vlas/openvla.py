"""Inference wrapper for the released JEPA-WAM continuous-action policy."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats

    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[Image, List[Image]],
        instruction: str,
        unnorm_key: Optional[str] = None,
        proprio: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        **_: str,
    ) -> np.ndarray:
        if proprio is None:
            raise ValueError("JEPA-WAM inference requires the LIBERO proprio state.")

        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        prompt_builder.add_turn("gpt", "")
        tokenized = self.llm_backbone.tokenizer(prompt_builder.get_prompt(), return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        placeholder_ids = torch.full(
            (input_ids.shape[0], self.action_placeholder_tokens),
            ACTION_TOKEN_BEGIN_IDX,
            dtype=input_ids.dtype,
            device=self.device,
        )
        input_ids = torch.cat([input_ids, placeholder_ids], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], self.action_placeholder_tokens),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        ).bool()

        images = image if isinstance(image, list) else [image]
        if len(images) != 2:
            raise ValueError("The released policy expects primary and wrist images.")
        image_transform = self.vision_backbone.get_image_transform()
        pixel_values = torch.stack([image_transform(frame) for frame in images], dim=0)[None].to(self.device)

        proprio_tensor = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        if proprio_tensor.ndim == 1:
            proprio_tensor = proprio_tensor.unsqueeze(0)
        key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        proprio_stats = self.norm_stats[key]["proprio"]
        proprio_low = torch.as_tensor(proprio_stats["q01"], device=self.device, dtype=torch.float32)
        proprio_high = torch.as_tensor(proprio_stats["q99"], device=self.device, dtype=torch.float32)
        proprio_mask = torch.as_tensor(
            proprio_stats.get("mask", np.ones_like(proprio_stats["q01"], dtype=bool)),
            device=self.device,
            dtype=torch.bool,
        )
        normalized_proprio = torch.where(
            proprio_mask,
            (2 * (proprio_tensor - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1).clamp(-1, 1),
            proprio_tensor,
        )

        with torch.autocast(
            "cuda",
            dtype=self.llm_backbone.half_precision_dtype,
            enabled=torch.cuda.is_available() and self.enable_mixed_precision_training,
        ):
            output = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            action_memory = output["llm_hidden"][:, -self.action_placeholder_tokens :, :]
            normalized_actions = self.action_head.predict_action(action_memory, normalized_proprio)

        normalized_actions = normalized_actions.float().cpu().numpy()[0]
        action_stats = self.norm_stats[key]["action"]
        action_low = np.asarray(action_stats["q01"])
        action_high = np.asarray(action_stats["q99"])
        action_mask = np.asarray(action_stats.get("mask", np.ones_like(action_low, dtype=bool)))
        return np.where(
            action_mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(f"Choose an unnorm_key from {sorted(norm_stats)}.")
            return next(iter(norm_stats))
        if unnorm_key not in norm_stats:
            raise ValueError(f"Unknown unnorm_key `{unnorm_key}`; choose from {sorted(norm_stats)}.")
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[key]["action"]

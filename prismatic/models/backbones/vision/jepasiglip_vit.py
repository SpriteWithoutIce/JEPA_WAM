"""
jepasiglip_vit.py

Vision backbone that returns concatenated features from both V-JEPA 2.1 and SigLIP.
"""

from dataclasses import dataclass
from functools import partial
from math import isqrt
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp.wrap import _or_policy

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone
from prismatic.models.backbones.vision.siglip_vit import SigLIPViTBackbone
from prismatic.models.backbones.vision.vjepa_vit import VJEPA21ViTBackbone

JEPASigLIP_VISION_BACKBONES = {
    "jepasiglip-vit-l-so-384px": {
        "vjepa": "vjepa2_1-vit-l-384px",
        "siglip": "siglip-vit-so400m-384px",
        "default_image_size": 384,
    },
}


@dataclass
class JEPASigLIPImageTransform:
    vjepa_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"vjepa": self.vjepa_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class JEPASigLIPViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 384,
        checkpoint_path: Optional[str] = None,
        siglip_local_path: Optional[str] = None,
    ) -> None:
        if vision_backbone_id not in JEPASigLIP_VISION_BACKBONES:
            raise ValueError(f"JEPA+SigLIP backbone `{vision_backbone_id}` is not supported!")

        cfg = JEPASigLIP_VISION_BACKBONES[vision_backbone_id]
        default_image_size = int(cfg["default_image_size"])
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.vjepa_backbone = VJEPA21ViTBackbone(
            str(cfg["vjepa"]),
            image_resize_strategy=image_resize_strategy,
            default_image_size=default_image_size,
            checkpoint_path=checkpoint_path,
        )
        self.siglip_backbone = SigLIPViTBackbone(
            str(cfg["siglip"]),
            image_resize_strategy=image_resize_strategy,
            default_image_size=default_image_size,
            local_path=siglip_local_path,
        )

        self.image_transform = JEPASigLIPImageTransform(
            self.vjepa_backbone.get_image_transform(), self.siglip_backbone.get_image_transform()
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            _or_policy,
            policies=[self.vjepa_backbone.get_fsdp_wrapping_policy(), self.siglip_backbone.get_fsdp_wrapping_policy()],
        )

    @staticmethod
    def _resize_patch_grid(patches: torch.Tensor, target_num_patches: int) -> torch.Tensor:
        src_side, target_side = isqrt(patches.shape[1]), isqrt(target_num_patches)
        if src_side * src_side != patches.shape[1]:
            raise ValueError(f"Expected a square patch grid, got {patches.shape[1]} patches.")
        if target_side * target_side != target_num_patches:
            raise ValueError(f"Expected a square target patch grid, got {target_num_patches} patches.")

        if src_side == target_side:
            return patches

        batch_size, _, channels = patches.shape
        patches_2d = patches.transpose(1, 2).reshape(batch_size, channels, src_side, src_side)
        resized = F.interpolate(patches_2d, size=(target_side, target_side), mode="bicubic", align_corners=False)
        return resized.reshape(batch_size, channels, target_num_patches).transpose(1, 2).contiguous()

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        vjepa_patches = self.vjepa_backbone(pixel_values["vjepa"])
        siglip_patches = self.siglip_backbone(pixel_values["siglip"])

        siglip_patches = self._resize_patch_grid(siglip_patches, vjepa_patches.shape[1])
        siglip_patches = siglip_patches.to(device=vjepa_patches.device, dtype=vjepa_patches.dtype)

        return torch.cat([vjepa_patches, siglip_patches], dim=2)

    def encode_future(self, future_pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vjepa_backbone.encode_future(future_pixel_values["vjepa"])

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.vjepa_backbone.default_image_resolution

    @property
    def embed_dim(self) -> int:
        return self.vjepa_backbone.embed_dim + self.siglip_backbone.embed_dim

    @property
    def num_patches(self) -> int:
        return self.vjepa_backbone.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

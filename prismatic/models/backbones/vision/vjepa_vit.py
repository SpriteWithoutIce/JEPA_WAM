"""
vjepa_vit.py

V-JEPA 2.1 vision backbone wrapper for Prismatic. This keeps the underlying
encoder architecture aligned with the AlphaBrain implementation and only adapts
it to the local VisionBackbone interface.
"""

import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone
from prismatic.models.backbones.vision.vjepa import Block, VisionTransformer
from prismatic.models.backbones.vision.vjepa import vision_transformer as vit_module

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_CHECKPOINT_KEYS = ("target_encoder", "ema_encoder", "encoder")

VJEPA_VISION_BACKBONES: Dict[str, Dict[str, object]] = {
    "vjepa2_1-vit-b-384px": {
        "model_size": "vjepa2_1_vit_base",
        "factory": "vit_base",
        "embed_dim": 768,
        "default_image_size": 384,
    },
    "vjepa2_1-vit-l-384px": {
        "model_size": "vjepa2_1_vit_large",
        "factory": "vit_large",
        "embed_dim": 1024,
        "default_image_size": 384,
    },
    "vjepa2_1-vit-g-384px": {
        "model_size": "vjepa2_1_vit_giant",
        "factory": "vit_giant_xformers",
        "embed_dim": 1408,
        "default_image_size": 384,
    },
    "vjepa2_1-vit-G-384px": {
        "model_size": "vjepa2_1_vit_gigantic",
        "factory": "vit_gigantic_xformers",
        "embed_dim": 1664,
        "default_image_size": 384,
    },
}


def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned[new_key] = value
    return cleaned


def _extract_encoder_state_dict(checkpoint: dict) -> dict:
    for key in _CHECKPOINT_KEYS:
        if key in checkpoint:
            return _clean_state_dict(checkpoint[key])
    return _clean_state_dict(checkpoint)


class VJEPA21ViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 384,
        checkpoint_path: Optional[str] = None,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_rope: bool = True,
        interpolate_rope: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        if vision_backbone_id not in VJEPA_VISION_BACKBONES:
            raise ValueError(f"V-JEPA backbone `{vision_backbone_id}` is not supported!")

        cfg = VJEPA_VISION_BACKBONES[vision_backbone_id]
        default_image_size = int(cfg["default_image_size"])
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_rope = use_rope
        self.interpolate_rope = interpolate_rope
        self.freeze_backbone = freeze_backbone
        self._embed_dim = int(cfg["embed_dim"])
        self.checkpoint_path = checkpoint_path or os.environ.get("VJEPA_CHECKPOINT_PATH")

        factory_name = str(cfg["factory"])
        factory_fn = getattr(vit_module, factory_name)
        encoder_kwargs = dict(
            patch_size=patch_size,
            img_size=(self.default_image_size, self.default_image_size),
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=use_rope,
            img_temporal_dim_size=1,
            interpolate_rope=interpolate_rope,
        )
        self.featurizer: VisionTransformer = factory_fn(**encoder_kwargs)

        if self.checkpoint_path is not None:
            self._load_weights(self.checkpoint_path)

        self.featurizer = self.featurizer.to(dtype=torch.bfloat16)
        if self.freeze_backbone:
            self.featurizer.requires_grad_(False)
            self.featurizer.eval()

        if self.image_resize_strategy not in {"resize-naive", "resize-crop"}:
            raise ValueError(
                f"Image Resize Strategy `{self.image_resize_strategy}` is not supported for V-JEPA. "
                "Use `resize-naive` to preserve AlphaBrain preprocessing behavior."
            )
        self.image_transform = Compose(
            [
                Resize((self.default_image_size, self.default_image_size), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _load_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_sd = _extract_encoder_state_dict(checkpoint)
        missing, unexpected = self.featurizer.load_state_dict(encoder_sd, strict=False)

        non_trivial_missing = [key for key in missing if "pos_embed" not in key]
        if non_trivial_missing:
            raise ValueError(f"Missing non-positional keys when loading V-JEPA checkpoint: {non_trivial_missing}")
        if unexpected:
            raise ValueError(f"Unexpected keys when loading V-JEPA checkpoint: {unexpected}")

    def get_fsdp_wrapping_policy(self) -> Callable:
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            x = pixel_values.unsqueeze(2)
            batch_size = pixel_values.shape[0]
            num_views = 1
        elif pixel_values.ndim == 5:
            batch_size, num_views, channels, height, width = pixel_values.shape
            x = pixel_values.reshape(batch_size * num_views, channels, height, width).unsqueeze(2)
        else:
            raise ValueError(
                "Expected image tensor of shape [B, 3, H, W] or [B, V, 3, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )

        x = x.to(device=next(self.featurizer.parameters()).device, dtype=next(self.featurizer.parameters()).dtype)
        out = self.featurizer(x)
        if isinstance(out, list):
            out = out[-1]
        if num_views > 1:
            out = out.reshape(batch_size, num_views, out.shape[1], out.shape[2]).reshape(
                batch_size, num_views * out.shape[1], out.shape[2]
            )
        return out

    def encode_future(self, future_pixel_values: torch.Tensor) -> torch.Tensor:
        if future_pixel_values.ndim == 5:
            batch_size, num_frames, channels, height, width = future_pixel_values.shape
            num_views = 1
            x = future_pixel_values.permute(0, 2, 1, 3, 4)
        elif future_pixel_values.ndim == 6:
            batch_size, num_views, num_frames, channels, height, width = future_pixel_values.shape
            x = future_pixel_values.reshape(batch_size * num_views, num_frames, channels, height, width).permute(
                0, 2, 1, 3, 4
            )
        else:
            raise ValueError(
                "Expected future image tensor of shape [B, T, 3, H, W] or [B, V, T, 3, H, W], "
                f"got {tuple(future_pixel_values.shape)}"
            )

        x = x.to(device=next(self.featurizer.parameters()).device, dtype=next(self.featurizer.parameters()).dtype)
        out = self.featurizer(x)
        if isinstance(out, list):
            out = out[-1]

        temporal_tokens = max(1, num_frames // self.tubelet_size)
        spatial_side = self.default_image_size // self.patch_size
        expected_tokens = temporal_tokens * spatial_side * spatial_side
        if out.shape[1] != expected_tokens:
            raise ValueError(
                f"Unexpected V-JEPA future token count: got {out.shape[1]}, expected {expected_tokens} "
                f"(T={num_frames}, tubelet={self.tubelet_size}, side={spatial_side})"
            )

        if num_views == 1:
            return out.reshape(batch_size, 1, temporal_tokens, spatial_side, spatial_side, out.shape[-1]).detach()

        return out.reshape(
            batch_size, num_views, temporal_tokens, spatial_side, spatial_side, out.shape[-1]
        ).detach()

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return (self.default_image_size // self.patch_size) ** 2

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

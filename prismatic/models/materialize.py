"""Factories for the single released JEPA-WAM backbone combination."""

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import LLMBackbone, Qwen25LLMBackbone
from prismatic.models.backbones.vision import ImageTransform, VJEPA21ViTBackbone, VisionBackbone
from prismatic.models.vlms import PrismaticVLM

VISION_BACKBONE_ID = "vjepa2_1-vit-l-384px"
LLM_BACKBONE_ID = "qwen25-0_5b-pure"


def get_vision_backbone_and_transform(
    vision_backbone_id: str,
    image_resize_strategy: str,
    checkpoint_path: Optional[str] = None,
) -> Tuple[VisionBackbone, ImageTransform]:
    if vision_backbone_id != VISION_BACKBONE_ID:
        raise ValueError(
            f"The public recipe only supports `{VISION_BACKBONE_ID}`, got `{vision_backbone_id}`."
        )
    backbone = VJEPA21ViTBackbone(
        vision_backbone_id,
        image_resize_strategy,
        checkpoint_path=checkpoint_path,
    )
    return backbone, backbone.get_image_transform()


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 32_768,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
    custom_hf_path: Optional[str] = None,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id != LLM_BACKBONE_ID:
        raise ValueError(f"The public recipe only supports `{LLM_BACKBONE_ID}`, got `{llm_backbone_id}`.")
    backbone = Qwen25LLMBackbone(
        llm_backbone_id,
        llm_max_length=llm_max_length,
        llm_path=custom_hf_path,
        hf_token=hf_token,
        inference_mode=inference_mode,
    )
    return backbone, backbone.get_tokenizer()


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
    **kwargs,
) -> PrismaticVLM:
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        **kwargs,
    )

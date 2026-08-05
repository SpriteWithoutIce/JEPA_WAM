"""
base_vlm.py

Minimal interface shared by the released JEPA-WAM model and inference wrapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone


# === JEPA-WAM VLM Interface ===
class VLM(nn.Module, ABC):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
    ) -> None:
        super().__init__()
        self.model_family, self.model_id = model_family, model_id
        self.vision_backbone, self.llm_backbone = vision_backbone, llm_backbone
        self.enable_mixed_precision_training = enable_mixed_precision_training

        self.all_module_keys, self.trainable_module_keys = None, None

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        **kwargs: str,
    ) -> VLM: ...

    @abstractmethod
    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder: ...

    @abstractmethod
    def freeze_for_training(self) -> None: ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> dict: ...

"""Projection module used by the released base VLM."""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(image_tokens)

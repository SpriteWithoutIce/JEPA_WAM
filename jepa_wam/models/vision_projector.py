"""Vision projector: maps V-JEPA patch embeddings into LLM embedding space."""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """2-layer MLP + learnable view position embedding.

    Input:  [B, V, N, D_jepa]   (N = num_patches per view, e.g. 196)
    Output: [B, V*N, D_llm]
    """

    def __init__(self, d_jepa: int, d_llm: int, num_views_max: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_jepa, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.view_pe = nn.Parameter(torch.randn(num_views_max, d_llm) * 0.02)

    def forward(self, vision_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_emb: [B, V, N, D_jepa]
        Returns:
            [B, V*N, D_llm]
        """
        B, V, N, _ = vision_emb.shape
        x = self.proj(vision_emb)                 # [B, V, N, D_llm]
        x = x + self.view_pe[:V].view(1, V, 1, -1)  # broadcast add
        x = x.reshape(B, V * N, -1)               # [B, V*N, D_llm]
        return x

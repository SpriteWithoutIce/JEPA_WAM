"""Auxiliary Head: predicts future-frame V-JEPA embeddings from LLM hidden states."""

import torch
import torch.nn as nn


class AuxQueries(nn.Module):
    """Learnable query embeddings with view / time / spatial PE."""

    def __init__(
        self,
        num_views_max: int = 3,
        T: int = 4,
        H: int = 14,
        W: int = 14,
        d: int = 768,
    ) -> None:
        super().__init__()
        self.view_pe = nn.Parameter(torch.randn(num_views_max, d) * 0.02)
        self.time_pe = nn.Parameter(torch.randn(T, d) * 0.02)
        self.spatial_pe = nn.Parameter(torch.randn(H, W, d) * 0.02)
        self.T, self.H, self.W = T, H, W

    def forward(self, B: int, V: int) -> torch.Tensor:
        # Broadcasted sum of view + time + spatial PE
        v = self.view_pe[:V][:, None, None, None, :]      # [V, 1, 1, 1, d]
        t = self.time_pe[None, :, None, None, :]          # [1, T, 1, 1, d]
        s = self.spatial_pe[None, None, :, :, :]          # [1, 1, H, W, d]
        q = v + t + s                                     # [V, T, H, W, d]
        q = q.reshape(1, V * self.T * self.H * self.W, -1)
        q = q.expand(B, -1, -1).contiguous()              # [B, N_q, d]
        return q


class AuxDecoderBlock(nn.Module):
    """Pre-norm decoder block: cross-attn -> self-attn -> FFN."""

    def __init__(self, d: int, n_heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Linear(d * ffn_ratio, d),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Cross-attn: queries attend to memory (LLM hidden states)
        h = self.norm1(queries)
        h, _ = self.cross_attn(h, memory, memory)
        queries = queries + h

        # Self-attn among queries
        h = self.norm2(queries)
        h, _ = self.self_attn(h, h, h)
        queries = queries + h

        # FFN
        h = self.norm3(queries)
        queries = queries + self.ffn(h)
        return queries


class AuxHead(nn.Module):
    """Predicts future-frame V-JEPA embeddings.

    Output shape: [B, V, T, H, W, D_jepa]
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.queries = AuxQueries(
            num_views_max=cfg.vision.num_views_max,
            T=cfg.aux_head.aux_T,
            H=cfg.aux_head.aux_H,
            W=cfg.aux_head.aux_W,
            d=cfg.aux_head.d_aux,
        )
        self.query_proj = nn.Linear(cfg.aux_head.d_aux, cfg.aux_head.d_aux)
        self.memory_proj = nn.Linear(cfg.llm.d_llm, cfg.aux_head.d_aux)

        self.blocks = nn.ModuleList([
            AuxDecoderBlock(
                cfg.aux_head.d_aux,
                cfg.aux_head.n_heads,
                cfg.aux_head.ffn_ratio,
            )
            for _ in range(cfg.aux_head.num_layers)
        ])

        self.final_norm = nn.LayerNorm(cfg.aux_head.d_aux)
        self.output_proj = nn.Linear(cfg.aux_head.d_aux, cfg.vision.d_jepa)

        self.T, self.H, self.W = cfg.aux_head.aux_T, cfg.aux_head.aux_H, cfg.aux_head.aux_W

    def forward(self, llm_hidden: torch.Tensor, V: int) -> torch.Tensor:
        """
        Args:
            llm_hidden: [B, L, D_llm]
            V: number of views in current batch
        Returns:
            [B, V, T, H, W, D_jepa]
        """
        B = llm_hidden.shape[0]

        queries = self.queries(B, V)              # [B, N_q, d_aux]
        queries = self.query_proj(queries)
        memory = self.memory_proj(llm_hidden)     # [B, L, d_aux]

        for block in self.blocks:
            queries = block(queries, memory)

        queries = self.final_norm(queries)
        out = self.output_proj(queries)           # [B, N_q, D_jepa]
        out = out.reshape(B, V, self.T, self.H, self.W, -1)
        return out

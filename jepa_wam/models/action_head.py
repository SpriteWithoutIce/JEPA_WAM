"""GR00T-style Flow Matching Action Head."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """Encode proprioception into a single state token."""

    def __init__(self, d_proprio: int, d_a: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_proprio, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        # proprio: [B, D_proprio]
        x = self.encoder(proprio)  # [B, D_a]
        return x.unsqueeze(1)      # [B, 1, D_a]


class NoisyActionEmbed(nn.Module):
    """Embed noisy action + timestep conditioning into transformer space."""

    def __init__(self, d_action: int, d_a: int, horizon: int) -> None:
        super().__init__()
        self.action_proj = nn.Linear(d_action, d_a)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_a),
            nn.SiLU(),
            nn.Linear(d_a, d_a),
        )
        self.pos_embed = nn.Parameter(torch.randn(horizon, d_a) * 0.02)

    def forward(self, action_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # action_noisy: [B, H_a, D_action]
        # t: [B]
        x = self.action_proj(action_noisy)          # [B, H_a, D_a]
        t_emb = self.time_mlp(t.unsqueeze(-1))      # [B, D_a]
        x = x + t_emb.unsqueeze(1)                  # broadcast time
        x = x + self.pos_embed.unsqueeze(0)         # add step position
        return x                                    # [B, H_a, D_a]


class SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN."""

    def __init__(self, d_a: int, n_heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.attn = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention (x attends to cond) + FFN."""

    def __init__(self, d_a: int, d_llm: int, n_heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_a)
        self.cond_proj = nn.Linear(d_llm, d_a)
        self.attn = nn.MultiheadAttention(d_a, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_a)
        self.ffn = nn.Sequential(
            nn.Linear(d_a, d_a * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_a * ffn_ratio, d_a),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, D_llm]  (last hidden state from LLM)
        h = self.norm1(x)
        k_v = self.cond_proj(cond).unsqueeze(1)   # [B, 1, D_a]
        h, _ = self.attn(h, k_v, k_v)             # cross-attn
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ActionHeadBackbone(nn.Module):
    """16 layers alternating self-attn (even index) and cross-attn (odd index)."""

    def __init__(
        self,
        d_a: int = 1024,
        d_llm: int = 896,
        n_heads: int = 16,
        num_layers: int = 16,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(num_layers):
            if i % 2 == 0:
                blocks.append(SelfAttnBlock(d_a, n_heads, ffn_ratio))
            else:
                blocks.append(CrossAttnBlock(d_a, d_llm, n_heads, ffn_ratio))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class ActionOutputHead(nn.Module):
    """Project action tokens back to continuous action space."""

    def __init__(self, d_a: int, d_action: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_a)
        self.proj = nn.Linear(d_a, d_action)
        # Zero-init final layer for training stability
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1 + H_a, D_a]
        x = x[:, 1:, :]          # drop state token
        x = self.norm(x)
        return self.proj(x)      # [B, H_a, D_action]


class ActionHead(nn.Module):
    """Complete action head with flow matching training and sampling."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_enc = StateEncoder(cfg.proprio_dim, cfg.action_head.d_a)
        self.noisy_emb = NoisyActionEmbed(
            cfg.action_dim, cfg.action_head.d_a, cfg.action_horizon
        )
        self.backbone = ActionHeadBackbone(
            d_a=cfg.action_head.d_a,
            d_llm=cfg.llm.d_llm,
            n_heads=cfg.action_head.n_heads,
            num_layers=cfg.action_head.num_layers,
            ffn_ratio=cfg.action_head.ffn_ratio,
        )
        self.out_head = ActionOutputHead(cfg.action_head.d_a, cfg.action_dim)

    def forward(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward.

        Args:
            z_action: [B, D_llm]  (LLM last hidden state)
            proprio:  [B, D_proprio]
            action_gt: [B, H_a, D_action]
        Returns:
            loss, v_pred
        """
        B = proprio.shape[0]
        device = proprio.device

        # 1. Sample flow matching timestep ~ Beta(alpha, beta)
        dist = torch.distributions.Beta(
            self.cfg.action_head.beta_alpha, self.cfg.action_head.beta_beta
        )
        t = dist.sample((B,)).to(device)  # [B]

        # 2. Add noise
        noise = torch.randn_like(action_gt)  # [B, H_a, D_action]
        action_noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action_gt
        velocity_gt = action_gt - noise

        # 3. Tokenize
        state_tok = self.state_enc(proprio)           # [B, 1, D_a]
        act_tok = self.noisy_emb(action_noisy, t)     # [B, H_a, D_a]
        x = torch.cat([state_tok, act_tok], dim=1)    # [B, 1+H_a, D_a]

        # 4. Backbone
        x = self.backbone(x, cond=z_action)           # [B, 1+H_a, D_a]

        # 5. Output
        v_pred = self.out_head(x)                     # [B, H_a, D_action]

        # 6. Loss
        loss = F.mse_loss(v_pred, velocity_gt)
        return loss, v_pred

    @torch.no_grad()
    def sample_action(
        self,
        z_action: torch.Tensor,
        proprio: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """Euler integration sampling for flow matching.

        Args:
            z_action: [B, D_llm]
            proprio:  [B, D_proprio]
            num_steps: override default flow steps
        Returns:
            [B, H_a, D_action]
        """
        if num_steps is None:
            num_steps = self.cfg.action_head.flow_steps_inference

        B = proprio.shape[0]
        device = proprio.device
        H_a = self.cfg.action_horizon
        D_action = self.cfg.action_dim

        # Start from pure noise
        x = torch.randn(B, H_a, D_action, device=device)
        dt = 1.0 / num_steps

        state_tok = self.state_enc(proprio)  # [B, 1, D_a]  -- constant across steps

        for step in range(num_steps):
            t = torch.full((B,), step * dt, device=device)
            act_tok = self.noisy_emb(x, t)
            tokens = torch.cat([state_tok, act_tok], dim=1)
            tokens = self.backbone(tokens, cond=z_action)
            v_pred = self.out_head(tokens)
            x = x + v_pred * dt

        return x

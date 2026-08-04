"""JEPA-WAM main model: assembles V-JEPA encoder, projector, LLM, action head, and aux head."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wam.models.action_head import ActionHead
from jepa_wam.models.aux_head import AuxHead
from jepa_wam.models.llm_wrapper import LLMWrapper
from jepa_wam.models.vision_projector import VisionProjector

# Lazy import to avoid hard dependency if vjepa21 not installed on CPU-only machines
_VJEPA_ENCODER = None


def _get_vjepa_encoder_class():
    global _VJEPA_ENCODER
    if _VJEPA_ENCODER is None:
        from vjepa21.extractor import VJEPA21Encoder

        _VJEPA_ENCODER = VJEPA21Encoder
    return _VJEPA_ENCODER


class VJEPAVisionEncoder(nn.Module):
    """Wrapper around VJEPA21Encoder for current-frame and future-frame encoding.

    We instantiate *one* encoder with num_frames=8; V-JEPA 2 uses RoPE which can
    generalize across temporal lengths, so it also works for 2-frame inputs.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        VJEPA21Encoder = _get_vjepa_encoder_class()
        self.encoder = VJEPA21Encoder(
            checkpoint_path=cfg.vision.checkpoint_path,
            model_name=cfg.vision.model_name,
            img_size=cfg.vision.img_size,
            num_frames=8,
            device="cpu",  # will be moved by model.to(device)
        )
        # Freeze permanently
        for p in self.encoder.model.parameters():
            p.requires_grad = False
        self.d_jepa = self.encoder.embed_dim

    def _ensure_device(self, target_device: torch.device) -> None:
        model_device = next(self.encoder.model.parameters()).device
        if model_device != target_device:
            self.encoder.model = self.encoder.model.to(target_device)

    @torch.no_grad()
    def encode_current(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, V, 3, H, W]
        Returns:
            [B, V, 196, D_jepa]
        """
        self._ensure_device(images.device)
        B, V, C, H, W = images.shape
        # Duplicate frame to satisfy T>=2 requirement
        x = images.unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)  # [B, V, 2, C, H, W]
        x = x.reshape(B * V, 2, C, H, W)
        emb = self.encoder.extract(x, return_type="all_tokens")  # [B*V, 196, D]
        return emb.reshape(B, V, 196, -1)

    @torch.no_grad()
    def encode_future(self, future_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            future_frames: [B, V, 8, 3, H, W]
        Returns:
            [B, V, 4, 14, 14, D_jepa]
        """
        self._ensure_device(future_frames.device)
        B, V, T, C, H, W = future_frames.shape
        x = future_frames.reshape(B * V, T, C, H, W)
        emb = self.encoder.extract(x, return_type="all_tokens")  # [B*V, 784, D]
        # 784 = 4 * 14 * 14
        emb = emb.reshape(B, V, 4, 14, 14, -1)
        return emb.detach()


class JepaVLA(nn.Module):
    """End-to-end JEPA-WAM model."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Vision encoder (frozen)
        self.vjepa = VJEPAVisionEncoder(cfg)
        assert self.vjepa.d_jepa == cfg.vision.d_jepa

        # 2. Vision projector (trainable)
        self.vision_proj = VisionProjector(
            d_jepa=cfg.vision.d_jepa,
            d_llm=cfg.llm.d_llm,
            num_views_max=cfg.vision.num_views_max,
        )

        # 3. LLM + LoRA
        self.llm = LLMWrapper(cfg)

        # 4. Action head (flow matching)
        self.action_head = ActionHead(cfg)

        # 5. Aux head (future frame prediction)
        self.aux_head = AuxHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward.

        Batch keys:
            images:         [B, V, 3, H, W]
            text_tokens:    [B, L_text]
            attention_mask: [B, L_text]   (optional, for padding)
            proprio:        [B, D_proprio]
            action:         [B, H_a, D_action]
            future_frames:  [B, V, 8, 3, H, W]   (optional, for aux loss)
        """
        images = batch["images"]
        text_tokens = batch["text_tokens"]
        proprio = batch["proprio"]
        action_gt = batch["action"]
        future_frames = batch.get("future_frames")
        attention_mask = batch.get("attention_mask")
        V = images.shape[1]

        # ---- Vision ----
        vision_emb = self.vjepa.encode_current(images)     # [B, V, 196, D_jepa]
        vision_tok = self.vision_proj(vision_emb)          # [B, V*196, D_llm]

        # ---- Language ----
        text_emb = self.llm.get_input_embeddings()(text_tokens)  # [B, L_text, D_llm]

        # ---- Concatenate: vision first, then text ----
        llm_input = torch.cat([vision_tok, text_emb], dim=1)     # [B, V*196 + L_text, D_llm]

        # Build attention mask for full sequence if needed
        if attention_mask is not None:
            vision_mask = torch.ones(
                vision_tok.shape[:2],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # ---- LLM ----
        llm_out = self.llm(
            inputs_embeds=llm_input,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        llm_hidden = llm_out.hidden_states[-1]             # [B, L, D_llm]

        # ---- Action head ----
        z_action = llm_hidden[:, -1, :]                    # [B, D_llm]  (last token)
        loss_action, _ = self.action_head(z_action, proprio, action_gt)

        losses = {"action": loss_action}

        # ---- Aux head (training only) ----
        if self.training and future_frames is not None:
            aux_pred = self.aux_head(llm_hidden, V=V)      # [B, V, 4, 14, 14, D_jepa]
            aux_target = self.vjepa.encode_future(future_frames)
            loss_aux = self.compute_aux_loss(aux_pred, aux_target)
            losses["aux"] = loss_aux

        return losses

    @torch.no_grad()
    def predict_action(
        self,
        batch: Dict[str, torch.Tensor],
        num_flow_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference-only action prediction.

        Returns:
            [B, H_a, D_action]
        """
        self.eval()
        images = batch["images"]
        text_tokens = batch["text_tokens"]
        proprio = batch["proprio"]
        attention_mask = batch.get("attention_mask")

        V = images.shape[1]

        # Vision
        vision_emb = self.vjepa.encode_current(images)
        vision_tok = self.vision_proj(vision_emb)

        # Language
        text_emb = self.llm.get_input_embeddings()(text_tokens)
        llm_input = torch.cat([vision_tok, text_emb], dim=1)

        if attention_mask is not None:
            vision_mask = torch.ones(
                vision_tok.shape[:2],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # LLM
        llm_out = self.llm(
            inputs_embeds=llm_input,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        llm_hidden = llm_out.hidden_states[-1]
        z_action = llm_hidden[:, -1, :]

        # Sample action via flow matching
        action = self.action_head.sample_action(
            z_action, proprio, num_steps=num_flow_steps
        )
        return action

    @staticmethod
    def compute_aux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Normalized per-patch MSE (no learnable parameters in layer_norm)."""
        pred_n = F.layer_norm(pred, pred.shape[-1:])
        target_n = F.layer_norm(target, target.shape[-1:])
        return F.mse_loss(pred_n, target_n)

    def get_lambda_aux(self, step: int, total_steps: int) -> float:
        """Cosine schedule for aux loss weight.

        Warm-up: first `warmup_ratio` steps -> 1.0
        Then cosine decay to `lambda_aux_final`.
        """
        cfg_loss = self.cfg.loss
        warmup_steps = int(total_steps * cfg_loss.warmup_ratio)
        if step < warmup_steps:
            return cfg_loss.lambda_aux_init
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg_loss.lambda_aux_final + (
            cfg_loss.lambda_aux_init - cfg_loss.lambda_aux_final
        ) * cosine

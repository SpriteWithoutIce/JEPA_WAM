from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F
from transformers import PretrainedConfig

from prismatic.models.flow_matching_head.action_encoder import ActionEncoder
from prismatic.models.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


@dataclass
class FlowMatchingActionJEPAHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=768)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    jepa_dim: int = field(default=None)
    jepa_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=4)
    num_target_vision_tokens: int = field(default=32)
    jepa_loss_weight: float = field(default=1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DIT_CONFIGS = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowMatchingActionJEPAHead(nn.Module):
    """
    Joint flow-matching head over actions and pooled future JEPA latents.

    Conditioning comes from:
      - current primary-view JEPA embedding
      - proprio state
      - learned future tokens
      - action placeholder hidden states from the VLM/LLM

    Predicted variables are:
      - action velocity
      - future JEPA delta velocity, where delta = future_jepa - current_jepa
    """

    def __init__(
        self,
        d_proprio: int,
        d_action: int,
        d_llm: int,
        d_jepa: int,
        horizon: int,
        jepa_horizon: int,
        fm_hidden_size: int = 1024,
        fm_action_model_type: str = "DiT-B",
        fm_num_layers: int = 16,
        fm_num_inference_timesteps: int = 4,
        fm_num_timestep_buckets: int = 1000,
        fm_noise_beta_alpha: float = 1.5,
        fm_noise_beta_beta: float = 1.0,
        fm_noise_s: float = 0.999,
        fm_num_target_vision_tokens: int = 32,
        fm_add_pos_embed: bool = True,
        fm_max_seq_len: int = 1024,
        fm_state_dropout: float = 0.5,
        fm_jepa_loss_weight: float = 1.0,
    ):
        super().__init__()

        action_model_cfg = DIT_CONFIGS[fm_action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = {
            **action_model_cfg,
            "cross_attention_dim": d_llm,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": fm_num_layers,
            "output_dim": fm_hidden_size,
            "positional_embeddings": None,
        }

        config = FlowMatchingActionJEPAHeadConfig(
            add_pos_embed=fm_add_pos_embed,
            diffusion_model_cfg=diffusion_model_cfg,
            input_embedding_dim=self.input_embedding_dim,
            hidden_size=fm_hidden_size,
            max_seq_len=fm_max_seq_len,
            action_dim=d_action,
            action_horizon=horizon,
            jepa_dim=d_jepa,
            jepa_horizon=jepa_horizon,
            noise_beta_alpha=fm_noise_beta_alpha,
            noise_beta_beta=fm_noise_beta_beta,
            noise_s=fm_noise_s,
            num_timestep_buckets=fm_num_timestep_buckets,
            num_inference_timesteps=fm_num_inference_timesteps,
            num_target_vision_tokens=fm_num_target_vision_tokens,
            jepa_loss_weight=fm_jepa_loss_weight,
        )
        self.config = config
        self.full_config = SimpleNamespace(framework=SimpleNamespace(action_model=config))

        self.hidden_size = config.hidden_size
        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.jepa_dim = config.jepa_dim
        self.jepa_horizon = config.jepa_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.jepa_loss_weight = config.jepa_loss_weight

        self.current_jepa_encoder = MLP(
            input_dim=d_jepa,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.state_encoder = MLP(
            input_dim=d_proprio,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.state_dropout = nn.Dropout(p=fm_state_dropout)
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.jepa_encoder = MLP(
            input_dim=d_jepa,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.jepa_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.jepa_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _prepare_state(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.dim() == 3:
            proprio = proprio[:, 0, :]
        return proprio

    def _pool_current_primary_jepa(self, current_vjepa: torch.Tensor, num_views: int) -> torch.Tensor:
        if current_vjepa.dim() != 3:
            raise ValueError(f"Expected current_vjepa shape [B, N, D], got {tuple(current_vjepa.shape)}")

        total_tokens = current_vjepa.shape[1]
        tokens_per_view = total_tokens // max(1, num_views)
        if tokens_per_view * max(1, num_views) != total_tokens:
            raise ValueError(
                f"Current JEPA token count {total_tokens} is not divisible by num_views={num_views}."
            )

        primary_tokens = current_vjepa[:, :tokens_per_view, :]
        return F.normalize(primary_tokens.mean(dim=1), dim=-1)

    def _pool_future_primary_jepa(self, future_jepa_target: torch.Tensor) -> torch.Tensor:
        if future_jepa_target.dim() != 6:
            raise ValueError(
                "Expected future_jepa_target shape [B, V, T, H, W, D], "
                f"got {tuple(future_jepa_target.shape)}"
            )

        primary_target = future_jepa_target[:, 0, ...]
        return F.normalize(primary_target.mean(dim=(2, 3)), dim=-1)

    def _future_jepa_delta(
        self,
        current_jepa_repr: torch.Tensor,
        future_jepa_repr: torch.Tensor,
    ) -> torch.Tensor:
        return future_jepa_repr - current_jepa_repr.unsqueeze(1)

    def _predict_joint_velocity(
        self,
        vl_embs: torch.Tensor,
        noisy_actions: torch.Tensor,
        noisy_jepa_delta: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        state: torch.Tensor,
        current_jepa_repr: torch.Tensor,
    ):
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        compute_dtype = vl_embs.dtype
        noisy_actions = noisy_actions.to(dtype=compute_dtype)
        noisy_jepa_delta = noisy_jepa_delta.to(dtype=compute_dtype)
        current_jepa_repr = current_jepa_repr.to(dtype=compute_dtype)
        if state is not None:
            state = state.to(dtype=compute_dtype)

        action_features = self.action_encoder(noisy_actions, timesteps_tensor)
        future_jepa_features = self.jepa_encoder(noisy_jepa_delta)
        current_jepa_feature = self.current_jepa_encoder(current_jepa_repr).unsqueeze(1)
        state_features = self.state_encoder(state).unsqueeze(1) if state is not None else None
        if state_features is not None:
            state_features = self.state_dropout(state_features)

        future_token_count = noisy_jepa_delta.shape[1]
        if future_token_count > self.future_tokens.num_embeddings:
            raise ValueError(
                f"Need {future_token_count} future tokens, but head was initialized with only "
                f"{self.future_tokens.num_embeddings}."
            )
        future_tokens = self.future_tokens.weight[:future_token_count].unsqueeze(0).expand(batch_size, -1, -1)

        token_segments = [current_jepa_feature]
        if state_features is not None:
            token_segments.append(state_features)
        token_segments.extend([future_tokens, action_features, future_jepa_features])
        joint_embs = torch.cat(token_segments, dim=1)

        if self.config.add_pos_embed:
            seq_len = joint_embs.shape[1]
            if seq_len > self.position_embedding.num_embeddings:
                raise ValueError(
                    f"Joint token length {seq_len} exceeds max positional embeddings "
                    f"{self.position_embedding.num_embeddings}."
                )
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            joint_embs = joint_embs + pos_embs

        model_output = self.model(
            hidden_states=joint_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
            return_all_hidden_states=False,
        )

        action_start = joint_embs.shape[1] - noisy_actions.shape[1] - noisy_jepa_delta.shape[1]
        action_end = action_start + noisy_actions.shape[1]
        jepa_start = action_end

        pred_actions = self.action_decoder(model_output[:, action_start:action_end, :])
        pred_jepa_delta = self.jepa_decoder(model_output[:, jepa_start:, :])
        return pred_actions, pred_jepa_delta

    def forward(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
        current_vjepa: torch.Tensor,
        future_jepa_target: torch.Tensor,
        num_views: int = 1,
        action_valid_mask: torch.Tensor = None,
        **_: object,
    ):
        if current_vjepa is None or future_jepa_target is None:
            raise ValueError("flow_gr00t_jepa requires both current_vjepa and future_jepa_target during training.")

        compute_dtype = vl_embs.dtype
        state = self._prepare_state(proprio).to(dtype=compute_dtype)
        action_gt = action_gt.to(dtype=compute_dtype)
        current_jepa_repr = self._pool_current_primary_jepa(
            current_vjepa,
            num_views=max(1, int(num_views)),
        ).to(dtype=compute_dtype)
        future_jepa_repr = self._pool_future_primary_jepa(future_jepa_target).to(dtype=compute_dtype)
        future_jepa_delta = self._future_jepa_delta(current_jepa_repr, future_jepa_repr)

        action_noise = torch.randn(action_gt.shape, device=action_gt.device, dtype=action_gt.dtype)
        jepa_noise = torch.randn(
            future_jepa_delta.shape,
            device=future_jepa_delta.device,
            dtype=future_jepa_delta.dtype,
        )

        t = self.sample_time(action_gt.shape[0], device=action_gt.device, dtype=action_gt.dtype)
        t = t[:, None, None]

        noisy_actions = (1 - t) * action_noise + t * action_gt
        noisy_jepa_delta = (1 - t) * jepa_noise + t * future_jepa_delta
        action_velocity = action_gt - action_noise
        jepa_velocity = future_jepa_delta - jepa_noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        pred_actions, pred_jepa_delta = self._predict_joint_velocity(
            vl_embs,
            noisy_actions,
            noisy_jepa_delta,
            t_discretized,
            state,
            current_jepa_repr,
        )

        squared_action_error = (pred_actions - action_velocity) ** 2
        if action_valid_mask is None:
            loss_action = squared_action_error.mean()
        else:
            if action_valid_mask.shape != squared_action_error.shape[:2]:
                raise ValueError(
                    f"action_valid_mask shape {tuple(action_valid_mask.shape)} does not match "
                    f"action trajectory shape {tuple(squared_action_error.shape[:2])}."
                )
            mask = action_valid_mask.to(
                device=squared_action_error.device,
                dtype=squared_action_error.dtype,
            ).unsqueeze(-1)
            loss_action = (squared_action_error * mask).sum() / (
                mask.sum() * squared_action_error.shape[-1]
            ).clamp_min(1.0)
        loss_jepa = ((pred_jepa_delta - jepa_velocity) ** 2).mean()
        loss = loss_action + self.jepa_loss_weight * loss_jepa
        return loss, pred_actions, pred_jepa_delta, loss_action, loss_jepa

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        current_vjepa: torch.Tensor,
        num_views: int = 1,
    ) -> torch.Tensor:
        state = self._prepare_state(proprio).to(dtype=vl_embs.dtype)
        current_jepa_repr = self._pool_current_primary_jepa(
            current_vjepa,
            num_views=max(1, int(num_views)),
        ).to(dtype=vl_embs.dtype)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        future_jepa_delta = torch.randn(
            size=(batch_size, self.jepa_horizon, self.jepa_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long)
            pred_action_velocity, pred_jepa_velocity = self._predict_joint_velocity(
                vl_embs,
                actions,
                future_jepa_delta,
                timesteps_tensor,
                state,
                current_jepa_repr,
            )
            actions = actions + dt * pred_action_velocity
            future_jepa_delta = future_jepa_delta + dt * pred_jepa_velocity

        return actions

    @torch.no_grad()
    def sample_action(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        current_vjepa: torch.Tensor,
        num_views: int = 1,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if num_steps is not None:
            old_steps = self.num_inference_timesteps
            self.num_inference_timesteps = num_steps
            try:
                return self.predict_action(vl_embs, proprio, current_vjepa, num_views=num_views)
            finally:
                self.num_inference_timesteps = old_steps
        return self.predict_action(vl_embs, proprio, current_vjepa, num_views=num_views)

"""Configuration dataclasses for JEPA-WAM."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class VisionConfig:
    """V-JEPA encoder & projector config."""
    checkpoint_path: str = "pretrained_models/vjepa2_1_vitl.pt"
    model_name: str = "vit_large"          # vit_base, vit_large, vit_giant_xformers, vit_gigantic_xformers
    img_size: int = 224
    d_jepa: int = 1024                     # encoder output dim
    num_views: int = 2
    num_views_max: int = 3                 # for position embedding reservation


@dataclass
class LLMConfig:
    """Qwen2.5 + LoRA config."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    d_llm: int = 896
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.0


@dataclass
class ActionHeadConfig:
    """GR00T-style Flow Matching Action Head."""
    d_a: int = 1024                        # internal action head dim
    n_heads: int = 16
    num_layers: int = 16                   # 8 self-attn + 8 cross-attn
    ffn_ratio: int = 4
    flow_steps_inference: int = 10
    beta_alpha: float = 1.5
    beta_beta: float = 1.0


@dataclass
class AuxHeadConfig:
    """Future-frame JEPA embedding prediction head."""
    d_aux: int = 1024
    n_heads: int = 12
    num_layers: int = 12
    ffn_ratio: int = 4
    aux_T: int = 4                         # temporal chunks after tubelet=2
    aux_H: int = 14
    aux_W: int = 14


@dataclass
class LossConfig:
    """Loss scheduling."""
    lambda_aux_init: float = 1.0
    lambda_aux_final: float = 0.2
    warmup_ratio: float = 0.1


@dataclass
class DataConfig:
    """Dataset & dataloader config."""
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    data_mix: str = "libero_spatial_no_noops"
    shuffle_buffer_size: int = 256_000
    image_aug: bool = False
    resize_resolution: Tuple[int, int] = (224, 224)
    window_size: int = 1                   # observation window for current frame
    future_action_window_size: int = 7     # = action_horizon - 1
    future_obs_window_size: int = 8        # how many future frames to extract for aux target
    num_workers: int = 0                   # RLDS handles parallelism internally


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # robot / task specifics
    proprio_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 8

    # optimization
    epochs: int = 100
    max_steps: Optional[int] = None
    global_batch_size: int = 64
    per_device_batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # training infra
    seed: int = 7
    save_interval: int = 2500
    run_root_dir: Path = Path("runs")
    run_id: Optional[str] = None
    trackers: Tuple[str, ...] = ("jsonl",)
    enable_mixed_precision_training: bool = True

    # device
    device: str = "cuda"

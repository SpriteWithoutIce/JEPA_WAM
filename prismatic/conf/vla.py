"""Fixed public JEPA-WAM training configuration."""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple, Union

from draccus import ChoiceRegistry

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


@dataclass
class VLAConfig(ChoiceRegistry):
    """The released V-JEPA 2.1 + Qwen2.5 + Flow-GR00T recipe."""

    vla_id: str = "jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-qwen25-vjepa21-vitl-384px+0_5b"

    data_mix: str = "libero_4_task_suites_no_noops"
    shuffle_buffer_size: int = 20_000

    max_steps: int = 40_000
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03

    vjepa_checkpoint_path: Optional[str] = None

    d_action: int = 7
    d_proprio: int = 8
    action_horizon: int = NUM_ACTIONS_CHUNK
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    fm_hidden_size: int = 1024
    fm_num_layers: int = 16
    fm_num_inference_timesteps: int = 4
    fm_num_timestep_buckets: int = 1_000
    fm_noise_beta_alpha: float = 1.5
    fm_noise_beta_beta: float = 1.0
    fm_noise_s: float = 0.999
    fm_num_target_vision_tokens: int = 32
    fm_add_pos_embed: bool = True
    fm_max_seq_len: int = 1_024
    fm_state_dropout: float = 0.5

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: Union[str, Tuple[str, ...]] = "all-linear"
    visual_token_pair_offset: int = 31
    lambda_visual_token_cosine: float = 0.5

    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True


Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90 = VLAConfig


@unique
class VLARegistry(Enum):
    JEPAVLA_QWEN25_VJEPA_224PX_0_5B_LIBERO_90 = VLAConfig

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


VLAConfig.register_subclass(VLAConfig.vla_id, VLAConfig)

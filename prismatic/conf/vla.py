"""
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple, Union

from draccus import ChoiceRegistry
from prismatic.vla.constants import NUM_TOKENS


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)
    save_every_n_steps: Optional[int]

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    min_learning_rate: float                       # Floor LR for cosine decay schedules
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")
    action_tokenizer: str

    image_sequence_len: int
    use_wrist_image: bool

    # Projector Fine-Tuning
    freeze_projector: bool = True                  # Preserve existing behavior unless explicitly overridden

    # V-JEPA Backbone Configuration
    vjepa_checkpoint_path: Optional[str] = None      # Path to V-JEPA 2.1 .pt checkpoint
    dino_local_path: Optional[str] = None            # Optional local DINOv2 checkpoint path/directory
    siglip_local_path: Optional[str] = None          # Optional local SigLIP checkpoint path/directory
    future_obs_window_size: int = 0                  # Number of future frames to extract for aux target
    future_obs_downsample_stride: int = 1            # Uniform stride for downsampling future supervision frames
    context: bool = False                             # Enable one randomly sampled in-episode state/action context chunk
    context_action_tokens: int = 4                   # Number of compressed action-context tokens
    strict_epoch_mode: bool = False                  # Disable infinite sampling; one epoch iterates each sample once
    shared_dataset_statistics: bool = False          # Normalize every dataset in a mixture with one shared statistic
    rank_shard_dataset_sources: bool = False        # Partition RLDS dataset sources across distributed ranks
    visual_token_pair_offset: int = 0                # Optional paired-frame offset (e.g. 31) with tail padding
    visual_token_cosine_target_future_only: bool = False  # Encode only t+offset instead of the [t, t+offset] pair
    stitch_primary_and_wrist_images: bool = False    # If True, stitch primary+wrist views horizontally into one image
    rotation_representation: str = "axis_angle"      # "axis_angle" or "rot6d"
    dataset_format: str = "rlds"                     # "rlds" or "lerobot"
    lerobot_primary_image_key: Optional[str] = None
    lerobot_wrist_image_keys: Tuple[str, ...] = ()
    lerobot_state_key: str = "observation.state"
    lerobot_action_key: str = "action"
    lerobot_use_quantile_normalization: bool = True
    lerobot_normalization_clip_value: float = 15.0
    lerobot_num_workers: int = 8
    lerobot_prefetch_factor: int = 3
    d_action: Optional[int] = None                   # Resolved action dim for continuous heads
    d_proprio: Optional[int] = None                  # Resolved proprio dim for action heads / projectors

    # LoRA / Adapter Fine-Tuning
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: Union[str, Tuple[str, ...]] = "all-linear"
    lora_unfreeze_last_n_llm_layers: int = 0
    use_vlm_peft: bool = False
    use_action_queries: bool = False

    # Action Head
    use_action_head: bool = True
    action_head_type: str = "flow_gr00t"            # "flow_gr00t", "flow_gr00t_jepa", or "l1"
    d_a: int = 1024                                  # Internal dim of action head
    n_heads_action: int = 16
    num_layers_action: int = 16
    ffn_ratio_action: int = 4
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    l1_use_pro_version: bool = False
    l1_num_blocks: int = 24
    fm_hidden_size: int = 1024
    fm_action_model_type: str = "DiT-L"
    fm_num_layers: int = 16
    fm_num_inference_timesteps: int = 4
    fm_num_timestep_buckets: int = 1000
    fm_noise_beta_alpha: float = 1.5
    fm_noise_beta_beta: float = 1.0
    fm_noise_s: float = 0.999
    fm_num_target_vision_tokens: int = 32
    fm_add_pos_embed: bool = True
    fm_max_seq_len: int = 1024
    fm_state_dropout: float = 0.5
    fm_jepa_loss_weight: float = 1.0
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    flow_gr00t_use_full_llm_hidden: bool = False
    use_llm_ce_loss: bool = False
    lambda_llm_ce: float = 0.1

    # Aux Head (Cross-Attention Decoder)
    use_aux_head: bool = True
    d_aux: int = 768
    n_heads_aux: int = 12
    num_layers_aux: int = 12
    ffn_ratio_aux: int = 4
    lambda_aux: float = 0.5
    use_visual_token_cosine_head: bool = False       # Supervise LLM visual tokens with paired-frame JEPA targets
    lambda_visual_token_cosine: float = 0.5
    visual_token_cosine_use_projector_target: bool = True
    visual_token_cosine_layer_idx: int = -1          # LLM hidden_states index for cosine supervision (-1 = final layer)
    visual_token_cosine_projection_type: str = "mlp" # "mlp" keeps old projection; "conv" uses iREPA-style conv + spatial norm

    # Optional optimizer overrides for module groups
    vlm_learning_rate: Optional[float] = None
    vlm_min_learning_rate: Optional[float] = None
    action_head_learning_rate: Optional[float] = None
    action_head_min_learning_rate: Optional[float] = None
    action_expert_warmup_steps: int = 0

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === OpenVLA Training Configurations ===


# = [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    image_sequence_len: int = 1
    use_wrist_image: bool = False

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None
    save_every_n_steps: Optional[int] = 25000

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"
    action_tokenizer: str = "action_tokenizer"


# = [8 GPU] SigLIP 224px Frozen Vision Backbone + Bridge =
@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True


# = [8 GPU] Fast Iteration =>> DINO-SigLIP 224px + Bridge =
@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    data_mix: str = "bridge"


# = [64 GPU] SigLIP 224px + OXE Magic Soup =
@dataclass
class Exp_SigLIP_224px_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "oxe_magic_soup"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# = [8 GPU] Qwen2.5 0.5B SigLIP 224px + OXE Magic Soup =
@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "prism-qwen25-extra-dinosiglip-224px+0_5b"

    data_mix: str = "oxe_magic_soup"
    action_tokenizer: str = "extra_action_tokenizer"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90"

    data_mix: str = "libero_90"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = "prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90"
    image_sequence_len: int = 2


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90"
    image_sequence_len: int = 2
    use_wrist_image: bool = True


## bridge Qwen


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-bridge"
    base_vlm: Union[str, Path] = "prism-qwen25-extra-dinosiglip-224px+0_5b"

    data_mix: str = "bridge_dataset"  # direct dataset
    action_tokenizer: str = "extra_action_tokenizer"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90(Exp_SigLIP_224px_Bridge):
    vla_id: str = "jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-qwen25-vjepa-224px+0_5b"

    image_sequence_len: int = 1
    use_wrist_image: bool = False
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    data_mix: str = "libero_90"
    action_tokenizer: str = "extra_action_tokenizer"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    # JEPA-specific overrides
    use_aux_head: bool = False
    future_obs_window_size: int = 0
    vjepa_checkpoint_path: Optional[str] = None


@dataclass
class Exp_JEPAVLA_Qwen3_VJEPA_1_7B_LIBERO_90(Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90):
    vla_id: str = "jepavla-qwen3-vjepa-224px+1_7b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-qwen3-vjepa21-vitl-384px+1_7b"


@dataclass
class Exp_JEPAVLA_Qwen25_JEPASigLIP_0_5B_LIBERO_90(Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90):
    vla_id: str = "jepavla-qwen25-jepasiglip-384px+0_5b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-jepasiglip+0_5b"


@dataclass
class Exp_DinoSigLIP_224px_LIBERO_90(Exp_DinoSigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-libero-90"

    data_mix: str = "libero_90"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


# = [64 GPU] DINO-SigLIP 224px + OXE Magic Soup++ =
@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-oxe-magic-soup-plus"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # Note =>> We adopt two stages, training on a mixture including DROID for 70% of training, before resampling!
    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# === OpenVLA Fine-tuning Configurations ===


# = [8 GPU] SigLIP 224px + T-DROID =
@dataclass
class Exp_SigLIP_224px_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_TDROID_PourCornInPot(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_pour_corn_in_pot"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_pour_corn_in_pot"


# = [8 GPU] SigLIP 224px + T-DROID -- Partial Finetuning =
@dataclass
class Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-last_layer+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-sandwich+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


# === [8 GPU] SigLIP 224px + FrankaWipe ===
@dataclass
class Exp_SigLIP_224px_Droid_Wipe(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-droid_wipe"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid_wipe"


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # Sanity Check Configurations =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_LIBERO_90 = Exp_DinoSigLIP_224px_LIBERO_90

    # SigLIP Frozen Backbone Experiment
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge

    # [OpenVLA v0.1 7B] SigLIP 224px + OXE Magic Soup
    SIGLIP_224PX_MX_OXE_MAGIC_SOUP = Exp_SigLIP_224px_OXE_Magic_Soup

    # [OpenVLA 7B] DINO + SigLIP 224px + OXE Magic Soup++
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus

    # [OpenVLA 0.5B] Qwen backbones
    QWEN25_DINOSIGLIP_224PX_0_5B_MX_OXE_MAGIC_SOUP = Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup
    QWEN25_DINOSIGLIP_224PX_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_T2_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_WRIST_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90

    QWEN25_DINOSIGLIP_224PX_0_5B_BRIDGE = Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge

    # === JEPA-VLA Configs ===
    JEPAVLA_QWEN25_VJEPA_224PX_0_5B_LIBERO_90 = Exp_JEPAVLA_Qwen25_VJEPA_0_5B_LIBERO_90
    JEPAVLA_QWEN3_VJEPA_224PX_1_7B_LIBERO_90 = Exp_JEPAVLA_Qwen3_VJEPA_1_7B_LIBERO_90
    JEPAVLA_QWEN25_JEPASIGLIP_384PX_0_5B_LIBERO_90 = Exp_JEPAVLA_Qwen25_JEPASigLIP_0_5B_LIBERO_90

    # === TDROID Fine-tuning Configs ===
    SIGLIP_224PX_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_TDROID_CarrotInBowl
    SIGLIP_224PX_MX_TDROID_POUR_CORN_IN_POT = Exp_SigLIP_224px_TDROID_PourCornInPot

    SIGLIP_224PX_ICY_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl
    SIGLIP_224PX_LASTLAYER_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl
    SIGLIP_224PX_SANDWICH_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl

    # === DROID Fine-tuning Configs ===
    SIGLIP_224PX_MX_DROID_WIPE = Exp_SigLIP_224px_Droid_Wipe

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)

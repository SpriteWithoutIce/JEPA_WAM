"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

# Force local repo imports ahead of any external `prismatic` checkout on PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.util.rotation_utils import resolve_vla_action_proprio_dims
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
from prismatic.vla.future_utils import compute_downsampled_future_frame_count
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from peft import LoraConfig, get_peft_model
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _normalize_lora_target_modules(target_modules):
    if isinstance(target_modules, tuple):
        return list(target_modules)
    if isinstance(target_modules, list):
        return target_modules
    if isinstance(target_modules, str):
        normalized = target_modules.strip()
        if normalized == "all-linear":
            return normalized
        if "," in normalized:
            return [module.strip() for module in normalized.split(",") if module.strip()]
        return normalized
    return target_modules


def apply_lora_to_vlm(vlm, vla_cfg: VLAConfig) -> None:
    if hasattr(vlm.llm_backbone.llm, "peft_config"):
        overwatch.info("LLM already wrapped with LoRA; skipping re-wrap.")
        return

    lora_config = LoraConfig(
        r=vla_cfg.lora_rank,
        lora_alpha=vla_cfg.lora_alpha,
        target_modules=_normalize_lora_target_modules(vla_cfg.lora_target_modules),
        lora_dropout=vla_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )
    vlm.llm_backbone.llm = get_peft_model(vlm.llm_backbone.llm, lora_config)
    vlm.llm_backbone.llm.print_trainable_parameters()


def _resolve_peft_target_modules(model: torch.nn.Module, target_modules, exclude_modules):
    if target_modules != "all-linear":
        return target_modules

    excluded_prefixes = tuple(exclude_modules)
    resolved = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if module_name.startswith(excluded_prefixes):
            continue
        if isinstance(module, torch.nn.Linear):
            resolved.append(module_name)

    if not resolved:
        raise ValueError("Could not resolve any linear modules for PEFT target_modules.")

    return sorted(set(resolved))


def apply_peft_to_full_vlm(vlm, vla_cfg: VLAConfig):
    if hasattr(vlm, "peft_config"):
        overwatch.info("VLM already wrapped with PEFT; skipping re-wrap.")
        return vlm

    target_modules = _normalize_lora_target_modules(vla_cfg.lora_target_modules)
    modules_to_save = []
    if getattr(vlm, "action_queries", None) is not None:
        # Keep action queries checkpointed with the PEFT adapter, but do not wrap action heads.
        modules_to_save.append("action_queries")

    excluded_modules = ["action_head", "aux_head", "visual_token_cosine_head"]
    if vla_cfg.freeze_vision_backbone:
        excluded_modules.append("vision_backbone")
    lora_kwargs = dict(
        r=vla_cfg.lora_rank,
        lora_alpha=vla_cfg.lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        lora_dropout=vla_cfg.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        init_lora_weights="gaussian",
    )
    if "exclude_modules" in inspect.signature(LoraConfig).parameters:
        lora_kwargs["exclude_modules"] = excluded_modules
    else:
        overwatch.info("Installed PEFT does not support `exclude_modules`; resolving filtered target_modules explicitly.")
        lora_kwargs["target_modules"] = _resolve_peft_target_modules(vlm, target_modules, excluded_modules)

    peft_vlm = get_peft_model(
        vlm,
        LoraConfig(**lora_kwargs),
    )
    peft_vlm.print_trainable_parameters()
    return peft_vlm

def build_vla_from_base_vlm(
    base_vlm_id_or_path: Union[str, Path],
    cfg: "TrainConfig",
    hf_token: Optional[str],
):
    """
    Load a base Prismatic VLM checkpoint, then rebuild it with JEPA-VLA heads attached.

    This is the right path when we start from a pretrained base VLM run directory
    and want to train a fresh VLA head stack (e.g. L1 regression head on Libero).
    """
    from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm

    base_vlm_path = Path(base_vlm_id_or_path)
    if base_vlm_path.is_dir():
        run_dir = base_vlm_path
        checkpoint_dir = run_dir / "checkpoints"
        latest_checkpoint = checkpoint_dir / "latest-checkpoint.pt"
        if latest_checkpoint.exists():
            checkpoint_path = latest_checkpoint
        else:
            checkpoint_candidates = sorted(checkpoint_dir.glob("step-*.pt"))
            if not checkpoint_candidates:
                raise ValueError(f"Could not find a base VLM checkpoint under `{checkpoint_dir}`")
            checkpoint_path = checkpoint_candidates[-1]
    elif base_vlm_path.is_file():
        checkpoint_path = base_vlm_path
        run_dir = checkpoint_path.parent.parent
    else:
        raise ValueError(
            "JEPA-VLA training expects `vla.base_vlm` to point to either a base VLM run directory "
            "with `config.json` and `checkpoints/latest-checkpoint.pt`, or directly to a checkpoint `.pt` file."
        )

    with open(run_dir / "config.json", "r") as f:
        model_cfg = json.load(f)["model"]

    model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if "llm_backbone" not in model_state_dict or "projector" not in model_state_dict:
        raise ValueError(
            f"Base VLM checkpoint `{checkpoint_path}` must contain `llm_backbone` and `projector` weights."
        )

    vision_checkpoint_path = cfg.vla.vjepa_checkpoint_path or model_cfg.get("vision_checkpoint_path")
    dino_local_path = cfg.vla.dino_local_path or model_cfg.get("dino_local_path")
    siglip_local_path = cfg.vla.siglip_local_path or model_cfg.get("siglip_local_path")
    llm_checkpoint_path = str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else model_cfg.get("llm_local_path")

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        cfg.vla.image_sequence_len,
        checkpoint_path=vision_checkpoint_path,
        dino_local_path=dino_local_path,
        siglip_local_path=siglip_local_path,
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=False,
        custom_hf_path=llm_checkpoint_path,
    )
    aux_target_dim = (
        vision_backbone.vjepa_backbone.embed_dim if hasattr(vision_backbone, "vjepa_backbone") else vision_backbone.embed_dim
    )
    aux_spatial_side = vision_backbone.default_image_size // getattr(vision_backbone, "patch_size", 16)
    effective_future_frames = compute_downsampled_future_frame_count(
        cfg.vla.future_obs_window_size,
        cfg.vla.future_obs_downsample_stride,
    )
    aux_temporal_tokens = max(1, effective_future_frames // getattr(vision_backbone, "tubelet_size", 2))
    vlm = get_vlm(
        model_cfg["model_id"],
        model_cfg["arch_specifier"],
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        use_action_head=cfg.vla.use_action_head,
        action_head_type=cfg.vla.action_head_type,
        use_aux_head=cfg.vla.use_aux_head,
        d_action=cfg.vla.d_action,
        d_proprio=cfg.vla.d_proprio,
        d_a=cfg.vla.d_a,
        n_heads_action=cfg.vla.n_heads_action,
        num_layers_action=cfg.vla.num_layers_action,
        ffn_ratio_action=cfg.vla.ffn_ratio_action,
        beta_alpha=cfg.vla.beta_alpha,
        beta_beta=cfg.vla.beta_beta,
        l1_use_pro_version=cfg.vla.l1_use_pro_version,
        l1_num_blocks=cfg.vla.l1_num_blocks,
        fm_hidden_size=cfg.vla.fm_hidden_size,
        fm_action_model_type=cfg.vla.fm_action_model_type,
        fm_num_inference_timesteps=cfg.vla.fm_num_inference_timesteps,
        fm_num_timestep_buckets=cfg.vla.fm_num_timestep_buckets,
        fm_noise_beta_alpha=cfg.vla.fm_noise_beta_alpha,
        fm_noise_beta_beta=cfg.vla.fm_noise_beta_beta,
        fm_noise_s=cfg.vla.fm_noise_s,
        fm_num_target_vision_tokens=cfg.vla.fm_num_target_vision_tokens,
        fm_add_pos_embed=cfg.vla.fm_add_pos_embed,
        fm_max_seq_len=cfg.vla.fm_max_seq_len,
        fm_state_dropout=cfg.vla.fm_state_dropout,
        fm_jepa_loss_weight=cfg.vla.fm_jepa_loss_weight,
        fm_jepa_horizon=aux_temporal_tokens,
        flow_gr00t_placeholder_tokens=cfg.vla.flow_gr00t_placeholder_tokens,
        flow_gr00t_use_full_llm_hidden=cfg.vla.flow_gr00t_use_full_llm_hidden,
        use_llm_ce_loss=cfg.vla.use_llm_ce_loss,
        lambda_llm_ce=cfg.vla.lambda_llm_ce,
        lora_unfreeze_last_n_llm_layers=cfg.vla.lora_unfreeze_last_n_llm_layers,
        d_aux=cfg.vla.d_aux,
        n_heads_aux=cfg.vla.n_heads_aux,
        num_layers_aux=cfg.vla.num_layers_aux,
        ffn_ratio_aux=cfg.vla.ffn_ratio_aux,
        lambda_aux=cfg.vla.lambda_aux,
        use_visual_token_cosine_head=cfg.vla.use_visual_token_cosine_head,
        lambda_visual_token_cosine=cfg.vla.lambda_visual_token_cosine,
        d_jepa=aux_target_dim,
        aux_T=aux_temporal_tokens,
        aux_H=aux_spatial_side,
        aux_W=aux_spatial_side,
    )

    # The base run only provides pretrained projector + LLM weights.
    # Vision weights should come from the explicit V-JEPA checkpoint path above,
    # while VLA heads are newly initialized for Libero training.
    vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
    vlm.projector.load_state_dict(model_state_dict["projector"])
    # V-JEPA checkpoints can come in bf16; keep training initialization consistent
    # with the rest of the codepath by materializing the full train-time model in fp32.
    vlm = vlm.to(dtype=torch.float32)
    return vlm


def log_module_parameter_breakdown(vlm) -> None:
    def summarize_module(module) -> tuple[int, int]:
        total = sum(param.numel() for param in module.parameters())
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        return total, trainable

    module_specs = [
        ("vision_backbone", getattr(vlm, "vision_backbone", None)),
        ("projector", getattr(vlm, "projector", None)),
        ("llm_backbone", getattr(vlm, "llm_backbone", None)),
        ("action_queries", getattr(vlm, "action_queries", None)),
        ("action_head", getattr(vlm, "action_head", None)),
        ("aux_head", getattr(vlm, "aux_head", None)),
    ]

    lines = ["Module Parameter Breakdown:"]
    for name, module in module_specs:
        if module is None:
            continue
        total, trainable = summarize_module(module)
        lines.append(
            f"  - {name}: total={total / 10**6:.3f}M, trainable={trainable / 10**6:.3f}M"
        )

    llm_module = getattr(getattr(vlm, "llm_backbone", None), "llm", None)
    if llm_module is not None and hasattr(llm_module, "named_parameters"):
        lora_total = sum(param.numel() for name, param in llm_module.named_parameters() if "lora_" in name)
        lora_trainable = sum(
            param.numel() for name, param in llm_module.named_parameters() if "lora_" in name and param.requires_grad
        )
        if lora_total > 0:
            lines.append(
                f"  - llm_lora: total={lora_total / 10**6:.3f}M, trainable={lora_trainable / 10**6:.3f}M"
            )

    overwatch.info("\n".join(lines))


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Custom Local Paths (for JEPA-VLA and local model checkpoints)
    llm_checkpoint_path: Optional[Path] = None                      # Local path to LLM (e.g., Qwen2.5-0.5B)
    fast_tokenizer_path: Optional[Path] = None                     # Local/HF path to FAST tokenizer for LM CE targets

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    save_every_n_epochs: Optional[int] = None                       # Interval for saving checkpoints (in epochs)
    image_aug: bool = False                                         # Whether to enable image augmentations
    use_wrist_image: bool = True                                    # Whether to include wrist camera images
    robotwin_aloha_mosaic: bool = False                             # Compose primary + two wrist views into one image
    debug_batch_shapes: bool = True                                 # Print first training batch tensor shapes
    debug_memory_stats: bool = False                                # Print CUDA memory breakdown during training
    debug_memory_stats_interval: int = 0                            # Log memory every N optimizer steps; 0 disables
    cpu_memory_log_interval: int = 10                               # Log CPU/cgroup memory every N optimizer steps; 0 disables
    debug_embedding_viz_interval: int = 1000                         # Save JEPA embedding heatmaps every N steps; 0 disables
    debug_embedding_viz_samples: int = 1                             # Number of batch samples to visualize
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "jepa-wam"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    use_wandb: bool = False

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        if self.vla.d_action is None or self.vla.d_proprio is None:
            self.vla.d_action, self.vla.d_proprio = resolve_vla_action_proprio_dims(
                self.vla.data_mix,
                self.vla.rotation_representation,
                default_action_dim=self.vla.d_action or ACTION_DIM,
                default_proprio_dim=self.vla.d_proprio or PROPRIO_DIM,
            )

        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.min_learning_rate = self.vla.min_learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy
        if self.cpu_memory_log_interval < 0:
            raise ValueError("cpu_memory_log_interval must be non-negative.")

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"
    from datetime import datetime
    cfg.run_id += f"--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip() if cfg.hf_token.exists() else None
    else:
        hf_token = os.environ.get(cfg.hf_token)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            checkpoint_name = cfg.pretrained_checkpoint.name
            if cfg.pretrained_checkpoint.is_dir() and checkpoint_name == "latest-checkpoint":
                metadata_path = cfg.pretrained_checkpoint / "checkpoint-metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        checkpoint_name = json.load(f).get("source_checkpoint", checkpoint_name)
            assert int(re.search("step-(.+?)-", checkpoint_name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", checkpoint_name).group(1)) == cfg.resume_epoch

        vlm = load_vla(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            base_vlm=cfg.vla.base_vlm,
            llm_checkpoint_path=str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else None,
            vjepa_checkpoint_path=cfg.vla.vjepa_checkpoint_path,
        )
        vlm = vlm.to(dtype=torch.float32)

    else:
        # For JEPA-VLA training we want a base VLM checkpoint plus fresh VLA heads.
        if cfg.vla.use_action_head or cfg.vla.use_aux_head:
            overwatch.info(
                f"Rebuilding base VLM `{cfg.vla.base_vlm}` with JEPA-VLA heads "
                f"(action_head_type={cfg.vla.action_head_type}, use_aux_head={cfg.vla.use_aux_head})"
            )
            vlm = build_vla_from_base_vlm(cfg.vla.base_vlm, cfg, hf_token)
        else:
            # Fallback: plain Prismatic VLM without JEPA-VLA heads.
            vlm = load(
                cfg.vla.base_vlm,
                hf_token=hf_token,
                load_for_training=True,
                image_sequence_len=cfg.vla.image_sequence_len,
            )

    if cfg.vla.use_vlm_peft:
        if not cfg.vla.use_lora:
            raise ValueError("`use_vlm_peft=True` requires `use_lora=True`.")
        if cfg.vla.lora_unfreeze_last_n_llm_layers > 0:
            raise ValueError("`use_vlm_peft=True` does not support `lora_unfreeze_last_n_llm_layers`.")
        overwatch.info(
            "Applying PEFT to the full VLM "
            f"(rank={cfg.vla.lora_rank}, alpha={cfg.vla.lora_alpha}, targets={cfg.vla.lora_target_modules})"
        )
        vlm = apply_peft_to_full_vlm(vlm, cfg.vla)
    elif cfg.vla.use_lora:
        overwatch.info(
            "Applying LoRA to LLM backbone "
            f"(rank={cfg.vla.lora_rank}, alpha={cfg.vla.lora_alpha}, targets={cfg.vla.lora_target_modules})"
        )
        apply_lora_to_vlm(vlm, cfg.vla)

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if cfg.vla.use_vlm_peft:
        stage = "vla-vlm-peft-frozen-vision-train" if cfg.vla.freeze_vision_backbone else "vla-vlm-peft-train"
    elif cfg.vla.use_lora:
        assert cfg.vla.freeze_vision_backbone, "LoRA training currently expects a frozen vision backbone."
        assert not cfg.vla.unfreeze_last_llm_layer, "Use `lora_unfreeze_last_n_llm_layers` for LoRA + layer unfreezing."
        if cfg.vla.lora_unfreeze_last_n_llm_layers > 0:
            stage = "vla-lora-last-n-train"
        else:
            stage = "vla-lora-train"
    elif not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train" if cfg.vla.freeze_projector else "vla-llm-projector-train"
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"freeze_projector: {cfg.vla.freeze_projector}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)
    vlm.debug_memory_stats = cfg.debug_memory_stats

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    log_module_parameter_breakdown(vlm)

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Dataset: format={cfg.vla.dataset_format} mixture={cfg.vla.data_mix}")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_proprio=True,
        use_wrist_image=cfg.use_wrist_image,
        action_head_type=cfg.vla.action_head_type,
        flow_gr00t_placeholder_tokens=cfg.vla.flow_gr00t_placeholder_tokens,
        use_llm_ce_loss=cfg.vla.use_llm_ce_loss,
        future_obs_window_size=cfg.vla.future_obs_window_size,
        future_obs_downsample_stride=cfg.vla.future_obs_downsample_stride,
        context=cfg.vla.context,
        strict_epoch_mode=cfg.vla.strict_epoch_mode,
        shared_dataset_statistics=cfg.vla.shared_dataset_statistics,
        rank_shard_dataset_sources=cfg.vla.rank_shard_dataset_sources,
        visual_token_pair_offset=cfg.vla.visual_token_pair_offset,
        stitch_primary_and_wrist_images=cfg.vla.stitch_primary_and_wrist_images,
        robotwin_aloha_mosaic=cfg.robotwin_aloha_mosaic,
        rotation_representation=cfg.vla.rotation_representation,
        fast_tokenizer_path=cfg.fast_tokenizer_path,
        dataset_format=cfg.vla.dataset_format,
        lerobot_primary_image_key=cfg.vla.lerobot_primary_image_key,
        lerobot_wrist_image_keys=cfg.vla.lerobot_wrist_image_keys,
        lerobot_state_key=cfg.vla.lerobot_state_key,
        lerobot_action_key=cfg.vla.lerobot_action_key,
        lerobot_use_quantile_normalization=cfg.vla.lerobot_use_quantile_normalization,
        lerobot_normalization_clip_value=cfg.vla.lerobot_normalization_clip_value,
        lerobot_num_workers=cfg.vla.lerobot_num_workers,
        lerobot_prefetch_factor=cfg.vla.lerobot_prefetch_factor,
        target_action_dim=cfg.vla.d_action,
        target_proprio_dim=cfg.vla.d_proprio,
    )

    global_dataset_length = getattr(vla_dataset, "global_dataset_length", len(vla_dataset))
    overwatch.info(
        "VLA dataset backend: format=%s class=%s global_examples=%d local_examples=%d",
        cfg.vla.dataset_format,
        type(vla_dataset).__name__,
        global_dataset_length,
        len(vla_dataset),
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        vlm_learning_rate=cfg.vla.vlm_learning_rate,
        vlm_min_learning_rate=cfg.vla.vlm_min_learning_rate,
        action_head_learning_rate=cfg.vla.action_head_learning_rate,
        action_head_min_learning_rate=cfg.vla.action_head_min_learning_rate,
        action_expert_warmup_steps=cfg.vla.action_expert_warmup_steps,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.cpu_memory_log_interval = cfg.cpu_memory_log_interval
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=global_dataset_length)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
        use_wandb=cfg.use_wandb
    )
    train_strategy.debug_batch_shapes = cfg.debug_batch_shapes
    train_strategy.debug_memory_stats = cfg.debug_memory_stats
    train_strategy.debug_memory_stats_interval = cfg.debug_memory_stats_interval
    train_strategy.debug_embedding_viz_interval = cfg.debug_embedding_viz_interval
    train_strategy.debug_embedding_viz_samples = cfg.debug_embedding_viz_samples
    train_strategy.save_adapter_dir_only = cfg.vla.use_vlm_peft

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset=vla_dataset,
        collator=collator,
        action_tokenizer=None,
        metrics=metrics,
        save_interval=cfg.save_interval,
        save_every_n_epochs=cfg.save_every_n_epochs,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()

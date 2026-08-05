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
import os
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
from prismatic.models import load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_fsdp_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
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


def build_vla_from_base_vlm(
    base_vlm_id_or_path: Union[str, Path],
    cfg: "TrainConfig",
    hf_token: Optional[str],
):
    """
    Load the released base VLM checkpoint and attach the fixed JEPA-WAM heads.
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
    llm_checkpoint_path = str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else model_cfg.get("llm_local_path")

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        checkpoint_path=vision_checkpoint_path,
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=False,
        custom_hf_path=llm_checkpoint_path,
    )
    vlm = get_vlm(
        model_cfg["model_id"],
        model_cfg["arch_specifier"],
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        d_action=cfg.vla.d_action,
        d_proprio=cfg.vla.d_proprio,
        fm_hidden_size=cfg.vla.fm_hidden_size,
        fm_num_layers=cfg.vla.fm_num_layers,
        fm_num_inference_timesteps=cfg.vla.fm_num_inference_timesteps,
        fm_num_timestep_buckets=cfg.vla.fm_num_timestep_buckets,
        fm_noise_beta_alpha=cfg.vla.fm_noise_beta_alpha,
        fm_noise_beta_beta=cfg.vla.fm_noise_beta_beta,
        fm_noise_s=cfg.vla.fm_noise_s,
        fm_num_target_vision_tokens=cfg.vla.fm_num_target_vision_tokens,
        fm_add_pos_embed=cfg.vla.fm_add_pos_embed,
        fm_max_seq_len=cfg.vla.fm_max_seq_len,
        fm_state_dropout=cfg.vla.fm_state_dropout,
        flow_gr00t_placeholder_tokens=cfg.vla.flow_gr00t_placeholder_tokens,
        lambda_visual_token_cosine=cfg.vla.lambda_visual_token_cosine,
        d_jepa=vision_backbone.embed_dim,
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
        ("action_head", getattr(vlm, "action_head", None)),
        ("visual_token_cosine_head", getattr(vlm, "visual_token_cosine_head", None)),
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
        default_factory=VLAConfig.get_choice_class(
            VLARegistry.JEPAVLA_QWEN25_VJEPA_224PX_0_5B_LIBERO_90.vla_id
        )
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Optional JEPA-WAM weights used to initialize a new run.
    initial_checkpoint: Optional[Path] = None

    # Custom Local Paths (for JEPA-VLA and local model checkpoints)
    llm_checkpoint_path: Optional[Path] = None                      # Local path to LLM (e.g., Qwen2.5-0.5B)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logs and checkpoints
    run_id_note: Optional[str] = None                               # Optional suffix for the run ID
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    debug_batch_shapes: bool = True                                 # Print first training batch tensor shapes
    debug_memory_stats: bool = False                                # Print CUDA memory breakdown during training
    debug_memory_stats_interval: int = 0                            # Log memory every N optimizer steps; 0 disables
    cpu_memory_log_interval: int = 10                               # Log CPU/cgroup memory every N optimizer steps; 0 disables
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "swanlab")
    swanlab_project: str = "jepa-wam"
    swanlab_entity: Optional[str] = None
    use_swanlab: bool = False

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.min_learning_rate = self.vla.min_learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.warmup_ratio = self.vla.warmup_ratio
        if self.cpu_memory_log_interval < 0:
            raise ValueError("cpu_memory_log_interval must be non-negative.")

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("JEPA-WAM Training :: Warming Up")

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
    from datetime import datetime
    cfg.run_id += f"--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Start =>> Build Directories and Set Randomness
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
    if cfg.initial_checkpoint is not None:
        vlm = load_vla(
            cfg.initial_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            base_vlm=cfg.vla.base_vlm,
            llm_checkpoint_path=str(cfg.llm_checkpoint_path) if cfg.llm_checkpoint_path else None,
            vjepa_checkpoint_path=cfg.vla.vjepa_checkpoint_path,
        )
        vlm = vlm.to(dtype=torch.float32)

    else:
        overwatch.info(f"Rebuilding base VLM `{cfg.vla.base_vlm}` with the fixed JEPA-WAM heads")
        vlm = build_vla_from_base_vlm(cfg.vla.base_vlm, cfg, hf_token)

    overwatch.info(
        "Applying Qwen LoRA (rank=%d, alpha=%d, targets=%s)",
        cfg.vla.lora_rank,
        cfg.vla.lora_alpha,
        cfg.vla.lora_target_modules,
    )
    apply_lora_to_vlm(vlm, cfg.vla)

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    overwatch.info("Freezing the base V-JEPA, projector, and Qwen weights")
    vlm.freeze_for_training()
    vlm.debug_memory_stats = cfg.debug_memory_stats

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    log_module_parameter_breakdown(vlm)

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating LIBERO RLDS dataset: mixture={cfg.vla.data_mix}")
    vla_dataset, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        visual_token_pair_offset=cfg.vla.visual_token_pair_offset,
        target_action_dim=cfg.vla.d_action,
        target_proprio_dim=cfg.vla.d_proprio,
    )

    global_dataset_length = getattr(vla_dataset, "global_dataset_length", len(vla_dataset))
    overwatch.info(
        "VLA dataset backend: class=%s global_examples=%d local_examples=%d",
        type(vla_dataset).__name__,
        global_dataset_length,
        len(vla_dataset),
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info("Initializing fixed FSDP full-shard strategy")
    train_strategy = get_fsdp_strategy(
        vlm=vlm,
        device_id=device_id,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.cpu_memory_log_interval = cfg.cpu_memory_log_interval
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=global_dataset_length)

    # Create Metrics =>> Handles JSONL and optional SwanLab tracking.
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        swanlab_project=cfg.swanlab_project,
        swanlab_entity=cfg.swanlab_entity,
        use_swanlab=cfg.use_swanlab,
    )
    train_strategy.debug_batch_shapes = cfg.debug_batch_shapes
    train_strategy.debug_memory_stats = cfg.debug_memory_stats
    train_strategy.debug_memory_stats_interval = cfg.debug_memory_stats_interval
    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset=vla_dataset,
        collator=collator,
        metrics=metrics,
        save_interval=cfg.save_interval,
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

"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

import json
import math
import shutil
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, NUM_ACTIONS_CHUNK, IGNORE_INDEX
NEWLINE_INDEX = 13  # '\n'
STOP_INDEX = 2  # '</s>'

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    base_lr: float,
):
    """Cosine schedule with linear warmup and an explicit learning-rate floor."""
    if base_lr <= 0:
        raise ValueError(f"`base_lr` must be positive, got {base_lr}.")
    if min_lr < 0:
        raise ValueError(f"`min_lr` must be non-negative, got {min_lr}.")
    if min_lr > base_lr:
        raise ValueError(f"`min_lr` ({min_lr}) must not exceed `base_lr` ({base_lr}).")

    floor_ratio = min_lr / base_lr

    def lr_lambda(current_step: int) -> float:
        if num_training_steps <= 0:
            return floor_ratio

        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup_and_group_min_lrs(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Cosine schedule with per-parameter-group LR floors."""
    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group["lr"])
        min_lr = group.get("min_lr", 0.0)

        if base_lr <= 0:
            raise ValueError(f"`base_lr` must be positive, got {base_lr}.")
        if min_lr < 0:
            raise ValueError(f"`min_lr` must be non-negative, got {min_lr}.")
        if min_lr > base_lr:
            raise ValueError(f"`min_lr` ({min_lr}) must not exceed `base_lr` ({base_lr}).")

        floor_ratio = min_lr / base_lr

        def lr_lambda(current_step: int, floor_ratio: float = floor_ratio) -> float:
            if num_training_steps <= 0:
                return floor_ratio

            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cosine

        lr_lambdas.append(lr_lambda)

    return LambdaLR(optimizer, lr_lambdas)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        min_learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.min_learning_rate = learning_rate, min_learning_rate
        self.weight_decay, self.max_grad_norm = weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio
        self.vlm_learning_rate = _.get("vlm_learning_rate")
        self.vlm_min_learning_rate = _.get("vlm_min_learning_rate")
        self.action_head_learning_rate = _.get("action_head_learning_rate")
        self.action_head_min_learning_rate = _.get("action_head_min_learning_rate")
        self.action_expert_warmup_steps = int(_.get("action_expert_warmup_steps", 0))

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn
        self.cpu_memory_log_interval = 10

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @staticmethod
    def _normalize_param_name(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "").replace("module.", "").replace("base_model.model.", "")

    @staticmethod
    def _extract_component_state_dicts(full_state_dict: dict, module_keys: list[str]) -> dict:
        model_state_dicts = {mkey: OrderedDict() for mkey in module_keys}

        for raw_key, param in full_state_dict.items():
            key = TrainingStrategy._normalize_param_name(raw_key)
            for mkey in model_state_dicts:
                if key.startswith(mprefix := f"{mkey}."):
                    model_state_dicts[mkey][key.removeprefix(mprefix)] = param
                    break

        return {mkey: state_dict for mkey, state_dict in model_state_dicts.items() if state_dict}

    @staticmethod
    def _read_cgroup_memory_bytes() -> tuple[Optional[int], Optional[int]]:
        """Return the current and maximum memory of the active cgroup when available."""
        candidates = (
            (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
            (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
        )
        for current_path, limit_path in candidates:
            if not current_path.exists():
                continue
            try:
                current_bytes = int(current_path.read_text().strip())
                raw_limit = limit_path.read_text().strip() if limit_path.exists() else "max"
                limit_bytes = None if raw_limit == "max" else int(raw_limit)
                return current_bytes, limit_bytes
            except (OSError, ValueError):
                continue
        return None, None

    def collect_cpu_memory_metrics(self) -> dict[str, float]:
        """Collect process RSS across all ranks and node/cgroup memory on rank zero."""
        try:
            local_rss_bytes = float(psutil.Process().memory_info().rss)
        except psutil.Error:
            local_rss_bytes = 0.0

        rss_sum_bytes = local_rss_bytes
        rss_max_bytes = local_rss_bytes
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            reduce_device = torch.device("cuda", self.device_id) if torch.cuda.is_available() else torch.device("cpu")
            rss = torch.tensor([local_rss_bytes], dtype=torch.float64, device=reduce_device)
            rss_sum = rss.clone()
            rss_max = rss.clone()
            dist.all_reduce(rss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(rss_max, op=dist.ReduceOp.MAX)
            rss_sum_bytes = float(rss_sum.item())
            rss_max_bytes = float(rss_max.item())
            world_size = dist.get_world_size()

        if not overwatch.is_rank_zero():
            return {}

        gib = float(1024 ** 3)
        metrics = {
            "System/CPU RSS Max Rank (GiB)": rss_max_bytes / gib,
            "System/CPU RSS Mean Rank (GiB)": rss_sum_bytes / world_size / gib,
            "System/CPU RSS Sum Ranks (GiB)": rss_sum_bytes / gib,
        }
        try:
            virtual_memory = psutil.virtual_memory()
            metrics["System/Host Memory Used (GiB)"] = virtual_memory.used / gib
            metrics["System/Host Memory Available (GiB)"] = virtual_memory.available / gib
            metrics["System/Host Memory Percent"] = float(virtual_memory.percent)
        except psutil.Error:
            pass

        cgroup_current, cgroup_limit = self._read_cgroup_memory_bytes()
        if cgroup_current is not None:
            metrics["System/Cgroup Memory Current (GiB)"] = cgroup_current / gib
        if cgroup_limit is not None and cgroup_limit > 0:
            metrics["System/Cgroup Memory Limit (GiB)"] = cgroup_limit / gib
            metrics["System/Cgroup Memory Percent"] = 100.0 * cgroup_current / cgroup_limit
        return metrics

    def build_optimizer_groups(self, named_parameters):
        use_custom_groups = self.vlm_learning_rate is not None or self.action_head_learning_rate is not None
        if not use_custom_groups:
            decay, no_decay = [], []
            for name, param in named_parameters:
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": decay, "weight_decay": self.weight_decay, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "default-decay"},
                {"params": no_decay, "weight_decay": 0.0, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "default-no-decay"},
            ]

        vlm_lr = self.vlm_learning_rate if self.vlm_learning_rate is not None else self.learning_rate
        vlm_min_lr = self.vlm_min_learning_rate if self.vlm_min_learning_rate is not None else self.min_learning_rate
        action_lr = self.action_head_learning_rate if self.action_head_learning_rate is not None else self.learning_rate
        action_min_lr = (
            self.action_head_min_learning_rate if self.action_head_min_learning_rate is not None else self.min_learning_rate
        )

        grouped_params = {
            "vlm-decay": {"params": [], "weight_decay": self.weight_decay, "base_lr": vlm_lr, "min_lr": vlm_min_lr, "name": "vlm-decay"},
            "vlm-no-decay": {"params": [], "weight_decay": 0.0, "base_lr": vlm_lr, "min_lr": vlm_min_lr, "name": "vlm-no-decay"},
            "action-decay": {"params": [], "weight_decay": self.weight_decay, "base_lr": action_lr, "min_lr": action_min_lr, "name": "action-decay"},
            "action-no-decay": {"params": [], "weight_decay": 0.0, "base_lr": action_lr, "min_lr": action_min_lr, "name": "action-no-decay"},
        }

        for raw_name, param in named_parameters:
            if not param.requires_grad:
                continue

            name = self._normalize_param_name(raw_name)
            is_action_group = (
                name.startswith("action_head.")
                or name.startswith("action_queries.")
                or name.startswith("aux_head.")
            )
            decay_key = "action-decay" if is_action_group else "vlm-decay"
            no_decay_key = "action-no-decay" if is_action_group else "vlm-no-decay"
            group_key = no_decay_key if param.ndim <= 1 or name.endswith(".bias") else decay_key
            grouped_params[group_key]["params"].append(param)

        return [group for group in grouped_params.values() if group["params"]]

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def _cuda_mem_snapshot(self, label: str) -> dict:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        return {
            "label": label,
            "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }

    def _log_cuda_mem_snapshot(self, label: str, step: int) -> None:
        if not overwatch.is_rank_zero() or not getattr(self, "debug_memory_stats", False):
            return
        snap = self._cuda_mem_snapshot(label)
        if not snap:
            return
        overwatch.info(
            f"[Mem][step={step:06d}] {label}: "
            f"alloc={snap['allocated_gb']:.2f}G reserved={snap['reserved_gb']:.2f}G max={snap['max_allocated_gb']:.2f}G"
        )

    def _save_adapter_checkpoint_dir(
        self,
        run_dir: Path,
        export_dir: Path,
        model_state_dicts: dict,
        full_vlm_state_dict: Optional[dict] = None,
    ) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("config.json", "config.yaml", "dataset_statistics.json"):
            src = run_dir / filename
            if src.exists():
                shutil.copy2(src, export_dir / filename)

        with open(export_dir / "checkpoint-metadata.json", "w") as f:
            json.dump(
                {
                    "source_checkpoint": export_dir.name,
                    "trainable_module_keys": list(self.trainable_module_keys),
                    "stage": self.stage,
                },
                f,
                indent=2,
            )

        vlm = getattr(self.vlm, "module", getattr(self.vlm, "_fsdp_wrapped_module", self.vlm))
        if not hasattr(vlm, "peft_config"):
            raise ValueError("Adapter-directory save requires the top-level VLM to be PEFT-wrapped.")

        from peft.utils.save_and_load import get_peft_model_state_dict

        adapter_dir = export_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_name = next(iter(vlm.peft_config.keys()))
        adapter_state_dict = get_peft_model_state_dict(
            vlm,
            state_dict=full_vlm_state_dict,
            adapter_name=adapter_name,
        )
        vlm.peft_config[adapter_name].save_pretrained(adapter_dir)
        torch.save(adapter_state_dict, adapter_dir / "adapter_model.bin")

        if "action_queries" in model_state_dicts:
            torch.save(model_state_dicts["action_queries"], export_dir / "action_queries--checkpoint.pt")
        if "action_head" in model_state_dicts:
            torch.save(model_state_dicts["action_head"], export_dir / "action_head--checkpoint.pt")
        if "aux_head" in model_state_dicts:
            torch.save(model_state_dicts["aux_head"], export_dir / "aux_head--checkpoint.pt")
        if "visual_token_cosine_head" in model_state_dicts:
            torch.save(
                model_state_dicts["visual_token_cosine_head"],
                export_dir / "visual_token_cosine_head--checkpoint.pt",
            )

    def _export_vla_checkpoint_dir(self, run_dir: Path, checkpoint_path: Path, model_state_dicts: dict) -> None:
        """
        Export a VLA-Adapter-style checkpoint directory next to the native training `.pt` checkpoint.

        The native `.pt` file remains the source of truth for training resume. This export directory exists so
        evaluation code can load LoRA adapters and auxiliary heads directly from a filesystem layout that mirrors
        the VLA-Adapter project.
        """
        export_dir = checkpoint_path.with_suffix("")
        export_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("config.json", "config.yaml", "dataset_statistics.json"):
            src = run_dir / filename
            if src.exists():
                shutil.copy2(src, export_dir / filename)

        with open(export_dir / "checkpoint-metadata.json", "w") as f:
            json.dump(
                {
                    "source_checkpoint": checkpoint_path.name,
                    "trainable_module_keys": list(self.trainable_module_keys),
                    "stage": self.stage,
                },
                f,
                indent=2,
            )

        vlm = getattr(self.vlm, "module", getattr(self.vlm, "_fsdp_wrapped_module", self.vlm))

        llm_module = getattr(vlm.llm_backbone, "llm", None)
        has_lora = llm_module is not None and hasattr(llm_module, "peft_config")
        if has_lora:
            from peft.utils.save_and_load import get_peft_model_state_dict

            adapter_dir = export_dir / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_name = next(iter(llm_module.peft_config.keys()))
            llm_state_dict = {
                key.removeprefix("llm."): value
                for key, value in model_state_dicts.get("llm_backbone", {}).items()
                if key.startswith("llm.")
            }
            adapter_state_dict = get_peft_model_state_dict(
                llm_module,
                state_dict=llm_state_dict,
                adapter_name=adapter_name,
            )
            llm_module.peft_config[adapter_name].save_pretrained(adapter_dir)
            torch.save(adapter_state_dict, adapter_dir / "adapter_model.bin")
        elif "llm_backbone" in model_state_dicts:
            torch.save(model_state_dicts["llm_backbone"], export_dir / "llm_backbone--checkpoint.pt")

        if "vision_backbone" in model_state_dicts:
            torch.save(model_state_dicts["vision_backbone"], export_dir / "vision_backbone--checkpoint.pt")
        if "projector" in model_state_dicts:
            torch.save(model_state_dicts["projector"], export_dir / "projector--checkpoint.pt")
        if "action_queries" in model_state_dicts:
            torch.save(model_state_dicts["action_queries"], export_dir / "action_queries--checkpoint.pt")
        if "action_head" in model_state_dicts:
            torch.save(model_state_dicts["action_head"], export_dir / "action_head--checkpoint.pt")
        if "aux_head" in model_state_dicts:
            torch.save(model_state_dicts["aux_head"], export_dir / "aux_head--checkpoint.pt")
        if "visual_token_cosine_head" in model_state_dicts:
            torch.save(
                model_state_dicts["visual_token_cosine_head"],
                export_dir / "visual_token_cosine_head--checkpoint.pt",
            )

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    @staticmethod
    def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(3, 1, 1)
        image = (image.detach() * std + mean).clamp(0, 1)
        array = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _embedding_to_heatmap(embedding: torch.Tensor, size: tuple[int, int]) -> Image.Image:
        heat = embedding.detach().float().norm(dim=-1)
        heat = heat - heat.min()
        heat = heat / heat.max().clamp_min(1e-6)
        heat = F.interpolate(heat[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        h = heat.cpu().numpy()

        r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
        color = np.stack([r, g, b], axis=-1)
        return Image.fromarray((color * 255).astype(np.uint8))

    @staticmethod
    def _overlay_heatmap(image: Image.Image, heatmap: Image.Image, alpha: float = 0.45) -> Image.Image:
        return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)

    @staticmethod
    def _concat_images(images: list[Image.Image]) -> Image.Image:
        widths, heights = zip(*(img.size for img in images))
        canvas = Image.new("RGB", (sum(widths), max(heights)))
        x = 0
        for img in images:
            canvas.paste(img.convert("RGB"), (x, 0))
            x += img.size[0]
        return canvas

    def _save_embedding_visualizations(
        self,
        batch: dict,
        output: dict,
        step: int,
        run_dir: Path,
        max_samples: int = 1,
    ) -> None:
        try:
            current_vjepa = output.get("current_vjepa")
            aux_pred = output.get("aux_pred")
            vjepa_target = output.get("vjepa_target")
            images = batch.get("pixel_values")
            future_images = batch.get("future_pixel_values")
            if current_vjepa is None or aux_pred is None or vjepa_target is None or images is None or future_images is None:
                return

            current_vjepa = current_vjepa.detach().cpu()
            aux_pred = aux_pred.detach().cpu()
            vjepa_target = vjepa_target.detach().cpu()
            images = images.detach().cpu()
            future_images = future_images.detach().cpu()

            if images.dim() == 4:
                images = images.unsqueeze(1)
            if future_images.dim() == 5:
                future_images = future_images.unsqueeze(1)

            B, V, _, H, W = images.shape
            _, _, T_aux, H_aux, W_aux, _ = aux_pred.shape
            current_vjepa = current_vjepa.reshape(B, V, H_aux * W_aux, -1).reshape(B, V, H_aux, W_aux, -1)

            out_dir = Path(run_dir) / "debug_embedding_viz" / f"step-{step:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for b in range(min(max_samples, B)):
                for v in range(V):
                    current_img = self._image_tensor_to_pil(images[b, v])
                    current_heat = self._embedding_to_heatmap(current_vjepa[b, v], current_img.size[::-1])
                    current_overlay = self._overlay_heatmap(current_img, current_heat)
                    current_overlay.save(out_dir / f"sample{b}_view{v}_current_jepa_overlay.png")

                    for t in range(T_aux):
                        frame_idx = min(future_images.shape[2] - 1, t * 2 + 1)
                        future_img = self._image_tensor_to_pil(future_images[b, v, frame_idx])
                        pred_heat = self._embedding_to_heatmap(aux_pred[b, v, t], future_img.size[::-1])
                        gt_heat = self._embedding_to_heatmap(vjepa_target[b, v, t], future_img.size[::-1])
                        pred_overlay = self._overlay_heatmap(future_img, pred_heat)
                        gt_overlay = self._overlay_heatmap(future_img, gt_heat)
                        self._concat_images([future_img, pred_overlay, gt_overlay]).save(
                            out_dir / f"sample{b}_view{v}_future_t{t}_raw_pred_gt.png"
                        )

            overwatch.info("Saved embedding visualizations to %s", out_dir)
        except Exception as e:
            overwatch.info("Skipping embedding visualization at step %d due to %s: %s", step, type(e).__name__, e)

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_every_n_epochs: Optional[int] = None,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"

        dataloader_num_workers = getattr(vla_dataset, "dataloader_num_workers", 0)
        dataloader_kwargs = {
            "dataset": vla_dataset,
            "batch_size": self.per_device_batch_size,
            "sampler": None,
            "collate_fn": collator,
            "num_workers": dataloader_num_workers,
            "worker_init_fn": self.worker_init_fn,
            "pin_memory": getattr(vla_dataset, "dataloader_pin_memory", False),
        }
        if dataloader_num_workers > 0:
            dataloader_kwargs.update(
                prefetch_factor=getattr(vla_dataset, "dataloader_prefetch_factor", 2),
                persistent_workers=True,
            )
        dataloader = DataLoader(**dataloader_kwargs)
        overwatch.info(
            "VLA DataLoader: num_workers=%d prefetch_factor=%s persistent_workers=%s pin_memory=%s",
            dataloader_num_workers,
            dataloader_kwargs.get("prefetch_factor"),
            dataloader_kwargs.get("persistent_workers", False),
            dataloader_kwargs["pin_memory"],
        )

        strict_epoch_mode = getattr(vla_dataset, "strict_epoch_mode", False)
        local_micro_batches_per_epoch = len(dataloader)
        if self.action_expert_warmup_steps > 0:
            overwatch.info(
                "Action-expert warmup enabled: first %d optimizer steps train only `action_head` with VLM features detached.",
                self.action_expert_warmup_steps,
            )
        if strict_epoch_mode:
            if local_micro_batches_per_epoch == 0:
                raise ValueError("Strict epoch mode requires a non-empty dataloader.")
            grad_steps_per_epoch = (local_micro_batches_per_epoch + self.grad_accumulation_steps - 1) // self.grad_accumulation_steps
            overwatch.info(
                "Strict epoch mode active: one epoch visits the full dataset once without repeat/sampling. "
                "local_micro_batches=%d grad_steps_per_epoch=%d",
                local_micro_batches_per_epoch,
                grad_steps_per_epoch,
            )
        else:
            grad_steps_per_epoch = len(dataloader)

        last_loss_scalar = {"value": None}

        def process_batch(batch, *, grad_step_ready: bool, accum_divisor: int, epoch_value: int) -> bool:
            should_log_memory = (
                getattr(self, "debug_memory_stats", False)
                and getattr(self, "debug_memory_stats_interval", 0) > 0
                and (metrics.global_step % self.debug_memory_stats_interval) == 0
            )
            if getattr(self, "debug_batch_shapes", False) and not getattr(self, "_printed_batch_shapes", False):
                if overwatch.is_rank_zero():
                    shape_lines = []
                    for key in (
                        "pixel_values",
                        "pixel_values_wrist",
                        "future_pixel_values",
                        "actions",
                        "action_valid_mask",
                        "action_valid_dim",
                        "proprio",
                        "input_ids",
                        "attention_mask",
                        "labels",
                    ):
                        value = batch.get(key)
                        if isinstance(value, torch.Tensor):
                            shape_lines.append(f"{key}={tuple(value.shape)} dtype={value.dtype}")
                        else:
                            shape_lines.append(f"{key}={type(value).__name__}")
                    dataset_names = batch.get("dataset_names")
                    if dataset_names is not None:
                        shape_lines.append(f"dataset_names[0]={dataset_names[0]!r}")
                    overwatch.info("First training batch shapes: %s", " | ".join(shape_lines))
                self._printed_batch_shapes = True

            with torch.autocast(
                "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
            ):
                action_expert_only = metrics.global_step < self.action_expert_warmup_steps
                output = self.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    future_pixel_values=batch.get("future_pixel_values"),
                    pair_pixel_values=batch.get("pair_pixel_values"),
                    actions=batch.get("actions"),
                    action_valid_mask=batch.get("action_valid_mask"),
                    action_valid_dim=batch.get("action_valid_dim"),
                    proprio=batch.get("proprio"),
                    action_expert_only=action_expert_only,
                )
                if isinstance(output, dict):
                    loss = output["loss"]
                    logits = output.get("logits")
                else:
                    loss = output.loss
                    logits = output.logits

            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_forward", metrics.global_step)
                if isinstance(output, dict) and "memory_stats" in output:
                    for entry in output["memory_stats"]:
                        overwatch.info(
                            f"[Mem][step={metrics.global_step:06d}] {entry['label']}: "
                            f"alloc={entry['allocated_gb']:.2f}G reserved={entry['reserved_gb']:.2f}G "
                            f"max={entry['max_allocated_gb']:.2f}G"
                        )

            if (
                isinstance(output, dict)
                and overwatch.is_rank_zero()
                and getattr(self, "debug_embedding_viz_interval", 0)
                and grad_step_ready
                and (metrics.global_step + 1) % self.debug_embedding_viz_interval == 0
            ):
                self._save_embedding_visualizations(
                    batch=batch,
                    output=output,
                    step=metrics.global_step + 1,
                    run_dir=metrics.run_dir,
                    max_samples=getattr(self, "debug_embedding_viz_samples", 1),
                )

            metrics.commit(loss=loss)
            last_loss_scalar["value"] = float(loss.item())
            (loss / accum_divisor).backward()
            if isinstance(output, dict):
                if "loss_action" in output:
                    metrics.commit(loss_action=output["loss_action"])
                if "loss_jepa" in output:
                    metrics.commit(loss_jepa=output["loss_jepa"])
                if "loss_aux" in output:
                    metrics.commit(loss_aux=output["loss_aux"])
                if "loss_visual_token_cosine" in output:
                    metrics.commit(loss_visual_token_cosine=output["loss_visual_token_cosine"])
                if "loss_llm_ce" in output:
                    metrics.commit(loss_llm_ce=output["loss_llm_ce"])
            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_backward", metrics.global_step)

            if logits is not None and action_tokenizer is not None and not getattr(self.vlm, "use_llm_ce_loss", False):
                predicted_token_ids = logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                ground_truth_token_ids = batch["labels"][:, 1:].to(predicted_token_ids.device)

                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
                action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)

                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
                next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)

                metrics.commit(
                    action_accuracy=action_accuracy,
                    l1_loss=action_l1_loss,
                    next_actions_accuracy=next_actions_accuracy,
                    next_actions_l1_loss=next_actions_l1_loss,
                    update_step_time=grad_step_ready,
                )

                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            pred_continuous_actions_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    predicted_token_ids[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    ground_truth_token_ids[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                pred_continuous_actions_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(),
                                action_accuracy=action_accuracy_ds,
                                l1_loss=action_l1_loss_ds,
                                next_actions_accuracy=next_actions_accuracy,
                                next_actions_l1_loss=next_actions_l1_loss,
                            )
            else:
                metrics.commit(update_step_time=grad_step_ready)

            if not grad_step_ready:
                return False

            self.clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_optimizer_step", metrics.global_step + 1)

            metrics.commit(global_step=metrics.global_step + 1, epoch=epoch_value, lr=self.lr_scheduler.get_last_lr()[0])
            should_log_cpu_memory = self.cpu_memory_log_interval > 0 and (
                metrics.global_step == 1 or metrics.global_step % self.cpu_memory_log_interval == 0
            )
            if should_log_cpu_memory:
                cpu_memory_metrics = self.collect_cpu_memory_metrics()
                if overwatch.is_rank_zero():
                    metrics.set_system_metrics(**cpu_memory_metrics)
            status = metrics.push()

            step_save_due = (
                save_interval is not None
                and save_interval > 0
                and not strict_epoch_mode
                and (metrics.global_step % save_interval) == 0
            )
            if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or step_save_due:
                self.save_checkpoint(
                    metrics.run_dir, metrics.global_step, epoch_value, loss.item(), only_trainable=not save_full_model
                )
                dist.barrier()
                if terminate:
                    return True

            progress.update()
            progress.set_description(status)
            return False

        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * grad_steps_per_epoch) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()
            self.optimizer.zero_grad()

            if strict_epoch_mode:
                remainder = local_micro_batches_per_epoch % self.grad_accumulation_steps
                final_block_size = remainder if remainder != 0 else self.grad_accumulation_steps

                for epoch_idx in range(self.epochs):
                    self.vlm.train()
                    self.optimizer.zero_grad()
                    epoch_value = epoch_idx + 1

                    for train_idx, batch in enumerate(dataloader):
                        in_final_partial_block = remainder != 0 and train_idx >= local_micro_batches_per_epoch - remainder
                        accum_divisor = final_block_size if in_final_partial_block else self.grad_accumulation_steps
                        grad_step_ready = (
                            ((train_idx + 1) % self.grad_accumulation_steps) == 0
                            or (train_idx + 1) == local_micro_batches_per_epoch
                        )
                        if process_batch(
                            batch,
                            grad_step_ready=grad_step_ready,
                            accum_divisor=accum_divisor,
                            epoch_value=epoch_value,
                        ):
                            return

                    should_save_on_epoch = (
                        self.max_steps is None
                        and save_every_n_epochs is not None
                        and save_every_n_epochs > 0
                        and (epoch_value % save_every_n_epochs) == 0
                    )
                    should_save_final_epoch = (
                        self.max_steps is None
                        and epoch_value == self.epochs
                        and save_every_n_epochs is not None
                        and save_every_n_epochs > 0
                        and (epoch_value % save_every_n_epochs) != 0
                    )
                    if should_save_on_epoch or should_save_final_epoch:
                        self.save_checkpoint(
                            metrics.run_dir,
                            metrics.global_step,
                            epoch_value,
                            last_loss_scalar["value"],
                            only_trainable=not save_full_model,
                        )
                        dist.barrier()
            else:
                global_dataset_length = getattr(vla_dataset, "global_dataset_length", len(vla_dataset))
                epoch_denominator = max(1, math.ceil(global_dataset_length / self.global_batch_size))
                for train_idx, batch in enumerate(dataloader):
                    grad_step_ready = ((train_idx + 1) % self.grad_accumulation_steps) == 0
                    epoch_value = (metrics.global_step + 1) // epoch_denominator
                    if process_batch(
                        batch,
                        grad_step_ready=grad_step_ready,
                        accum_divisor=self.grad_accumulation_steps,
                        epoch_value=epoch_value,
                    ):
                        return

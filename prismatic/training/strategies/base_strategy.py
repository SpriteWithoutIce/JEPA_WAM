"""
base_strategy.py

Shared optimizer, checkpoint, and fixed VLA training-loop logic for the FSDP strategy.
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


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
        max_steps: int,
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        min_learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.max_steps = max_steps
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.min_learning_rate = learning_rate, min_learning_rate
        self.weight_decay, self.max_grad_norm = weight_decay, max_grad_norm
        self.warmup_ratio = warmup_ratio

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
        decay, no_decay = [], []
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.weight_decay, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": self.learning_rate, "min_lr": self.min_learning_rate, "name": "no-decay"},
        ]
        return [group for group in groups if group["params"]]

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

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
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
                        "pair_pixel_values",
                        "actions",
                        "proprio",
                        "input_ids",
                        "attention_mask",
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
                output = self.vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    pair_pixel_values=batch.get("pair_pixel_values"),
                    actions=batch.get("actions"),
                    proprio=batch.get("proprio"),
                )
                loss = output["loss"]

            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_forward", metrics.global_step)
                if isinstance(output, dict) and "memory_stats" in output:
                    for entry in output["memory_stats"]:
                        overwatch.info(
                            f"[Mem][step={metrics.global_step:06d}] {entry['label']}: "
                            f"alloc={entry['allocated_gb']:.2f}G reserved={entry['reserved_gb']:.2f}G "
                            f"max={entry['max_allocated_gb']:.2f}G"
                        )

            metrics.commit(loss=loss)
            (loss / accum_divisor).backward()
            if "loss_action" in output:
                metrics.commit(loss_action=output["loss_action"])
            if "loss_visual_token_cosine" in output:
                metrics.commit(loss_visual_token_cosine=output["loss_visual_token_cosine"])
            if should_log_memory and overwatch.is_rank_zero():
                self._log_cuda_mem_snapshot("after_backward", metrics.global_step)

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
                and (metrics.global_step % save_interval) == 0
            )
            if (terminate := metrics.global_step >= self.max_steps) or step_save_due:
                self.save_checkpoint(
                    metrics.run_dir, metrics.global_step, epoch_value, loss.item(), only_trainable=False
                )
                dist.barrier()
                if terminate:
                    return True

            progress.update()
            progress.set_description(status)
            return False

        status = metrics.get_status()
        with tqdm(
            total=self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()
            self.optimizer.zero_grad()

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

        raise RuntimeError("LIBERO dataloader ended before max_steps was reached.")

"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import (
    TrainingStrategy,
    get_cosine_schedule_with_warmup_and_group_min_lrs,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    def __init__(
        self,
        vlm,
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
        worker_init_fn=None,
        vlm_learning_rate: Optional[float] = None,
        vlm_min_learning_rate: Optional[float] = None,
        action_head_learning_rate: Optional[float] = None,
        action_head_min_learning_rate: Optional[float] = None,
        action_expert_warmup_steps: int = 0,
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            vlm_learning_rate=vlm_learning_rate,
            vlm_min_learning_rate=vlm_min_learning_rate,
            action_head_learning_rate=action_head_learning_rate,
            action_head_min_learning_rate=action_head_min_learning_rate,
            action_expert_warmup_steps=action_expert_warmup_steps,
        )

    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        if getattr(self, "save_adapter_dir_only", False):
            checkpoint_dir = run_dir / "checkpoints"
            export_dir = checkpoint_dir / (
                f"step-{global_step:06d}-epoch-{epoch:02d}-loss={'inf' if train_loss is None else f'{train_loss:.4f}'}"
            )
            full_vlm_state_dict = self.vlm.module.state_dict()
            sidecar_module_keys = [
                mkey
                for mkey in ("action_head", "aux_head", "visual_token_cosine_head")
                if mkey in self.trainable_module_keys and getattr(self.vlm.module, mkey, None) is not None
            ]
            self._save_adapter_checkpoint_dir(
                run_dir,
                export_dir,
                model_state_dicts=self._extract_component_state_dicts(full_vlm_state_dict, sidecar_module_keys),
                full_vlm_state_dict=full_vlm_state_dict,
            )

            latest_dir = checkpoint_dir / "latest-checkpoint"
            if latest_dir.exists() or latest_dir.is_symlink():
                shutil.rmtree(latest_dir, ignore_errors=True)
            shutil.copytree(export_dir, latest_dir)
            return

        # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
        self._export_vla_checkpoint_dir(run_dir, checkpoint_path, model_state_dicts)

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()

        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            groups = self.build_optimizer_groups(self.vlm.named_parameters())
            optimizer_groups = []
            for group in groups:
                optimizer_groups.append(
                    {
                        "params": group["params"],
                        "weight_decay": group["weight_decay"],
                        "lr": group["base_lr"],
                        "base_lr": group["base_lr"],
                        "min_lr": group["min_lr"],
                        "name": group["name"],
                    }
                )

            self.optimizer = AdamW(optimizer_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
            if any(group["min_lr"] > 0 for group in optimizer_groups):
                self.lr_scheduler = get_cosine_schedule_with_warmup_and_group_min_lrs(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps, num_training_steps
                )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            groups = self.build_optimizer_groups(self.vlm.named_parameters())
            optimizer_groups = []
            for group in groups:
                optimizer_groups.append(
                    {
                        "params": group["params"],
                        "weight_decay": group["weight_decay"],
                        "lr": group["base_lr"],
                        "base_lr": group["base_lr"],
                        "min_lr": group["min_lr"],
                        "name": group["name"],
                    }
                )
            self.optimizer = AdamW(optimizer_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> Min Cosine LR = {self.min_learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )
        for group in self.optimizer.param_groups:
            overwatch.info(
                "         |-> Optimizer Group `%s`: base_lr=%s min_lr=%s weight_decay=%s params=%d",
                group.get("name", "unnamed"),
                group.get("base_lr", group["lr"]),
                group.get("min_lr", 0.0),
                group["weight_decay"],
                len(group["params"]),
            )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)

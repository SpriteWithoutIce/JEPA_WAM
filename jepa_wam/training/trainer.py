"""JEPA-WAM training loop."""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from jepa_wam.models.jepa_vla import JepaVLA
from jepa_wam.training.metrics import Metrics


class JEPAWTrainer:
    """Simple trainer supporting AMP, gradient clipping, and cosine LR."""

    def __init__(
        self,
        model: JepaVLA,
        cfg,
        device_id: int = 0,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device_id = device_id

        # Move model to device
        self.model.to(device_id)

        # Optimizer: only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler (cosine with warmup)
        self.total_steps = cfg.max_steps or 100000
        self.warmup_steps = int(self.total_steps * cfg.warmup_ratio)
        self.lr_scheduler = self._build_scheduler()

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.enable_mixed_precision_training)
        self.enable_amp = cfg.enable_mixed_precision_training

    def _build_scheduler(self):
        """Cosine decay with linear warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        dataset: IterableDataset,
        collator,
        metrics: Metrics,
        save_interval: int = 2500,
        run_dir: Optional[Path] = None,
    ) -> None:
        """Main training loop."""
        self.model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

        progress = tqdm(
            total=self.total_steps,
            desc="Training",
            leave=False,
        )

        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device_id) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                losses = self.model(batch)
                loss_action = losses["action"]
                loss_aux = losses.get("aux", torch.tensor(0.0, device=self.device_id))

                lambda_aux = self.model.get_lambda_aux(metrics.global_step, self.total_steps)
                total_loss = loss_action + lambda_aux * loss_aux

            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            # Logging
            metrics.commit(
                total_loss=total_loss.item(),
                action_loss=loss_action.item(),
                aux_loss=loss_aux.item(),
                lambda_aux=lambda_aux,
                lr=self.lr_scheduler.get_last_lr()[0],
            )
            metrics.step()

            progress.update()
            progress.set_description(metrics.get_status())

            # Checkpointing
            if run_dir is not None and metrics.global_step % save_interval == 0:
                self.save_checkpoint(run_dir, metrics.global_step)

            if metrics.global_step >= self.total_steps:
                break

        progress.close()

    def save_checkpoint(self, run_dir: Path, step: int) -> None:
        """Save trainable weights only."""
        ckpt_path = run_dir / "checkpoints" / f"step-{step}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, ckpt_path: Path) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["scheduler"])
        print(f"Loaded checkpoint from {ckpt_path}")

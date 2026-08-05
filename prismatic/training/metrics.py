"""
metrics.py

Metrics and JSONL/SwanLab trackers for the fixed JEPA-WAM training loop.
"""

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import jsonlines
import numpy as np
import torch
import swanlab

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None: ...

    def finalize(self) -> None: ...


# === Individual Tracker Definitions ===
class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return


class SwanLabTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "prismatic",
        entity: Optional[str] = None,
        group: str = "align",
        enabled: bool = False,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        self.enabled = enabled

        self.project, self.entity, self.group, self.log_dir = project, entity, group, self.run_dir

        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        if self.enabled:
            swanlab.init(
                name=self.run_id,
                dir=self.log_dir,
                config=self.hparams,
                project=self.project,
                entity=self.entity,
                group=self.group,
            )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        if self.enabled:
            swanlab.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        if self.enabled:
            swanlab.log(metrics, step=global_step)

    def finalize(self) -> None:
        if overwatch.is_rank_zero() and self.enabled:
            swanlab.finish()


# === Fixed VLA Metrics ===


class VLAMetrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        swanlab_project: str = "jepa-wam",
        swanlab_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        use_swanlab: bool = False,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "swanlab":
                tracker = SwanLabTracker(
                    run_id,
                    run_dir,
                    hparams,
                    project=swanlab_project,
                    entity=swanlab_entity,
                    group="vla-train",
                    enabled=use_swanlab,
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0
        self.epoch = 0
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
            "loss_action": deque(maxlen=window_size),
            "loss_visual_token_cosine": deque(maxlen=window_size),
        }

        # Latest system metrics are emitted on the next tracker write, then cleared.
        self.system_metrics: Dict[str, float] = {}

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def set_system_metrics(self, **metrics: float) -> None:
        """Attach scalar system metrics to the next tracker write on rank zero."""
        if overwatch.is_rank_zero():
            self.system_metrics.update({key: float(value) for key, value in metrics.items()})

    def get_status(
        self,
        loss: Optional[torch.Tensor] = None,
        loss_action: Optional[torch.Tensor] = None,
        loss_visual_token_cosine: Optional[torch.Tensor] = None,
    ) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return (
            f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} "
            f"- Loss :: {loss:.4f} - Action :: {loss_action:.4f} "
            f"- Visual Cos :: {loss_visual_token_cosine:.4f}"
        )

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value)

    @overwatch.rank_zero_only
    def push(self) -> str:
        def _mean_or_default(values, default: float = 0.0) -> float:
            if len(values) == 0:
                return default
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.stack(list(values)).mean().item()
            return float(np.mean(list(values)))

        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = _mean_or_default(self.state["loss_raw"])
        loss = _mean_or_default(self.state["loss"])
        loss_action = _mean_or_default(self.state["loss_action"])
        loss_visual_token_cosine = _mean_or_default(self.state["loss_visual_token_cosine"])
        step_time = _mean_or_default(self.state["step_time"])
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0.0
        status = self.get_status(loss, loss_action, loss_visual_token_cosine)

        system_metrics = self.system_metrics
        self.system_metrics = {}

        # Fire to Trackers
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Epoch": self.epoch,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss Action": loss_action,
                f"{prefix}/Loss Visual Cosine": loss_visual_token_cosine,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
                **system_metrics,
            },
        )
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.finalize()

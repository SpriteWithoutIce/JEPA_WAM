"""
materialize.py

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from typing import Callable, Optional

import torch

from prismatic.models.vlms import PrismaticVLM
from prismatic.training.strategies import FSDPStrategy, TrainingStrategy

def get_fsdp_strategy(
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
) -> TrainingStrategy:
    return FSDPStrategy(
        vlm=vlm,
        device_id=device_id,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        enable_mixed_precision_training=enable_mixed_precision_training,
        reduce_in_full_precision=reduce_in_full_precision,
        mixed_precision_dtype=mixed_precision_dtype,
        worker_init_fn=worker_init_fn,
    )

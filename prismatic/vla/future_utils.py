"""
future_utils.py

Helpers for configuring and materializing future-frame supervision targets.
"""

from typing import Sequence, TypeVar

T = TypeVar("T")


def normalize_future_downsample_stride(stride: int) -> int:
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"future_obs_downsample_stride must be >= 1, got {stride}")
    return stride


def compute_downsampled_future_frame_count(window_size: int, downsample_stride: int) -> int:
    """Return how many future frames remain after uniform stride-based downsampling."""
    window_size = int(window_size)
    if window_size <= 0:
        return 0

    stride = normalize_future_downsample_stride(downsample_stride)
    return (window_size + stride - 1) // stride


def downsample_future_sequence(values: Sequence[T], downsample_stride: int) -> list[T]:
    """Keep every `downsample_stride`-th future frame, preserving order."""
    stride = normalize_future_downsample_stride(downsample_stride)
    if stride == 1:
        return list(values)
    return list(values[::stride])

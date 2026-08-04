"""Utilities for switching between axis-angle and 6D rotation representations."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import numpy as np

AXIS_ANGLE = "axis_angle"
ROT6D = "rot6d"
SUPPORTED_ROTATION_REPRESENTATIONS = (AXIS_ANGLE, ROT6D)
_EPS = 1e-8


def validate_rotation_representation(rotation_representation: str) -> str:
    rotation_representation = str(rotation_representation).lower()
    if rotation_representation not in SUPPORTED_ROTATION_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported rotation representation `{rotation_representation}`. "
            f"Choose from {SUPPORTED_ROTATION_REPRESENTATIONS}."
        )
    return rotation_representation


def is_libero_data_mix(data_mix: str) -> bool:
    return "libero" in str(data_mix).lower()


def resolve_vla_action_proprio_dims(
    data_mix: str,
    rotation_representation: str,
    *,
    default_action_dim: int = 7,
    default_proprio_dim: int = 8,
) -> Tuple[int, int]:
    rotation_representation = validate_rotation_representation(rotation_representation)
    if rotation_representation == AXIS_ANGLE:
        if is_libero_data_mix(data_mix):
            return 7, 8
        return default_action_dim, default_proprio_dim

    if not is_libero_data_mix(data_mix):
        raise ValueError(
            "`rot6d` is currently implemented only for LIBERO data/eval in this repository."
        )

    return 10, 11


def get_libero_eval_proprio_dim(rotation_representation: str) -> int:
    _, proprio_dim = resolve_vla_action_proprio_dims("libero", rotation_representation)
    return proprio_dim


def get_libero_eval_action_dim(rotation_representation: str) -> int:
    action_dim, _ = resolve_vla_action_proprio_dims("libero", rotation_representation)
    return action_dim


def standardize_fn_hash_key(standardize_fn: Any) -> str:
    if standardize_fn is None:
        return ""

    fn_obj = getattr(standardize_fn, "func", standardize_fn)
    payload: Dict[str, Any] = {
        "module": getattr(fn_obj, "__module__", ""),
        "qualname": getattr(fn_obj, "__qualname__", repr(fn_obj)),
    }

    args = getattr(standardize_fn, "args", ())
    keywords = getattr(standardize_fn, "keywords", None) or {}
    if args:
        payload["args"] = [repr(arg) for arg in args]
    if keywords:
        payload["keywords"] = {key: repr(value) for key, value in sorted(keywords.items())}

    return json.dumps(payload, sort_keys=True)


def _normalize_np(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    single = axis_angle.ndim == 1
    if single:
        axis_angle = axis_angle[None, :]

    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.maximum(theta, _EPS)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), k.shape)
    sin_theta = np.sin(theta)[..., None]
    cos_theta = np.cos(theta)[..., None]
    rot = eye + sin_theta * k + (1.0 - cos_theta) * np.matmul(k, k)

    small = (theta[..., 0] < _EPS)[..., None, None]
    rot = np.where(small, eye, rot)
    return rot[0] if single else rot


def quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    single = quat.ndim == 1
    if single:
        quat = quat[None, :]

    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), _EPS)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))
    return rot[0] if single else rot


def matrix_to_rot6d_np(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def axis_angle_to_rot6d_np(axis_angle: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d_np(axis_angle_to_matrix_np(axis_angle))


def quat_to_rot6d_np(quat: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d_np(quat_to_matrix_np(quat))


def rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., 0:5:2]
    a2 = rot6d[..., 1:6:2]
    b1 = _normalize_np(a1)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_np(a2 - proj)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def matrix_to_quat_np(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    single = matrix.ndim == 2
    if single:
        matrix = matrix[None, ...]

    quats = []
    for mat in matrix:
        trace = float(np.trace(mat))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (mat[2, 1] - mat[1, 2]) / s
            y = (mat[0, 2] - mat[2, 0]) / s
            z = (mat[1, 0] - mat[0, 1]) / s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w], dtype=np.float64)
        quat = quat / np.maximum(np.linalg.norm(quat), _EPS)
        quats.append(quat)

    quat_array = np.stack(quats, axis=0)
    return quat_array[0] if single else quat_array


def quat_to_axis_angle_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    single = quat.ndim == 1
    if single:
        quat = quat[None, :]

    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), _EPS)
    w = np.clip(quat[..., 3], -1.0, 1.0)
    den = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    angle = 2.0 * np.arccos(w)
    axis_angle = np.zeros(quat.shape[:-1] + (3,), dtype=np.float64)
    valid = den > _EPS
    axis_angle[valid] = quat[valid, :3] * (angle[valid, None] / den[valid, None])
    return axis_angle[0] if single else axis_angle


def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    return quat_to_axis_angle_np(matrix_to_quat_np(rot6d_to_matrix_np(rot6d)))


def axis_angle_to_rot6d_tf(axis_angle):
    import tensorflow as tf

    axis_angle = tf.cast(axis_angle, tf.float32)
    theta = tf.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_theta = tf.maximum(theta, tf.cast(_EPS, axis_angle.dtype))
    axis = axis_angle / safe_theta

    x, y, z = tf.unstack(axis, axis=-1)
    zeros = tf.zeros_like(x)
    k = tf.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=-1,
    )
    k = tf.reshape(k, tf.concat([tf.shape(axis)[:-1], tf.constant([3, 3], dtype=tf.int32)], axis=0))
    eye = tf.eye(3, batch_shape=tf.shape(axis)[:-1], dtype=axis_angle.dtype)
    sin_theta = tf.reshape(
        tf.sin(theta),
        tf.concat([tf.shape(theta)[:-1], tf.constant([1, 1], dtype=tf.int32)], axis=0),
    )
    cos_theta = tf.reshape(
        tf.cos(theta),
        tf.concat([tf.shape(theta)[:-1], tf.constant([1, 1], dtype=tf.int32)], axis=0),
    )
    rot = eye + sin_theta * k + (1.0 - cos_theta) * tf.linalg.matmul(k, k)

    small = tf.reshape(
        theta[..., 0] < tf.cast(_EPS, axis_angle.dtype),
        tf.concat([tf.shape(theta)[:-1], tf.constant([1, 1], dtype=tf.int32)], axis=0),
    )
    rot = tf.where(small, eye, rot)
    return tf.reshape(rot[..., :, :2], tf.concat([tf.shape(rot)[:-2], tf.constant([6], dtype=tf.int32)], axis=0))

"""Geometry helpers for Dataclean standard fields (rotvec / quaternion)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """(T, 4) xyzw → (T, 3) rotvec."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return Rotation.from_quat(q).as_rotvec()


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """(T, 4) xyzw → (T, 4) wxyz."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return np.concatenate([q[:, 3:4], q[:, 0:3]], axis=1)


def xyz_quat_xyzw_to_pose6(block7: np.ndarray) -> np.ndarray:
    """(T, 7) xyz+quat_xyzw → (T, 6) xyz+rotvec."""
    xyz = block7[:, 0:3]
    rotvec = quat_xyzw_to_rotvec(block7[:, 3:7])
    return np.concatenate([xyz, rotvec], axis=1).astype(np.float32)


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """(T, 6) → (T, 3) rotvec."""
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=-1)
    return Rotation.from_matrix(mat).as_rotvec().astype(np.float32)


def as_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

"""Shared helpers for egodex_lerobot_v21 manual_review scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(
    "/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_lerobot_v21"
)
# Local fallback used when production root is unavailable.
LOCAL_FALLBACK_DATASET = Path("/mnt/pfs/datasets/egodex/part1_lerobot")
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "manual_review" / "results"

CAMERA_KEY = "observation.images.camera_top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXTRINSICS_KEY = "observation.camera_extrinsics_world"


def default_dataset_root() -> Path:
    """Prefer config.yaml root; fall back to LOCAL_FALLBACK_DATASET."""
    cfg_path = PACKAGE_DIR / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            root = Path((cfg.get("dataset") or {}).get("root") or DEFAULT_DATASET)
            if root.exists():
                return root
        except Exception:
            pass
    if LOCAL_FALLBACK_DATASET.exists():
        return LOCAL_FALLBACK_DATASET
    return DEFAULT_DATASET


# Keep symbol for older scripts that import DEFAULT_DATASET as a Path constant.
if not DEFAULT_DATASET.exists() and LOCAL_FALLBACK_DATASET.exists():
    DEFAULT_DATASET = LOCAL_FALLBACK_DATASET

LEFT_GRIPPER_IDX = 9
RIGHT_GRIPPER_IDX = 19

STATE_ACTION_NAMES = [
    "left_px",
    "left_py",
    "left_pz",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_width",
    "right_px",
    "right_py",
    "right_pz",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_width",
]


def resolve_dataset_root(root: Path) -> Path:
    """Return a LeRobot task root (with meta/info.json).

    For part/task layout containers, pick the first task directory that has info.json.
    """
    root = Path(root)
    if (root / "meta" / "info.json").exists():
        return root
    for part in sorted(p for p in root.iterdir() if p.is_dir()):
        for task in sorted(p for p in part.iterdir() if p.is_dir()):
            if (task / "meta" / "info.json").exists():
                return task
    return root


def load_info(dataset_root: Path) -> dict[str, Any]:
    dataset_root = resolve_dataset_root(dataset_root)
    path = dataset_root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; for part/task layout pass a task dir, e.g. "
            f"{DEFAULT_DATASET}/part1/add_remove_lid"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def episode_parquet(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    dataset_root = resolve_dataset_root(dataset_root)
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    path = dataset_root / info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )
    if path.exists():
        return path
    matches = list(dataset_root.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    return path


def episode_video(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    dataset_root = resolve_dataset_root(dataset_root)
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return dataset_root / info["video_path"].format(
        episode_chunk=chunk, video_key=CAMERA_KEY, episode_index=episode_index
    )


def episode_paths(
    dataset_root: Path, info: dict[str, Any], episode_index: int
) -> tuple[Path, Path]:
    return episode_parquet(dataset_root, info, episode_index), episode_video(
        dataset_root, info, episode_index
    )


def array_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"{key!r} not in columns: {list(df.columns)}")
    values = df[key].to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float32)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float32)
    return values.astype(np.float32)[:, None]


def as_lr_gripper(arr: np.ndarray, *, path: Path | None = None) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] < 20:
        where = f" in {path}" if path else ""
        raise ValueError(f"expected (T,20) egodex vector, got {arr.shape}{where}")
    return np.stack([arr[:, LEFT_GRIPPER_IDX], arr[:, RIGHT_GRIPPER_IDX]], axis=1).astype(
        np.float32
    )


def load_grippers(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=[ACTION_KEY, STATE_KEY])
    action = as_lr_gripper(array_column(df, ACTION_KEY), path=parquet_path)
    state = as_lr_gripper(array_column(df, STATE_KEY), path=parquet_path)
    n = min(action.shape[0], state.shape[0])
    return action[:n], state[:n]


def split_hand_pose20(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right (T,7) xyz+quat_xyzw from egodex 20d vectors."""
    from scipy.spatial.transform import Rotation

    from robot_data_processing.transforms import rot6d_to_matrix

    if arr.ndim == 1:
        arr = arr[None, :]
    left_xyz = arr[:, 0:3]
    right_xyz = arr[:, 10:13]
    left_quat = Rotation.from_matrix(rot6d_to_matrix(arr[:, 3:9])).as_quat()
    right_quat = Rotation.from_matrix(rot6d_to_matrix(arr[:, 13:19])).as_quat()
    left = np.concatenate([left_xyz, left_quat], axis=1).astype(np.float32)
    right = np.concatenate([right_xyz, right_quat], axis=1).astype(np.float32)
    return left, right


def load_eef_poses_camera(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """EEF poses from observation.state (already in camera / aligned frame)."""
    df = pd.read_parquet(parquet_path, columns=[STATE_KEY])
    state = array_column(df, STATE_KEY)
    return split_hand_pose20(state)


def even_sample_indices(total: int, n: int) -> list[int]:
    n = min(n, total)
    if n == total:
        return list(range(total))
    raw = np.linspace(0, total - 1, n)
    idxs = np.unique(np.rint(raw).astype(int))
    if len(idxs) < n:
        unused = np.setdiff1d(np.arange(total), idxs, assume_unique=False)
        need = n - len(idxs)
        fill = unused[np.linspace(0, len(unused) - 1, need).astype(int)]
        idxs = np.sort(np.concatenate([idxs, fill]))
    return idxs.tolist()

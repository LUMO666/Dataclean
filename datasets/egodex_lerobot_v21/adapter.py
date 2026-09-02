"""EgoDex LeRobot v21 → Dataclean standard adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_data_processing.normalize.standard_types import EpisodeRef, StandardEpisode
from robot_data_processing.normalize.transforms_standard import as_col
from robot_data_processing.transforms import egodex_to_canonical


def _pick(raw: dict[str, np.ndarray], *keys: str) -> np.ndarray | None:
    for k in keys:
        if k in raw:
            return np.asarray(raw[k])
    return None


def _hand_blocks_from_20(x: np.ndarray, hand: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz Nx3, rot6d Nx6, grip Nx1) from packed 20d."""
    if hand == "left":
        xyz, rot6d, grip = x[:, 0:3], x[:, 3:9], x[:, 9:10]
    else:
        xyz, rot6d, grip = x[:, 10:13], x[:, 13:19], x[:, 19:20]
    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rot6d, dtype=np.float32),
        as_col(grip),
    )


def _load_intrinsics(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    arr = _pick(raw, "observation.camera_intrinsics", "camera_intrinsics")
    if arr is None:
        return {}
    mat = np.asarray(arr, dtype=np.float64).reshape(-1, 9)[0].reshape(3, 3)
    return {
        "camera_top.matrix": mat.tolist(),
        "camera_top.dist_coeffs": [0.0] * 5,
    }


def _load_extrinsics(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arr = _pick(raw, "observation.camera_extrinsics_world", "camera_extrinsics_world")
    if arr is None:
        return {}
    mats = np.asarray(arr, dtype=np.float32).reshape(-1, 4, 4)
    return {"observation.camera_extrinsics_world": mats}


class EgoDexAdapter:
    def __init__(self, package_dir: Path):
        self.package_dir = Path(package_dir)
        keymap_path = self.package_dir / "keymap.json"
        self.keymap = json.loads(keymap_path.read_text(encoding="utf-8")) if keymap_path.exists() else {}

    def camera_name_map(self) -> dict[str, str]:
        return {
            "observation.images.camera_top": "camera_top",
            "camera_top": "camera_top",
        }

    def to_standard_episode(
        self, ref: EpisodeRef, raw: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> StandardEpisode:
        state20 = np.asarray(raw["observation.state"], dtype=np.float64)
        action20 = np.asarray(raw.get("action", state20), dtype=np.float64)
        if state20.ndim != 2 or state20.shape[1] < 20:
            raise ValueError(f"observation.state must be Tx20, got {state20.shape}")
        if action20.ndim != 2 or action20.shape[1] < 20:
            raise ValueError(f"action must be Tx20, got {action20.shape}")
        T = int(state20.shape[0])

        s_l_xyz, s_l_rot, sg_l = _hand_blocks_from_20(state20, "left")
        s_r_xyz, s_r_rot, sg_r = _hand_blocks_from_20(state20, "right")
        a_l_xyz, a_l_rot, ag_l = _hand_blocks_from_20(action20, "left")
        a_r_xyz, a_r_rot, ag_r = _hand_blocks_from_20(action20, "right")

        fields: dict[str, np.ndarray] = {
            "observation.state.eef.left.position": s_l_xyz,
            "observation.state.eef.left.rotation_6d": s_l_rot,
            "observation.state.eef.right.position": s_r_xyz,
            "observation.state.eef.right.rotation_6d": s_r_rot,
            "observation.state.gripper.left.closedness": sg_l,
            "observation.state.gripper.right.closedness": sg_r,
            "action.eef.left.position": a_l_xyz,
            "action.eef.left.rotation_6d": a_l_rot,
            "action.eef.right.position": a_r_xyz,
            "action.eef.right.rotation_6d": a_r_rot,
            "action.gripper.left.closedness": ag_l,
            "action.gripper.right.closedness": ag_r,
        }
        if "observation.state.hand_features" in raw or "hand_features" in raw:
            hf = raw.get("observation.state.hand_features", raw.get("hand_features"))
            if hf is not None:
                fields["observation.state.hand_features"] = np.asarray(hf, dtype=np.float32)

        intrinsic = _load_intrinsics(raw)
        extrinsic = _load_extrinsics(raw)
        ep_meta = {
            **meta,
            "tasks": meta.get("tasks") or ([ref.task] if ref.task else []),
            "source_episode_index": ref.episode_index,
            "calibration_loaded": bool(intrinsic) or bool(extrinsic),
        }
        return StandardEpisode(
            episode_index=ref.episode_index,
            num_frames=T,
            fields=fields,
            camera_keys=["observation.images.camera_top"],
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            meta=ep_meta,
            has_camera_top=True,
        )

    def to_quality_arrays(self, standard: StandardEpisode) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild EgoDex 20d then canonical 14d (xyz+rpy+gripper) for quality stages."""
        T = standard.num_frames
        packed = np.zeros((T, 20), dtype=np.float64)

        def _fill(prefix: str, offset: int) -> None:
            pos = standard.fields.get(f"{prefix}.position")
            rot = standard.fields.get(f"{prefix}.rotation_6d")
            if pos is None or rot is None:
                return
            packed[:, offset : offset + 3] = np.asarray(pos, dtype=np.float64)
            packed[:, offset + 3 : offset + 9] = np.asarray(rot, dtype=np.float64)

        _fill("observation.state.eef.left", 0)
        g = standard.fields.get("observation.state.gripper.left.closedness")
        if g is not None:
            packed[:, 9] = np.asarray(g).reshape(-1)
        _fill("observation.state.eef.right", 10)
        g = standard.fields.get("observation.state.gripper.right.closedness")
        if g is not None:
            packed[:, 19] = np.asarray(g).reshape(-1)

        state14 = egodex_to_canonical(packed)

        packed_a = np.zeros((T, 20), dtype=np.float64)

        def _fill_a(prefix: str, offset: int) -> None:
            pos = standard.fields.get(f"{prefix}.position")
            rot = standard.fields.get(f"{prefix}.rotation_6d")
            if pos is None or rot is None:
                return
            packed_a[:, offset : offset + 3] = np.asarray(pos, dtype=np.float64)
            packed_a[:, offset + 3 : offset + 9] = np.asarray(rot, dtype=np.float64)

        _fill_a("action.eef.left", 0)
        g = standard.fields.get("action.gripper.left.closedness")
        if g is not None:
            packed_a[:, 9] = np.asarray(g).reshape(-1)
        _fill_a("action.eef.right", 10)
        g = standard.fields.get("action.gripper.right.closedness")
        if g is not None:
            packed_a[:, 19] = np.asarray(g).reshape(-1)
        action14 = egodex_to_canonical(packed_a)
        return state14, action14


def build_adapter(package_dir: Path) -> EgoDexAdapter:
    return EgoDexAdapter(package_dir)

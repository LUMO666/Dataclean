"""Humanoid merged → Dataclean standard adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_data_processing.normalize.calibration import load_mapped_calibration
from robot_data_processing.normalize.standard_types import EpisodeRef, StandardEpisode
from robot_data_processing.normalize.transforms_standard import (
    as_col,
    rotation_6d_to_matrices,
    xyz_quat_xyzw_to_position_rotation_6d,
)


def _pick(raw: dict[str, np.ndarray], *keys: str) -> np.ndarray | None:
    for k in keys:
        if k in raw:
            return np.asarray(raw[k])
    return None


class HumanoidAdapter:
    def __init__(self, package_dir: Path):
        self.package_dir = Path(package_dir)
        self.keymap = json.loads((self.package_dir / "keymap.json").read_text(encoding="utf-8"))

    def camera_name_map(self) -> dict[str, str]:
        # source video folder key → standard camera short name
        return {
            "observation.images.camera_top": "camera_top",
            "observation.images.camera_wrist_left": "camera_wrist_left",
            "observation.images.camera_wrist_right": "camera_wrist_right",
        }

    def to_standard_episode(
        self, ref: EpisodeRef, raw: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> StandardEpisode:
        end = _pick(raw, "observation.state.end.position")
        if end is None:
            state = _pick(raw, "observation.state")
            if state is None or state.shape[1] < 28:
                raise ValueError("missing observation.state.end.position / observation.state")
            end = state[:, 14:28]

        T = end.shape[0]
        arm_act = _pick(raw, "action.arm.position", "action")
        grip_act = _pick(raw, "action.effector.position")
        if arm_act is not None and arm_act.shape[1] >= 14 and grip_act is None:
            # packed action: arm(12)+gripper(2)
            grip_act = arm_act[:, 12:14]
            arm_act = arm_act[:, 0:12]
        elif arm_act is not None and arm_act.shape[1] >= 12:
            arm_act = arm_act[:, 0:12]

        arm_obs = _pick(raw, "observation.state.arm.position")
        grip_obs = _pick(raw, "observation.state.effector.position")
        if arm_obs is None:
            state = _pick(raw, "observation.state")
            if state is not None and state.shape[1] >= 14:
                arm_obs = state[:, 0:12]
                grip_obs = state[:, 12:14] if grip_obs is None else grip_obs

        left_pos, left_rot6d = xyz_quat_xyzw_to_position_rotation_6d(end[:, 0:7])
        right_pos, right_rot6d = xyz_quat_xyzw_to_position_rotation_6d(end[:, 7:14])
        fields: dict[str, np.ndarray] = {
            "observation.state.eef.left.position": left_pos,
            "observation.state.eef.left.rotation_6d": left_rot6d,
            "observation.state.eef.right.position": right_pos,
            "observation.state.eef.right.rotation_6d": right_rot6d,
        }
        # action.eef is intentionally omitted when source has no separate EE action;
        # Stage2 fill_missing_action_eef will set action.eef[t]=state.eef[t+global_lag].
        if arm_act is not None:
            fields["action.arm.left.joint_position"] = arm_act[:, 0:6].astype(np.float32)
            fields["action.arm.right.joint_position"] = arm_act[:, 6:12].astype(np.float32)
        if grip_act is not None:
            fields["action.gripper.left.closedness"] = as_col(grip_act[:, 0])
            fields["action.gripper.right.closedness"] = as_col(grip_act[:, 1])
        if arm_obs is not None:
            fields["observation.state.arm.left.joint_position"] = arm_obs[:, 0:6].astype(np.float32)
            fields["observation.state.arm.right.joint_position"] = arm_obs[:, 6:12].astype(np.float32)
        if grip_obs is not None:
            fields["observation.state.gripper.left.closedness"] = as_col(grip_obs[:, 0])
            fields["observation.state.gripper.right.closedness"] = as_col(grip_obs[:, 1])

        cams = [
            "observation.images.camera_top",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        ]
        # Extrinsics / intrinsices live in parameters/.../calibration_bundle_optimized.json
        # (not parquet columns); map via keymap.json.
        intrinsic, extrinsic, calib_path = load_mapped_calibration(
            ref.dataset_root, ref.episode_index, self.keymap
        )
        ep_meta = {
            **meta,
            "tasks": meta.get("tasks") or [],
            "source_episode_index": ref.episode_index,
            "calibration_bundle": str(calib_path) if calib_path is not None else None,
            "calibration_loaded": bool(extrinsic) or bool(intrinsic),
        }
        return StandardEpisode(
            episode_index=ref.episode_index,
            num_frames=T,
            fields=fields,
            camera_keys=cams,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            meta=ep_meta,
            has_camera_top=True,
        )

    def to_quality_arrays(self, standard: StandardEpisode) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild approximate humanoid 28d state / 14d action for quality stages."""
        T = standard.num_frames
        state = np.zeros((T, 28), dtype=np.float64)
        action = np.zeros((T, 14), dtype=np.float64)
        la = standard.fields.get("observation.state.arm.left.joint_position")
        ra = standard.fields.get("observation.state.arm.right.joint_position")
        lg = standard.fields.get("observation.state.gripper.left.closedness")
        rg = standard.fields.get("observation.state.gripper.right.closedness")
        le_pos = standard.fields.get("observation.state.eef.left.position")
        le_rot = standard.fields.get("observation.state.eef.left.rotation_6d")
        re_pos = standard.fields.get("observation.state.eef.right.position")
        re_rot = standard.fields.get("observation.state.eef.right.rotation_6d")
        if la is not None:
            state[:, 0:6] = la
        if ra is not None:
            state[:, 6:12] = ra
        if lg is not None:
            state[:, 12] = lg.reshape(-1)
        if rg is not None:
            state[:, 13] = rg.reshape(-1)
        if le_pos is not None:
            state[:, 14:17] = le_pos
        if le_rot is not None:
            state[:, 17:21] = Rotation.from_matrix(
                rotation_6d_to_matrices(le_rot, name="left EEF")
            ).as_quat()
        if re_pos is not None:
            state[:, 21:24] = re_pos
        if re_rot is not None:
            state[:, 24:28] = Rotation.from_matrix(
                rotation_6d_to_matrices(re_rot, name="right EEF")
            ).as_quat()
        aa = standard.fields.get("action.arm.left.joint_position")
        ab = standard.fields.get("action.arm.right.joint_position")
        ag = standard.fields.get("action.gripper.left.closedness")
        ah = standard.fields.get("action.gripper.right.closedness")
        if aa is not None:
            action[:, 0:6] = aa
        if ab is not None:
            action[:, 6:12] = ab
        if ag is not None:
            action[:, 12] = ag.reshape(-1)
        if ah is not None:
            action[:, 13] = ah.reshape(-1)
        return state, action


def build_adapter(package_dir: Path) -> HumanoidAdapter:
    return HumanoidAdapter(package_dir)

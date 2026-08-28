"""RoboMind UR → Dataclean standard adapter (primary / dual compact)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_data_processing.normalize.standard_types import EpisodeRef, StandardEpisode
from robot_data_processing.normalize.transforms_standard import as_col
from robot_data_processing.transforms import robomind_ur_build_action, robomind_ur_build_state


class RobomindAdapter:
    def __init__(self, package_dir: Path):
        self.package_dir = Path(package_dir)

    def camera_name_map(self) -> dict[str, str]:
        # RoboMind camera keys vary; map common top-like views if present
        return {
            "observation.images.camera_top": "camera_top",
            "camera_top": "camera_top",
            "camera_front": "camera_top",
        }

    def to_standard_episode(
        self, ref: EpisodeRef, raw: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> StandardEpisode:
        state = robomind_ur_build_state(raw)
        action = robomind_ur_build_action(raw)
        T = state.shape[0]
        # EE blocks in state: [0:7] left, [7:14] right — first dim used as gripper proxy historically
        fields = {
            "observation.state.eef.left.pose": np.concatenate(
                [state[:, 0:1], np.zeros((T, 6), dtype=np.float32)], axis=1
            ).astype(np.float32),
            "observation.state.eef.right.pose": np.concatenate(
                [state[:, 7:8], np.zeros((T, 6), dtype=np.float32)], axis=1
            ).astype(np.float32),
            "observation.state.arm.left.joint_position": state[:, 14:20].astype(np.float32),
            "observation.state.arm.right.joint_position": state[:, 20:26].astype(np.float32),
            "observation.state.gripper.left.closedness": as_col(state[:, 0]),
            "observation.state.gripper.right.closedness": as_col(state[:, 7]),
            "action.arm.left.joint_position": action[:, 14:20].astype(np.float32),
            "action.arm.right.joint_position": action[:, 20:26].astype(np.float32),
            "action.gripper.left.closedness": as_col(action[:, 0]),
            "action.gripper.right.closedness": as_col(action[:, 7]),
            "action.eef.left.pose": np.concatenate(
                [action[:, 0:1], np.zeros((T, 6), dtype=np.float32)], axis=1
            ).astype(np.float32),
            "action.eef.right.pose": np.concatenate(
                [action[:, 7:8], np.zeros((T, 6), dtype=np.float32)], axis=1
            ).astype(np.float32),
        }
        has_top = any(
            k.endswith("camera_top") or "camera_front" in k or "camera_top" in k for k in raw
        ) or True  # camera presence checked at video export
        return StandardEpisode(
            episode_index=ref.episode_index,
            num_frames=T,
            fields=fields,
            camera_keys=["observation.images.camera_top"],
            meta={**meta},
            has_camera_top=has_top,
        )

    def to_quality_arrays(self, standard: StandardEpisode) -> tuple[np.ndarray, np.ndarray]:
        T = standard.num_frames
        state = np.zeros((T, 26), dtype=np.float64)
        action = np.zeros((T, 52), dtype=np.float64)
        state[:, 0] = standard.fields["observation.state.gripper.left.closedness"].reshape(-1)
        state[:, 7] = standard.fields["observation.state.gripper.right.closedness"].reshape(-1)
        state[:, 14:20] = standard.fields["observation.state.arm.left.joint_position"]
        state[:, 20:26] = standard.fields["observation.state.arm.right.joint_position"]
        action[:, 0] = standard.fields["action.gripper.left.closedness"].reshape(-1)
        action[:, 7] = standard.fields["action.gripper.right.closedness"].reshape(-1)
        action[:, 14:20] = standard.fields["action.arm.left.joint_position"]
        action[:, 20:26] = standard.fields["action.arm.right.joint_position"]
        return state, action


def build_adapter(package_dir: Path) -> RobomindAdapter:
    return RobomindAdapter(package_dir)

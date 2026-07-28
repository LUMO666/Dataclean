"""EgoDex LeRobot v21 → Dataclean standard adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_data_processing.normalize.standard_types import EpisodeRef, StandardEpisode
from robot_data_processing.normalize.transforms_standard import as_col, rot6d_to_rotvec


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
        T = state20.shape[0]

        def pose_from_20(x: np.ndarray, hand: str) -> np.ndarray:
            if hand == "left":
                xyz, rot6d, grip = x[:, 0:3], x[:, 3:9], x[:, 9:10]
            else:
                xyz, rot6d, grip = x[:, 10:13], x[:, 13:19], x[:, 19:20]
            rotvec = rot6d_to_rotvec(rot6d)
            return np.concatenate([xyz, rotvec], axis=1).astype(np.float32), as_col(grip)

        a_l, ag_l = pose_from_20(action20, "left")
        a_r, ag_r = pose_from_20(action20, "right")
        s_l, sg_l = pose_from_20(state20, "left")
        s_r, sg_r = pose_from_20(state20, "right")

        fields = {
            "action.eef.left.pose": a_l,
            "action.eef.right.pose": a_r,
            "action.gripper.left.closedness": ag_l,
            "action.gripper.right.closedness": ag_r,
            "observation.state.eef.left.pose": s_l,
            "observation.state.eef.right.pose": s_r,
            "observation.state.gripper.left.closedness": sg_l,
            "observation.state.gripper.right.closedness": sg_r,
        }
        if "observation.state.hand_features" in raw or "hand_features" in raw:
            hf = raw.get("observation.state.hand_features", raw.get("hand_features"))
            if hf is not None:
                fields["observation.state.hand_features"] = np.asarray(hf, dtype=np.float32)

        return StandardEpisode(
            episode_index=ref.episode_index,
            num_frames=T,
            fields=fields,
            camera_keys=["observation.images.camera_top"],
            meta={**meta, "tasks": [ref.task] if ref.task else []},
            has_camera_top=True,
        )

    def to_quality_arrays(self, standard: StandardEpisode) -> tuple[np.ndarray, np.ndarray]:
        # Reconstruct approximate 20d then canonical 14d is done by engine loader normally.
        # Here build 14d pose_gripper layout from standard fields.
        T = standard.num_frames
        state = np.zeros((T, 14), dtype=np.float64)
        action = np.zeros((T, 14), dtype=np.float64)
        for arr, pose_l, pose_r, g_l, g_r in (
            (
                state,
                "observation.state.eef.left.pose",
                "observation.state.eef.right.pose",
                "observation.state.gripper.left.closedness",
                "observation.state.gripper.right.closedness",
            ),
            (
                action,
                "action.eef.left.pose",
                "action.eef.right.pose",
                "action.gripper.left.closedness",
                "action.gripper.right.closedness",
            ),
        ):
            pl = standard.fields[pose_l]
            pr = standard.fields[pose_r]
            # store xyz + rotvec as xyz+rpy slots for quality (approximation)
            arr[:, 0:6] = pl
            arr[:, 6] = standard.fields[g_l].reshape(-1)
            arr[:, 7:13] = pr
            arr[:, 13] = standard.fields[g_r].reshape(-1)
        return state, action


def build_adapter(package_dir: Path) -> EgoDexAdapter:
    return EgoDexAdapter(package_dir)

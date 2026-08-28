"""Humanoid episode-frame geometry (P6): arm-local → camera_reference @ frame 0."""
from __future__ import annotations

from typing import Any

import numpy as np

from robot_data_processing.normalize.episode_frame_geometry import (
    build_scaled_intrinsics_record,
    invert_se3,
    target_short_side_size,
)
from robot_data_processing.normalize.standard_types import (
    EpisodeGeometryResult,
    EpisodeRef,
    StandardEpisode,
)
from robot_data_processing.normalize.transforms_standard import (
    matrices_to_position_rotation_6d,
    position_rotation_6d_to_matrices,
)

SCHEMA_VERSION = "episode_camera_geometry_v1"
EPISODE_FRAME_DEFINITION = "camera_reference at frame_index=0"

NEW_EXTRINSIC_KEYS = {
    "reference": "extrinsic.camera_reference.T_Episode_CameraReference",
    "aux_0": "extrinsic.camera_aux_0.T_Episode_CameraAux0",
    "aux_1": "extrinsic.camera_aux_1.T_Episode_CameraAux1",
}

DEFAULT_SOURCE_CALIBRATION_SIZE = (1280, 720)
DEFAULT_VIDEO_SHORT_SIDE = 384


def _strip_intrinsic_prefix(intrinsic: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in intrinsic.items():
        short = str(key)
        if short.startswith("intrinsic."):
            short = short[len("intrinsic.") :]
        out[short] = val
    return out


def _require_eef(fields: dict[str, np.ndarray], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    pos_key = f"{prefix}.position"
    rot_key = f"{prefix}.rotation_6d"
    if pos_key not in fields or rot_key not in fields:
        raise KeyError(f"Missing required fields {pos_key} / {rot_key}")
    position = np.asarray(fields[pos_key], dtype=np.float64)
    rotation_6d = np.asarray(fields[rot_key], dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"{pos_key} must be Nx3, got {position.shape}")
    if rotation_6d.ndim != 2 or rotation_6d.shape[1] != 6:
        raise ValueError(f"{rot_key} must be Nx6, got {rotation_6d.shape}")
    return position, rotation_6d


def _write_eef_from_matrices(
    out_fields: dict[str, np.ndarray],
    prefix: str,
    matrices: np.ndarray,
    *,
    name: str,
) -> None:
    position, rotation_6d = matrices_to_position_rotation_6d(matrices, name=name)
    out_fields[f"{prefix}.position"] = position
    out_fields[f"{prefix}.rotation_6d"] = rotation_6d
    out_fields.pop(f"{prefix}.pose", None)


class HumanoidEpisodeGeometry:
    def __init__(self, package_dir):
        self.package_dir = package_dir

    def apply_episode_frame_geometry(
        self,
        standard: StandardEpisode,
        *,
        ref: EpisodeRef,
        geometry_cfg: dict[str, Any],
        keymap: dict[str, Any],
        stored_video_size: tuple[int, int] | None = None,
    ) -> EpisodeGeometryResult:
        del keymap
        cameras = geometry_cfg.get("cameras") or {}
        mapping = dict(cameras.get("mapping") or {})
        wrist_view = list(cameras.get("wrist_view") or ["camera_aux_0", "camera_aux_1"])
        extr_cfg = geometry_cfg.get("extrinsics") or {}
        short_side = int(geometry_cfg.get("video_short_side", DEFAULT_VIDEO_SHORT_SIDE))
        src_calib = geometry_cfg.get("source_calibration_size", list(DEFAULT_SOURCE_CALIBRATION_SIZE))
        src_w, src_h = int(src_calib[0]), int(src_calib[1])
        if stored_video_size is None:
            stored_w, stored_h = target_short_side_size(src_w, src_h, short_side=short_side)
        else:
            stored_w, stored_h = stored_video_size

        ext = standard.extrinsic
        ref_left_key = extr_cfg.get("reference_to_arm", {}).get("left", "camera_top.T_ArmLeft_CameraTop")
        ref_right_key = extr_cfg.get("reference_to_arm", {}).get(
            "right", "camera_top.T_ArmRight_CameraTop"
        )
        hand_eye = extr_cfg.get("hand_eye") or {}
        aux0_key = hand_eye.get("camera_aux_0", "camera_top.T_ArmLeft_CameraWristLeft")
        aux1_key = hand_eye.get("camera_aux_1", "camera_top.T_ArmRight_CameraWristRight")

        missing = [
            k
            for k in (ref_left_key, ref_right_key, aux0_key, aux1_key)
            if k not in ext
        ]
        if missing:
            return EpisodeGeometryResult(
                fields={},
                extrinsic={},
                camera_keys=[],
                camera_mapping=mapping,
                camera_intrinsics={},
                video_export_map={},
                episode_frame_definition=EPISODE_FRAME_DEFINITION,
                discard=True,
                discard_reasons=[f"missing_extrinsic:{','.join(missing)}"],
            )

        T_arm_left_ref = np.asarray(ext[ref_left_key], dtype=np.float64)
        T_arm_right_ref = np.asarray(ext[ref_right_key], dtype=np.float64)
        T_arm_left_aux0 = np.asarray(ext[aux0_key], dtype=np.float64)
        T_arm_right_aux1 = np.asarray(ext[aux1_key], dtype=np.float64)

        T_episode_arm_left = invert_se3(T_arm_left_ref)
        T_episode_arm_right = invert_se3(T_arm_right_ref)

        rows = standard.num_frames
        state_left_pos, state_left_rot = _require_eef(
            standard.fields, "observation.state.eef.left"
        )
        state_right_pos, state_right_rot = _require_eef(
            standard.fields, "observation.state.eef.right"
        )

        state_left_mat = np.einsum(
            "ij,njk->nik",
            T_episode_arm_left,
            position_rotation_6d_to_matrices(
                state_left_pos, state_left_rot, name="left EEF state"
            ),
        )
        state_right_mat = np.einsum(
            "ij,njk->nik",
            T_episode_arm_right,
            position_rotation_6d_to_matrices(
                state_right_pos, state_right_rot, name="right EEF state"
            ),
        )

        reference = np.broadcast_to(np.eye(4, dtype=np.float64), (rows, 4, 4)).copy()
        aux_0 = state_left_mat @ T_arm_left_aux0
        aux_1 = state_right_mat @ T_arm_right_aux1

        out_fields = dict(standard.fields)
        for legacy_key in list(out_fields):
            if legacy_key.endswith(".pose") or legacy_key.startswith("observation.geometry.") or legacy_key.startswith(
                "action.geometry."
            ):
                del out_fields[legacy_key]

        _write_eef_from_matrices(
            out_fields,
            "observation.state.eef.left",
            state_left_mat,
            name="left EEF state",
        )
        _write_eef_from_matrices(
            out_fields,
            "observation.state.eef.right",
            state_right_mat,
            name="right EEF state",
        )

        for side, prefix, T_e in (
            ("left", "action.eef.left", T_episode_arm_left),
            ("right", "action.eef.right", T_episode_arm_right),
        ):
            pos_key = f"{prefix}.position"
            rot_key = f"{prefix}.rotation_6d"
            if pos_key in standard.fields and rot_key in standard.fields:
                pos, rot = _require_eef(standard.fields, prefix)
                action_mat = np.einsum(
                    "ij,njk->nik",
                    T_e,
                    position_rotation_6d_to_matrices(pos, rot, name=prefix),
                )
                _write_eef_from_matrices(out_fields, prefix, action_mat, name=prefix)

        out_extrinsic = {
            NEW_EXTRINSIC_KEYS["reference"]: reference.astype(np.float32),
            NEW_EXTRINSIC_KEYS["aux_0"]: aux_0.astype(np.float32),
            NEW_EXTRINSIC_KEYS["aux_1"]: aux_1.astype(np.float32),
        }

        src_cam = cameras.get("reference", "observation.images.camera_top")
        aux_src = list(
            cameras.get("aux")
            or [
                "observation.images.camera_wrist_left",
                "observation.images.camera_wrist_right",
            ]
        )
        video_export_map = {src_cam: "camera_reference"}
        for i, aux_key in enumerate(aux_src):
            video_export_map[aux_key] = f"camera_aux_{i}"

        camera_keys = [
            "observation.images.camera_reference",
            "observation.images.camera_aux_0",
            "observation.images.camera_aux_1",
        ]

        intrinsic_short = _strip_intrinsic_prefix(standard.intrinsic)
        calib_intrinsic_keys = {
            "camera_reference": "camera_top",
            "camera_aux_0": "camera_wrist_left",
            "camera_aux_1": "camera_wrist_right",
        }
        camera_intrinsics = build_scaled_intrinsics_record(
            intrinsic_short,
            output_cameras=calib_intrinsic_keys,
            source_calibration_size=(src_w, src_h),
            stored_size=(stored_w, stored_h),
        )

        ref_err = float(np.max(np.abs(reference - np.eye(4))))
        geometry_meta = {
            "schema_version": SCHEMA_VERSION,
            "episode_index": ref.episode_index,
            "rows": rows,
            "reference_identity_max_abs_error": ref_err,
            "stored_video_size": {"width": stored_w, "height": stored_h},
        }

        info_extras = {
            "camera_geometry": {
                "schema_version": SCHEMA_VERSION,
                "episode_frame": EPISODE_FRAME_DEFINITION,
                "transform_convention": "T_A_B maps coordinates from B to A",
                "translation_unit": "meter",
                "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
                "intrinsic_storage": "meta/episodes.jsonl::camera_intrinsics",
                "per_frame_extrinsic_keys": list(NEW_EXTRINSIC_KEYS.values()),
                "eef_position_format": "absolute xyz in meters",
                "eef_rotation_format": "absolute Rotation 6D, first two matrix rows flattened",
            },
            "camera_mapping": mapping,
            "action_target": geometry_cfg.get("action") or {},
        }

        return EpisodeGeometryResult(
            fields=out_fields,
            extrinsic=out_extrinsic,
            camera_keys=camera_keys,
            camera_mapping=mapping,
            camera_intrinsics=camera_intrinsics,
            video_export_map=video_export_map,
            episode_frame_definition=EPISODE_FRAME_DEFINITION,
            geometry_meta=geometry_meta,
            wrist_view_cameras=wrist_view,
            info_extras=info_extras,
        )


def build_episode_geometry(package_dir):
    return HumanoidEpisodeGeometry(package_dir)

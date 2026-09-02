"""EgoDex episode-frame geometry (P6): single head camera as camera_reference."""
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
REFERENCE_EXTRINSIC_KEY = "extrinsic.camera_reference.T_Episode_CameraReference"

DEFAULT_SOURCE_CALIBRATION_SIZE = (1920, 1080)
DEFAULT_VIDEO_SHORT_SIDE = 384
DEFAULT_EXTRINSIC_SOURCE = "observation.camera_extrinsics_world"


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


def _strip_legacy_pose_keys(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(fields)
    for key in list(out):
        if key.endswith(".pose") or key.startswith("observation.geometry.") or key.startswith(
            "action.geometry."
        ):
            del out[key]
    return out


class EgoDexEpisodeGeometry:
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
        mapping = dict(cameras.get("mapping") or {"camera_reference": "camera_top"})
        wrist_view = list(cameras.get("wrist_view") or [])
        extr_cfg = geometry_cfg.get("extrinsics") or {}
        short_side = int(geometry_cfg.get("video_short_side", DEFAULT_VIDEO_SHORT_SIDE))
        src_calib = geometry_cfg.get(
            "source_calibration_size", list(DEFAULT_SOURCE_CALIBRATION_SIZE)
        )
        src_w, src_h = int(src_calib[0]), int(src_calib[1])
        if stored_video_size is None:
            stored_w, stored_h = target_short_side_size(src_w, src_h, short_side=short_side)
        else:
            stored_w, stored_h = stored_video_size

        ext_key = str(extr_cfg.get("source", DEFAULT_EXTRINSIC_SOURCE))
        ext = standard.extrinsic or {}
        if ext_key not in ext:
            return EpisodeGeometryResult(
                fields={},
                extrinsic={},
                camera_keys=[],
                camera_mapping=mapping,
                camera_intrinsics={},
                video_export_map={},
                episode_frame_definition=EPISODE_FRAME_DEFINITION,
                discard=True,
                discard_reasons=[f"missing_extrinsic:{ext_key}"],
            )

        T_world_cam = np.asarray(ext[ext_key], dtype=np.float64).reshape(-1, 4, 4)
        rows = int(standard.num_frames)
        if T_world_cam.shape[0] == 1:
            T_world_cam = np.broadcast_to(T_world_cam[0], (rows, 4, 4)).copy()
        elif T_world_cam.shape[0] != rows:
            return EpisodeGeometryResult(
                fields={},
                extrinsic={},
                camera_keys=[],
                camera_mapping=mapping,
                camera_intrinsics={},
                video_export_map={},
                episode_frame_definition=EPISODE_FRAME_DEFINITION,
                discard=True,
                discard_reasons=[
                    f"extrinsic_length_mismatch:{T_world_cam.shape[0]}!={rows}"
                ],
            )

        T_world_cam0 = T_world_cam[0]
        T_episode_world = invert_se3(T_world_cam0)
        reference = np.einsum("ij,njk->nik", T_episode_world, T_world_cam)

        poses_in_episode = bool(geometry_cfg.get("poses_in_episode_frame", True))
        out_fields = _strip_legacy_pose_keys(standard.fields)

        for prefix in (
            "observation.state.eef.left",
            "observation.state.eef.right",
            "action.eef.left",
            "action.eef.right",
        ):
            if f"{prefix}.position" not in standard.fields:
                continue
            pos, rot = _require_eef(standard.fields, prefix)
            if poses_in_episode:
                out_fields[f"{prefix}.position"] = pos.astype(np.float32)
                out_fields[f"{prefix}.rotation_6d"] = rot.astype(np.float32)
                out_fields.pop(f"{prefix}.pose", None)
            else:
                mats = np.einsum(
                    "ij,njk->nik",
                    T_episode_world,
                    position_rotation_6d_to_matrices(pos, rot, name=prefix),
                )
                _write_eef_from_matrices(out_fields, prefix, mats, name=prefix)

        out_extrinsic = {
            REFERENCE_EXTRINSIC_KEY: reference.astype(np.float32),
        }

        src_cam = cameras.get("reference", "observation.images.camera_top")
        video_export_map = {src_cam: "camera_reference"}
        camera_keys = ["observation.images.camera_reference"]

        intrinsic_short = _strip_intrinsic_prefix(standard.intrinsic)
        camera_intrinsics = build_scaled_intrinsics_record(
            intrinsic_short,
            output_cameras={"camera_reference": "camera_top"},
            source_calibration_size=(src_w, src_h),
            stored_size=(stored_w, stored_h),
        )

        ref_err = float(np.max(np.abs(reference[0] - np.eye(4))))
        geometry_meta = {
            "schema_version": SCHEMA_VERSION,
            "episode_index": ref.episode_index,
            "rows": rows,
            "poses_in_episode_frame": poses_in_episode,
            "reference_identity_max_abs_error": ref_err,
            "stored_video_size": {"width": stored_w, "height": stored_h},
            "extrinsic_source": ext_key,
        }

        info_extras = {
            "camera_geometry": {
                "schema_version": SCHEMA_VERSION,
                "episode_frame": EPISODE_FRAME_DEFINITION,
                "transform_convention": "T_A_B maps coordinates from B to A",
                "translation_unit": "meter",
                "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
                "intrinsic_storage": "meta/episodes.jsonl::camera_intrinsics",
                "per_frame_extrinsic_keys": [REFERENCE_EXTRINSIC_KEY],
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
    return EgoDexEpisodeGeometry(package_dir)

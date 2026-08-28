"""Stub episode-frame geometry module (P6).

Copy to ``datasets/<your_dataset>/episode_geometry.py`` and implement
``apply_episode_frame_geometry`` for your calibration / camera layout.

Engine helpers: ``robot_data_processing.normalize.episode_frame_geometry``
"""
from __future__ import annotations

from typing import Any

from robot_data_processing.normalize.standard_types import (
    EpisodeGeometryResult,
    EpisodeRef,
    StandardEpisode,
)


class TemplateEpisodeGeometry:
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
        del stored_video_size, keymap, ref
        if not geometry_cfg.get("enabled", False):
            cameras = geometry_cfg.get("cameras") or {}
            mapping = dict(cameras.get("mapping") or {})
            return EpisodeGeometryResult(
                fields=dict(standard.fields),
                extrinsic=dict(standard.extrinsic),
                camera_keys=list(standard.camera_keys),
                camera_mapping=mapping,
                camera_intrinsics={},
                video_export_map={},
                episode_frame_definition=str(
                    geometry_cfg.get("episode_frame", "passthrough")
                ),
            )
        raise NotImplementedError(
            "Implement apply_episode_frame_geometry for this dataset "
            "(see humanoid_merged/episode_geometry.py)."
        )


def build_episode_geometry(package_dir):
    return TemplateEpisodeGeometry(package_dir)

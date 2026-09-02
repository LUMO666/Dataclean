"""P7 StandardExport: write Dataclean-standard LeRobot dataset."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_processing.normalize.standard_types import StandardEpisode
from robot_data_processing.frame_alignment import (
    check_export_frame_alignment,
    export_frame_indices,
    probe_video_frames,
)
from robot_data_processing.phases.media import export_normalized_videos
from robot_data_processing.types import EpisodeResult

_EXPORT_POOL_STATE: dict[str, Any] = {}


def load_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # Strip // line comments (standard_info.json uses them)
    cleaned = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return json.loads(cleaned)


def _feature_spec(shape: list[int], names: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": "float32", "shape": shape}
    if names:
        out["names"] = names
    return out


def build_info_features(fields_present: set[str], camera_keys: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in sorted(fields_present):
        # shape inferred at write time; placeholder updated per column
        features[name] = {"dtype": "float32", "shape": [1]}
    for cam in camera_keys:
        key = cam if cam.startswith("observation.images.") else f"observation.images.{cam}"
        features[key] = _video_feature_spec(384, 384)
    for idx_name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[idx_name] = {
            "dtype": "float32" if idx_name == "timestamp" else "int64",
            "shape": [1],
        }
    return features


def _video_feature_spec(
    height: int,
    width: int,
    *,
    fps: float = 30.0,
) -> dict[str, Any]:
    """LeRobot / standard_info video feature: shape [H, W, C]."""
    return {
        "dtype": "video",
        "shape": [int(height), int(width), 3],
        "names": ["height", "width", "channels"],
        "info": {
            "has_audio": False,
            "video.channels": 3,
            "video.codec": "h264",
            "video.fps": float(fps),
            "video.height": int(height),
            "video.is_depth_map": False,
            "video.pix_fmt": "yuv420p",
            "video.width": int(width),
        },
    }


def _camera_short_name(cam_name: str) -> str:
    return cam_name.split(".")[-1] if "." in cam_name else cam_name


def _apply_wrist_view_flags(
    features: dict[str, Any],
    camera_sizes: dict[str, tuple[int, int]],
    wrist_view_cameras: list[str] | None,
) -> None:
    """Annotate every exported video camera with ``camera.is_wrist_view``."""
    wrist_set = {_camera_short_name(name) for name in (wrist_view_cameras or [])}
    for cam_name in camera_sizes:
        key = cam_name if cam_name.startswith("observation.images.") else f"observation.images.{cam_name}"
        spec = features.get(key)
        if not isinstance(spec, dict):
            continue
        info = spec.setdefault("info", {})
        if not isinstance(info, dict):
            info = {}
            spec["info"] = info
        info["camera.is_wrist_view"] = _camera_short_name(cam_name) in wrist_set


def _extrinsic_column_name(key: str) -> str:
    return key if key.startswith("extrinsic.") else f"extrinsic.{key}"


def _ndarray_series_to_pa(series: np.ndarray) -> pa.Array:
    """Write (n, D) as list[float], or (n, H, W) as list[list[float]] (matrix cells)."""
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 3:
        return pa.array(series.tolist(), type=pa.list_(pa.list_(pa.float32())))
    if series.ndim == 2:
        return pa.array(series.tolist(), type=pa.list_(pa.float32()))
    if series.ndim == 1:
        return pa.array(series.reshape(-1, 1).tolist(), type=pa.list_(pa.float32()))
    raise ValueError(f"unsupported series ndim={series.ndim} shape={series.shape}")


def _broadcast_extrinsics(
    extrinsic: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Broadcast extrinsics to kept frames as (n, 4, 4)."""
    n = int(indices.size)
    out: dict[str, np.ndarray] = {}
    for key, val in extrinsic.items():
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (4, 4):
            series = np.repeat(arr[None, :, :], n, axis=0)
        elif arr.ndim == 3 and arr.shape[-2:] == (4, 4):
            series = arr[indices]
        elif arr.ndim == 2 and arr.shape[1] == 16:
            series = arr[indices].reshape(n, 4, 4)
        else:
            flat = arr.reshape(-1)
            if flat.size != 16:
                continue
            series = np.repeat(flat.reshape(1, 4, 4), n, axis=0)
        out[_extrinsic_column_name(key)] = series.astype(np.float32)
    return out


def normalize_intrinsic_payload(intrinsic: dict[str, Any] | None) -> dict[str, Any]:
    """Episode-level intrinsic dict with standard_info short keys (e.g. camera_top.matrix)."""
    out: dict[str, Any] = {}
    for key, val in (intrinsic or {}).items():
        short = str(key)
        if short.startswith("intrinsic."):
            short = short[len("intrinsic.") :]
        arr = np.asarray(val)
        out[short] = arr.tolist()
    return out


def intrinsic_feature_specs(intrinsic: dict[str, Any] | None) -> dict[str, Any]:
    """dtype/shape specs for info.json ``intrinsic`` (same level as fps / embodiment)."""
    specs: dict[str, Any] = {}
    for key, val in normalize_intrinsic_payload(intrinsic).items():
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim >= 2:
            shape = [int(x) for x in arr.shape]
        else:
            shape = [int(arr.size)]
        specs[key] = _feature_spec(shape)
    return specs


def v2_intrinsic_feature_specs(camera_intrinsics: dict[str, Any] | None) -> dict[str, Any]:
    """Build info.json intrinsic specs for v2 episode geometry (reference/aux cameras)."""
    specs: dict[str, Any] = {}
    for cam, payload in (camera_intrinsics or {}).items():
        if not isinstance(payload, dict):
            continue
        if "matrix" in payload:
            specs[f"{cam}.matrix"] = _feature_spec([3, 3])
        if "dist_coeffs" in payload:
            dist = payload["dist_coeffs"]
            n = len(dist) if isinstance(dist, list) else 5
            specs[f"{cam}.dist_coeffs"] = _feature_spec([n])
    return specs


def standard_episode_to_table(
    ep: StandardEpisode,
    *,
    keep_mask: np.ndarray | None = None,
    global_index_start: int = 0,
    task_index: int = 0,
    fps: float = 30.0,
) -> pa.Table:
    T = ep.num_frames
    if keep_mask is None:
        indices = np.arange(T, dtype=np.int64)
    else:
        indices = np.flatnonzero(np.asarray(keep_mask).reshape(-1) != 0).astype(np.int64)
    n = int(indices.size)
    arrays: dict[str, pa.Array] = {}
    for name, arr in ep.fields.items():
        sub = np.asarray(arr, dtype=np.float32)[indices]
        if sub.ndim == 1:
            sub = sub.reshape(-1, 1)
        arrays[name] = _ndarray_series_to_pa(sub)

    for name, series in _broadcast_extrinsics(ep.extrinsic, indices).items():
        arrays[name] = _ndarray_series_to_pa(series)

    frame_index = np.arange(n, dtype=np.int64)
    arrays["frame_index"] = pa.array(frame_index)
    arrays["episode_index"] = pa.array(np.full(n, ep.episode_index, dtype=np.int64))
    arrays["index"] = pa.array(np.arange(global_index_start, global_index_start + n, dtype=np.int64))
    arrays["task_index"] = pa.array(np.full(n, task_index, dtype=np.int64))
    arrays["timestamp"] = pa.array((frame_index / max(fps, 1e-6)).astype(np.float32))
    return pa.table(arrays)


def write_standard_episode_parquet(
    output_root: Path,
    ep: StandardEpisode,
    *,
    keep_mask: np.ndarray | None = None,
    global_index_start: int = 0,
    fps: float = 30.0,
    task_index: int = 0,
) -> tuple[Path, int]:
    table = standard_episode_to_table(
        ep,
        keep_mask=keep_mask,
        global_index_start=global_index_start,
        fps=fps,
        task_index=task_index,
    )
    # Episode-level constants (same tier as fps / embodiment): one payload per parquet file.
    meta: dict[bytes, bytes] = dict(table.schema.metadata or {})
    meta[b"fps"] = str(float(fps)).encode("utf-8")
    embodiment = ep.meta.get("embodiment")
    if embodiment is not None:
        meta[b"embodiment"] = str(embodiment).encode("utf-8")
    intr = normalize_intrinsic_payload(ep.intrinsic)
    if intr:
        meta[b"intrinsic"] = json.dumps(intr, ensure_ascii=False).encode("utf-8")
    table = table.replace_schema_metadata(meta)

    chunk = ep.episode_index // 1000
    out_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep.episode_index:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return out_path, table.num_rows


def write_parameters(
    output_root: Path,
    ep: StandardEpisode,
) -> None:
    chunk = ep.episode_index // 1000
    out_dir = output_root / "parameters" / f"chunk-{chunk:03d}" / f"episode_{ep.episode_index:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "intrinsic": normalize_intrinsic_payload(ep.intrinsic),
        "extrinsic": {},
    }
    for k, v in ep.extrinsic.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
            arr = arr[0]
        payload["extrinsic"][k] = arr.tolist()
    if ep.meta.get("source_episode_index") is not None:
        payload["source_episode_index"] = ep.meta.get("source_episode_index")
    if ep.meta.get("calibration_bundle"):
        payload["source_calibration"] = ep.meta.get("calibration_bundle")
    with (out_dir / "calibration_standard.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _kept_length(ep: StandardEpisode, result: EpisodeResult | None) -> int | None:
    if result is not None and result.discard:
        return None
    if result is not None and result.kept_frames == 0:
        return None
    keep_mask = None if result is None else result.step_validity_mask
    if keep_mask is not None:
        n = int(np.asarray(keep_mask).sum())
        return n if n > 0 else None
    return int(ep.num_frames)


def _default_export_workers(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    return min(16, os.cpu_count() or 4)


def _init_export_pool(state: dict[str, Any]) -> None:
    _EXPORT_POOL_STATE.clear()
    _EXPORT_POOL_STATE.update(state)


def _export_one_task(
    task: tuple[int, StandardEpisode, np.ndarray | None, int, int],
) -> tuple[dict[str, Any], dict[str, tuple[int, int]], set[str]]:
    """Worker: write one episode parquet + parameters + videos."""
    out_ep_index, ep, keep_mask, global_index_start, task_index = task
    # Ensure episode_index matches output id (may already be remapped by orchestrator)
    ep.episode_index = int(out_ep_index)
    output_root = Path(_EXPORT_POOL_STATE["output_root"])
    camera_map: dict[str, str] = _EXPORT_POOL_STATE["camera_map"]
    fps = float(_EXPORT_POOL_STATE["fps"])
    target_height = int(_EXPORT_POOL_STATE["target_height"])
    target_short_side = _EXPORT_POOL_STATE.get("target_short_side")
    max_keyframe_interval = int(_EXPORT_POOL_STATE["max_keyframe_interval"])
    normalize_videos = bool(_EXPORT_POOL_STATE["normalize_videos"])
    parallel_videos = bool(_EXPORT_POOL_STATE.get("parallel_videos", True))
    skip_parameters = bool(_EXPORT_POOL_STATE.get("skip_parameters", False))
    default_source_root = Path(_EXPORT_POOL_STATE["source_root"])

    path, n = write_standard_episode_parquet(
        output_root,
        ep,
        keep_mask=keep_mask,
        global_index_start=global_index_start,
        fps=fps,
        task_index=task_index,
    )
    del path
    if not skip_parameters:
        write_parameters(output_root, ep)
    camera_sizes: dict[str, tuple[int, int]] = {}
    keep_indices = export_frame_indices(ep.num_frames, keep_mask)
    if normalize_videos and camera_map:
        src_root = (
            default_source_root
            if ep.meta.get("dataset_root") is None
            else Path(ep.meta["dataset_root"])
        )
        src_ep = int(ep.meta.get("source_episode_index", ep.episode_index))
        camera_sizes = export_normalized_videos(
            src_root,
            output_root,
            ep.episode_index,
            camera_map,
            target_height=target_height,
            target_short_side=(
                int(target_short_side) if target_short_side is not None else None
            ),
            max_keyframe_interval=max_keyframe_interval,
            source_episode_index=src_ep,
            parallel_videos=parallel_videos,
            keep_indices=keep_indices,
            fps=fps,
            num_frames=int(ep.num_frames),
        )
        out_chunk = ep.episode_index // 1000
        for src_key, std_name in camera_map.items():
            out_video = (
                output_root
                / "videos"
                / f"chunk-{out_chunk:03d}"
                / f"observation.images.{std_name}"
                / f"episode_{ep.episode_index:06d}.mp4"
            )
            if not out_video.exists():
                raise FileNotFoundError(f"Missing exported video: {out_video}")
            n_vid = probe_video_frames(out_video)
            if n_vid != n:
                raise ValueError(
                    f"exported video frame mismatch observation.images.{std_name}: "
                    f"video={n_vid} parquet={n}"
                )
    row = {
        "episode_index": ep.episode_index,
        "tasks": list(ep.meta.get("tasks") or []),
        "length": n,
        "source_episode_index": ep.meta.get("source_episode_index", ep.episode_index),
        "part": ep.meta.get("part"),
        "task": ep.meta.get("task"),
        "dataset_root": ep.meta.get("dataset_root"),
        "task_index": int(task_index),
    }
    if ep.meta.get("camera_intrinsics"):
        row["camera_intrinsics"] = ep.meta["camera_intrinsics"]
    field_names = set(ep.fields.keys())
    field_names.update(_extrinsic_column_name(k) for k in ep.extrinsic.keys())
    return row, camera_sizes, field_names


def _build_info_dict(
    *,
    episodes_out: list[dict[str, Any]],
    kept: list[tuple[StandardEpisode, np.ndarray | None, int]],
    fields_present: set[str],
    camera_sizes: dict[str, tuple[int, int]],
    camera_map: dict[str, str],
    fps: float,
    embodiment: str,
    state_action_delay: int,
    tasks_out: list[dict[str, Any]],
    total_frames: int,
    skip_parameters: bool,
    info_extras: dict[str, Any] | None,
    wrist_view_cameras: list[str] | None,
    standard_info_path: Path | None,
) -> dict[str, Any]:
    camera_list = sorted(camera_sizes.keys()) or sorted(camera_map.values())
    features = build_info_features(fields_present, camera_list)
    for cam_name, (h, w) in camera_sizes.items():
        key = cam_name if cam_name.startswith("observation.images.") else f"observation.images.{cam_name}"
        features[key] = _video_feature_spec(h, w, fps=fps)
    _apply_wrist_view_flags(features, camera_sizes, wrist_view_cameras)
    info_intrinsic: dict[str, Any] = {}
    if kept:
        ep0 = kept[0][0]
        for name, arr in ep0.fields.items():
            a = np.asarray(arr)
            d = 1 if a.ndim == 1 else int(a.shape[-1])
            features[name] = _feature_spec([d])
        for key in ep0.extrinsic.keys():
            features[_extrinsic_column_name(key)] = _feature_spec([4, 4])
        if skip_parameters and ep0.meta.get("camera_intrinsics"):
            info_intrinsic = v2_intrinsic_feature_specs(ep0.meta["camera_intrinsics"])
        else:
            info_intrinsic = intrinsic_feature_specs(ep0.intrinsic)

    info: dict[str, Any] = {
        "codebase_version": "v2.1",
        "robot_type": embodiment,
        "embodiment": embodiment,
        "fps": fps,
        "state_action_delay": state_action_delay,
        "camera_view_direction": "arm_side",
        "total_episodes": len(episodes_out),
        "total_frames": total_frames,
        "total_tasks": max(len(tasks_out), 1),
        "total_videos": len(episodes_out) * max(len(camera_list), 1),
        "total_chunks": max(
            1,
            math.ceil((max((ep.episode_index for ep, _, _ in kept), default=0) + 1) / 1000),
        ),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    if info_extras:
        for key, val in info_extras.items():
            if val is not None:
                info[key] = val
    if info_intrinsic:
        info["intrinsic"] = info_intrinsic
    if standard_info_path and standard_info_path.exists():
        try:
            template = load_json_with_comments(standard_info_path)
            tmpl_feats = template.get("features") or {}
            for k, v in tmpl_feats.items():
                if k.startswith("observation.images."):
                    continue
                if k in features:
                    features[k] = v
            nested_ext = tmpl_feats.get("extrinsic") if isinstance(tmpl_feats.get("extrinsic"), dict) else None
            if nested_ext:
                for ek, ev in nested_ext.items():
                    col = _extrinsic_column_name(ek)
                    if col in features and isinstance(ev, dict):
                        features[col] = {
                            "dtype": ev.get("dtype", "float32"),
                            "shape": list(ev.get("shape") or [4, 4]),
                            "names": ev.get("names"),
                        }
            for k in list(features):
                if k.startswith("observation.images.") and k not in {
                    (n if n.startswith("observation.images.") else f"observation.images.{n}")
                    for n in camera_sizes
                }:
                    del features[k]
            for cam_name, (h, w) in camera_sizes.items():
                key = (
                    cam_name if cam_name.startswith("observation.images.") else f"observation.images.{cam_name}"
                )
                features[key] = _video_feature_spec(h, w, fps=fps)
            _apply_wrist_view_flags(features, camera_sizes, wrist_view_cameras)
            info["features"] = features
            tmpl_intr = template.get("intrinsic")
            if isinstance(tmpl_intr, dict) and not skip_parameters:
                merged = dict(info_intrinsic)
                for ik, iv in tmpl_intr.items():
                    if not isinstance(iv, dict):
                        continue
                    if ik in merged:
                        merged[ik] = {
                            "dtype": iv.get("dtype", merged[ik].get("dtype", "float32")),
                            "shape": list(iv.get("shape") or merged[ik].get("shape")),
                            "names": iv.get("names"),
                        }
                    else:
                        merged[ik] = {
                            "dtype": iv.get("dtype", "float32"),
                            "shape": list(iv.get("shape") or [1]),
                            "names": iv.get("names"),
                        }
                info["intrinsic"] = merged
            for key in ("prompt_template", "horizon"):
                if key in template:
                    info[key] = template[key]
        except Exception:
            pass
    info["total_tasks"] = max(len(tasks_out), 1)
    return info


def _write_v21_meta_files(
    dataset_root: Path,
    *,
    info: dict[str, Any],
    episodes_out: list[dict[str, Any]],
    tasks_out: list[dict[str, Any]],
    embodiment: str,
) -> None:
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_out:
            ep_row: dict[str, Any] = {
                "episode_index": int(row["episode_index"]),
                "tasks": list(row.get("tasks") or []),
                "length": int(row["length"]),
            }
            if row.get("camera_intrinsics"):
                ep_row["camera_intrinsics"] = row["camera_intrinsics"]
            f.write(json.dumps(ep_row, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        rows = tasks_out or [{"task_index": 0, "task": embodiment}]
        for row in rows:
            f.write(
                json.dumps(
                    {"task_index": int(row["task_index"]), "task": str(row["task"])},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (meta_dir / "episode_source_map.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_out:
            f.write(
                json.dumps(
                    {
                        "episode_index": row["episode_index"],
                        "source_episode_index": row.get("source_episode_index"),
                        "part": row.get("part"),
                        "task": row.get("task"),
                        "task_index": row.get("task_index"),
                        "dataset_root": row.get("dataset_root"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def export_standard_dataset(
    *,
    output_root: Path,
    episodes: list[tuple[StandardEpisode, EpisodeResult | None]],
    camera_map: dict[str, str],
    source_root: Path,
    fps: float = 30.0,
    embodiment: str = "unknown",
    state_action_delay: int = 1,
    target_height: int = 384,
    target_short_side: int | None = None,
    max_keyframe_interval: int = 10,
    standard_info_path: Path | None = None,
    normalize_videos: bool = True,
    export_workers: int | None = None,
    parallel_videos: bool = True,
    renumber_episodes: bool = False,
    task_name_to_index: dict[str, int] | None = None,
    show_progress: bool = True,
    skip_parameters: bool = False,
    info_extras: dict[str, Any] | None = None,
    wrist_view_cameras: list[str] | None = None,
    lerobot_version: str = "v2.1",
    v30_output_root: Path | None = None,
    v30_data_file_size_in_mb: int = 100,
    v30_video_file_size_in_mb: int = 200,
    keep_all_frames: bool = False,
) -> dict[str, Any]:
    """Export filtered standard episodes (+ optional video normalize).

    Always writes a single LeRobot root under ``output_root`` (no per-task subdirs).
    When ``renumber_episodes`` is True, output episode ids become contiguous 0..N-1
    while ``meta.source_episode_index`` keeps the source id for video lookup.

    ``lerobot_version`` controls the on-disk LeRobot layout:
    - ``v2.1``: per-episode parquet/mp4 (default)
    - ``v3.0``: merged chunk files; episode data is written once then finalized in-place
    - ``both``: v2.1 at ``output_root`` and v3.0 at ``v30_output_root`` (or ``{output_root}_v30``)
    """
    from tqdm import tqdm

    version = str(lerobot_version or "v2.1").lower()
    if version not in ("v2.1", "v3.0", "both"):
        raise ValueError(f"Unsupported lerobot_version: {lerobot_version!r}")
    write_v30 = version in ("v3.0", "both")
    if version == "v3.0":
        renumber_episodes = True

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cache_root: Path | None = None
    if version == "v3.0":
        cache_root = output_root / ".lerobot_export_cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        dataset_root = cache_root
    else:
        dataset_root = output_root

    # Filter + assign lengths / global indices up front so workers can run independently.
    kept: list[tuple[StandardEpisode, np.ndarray | None, int]] = []
    skipped = 0
    skipped_frame_alignment = 0
    for ep, result in episodes:
        if result is not None and result.discard:
            skipped += 1
            continue
        if keep_all_frames:
            n = int(ep.num_frames)
            if n <= 0:
                skipped += 1
                continue
        else:
            n = _kept_length(ep, result)
            if n is None:
                skipped += 1
                continue
        keep_mask = None
        if not keep_all_frames and result is not None:
            keep_mask = result.step_validity_mask
        src_root = Path(ep.meta.get("dataset_root", source_root))
        src_ep = int(ep.meta.get("source_episode_index", ep.episode_index))
        align = check_export_frame_alignment(
            num_frames=int(ep.num_frames),
            fields=ep.fields,
            extrinsic=ep.extrinsic,
            keep_mask=keep_mask,
            dataset_root=src_root,
            source_episode_index=src_ep,
            camera_map=camera_map if normalize_videos else {},
            fps=float(fps),
        )
        if not align.ok:
            skipped += 1
            skipped_frame_alignment += 1
            continue
        # Ensure source index is recorded before optional renumber
        ep.meta.setdefault("source_episode_index", ep.episode_index)
        kept.append((ep, keep_mask, n))

    if renumber_episodes:
        for new_i, (ep, _mask, _n) in enumerate(kept):
            ep.meta.setdefault("source_episode_index", ep.episode_index)
            ep.episode_index = new_i

    # Task index map: prefer source tasks.jsonl indices; else assign densely.
    source_task_map = dict(task_name_to_index or {})
    task_to_index: dict[str, int] = {}
    tasks_out: list[dict[str, Any]] = []
    used_task_index_to_name: dict[int, str] = {}
    next_synth_index = (max(source_task_map.values()) + 1) if source_task_map else 0

    def _task_names_for(ep: StandardEpisode) -> list[str]:
        names = ep.meta.get("tasks") or []
        if isinstance(names, str):
            names = [names]
        names = [str(x) for x in names if x is not None and str(x) != ""]
        if not names and ep.meta.get("task"):
            names = [str(ep.meta["task"])]
        if not names:
            names = [embodiment]
        return names

    def _task_index_for(ep: StandardEpisode) -> int:
        nonlocal next_synth_index
        names = _task_names_for(ep)
        # Keep full list on meta for episodes.jsonl
        ep.meta["tasks"] = names
        name = names[0]
        if name in task_to_index:
            return task_to_index[name]
        if name in source_task_map:
            idx = int(source_task_map[name])
        else:
            idx = int(next_synth_index)
            next_synth_index += 1
        task_to_index[name] = idx
        used_task_index_to_name[idx] = name
        return idx

    export_tasks: list[tuple[int, StandardEpisode, np.ndarray | None, int, int]] = []
    global_index = 0
    for ep, keep_mask, n in kept:
        ti = _task_index_for(ep)
        export_tasks.append((ep.episode_index, ep, keep_mask, global_index, ti))
        global_index += n

    tasks_out = [
        {"task_index": int(ti), "task": used_task_index_to_name[ti]}
        for ti in sorted(used_task_index_to_name)
    ]

    workers = _default_export_workers(export_workers)
    episodes_out: list[dict[str, Any]] = []
    fields_present: set[str] = set()
    # camera_name → (height, width); first-seen wins (episodes share scale policy)
    camera_sizes: dict[str, tuple[int, int]] = {}
    total_frames = 0

    pool_state = {
        "output_root": str(dataset_root),
        "camera_map": camera_map,
        "fps": fps,
        "target_height": target_height,
        "target_short_side": target_short_side,
        "max_keyframe_interval": max_keyframe_interval,
        "normalize_videos": normalize_videos,
        "parallel_videos": parallel_videos,
        "skip_parameters": skip_parameters,
        "source_root": str(source_root),
    }

    def _merge_camera_sizes(sizes: dict[str, tuple[int, int]]) -> None:
        for name, hw in sizes.items():
            camera_sizes.setdefault(name, (int(hw[0]), int(hw[1])))

    if workers <= 1 or len(export_tasks) <= 1:
        _init_export_pool(pool_state)
        iterator: Any = export_tasks
        if show_progress:
            iterator = tqdm(export_tasks, desc="export standard")
        for task in iterator:
            row, sizes, field_names = _export_one_task(task)
            episodes_out.append(row)
            _merge_camera_sizes(sizes)
            fields_present.update(field_names)
            total_frames += int(row["length"])
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_export_pool,
            initargs=(pool_state,),
        ) as pool:
            futures = {pool.submit(_export_one_task, task): task[0] for task in export_tasks}
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"export standard ({workers}w)")
            results_by_ep: dict[int, tuple[dict[str, Any], dict[str, tuple[int, int]], set[str]]] = {}
            for fut in iterator:
                ep_id = futures[fut]
                results_by_ep[ep_id] = fut.result()
        for ep_id, _ep, _mask, _g0, _ti in export_tasks:
            row, sizes, field_names = results_by_ep[ep_id]
            episodes_out.append(row)
            _merge_camera_sizes(sizes)
            fields_present.update(field_names)
            total_frames += int(row["length"])

    camera_list = sorted(camera_sizes.keys()) or sorted(camera_map.values())
    info = _build_info_dict(
        episodes_out=episodes_out,
        kept=kept,
        fields_present=fields_present,
        camera_sizes=camera_sizes,
        camera_map=camera_map,
        fps=fps,
        embodiment=embodiment,
        state_action_delay=state_action_delay,
        tasks_out=tasks_out,
        total_frames=total_frames,
        skip_parameters=skip_parameters,
        info_extras=info_extras,
        wrist_view_cameras=wrist_view_cameras,
        standard_info_path=standard_info_path,
    )
    _write_v21_meta_files(
        dataset_root,
        info=info,
        episodes_out=episodes_out,
        tasks_out=tasks_out,
        embodiment=embodiment,
    )

    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "lerobot_version": version,
        "total_episodes": len(episodes_out),
        "total_frames": total_frames,
        "skipped_episodes": skipped,
        "skipped_frame_alignment": skipped_frame_alignment,
        "cameras": camera_list,
        "state_action_delay": state_action_delay,
        "export_workers": workers,
        "parallel_videos": parallel_videos,
        "renumber_episodes": renumber_episodes,
        "keep_all_frames": keep_all_frames,
        "codebase_version": "v2.1" if version == "v2.1" else None,
    }

    if write_v30:
        from robot_data_processing.export_lerobot_v30 import (
            install_v30_dataset,
            install_v30_dataset_into_root,
        )

        if version == "v3.0":
            assert cache_root is not None
            v30_summary = install_v30_dataset_into_root(
                cache_root,
                output_root,
                data_file_size_in_mb=int(v30_data_file_size_in_mb),
                video_file_size_in_mb=int(v30_video_file_size_in_mb),
            )
            if cache_root.exists():
                shutil.rmtree(cache_root)
            summary["codebase_version"] = v30_summary.get("codebase_version")
        else:
            v30_root = Path(v30_output_root) if v30_output_root is not None else output_root.parent / f"{output_root.name}_v30"
            v30_summary = install_v30_dataset(
                dataset_root,
                v30_root,
                data_file_size_in_mb=int(v30_data_file_size_in_mb),
                video_file_size_in_mb=int(v30_video_file_size_in_mb),
            )
            (v30_root / "lerobot_v30_export.json").write_text(
                json.dumps(v30_summary, indent=2),
                encoding="utf-8",
            )
        summary["lerobot_v30"] = v30_summary
        if version == "v3.0":
            (output_root / "lerobot_v30_export.json").write_text(
                json.dumps(v30_summary, indent=2),
                encoding="utf-8",
            )

    report_path = output_root / "export_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary

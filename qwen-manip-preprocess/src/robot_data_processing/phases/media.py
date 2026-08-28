"""P5 MediaNormalize: rename cameras, height≤target (keep AR, no upscale), GOP≤10."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from robot_data_processing.lerobot_export import _video_encoder_args, extract_video_by_indices, trim_video


def probe_video_size(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` via ffprobe. Raises on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    h = int(stream["height"])
    w = int(stream["width"])
    if h <= 0 or w <= 0:
        raise RuntimeError(f"invalid video size {w}x{h} for {path}")
    return h, w


def _even(n: int) -> int:
    return n if n % 2 == 0 else max(n - 1, 2)


def normalize_video(
    src: Path,
    dst: Path,
    *,
    target_height: int = 384,
    max_keyframe_interval: int = 10,
) -> tuple[int, int]:
    """Re-encode to H.264; scale height down to ``target_height`` (never upscale).

    Returns final ``(height, width)``.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)

    try:
        src_h, src_w = probe_video_size(src)
    except Exception:
        src_h, src_w = 0, 0

    # Only downscale when taller than target; otherwise keep native resolution.
    if src_h > 0 and src_h <= int(target_height):
        out_h, out_w = _even(src_h), _even(src_w)
        # Keep size (force even dims for yuv420) but still re-encode for GOP.
        vf = f"scale={out_w}:{out_h}"
    elif src_h > int(target_height):
        out_h = int(target_height)
        # width auto, even
        vf = f"scale=-2:{out_h}"
        out_w = 0  # filled after encode via probe
    else:
        # Unknown source size: attempt target-height scale (ffmpeg may fail → copy)
        vf = f"scale=-2:{int(target_height)}"
        out_h, out_w = int(target_height), 0

    enc_args = _video_encoder_args()
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        *enc_args,
        "-g",
        str(max_keyframe_interval),
        "-keyint_min",
        str(max_keyframe_interval),
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        shutil.copy2(src, dst)

    try:
        return probe_video_size(dst)
    except Exception:
        if out_w > 0:
            return out_h, out_w
        if src_h > 0 and src_w > 0:
            return src_h, src_w
        return int(target_height), int(target_height)


def normalize_video_short_side(
    src: Path,
    dst: Path,
    *,
    target_short_side: int = 384,
    max_keyframe_interval: int = 10,
) -> tuple[int, int]:
    """Re-encode to H.264; scale so the shorter side equals ``target_short_side``."""
    from robot_data_processing.normalize.episode_frame_geometry import target_short_side_size

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)

    try:
        src_h, src_w = probe_video_size(src)
    except Exception:
        src_h, src_w = 0, 0

    if src_h > 0 and src_w > 0:
        out_w, out_h = target_short_side_size(src_w, src_h, short_side=target_short_side)
        vf = f"scale={out_w}:{out_h}"
    else:
        vf = f"scale=-2:{int(target_short_side)}"
        out_h, out_w = int(target_short_side), 0

    enc_args = _video_encoder_args()
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        *enc_args,
        "-g",
        str(max_keyframe_interval),
        "-keyint_min",
        str(max_keyframe_interval),
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        shutil.copy2(src, dst)

    try:
        return probe_video_size(dst)
    except Exception:
        if out_w > 0:
            return out_h, out_w
        return int(target_short_side), int(target_short_side)


def resolve_source_video(
    dataset_root: Path,
    episode_index: int,
    source_video_key: str,
) -> Path:
    chunk = episode_index // 1000
    return (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / source_video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def _prepare_video_source(
    src: Path,
    *,
    keep_indices: np.ndarray | None,
    fps: float,
    num_frames: int,
) -> tuple[Path, Path | None]:
    """Return ``(source_path, temp_path_to_delete)`` optionally trimmed to kept frames."""
    src = Path(src)
    if keep_indices is None:
        return src, None
    indices = np.asarray(keep_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"Cannot export video with empty keep_indices: {src}")
    if int(indices.size) == int(num_frames) and np.array_equal(indices, np.arange(int(num_frames))):
        return src, None

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    tmp = Path(tmp_name)
    if np.array_equal(indices, np.arange(indices.size)) and int(indices[0]) == 0:
        trim_video(src, tmp, int(indices.size), fps)
    else:
        extract_video_by_indices(src, tmp, indices, fps)
    return tmp, tmp


def export_normalized_videos(
    dataset_root: Path,
    output_root: Path,
    episode_index: int,
    camera_map: dict[str, str],
    *,
    target_height: int = 384,
    target_short_side: int | None = None,
    max_keyframe_interval: int = 10,
    source_episode_index: int | None = None,
    parallel_videos: bool = True,
    keep_indices: np.ndarray | None = None,
    fps: float = 30.0,
    num_frames: int | None = None,
) -> dict[str, tuple[int, int]]:
    """camera_map: source_key → standard camera name (e.g. camera_top).

    When ``target_short_side`` is set, resize so the shorter side equals that value
    (v2 episode geometry contract). Otherwise scale by ``target_height``.

    Returns ``{camera_name: (height, width)}`` for written videos.
    """
    src_ep = int(episode_index if source_episode_index is None else source_episode_index)
    out_chunk = episode_index // 1000
    frame_count = int(num_frames if num_frames is not None else (keep_indices.size if keep_indices is not None else 0))

    jobs: list[tuple[str, Path, Path]] = []
    for src_key, std_name in camera_map.items():
        src = resolve_source_video(dataset_root, src_ep, src_key)
        if not src.exists():
            alt = resolve_source_video(dataset_root, src_ep, f"observation.images.{src_key}")
            src = alt if alt.exists() else src
        if not src.exists():
            continue
        dst = (
            output_root
            / "videos"
            / f"chunk-{out_chunk:03d}"
            / f"observation.images.{std_name}"
            / f"episode_{episode_index:06d}.mp4"
        )
        jobs.append((std_name, src, dst))

    if not jobs:
        return {}

    def _one(job: tuple[str, Path, Path]) -> tuple[str, int, int]:
        std_name, src, dst = job
        trim_src, trim_tmp = _prepare_video_source(
            src,
            keep_indices=keep_indices,
            fps=fps,
            num_frames=frame_count,
        )
        try:
            if target_short_side is not None:
                h, w = normalize_video_short_side(
                    trim_src,
                    dst,
                    target_short_side=int(target_short_side),
                    max_keyframe_interval=max_keyframe_interval,
                )
            else:
                h, w = normalize_video(
                    trim_src,
                    dst,
                    target_height=target_height,
                    max_keyframe_interval=max_keyframe_interval,
                )
            return std_name, h, w
        finally:
            if trim_tmp is not None and trim_tmp.exists():
                trim_tmp.unlink(missing_ok=True)

    sizes: dict[str, tuple[int, int]] = {}
    if parallel_videos and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                name, h, w = fut.result()
                sizes[name] = (h, w)
    else:
        for job in jobs:
            name, h, w = _one(job)
            sizes[name] = (h, w)
    return sizes

"""Optional LeRobot v2.1 → v3.0 export conversion for Dataclean P7 output."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from robot_data_processing.lerobot_export import (
    _stats_1d,
    _video_stats_from_samples,
    _write_jsonl,
    sample_video_pixels,
)
from robot_data_processing.schema import stack_column

logger = logging.getLogger(__name__)

V21 = "v2.1"
V30 = "v3.0"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _unflatten_dict(d: dict, sep: str = "/") -> dict:
    out: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def _serialize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in _flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        elif isinstance(value, (int, float, list)):
            serialized[key] = value
        else:
            raise NotImplementedError(f"Unsupported stats value type {type(value)} for key {key}")
    return _unflatten_dict(serialized)


def _cast_stats_to_numpy(stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    flat = {key: np.atleast_1d(np.asarray(value)) for key, value in _flatten_dict(stats).items()}
    return _unflatten_dict(flat)


def _aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count
    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]
        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                q_percent = int(q_key[1:])
                if q_percent <= 50:
                    aggregated[q_key] = np.min(quantile_values, axis=0)
                else:
                    aggregated[q_key] = np.max(quantile_values, axis=0)
    return aggregated


def _aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats: dict[str, dict[str, np.ndarray]] = {key: {} for key in data_keys}
    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = _aggregate_feature_stats(stats_with_key)
    return aggregated_stats


def _load_info(dataset_root: Path) -> dict[str, Any]:
    return _load_json(dataset_root / INFO_PATH)


def _get_parquet_file_size_in_mb(parquet_path: Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            total_uncompressed_size += rg_metadata.column(column).total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def _get_parquet_num_frames(parquet_path: Path) -> int:
    return pq.read_metadata(parquet_path).num_rows


def _get_file_size_in_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024**2)


def _update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


def _load_jsonlines(fpath: Path) -> list[Any]:
    rows: list[Any] = []
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _list_v21_episode_parquets(dataset_root: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    data_dir = dataset_root / DATA_DIR
    for path in sorted(data_dir.glob("chunk-*/*.parquet")):
        if not path.name.startswith("episode_"):
            continue
        ep = int(path.stem.split("_")[1])
        rows.append((ep, path))
    return sorted(rows, key=lambda x: x[0])


def _video_paths_for_episode(
    dataset_root: Path,
    episode_index: int,
    video_keys: list[str],
) -> dict[str, Path]:
    chunk = episode_index // 1000
    out: dict[str, Path] = {}
    for key in video_keys:
        p = dataset_root / VIDEO_DIR / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        if p.exists():
            out[key] = p
    return out


def _stats_for_table_column(table, name: str) -> dict[str, Any]:
    col = table.column(name).combine_chunks()
    if name.startswith("extrinsic."):
        values = col.to_pylist()
        arr = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in values], axis=0)
        return _stats_1d(arr)
    values = col.to_numpy(zero_copy_only=False)
    if isinstance(values, np.ndarray) and values.dtype == object:
        try:
            arr = stack_column(values)
        except ValueError:
            arr = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in values], axis=0)
        return _stats_1d(arr)
    return _stats_1d(np.asarray(values, dtype=np.float64))


def write_episodes_stats_jsonl(dataset_root: Path) -> Path:
    """Create ``meta/episodes_stats.jsonl`` for a v2.1 export if missing."""
    dataset_root = Path(dataset_root)
    out_path = dataset_root / LEGACY_EPISODES_STATS_PATH
    info = _load_info(dataset_root)
    video_keys = [
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    rows: list[dict[str, Any]] = []
    for episode_index, parquet_path in _list_v21_episode_parquets(dataset_root):
        table = pq.read_table(parquet_path)
        video_paths = _video_paths_for_episode(dataset_root, episode_index, video_keys)
        stats: dict[str, Any] = {}
        for name in table.column_names:
            if name in video_paths:
                stats[name] = _video_stats_from_samples(sample_video_pixels(video_paths[name]))
            else:
                stats[name] = _stats_for_table_column(table, name)
        rows.append({"episode_index": int(episode_index), "stats": stats})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path, rows)
    return out_path


def _validate_local_dataset_version(local_path: Path) -> None:
    info = _load_info(local_path)
    dataset_version = info.get("codebase_version") or "unknown"
    if dataset_version != V21:
        raise ValueError(
            f"Local dataset has codebase_version '{dataset_version}', expected '{V21}'. "
            "This converter only supports v2.1 → v3.0."
        )


def _legacy_load_episodes(local_dir: Path) -> dict[int, dict[str, Any]]:
    episodes = _load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def _legacy_load_episodes_stats(local_dir: Path) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    episodes_stats = _load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: _cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def _legacy_load_tasks(local_dir: Path) -> dict[int, str]:
    tasks = _load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    return {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}


def _get_video_keys(root: Path) -> list[str]:
    info = _load_info(root)
    return sorted(
        key for key, ft in (info.get("features") or {}).items() if isinstance(ft, dict) and ft.get("dtype") == "video"
    )


def _get_image_keys(root: Path) -> list[str]:
    info = _load_info(root)
    return [
        key for key, ft in (info.get("features") or {}).items() if isinstance(ft, dict) and ft.get("dtype") == "image"
    ]


def _concatenate_video_files(input_video_paths: list[Path], output_video_path: Path) -> None:
    if not input_video_paths:
        raise FileNotFoundError("No input video paths provided.")
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False, encoding="utf-8") as tmp:
        tmp.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp.write(f"file '{Path(input_path).resolve()}'\n")
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            tmp_path,
            "-c",
            "copy",
            "-movflags",
            "faststart",
            str(output_video_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _get_video_duration_in_s(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def _convert_info(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int,
) -> None:
    info = _load_json(root / INFO_PATH)
    info["codebase_version"] = V30
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = data_file_size_in_mb
    info["video_files_size_in_mb"] = video_file_size_in_mb
    info["data_path"] = DEFAULT_DATA_PATH
    if info.get("video_path") is not None:
        info["video_path"] = DEFAULT_VIDEO_PATH
    info["fps"] = int(info["fps"])
    for key in info.get("features", {}):
        if info["features"][key].get("dtype") == "video":
            continue
        info["features"][key]["fps"] = info["fps"]
    logger.info("Converting info from %s to %s", root, new_root)
    _write_json(new_root / INFO_PATH, info)


def _convert_tasks(root: Path, new_root: Path) -> None:
    logger.info("Converting tasks from %s to %s", root, new_root)
    tasks = _legacy_load_tasks(root)
    df_tasks = pd.DataFrame({"task_index": list(tasks.keys())}, index=pd.Index(list(tasks.values()), name="task"))
    out_path = new_root / DEFAULT_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_tasks.to_parquet(out_path)


def _concat_data_files(
    paths_to_cat: list[Path],
    new_root: Path,
    chunk_idx: int,
    file_idx: int,
    image_keys: list[str],
) -> None:
    dataframes = [pd.read_parquet(file) for file in paths_to_cat]
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = None
    if image_keys:
        schema = pa.Schema.from_pandas(concatenated_df)
    concatenated_df.to_parquet(path, index=False, schema=schema)


def _episode_index_from_stem(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _convert_data(root: Path, new_root: Path, data_file_size_in_mb: int) -> list[dict[str, Any]]:
    data_dir = root / DATA_DIR
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    image_keys = _get_image_keys(root)

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    num_frames = 0
    paths_to_cat: list[Path] = []
    episodes_metadata: list[dict[str, Any]] = []

    logger.info("Converting data files from %d episodes", len(ep_paths))
    for ep_path in tqdm.tqdm(ep_paths, desc="convert data files"):
        episode_index = _episode_index_from_stem(ep_path)
        ep_size_in_mb = _get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = _get_parquet_num_frames(ep_path)

        if size_in_mb + ep_size_in_mb >= data_file_size_in_mb and paths_to_cat:
            _concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
            chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0.0
            paths_to_cat = []

        episodes_metadata.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": num_frames,
                "dataset_to_index": num_frames + ep_num_frames,
            }
        )
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        paths_to_cat.append(ep_path)

    if paths_to_cat:
        _concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
    return episodes_metadata


def _convert_videos_of_camera(
    root: Path,
    new_root: Path,
    video_key: str,
    video_file_size_in_mb: int,
) -> list[dict[str, Any]]:
    videos_dir = root / VIDEO_DIR
    ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    duration_in_s = 0.0
    paths_to_cat: list[Path] = []
    episodes_metadata: list[dict[str, Any]] = []
    row_indices_in_batch: list[int] = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        episode_index = _episode_index_from_stem(ep_path)
        ep_size_in_mb = _get_file_size_in_mb(ep_path)
        ep_duration_in_s = _get_video_duration_in_s(ep_path)

        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and paths_to_cat:
            _concatenate_video_files(
                paths_to_cat,
                new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
            )
            for past_row_idx in row_indices_in_batch:
                episodes_metadata[past_row_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_row_idx][f"videos/{video_key}/file_index"] = file_idx
            chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0.0
            duration_in_s = 0.0
            paths_to_cat = []
            row_indices_in_batch = []

        row_idx = len(episodes_metadata)
        episodes_metadata.append(
            {
                "episode_index": episode_index,
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": duration_in_s,
                f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
            }
        )
        paths_to_cat.append(ep_path)
        row_indices_in_batch.append(row_idx)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s

    if paths_to_cat:
        _concatenate_video_files(
            paths_to_cat,
            new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
        )
        for past_row_idx in row_indices_in_batch:
            episodes_metadata[past_row_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_row_idx][f"videos/{video_key}/file_index"] = file_idx
    return episodes_metadata


def _convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int) -> list[dict[str, Any]] | None:
    logger.info("Converting videos from %s to %s", root, new_root)
    video_keys = _get_video_keys(root)
    if not video_keys:
        return None

    eps_metadata_per_cam = [_convert_videos_of_camera(root, new_root, camera, video_file_size_in_mb) for camera in video_keys]
    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cameras must have the same number of episodes ({num_eps_per_cam}).")

    episodes_metadata: list[dict[str, Any]] = []
    num_episodes = num_eps_per_cam[0]
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="convert videos"):
        ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(len(video_keys))]
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")
        ep_dict: dict[str, Any] = {"episode_index": ep_ids[0]}
        for cam_idx in range(len(video_keys)):
            cam_meta = dict(eps_metadata_per_cam[cam_idx][ep_idx])
            cam_meta.pop("episode_index", None)
            ep_dict.update(cam_meta)
        episodes_metadata.append(ep_dict)
    return episodes_metadata


def _generate_episode_metadata_dict(
    episodes_legacy_metadata: dict[int, dict[str, Any]],
    episodes_metadata: list[dict[str, Any]],
    episodes_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    episodes_videos: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    videos_by_episode: dict[int, dict[str, Any]] = {}
    if episodes_videos is not None:
        videos_by_episode = {row["episode_index"]: row for row in episodes_videos}

    for ep_metadata in episodes_metadata:
        episode_index = ep_metadata["episode_index"]
        ep_legacy_metadata = episodes_legacy_metadata[episode_index]
        ep_stats = episodes_stats[episode_index]
        ep_video = videos_by_episode.get(episode_index, {})
        ep_ids_set = {episode_index, ep_legacy_metadata["episode_index"], ep_metadata["episode_index"]}
        if episodes_videos is not None and ep_video:
            ep_ids_set.add(ep_video["episode_index"])
        if len(ep_ids_set) != 1:
            raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")
        ep_dict = {
            **ep_metadata,
            **ep_video,
            **ep_legacy_metadata,
            **_flatten_dict({"stats": ep_stats}),
        }
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def _write_episodes(episodes_rows: list[dict[str, Any]], new_root: Path) -> None:
    df = pd.DataFrame(episodes_rows)
    fpath = new_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fpath, index=False)


def _write_stats(stats: dict[str, dict[str, np.ndarray]], new_root: Path) -> None:
    _write_json(new_root / STATS_PATH, _serialize_stats(stats))


def _convert_episodes_metadata(
    root: Path,
    new_root: Path,
    episodes_metadata: list[dict[str, Any]],
    episodes_video_metadata: list[dict[str, Any]] | None = None,
) -> None:
    logger.info("Converting episodes metadata from %s to %s", root, new_root)
    episodes_legacy_metadata = _legacy_load_episodes(root)
    episodes_stats = _legacy_load_episodes_stats(root)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))
    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    episodes_rows = list(
        _generate_episode_metadata_dict(
            episodes_legacy_metadata,
            episodes_metadata,
            episodes_stats,
            episodes_video_metadata,
        )
    )
    _write_episodes(episodes_rows, new_root)
    _write_stats(_aggregate_stats(list(episodes_stats.values())), new_root)


def convert_v21_directory_to_v30(
    v21_root: Path,
    v30_root: Path,
    *,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> Path:
    """Convert a local v2.1 dataset directory to v3.0 layout under ``v30_root``."""
    v21_root = Path(v21_root)
    v30_root = Path(v30_root)
    if v30_root.exists():
        shutil.rmtree(v30_root)
    v30_root.mkdir(parents=True, exist_ok=True)

    info = _load_info(v21_root)
    if str(info.get("codebase_version", V21)) != V21:
        raise ValueError(
            f"Expected codebase_version {V21}, got {info.get('codebase_version')!r} in {v21_root}"
        )

    stats_path = v21_root / LEGACY_EPISODES_STATS_PATH
    if not stats_path.exists():
        write_episodes_stats_jsonl(v21_root)

    _validate_local_dataset_version(v21_root)
    _convert_info(v21_root, v30_root, data_file_size_in_mb, video_file_size_in_mb)
    _convert_tasks(v21_root, v30_root)
    episodes_metadata = _convert_data(v21_root, v30_root, data_file_size_in_mb)
    episodes_videos_metadata = _convert_videos(v21_root, v30_root, video_file_size_in_mb)
    _convert_episodes_metadata(v21_root, v30_root, episodes_metadata, episodes_videos_metadata)
    return v30_root


def _v30_summary(output_root: Path, *, v21_source: Path | None = None) -> dict[str, Any]:
    info = _load_info(output_root)
    return {
        "lerobot_version": V30,
        "output_root": str(output_root),
        "v21_source": str(v21_source) if v21_source is not None else None,
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
    }


def install_v30_dataset(
    v21_root: Path,
    v30_root: Path,
    *,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> dict[str, Any]:
    """Convert v2.1 layout at ``v21_root`` into a standalone v3.0 dataset at ``v30_root``."""
    convert_v21_directory_to_v30(
        Path(v21_root),
        Path(v30_root),
        data_file_size_in_mb=int(data_file_size_in_mb),
        video_file_size_in_mb=int(video_file_size_in_mb),
    )
    return _v30_summary(v30_root, v21_source=Path(v21_root))


def install_v30_dataset_into_root(
    v21_root: Path,
    output_root: Path,
    *,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> dict[str, Any]:
    """Convert v2.1 layout and install ``meta/``, ``data/``, ``videos/`` under ``output_root``."""
    output_root = Path(output_root)
    staging = output_root.parent / f".{output_root.name}_v30_staging"
    if staging.exists():
        shutil.rmtree(staging)
    install_v30_dataset(
        v21_root,
        staging,
        data_file_size_in_mb=data_file_size_in_mb,
        video_file_size_in_mb=video_file_size_in_mb,
    )
    for sub in ("meta", "data", "videos"):
        src = staging / sub
        dst = output_root / sub
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    if staging.exists():
        shutil.rmtree(staging)
    return _v30_summary(output_root, v21_source=Path(v21_root))


def export_lerobot_v30_from_v21(
    v21_root: Path,
    *,
    output_root: Path | None = None,
    lerobot_root: Path | None = None,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    replace_v21: bool = False,
) -> dict[str, Any]:
    """Convert an existing v2.1 export to v3.0.

    ``lerobot_root`` is deprecated and ignored; conversion is implemented locally.
    """
    del lerobot_root
    v21_root = Path(v21_root)
    if output_root is None:
        output_root = v21_root.parent / f"{v21_root.name}_v30"
    output_root = Path(output_root)

    summary = install_v30_dataset(
        v21_root,
        output_root,
        data_file_size_in_mb=int(data_file_size_in_mb),
        video_file_size_in_mb=int(video_file_size_in_mb),
    )

    if replace_v21 and output_root.resolve() != v21_root.resolve():
        backup = v21_root.parent / f"{v21_root.name}_v21_backup"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(v21_root), str(backup))
        shutil.move(str(output_root), str(v21_root))
        output_root = v21_root
        backup_note = str(backup)
        summary["output_root"] = str(output_root)
        summary["v21_backup"] = backup_note
    else:
        backup_note = None
        summary["v21_backup"] = None
    return summary

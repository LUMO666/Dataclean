#!/usr/bin/env python3
"""Merge LeRobot v2.1 shard exports → renumber → v3.0 → relative_statistics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

from robot_data_processing.export_lerobot_v30 import (  # noqa: E402
    install_v30_dataset_into_root,
)
from robot_data_processing.relative_statistics import (  # noqa: E402
    prepare_relative_action_statistics,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _rewrite_parquet_indices(
    src: Path,
    dst: Path,
    *,
    new_episode_index: int,
    global_index_start: int,
) -> int:
    table = pq.read_table(src)
    n = table.num_rows
    cols: dict[str, pa.Array] = {}
    for name in table.column_names:
        if name == "episode_index":
            cols[name] = pa.array(np.full(n, int(new_episode_index), dtype=np.int64))
        elif name == "index":
            cols[name] = pa.array(
                np.arange(global_index_start, global_index_start + n, dtype=np.int64)
            )
        else:
            cols[name] = table[name]
    out = pa.table(cols, metadata=table.schema.metadata)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)
    return int(n)


def _v21_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _v21_video_path(root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def _collect_shard_episodes(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_src: set[int] = set()
    for shard in shard_dirs:
        info_path = shard / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing {info_path} (shard must be v2.1 export)")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if str(info.get("codebase_version", "v2.1")) != "v2.1":
            raise ValueError(
                f"Shard {shard} has codebase_version={info.get('codebase_version')!r}; "
                "expected v2.1 (use --export-shard-only)"
            )
        episodes = _load_jsonl(shard / "meta" / "episodes.jsonl")
        source_map = {
            int(r["episode_index"]): r
            for r in _load_jsonl(shard / "meta" / "episode_source_map.jsonl")
        }
        video_keys = [
            key
            for key, spec in (info.get("features") or {}).items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ]
        for row in episodes:
            ep_id = int(row["episode_index"])
            src_row = source_map.get(ep_id, {})
            src_id = int(src_row.get("source_episode_index", ep_id))
            if src_id in seen_src:
                raise ValueError(f"Duplicate source_episode_index={src_id} across shards")
            seen_src.add(src_id)
            parquet = _v21_parquet_path(shard, ep_id)
            if not parquet.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet}")
            videos = {}
            for key in video_keys:
                vp = _v21_video_path(shard, ep_id, key)
                if not vp.exists():
                    raise FileNotFoundError(f"Missing video: {vp}")
                videos[key] = vp
            collected.append(
                {
                    "source_episode_index": src_id,
                    "shard_episode_index": ep_id,
                    "shard_root": shard,
                    "length": int(row["length"]),
                    "tasks": list(row.get("tasks") or []),
                    "camera_intrinsics": row.get("camera_intrinsics"),
                    "parquet": parquet,
                    "videos": videos,
                    "task_index": src_row.get("task_index"),
                    "part": src_row.get("part"),
                    "task": src_row.get("task"),
                    "dataset_root": src_row.get("dataset_root"),
                    "info": info,
                }
            )
    collected.sort(key=lambda r: int(r["source_episode_index"]))
    return collected


def _merge_tasks(shard_dirs: list[Path], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union tasks.jsonl from shards; fall back to names seen on episodes."""
    index_to_name: dict[int, str] = {}
    name_to_index: dict[str, int] = {}
    for shard in shard_dirs:
        for row in _load_jsonl(shard / "meta" / "tasks.jsonl"):
            idx = int(row["task_index"])
            name = str(row["task"])
            index_to_name.setdefault(idx, name)
            name_to_index.setdefault(name, idx)
    next_idx = (max(index_to_name) + 1) if index_to_name else 0
    for row in rows:
        tasks = row.get("tasks") or []
        if not tasks and row.get("task"):
            tasks = [row["task"]]
        for name in tasks:
            name = str(name)
            if name not in name_to_index:
                name_to_index[name] = next_idx
                index_to_name[next_idx] = name
                next_idx += 1
    if not index_to_name:
        index_to_name[0] = "unknown"
        name_to_index["unknown"] = 0
    return [
        {"task_index": idx, "task": index_to_name[idx]}
        for idx in sorted(index_to_name)
    ]


def _build_v21_info(
    template: dict[str, Any],
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    info = dict(template)
    info["codebase_version"] = "v2.1"
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["chunks_size"] = 1000
    info["total_chunks"] = max(1, (total_episodes + 999) // 1000)
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    return info


def merge_shards(
    *,
    shard_dirs: list[Path],
    output_dir: Path,
    config_path: Path | None = None,
    keep_v21: bool = False,
    skip_relative: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    shard_dirs = [Path(p).resolve() for p in shard_dirs]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    export_cfg = cfg.get("export") or {}
    v30_cfg = export_cfg.get("v30") or {}
    rel_cfg = cfg.get("relative_statistics") or {}

    print(f"[merge] collecting {len(shard_dirs)} shards...", flush=True)
    rows = _collect_shard_episodes(shard_dirs)
    if not rows:
        raise RuntimeError("No episodes found in shards")
    print(f"[merge] kept episodes={len(rows)}", flush=True)

    v21_root = output_dir / ".merge_v21_staging"
    if v21_root.exists():
        shutil.rmtree(v21_root)
    v21_root.mkdir(parents=True, exist_ok=True)

    tasks_out = _merge_tasks(shard_dirs, rows)
    task_name_to_index = {r["task"]: int(r["task_index"]) for r in tasks_out}
    episodes_out: list[dict[str, Any]] = []
    source_map_out: list[dict[str, Any]] = []
    global_index = 0
    link_mode_counts = {"hardlink": 0, "copy": 0}

    for new_i, row in enumerate(rows):
        n = _rewrite_parquet_indices(
            row["parquet"],
            _v21_parquet_path(v21_root, new_i),
            new_episode_index=new_i,
            global_index_start=global_index,
        )
        for key, src_video in row["videos"].items():
            mode = _link_or_copy(src_video, _v21_video_path(v21_root, new_i, key))
            link_mode_counts[mode] = link_mode_counts.get(mode, 0) + 1
        tasks = list(row.get("tasks") or [])
        if not tasks and row.get("task"):
            tasks = [str(row["task"])]
        if not tasks:
            tasks = [next(iter(task_name_to_index))]
        ep_row: dict[str, Any] = {
            "episode_index": new_i,
            "tasks": tasks,
            "length": n,
        }
        if row.get("camera_intrinsics"):
            ep_row["camera_intrinsics"] = row["camera_intrinsics"]
        episodes_out.append(ep_row)
        source_map_out.append(
            {
                "episode_index": new_i,
                "source_episode_index": row["source_episode_index"],
                "part": row.get("part"),
                "task": row.get("task") or (tasks[0] if tasks else None),
                "task_index": task_name_to_index.get(tasks[0]),
                "dataset_root": row.get("dataset_root"),
                "shard_root": str(row["shard_root"]),
                "shard_episode_index": row["shard_episode_index"],
            }
        )
        global_index += n
        if (new_i + 1) % 50 == 0 or new_i + 1 == len(rows):
            print(f"[merge] staged {new_i + 1}/{len(rows)}", flush=True)

    info = _build_v21_info(
        rows[0]["info"],
        total_episodes=len(rows),
        total_frames=global_index,
        total_tasks=len(tasks_out),
    )
    meta_dir = v21_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in tasks_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (meta_dir / "episode_source_map.jsonl").open("w", encoding="utf-8") as f:
        for row in source_map_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t_stage = time.perf_counter() - t0
    print(
        f"[merge] v2.1 staging done in {t_stage:.1f}s "
        f"(episodes={len(rows)} frames={global_index} link={link_mode_counts})",
        flush=True,
    )

    t_v30 = time.perf_counter()
    v30_summary = install_v30_dataset_into_root(
        v21_root,
        output_dir,
        data_file_size_in_mb=int(v30_cfg.get("data_file_size_in_mb", 100)),
        video_file_size_in_mb=int(v30_cfg.get("video_file_size_in_mb", 200)),
    )
    # Preserve source map alongside v3 meta
    shutil.copy2(
        v21_root / "meta" / "episode_source_map.jsonl",
        output_dir / "meta" / "episode_source_map.jsonl",
    )
    (output_dir / "lerobot_v30_export.json").write_text(
        json.dumps(v30_summary, indent=2), encoding="utf-8"
    )
    print(
        f"[merge] v3.0 done in {time.perf_counter() - t_v30:.1f}s → {output_dir}",
        flush=True,
    )

    rel_payload = None
    if not skip_relative and bool(rel_cfg.get("enabled", True)):
        horizon_s = float(rel_cfg.get("statistics_horizon_seconds", 2.0))
        t_rel = time.perf_counter()
        rel_payload = prepare_relative_action_statistics(
            output_dir,
            statistics_horizon_seconds=horizon_s,
            rebuild=True,
        )
        print(
            f"[merge] relative_statistics in {time.perf_counter() - t_rel:.1f}s "
            f"anchors={rel_payload['anchor_count']}",
            flush=True,
        )

    if keep_v21:
        final_v21 = output_dir / "v21_merged"
        if final_v21.exists():
            shutil.rmtree(final_v21)
        shutil.move(str(v21_root), str(final_v21))
    else:
        shutil.rmtree(v21_root)

    summary = {
        "output_dir": str(output_dir),
        "shard_dirs": [str(p) for p in shard_dirs],
        "total_episodes": len(rows),
        "total_frames": global_index,
        "link_mode_counts": link_mode_counts,
        "lerobot_v30": v30_summary,
        "relative_statistics": (
            {
                "anchor_count": rel_payload["anchor_count"],
                "source_frame_count": rel_payload["source_frame_count"],
            }
            if rel_payload
            else None
        ),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (output_dir / "merge_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[merge] done in {summary['elapsed_sec']}s", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        action="append",
        dest="shard_dirs",
        type=Path,
        required=True,
        help="Shard output directory (repeat for each shard)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "run_config_v30_n350.yaml",
        help="Dataclean yaml for v30 sizes / relative_statistics",
    )
    parser.add_argument(
        "--keep-v21",
        action="store_true",
        help="Keep merged v2.1 staging under output_dir/v21_merged",
    )
    parser.add_argument(
        "--skip-relative",
        action="store_true",
        help="Skip statistics_relative.json",
    )
    args = parser.parse_args()
    merge_shards(
        shard_dirs=list(args.shard_dirs),
        output_dir=args.output_dir,
        config_path=args.config,
        keep_v21=bool(args.keep_v21),
        skip_relative=bool(args.skip_relative),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

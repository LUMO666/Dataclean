#!/usr/bin/env python3
"""Sample episodes and compare measured FPS from timestamp / timestamps vs info.json fps."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/project_rlinf_hs/liuweilin/Dataclean/datasets/humanoid_merged/manual_review/results"
)


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def episode_parquet(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_root / info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )


def as_1d(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        arr = np.stack(values)
    elif isinstance(first, (list, tuple)):
        arr = np.asarray(values.tolist())
    else:
        arr = np.asarray(values)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def fps_from_diffs(diffs: np.ndarray, *, scale: float = 1.0) -> dict[str, float]:
    """scale: multiply raw diffs to convert to seconds (e.g. 1e-9 for nanoseconds)."""
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return {
            "fps_median": float("nan"),
            "fps_mean": float("nan"),
            "dt_median_s": float("nan"),
            "dt_mean_s": float("nan"),
            "dt_std_s": float("nan"),
            "dt_min_s": float("nan"),
            "dt_max_s": float("nan"),
            "n_intervals": 0.0,
        }
    dt = diffs * scale
    return {
        "fps_median": float(1.0 / np.median(dt)),
        "fps_mean": float(1.0 / np.mean(dt)),
        "dt_median_s": float(np.median(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "dt_std_s": float(np.std(dt)),
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "n_intervals": float(len(dt)),
    }


def infer_timestamps_scale(diffs: np.ndarray) -> float:
    """Return multiplier to convert timestamps diffs to seconds."""
    med = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
    # ~1/30 s intervals encoded as ns / us / ms / s
    candidates = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
    }
    best_name, best_scale, best_err = "ns", 1e-9, float("inf")
    for name, scale in candidates.items():
        dt = med * scale
        if dt <= 0:
            continue
        err = abs(1.0 / dt - 30.0)  # proximity to common robot fps; only for unit guess
        # also accept if dt is in a sane frame-interval band
        if 1e-4 < dt < 1.0 and err < best_err:
            best_name, best_scale, best_err = name, scale, err
    return best_scale


def analyze_episode(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    *,
    declared_fps: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    path = episode_parquet(dataset_root, info, episode_index)
    df = pd.read_parquet(path, columns=["timestamp", "timestamps"])
    ts_rel = as_1d(df["timestamp"])
    ts_abs = as_1d(df["timestamps"])
    n = len(ts_rel)

    rel_diffs = np.diff(ts_rel)
    abs_diffs = np.diff(ts_abs)
    abs_scale = infer_timestamps_scale(abs_diffs)

    rel_stats = fps_from_diffs(rel_diffs, scale=1.0)
    abs_stats = fps_from_diffs(abs_diffs, scale=abs_scale)

    # span-based FPS: (n-1) / duration
    rel_span_fps = float((n - 1) / (ts_rel[-1] - ts_rel[0])) if n > 1 and ts_rel[-1] > ts_rel[0] else float("nan")
    abs_dur = (ts_abs[-1] - ts_abs[0]) * abs_scale
    abs_span_fps = float((n - 1) / abs_dur) if n > 1 and abs_dur > 0 else float("nan")

    def matches(fps: float) -> bool:
        if not np.isfinite(fps):
            return False
        return abs(fps - declared_fps) <= max(atol, rtol * abs(declared_fps))

    row = {
        "episode_index": episode_index,
        "num_frames": n,
        "declared_fps": declared_fps,
        "timestamp_fps_median": rel_stats["fps_median"],
        "timestamp_fps_mean": rel_stats["fps_mean"],
        "timestamp_fps_span": rel_span_fps,
        "timestamp_dt_median_s": rel_stats["dt_median_s"],
        "timestamp_dt_std_s": rel_stats["dt_std_s"],
        "timestamps_unit_scale": abs_scale,
        "timestamps_fps_median": abs_stats["fps_median"],
        "timestamps_fps_mean": abs_stats["fps_mean"],
        "timestamps_fps_span": abs_span_fps,
        "timestamps_dt_median_s": abs_stats["dt_median_s"],
        "timestamps_dt_std_s": abs_stats["dt_std_s"],
        "timestamps_dt_min_s": abs_stats["dt_min_s"],
        "timestamps_dt_max_s": abs_stats["dt_max_s"],
        "match_timestamp_median": matches(rel_stats["fps_median"]),
        "match_timestamps_median": matches(abs_stats["fps_median"]),
        "match_timestamp_span": matches(rel_span_fps),
        "match_timestamps_span": matches(abs_span_fps),
        "parquet": str(path),
    }
    return row


def sample_episodes(
    dataset_root: Path,
    info: dict[str, Any],
    n: int,
    seed: int,
    *,
    probe_factor: int = 5,
) -> list[int]:
    total = int(info["total_episodes"])
    rng = random.Random(seed)
    picked: list[int] = []
    for ep in rng.sample(range(total), min(total, n * probe_factor)):
        if episode_parquet(dataset_root, info, ep).exists():
            picked.append(ep)
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise RuntimeError(f"only found {len(picked)} readable episodes, need {n}")
    return picked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=0.5, help="Absolute FPS tolerance")
    p.add_argument("--rtol", type=float, default=0.02, help="Relative FPS tolerance")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset_root)
    declared = float(info["fps"])
    episodes = args.episodes or sample_episodes(
        args.dataset_root, info, args.num_samples, args.seed
    )

    rows = [
        analyze_episode(
            args.dataset_root,
            info,
            ep,
            declared_fps=declared,
            atol=args.atol,
            rtol=args.rtol,
        )
        for ep in episodes
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "fps_check_sample20.json"
    out_csv = args.output_dir / "fps_check_sample20.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    n = len(rows)
    match_ts = sum(r["match_timestamp_median"] for r in rows)
    match_abs = sum(r["match_timestamps_median"] for r in rows)
    match_ts_span = sum(r["match_timestamp_span"] for r in rows)
    match_abs_span = sum(r["match_timestamps_span"] for r in rows)

    print(f"dataset={args.dataset_root}")
    print(f"declared_fps={declared}")
    print(f"tolerance: atol={args.atol}, rtol={args.rtol}")
    print(f"episodes={episodes}")
    print()
    print(
        f"{'ep':>8} {'N':>5} {'ts_med':>8} {'ts_span':>8} "
        f"{'abs_med':>8} {'abs_span':>8} {'abs_dtσ':>8} {'ts?':>4} {'abs?':>4}"
    )
    for r in rows:
        print(
            f"{r['episode_index']:8d} {r['num_frames']:5d} "
            f"{r['timestamp_fps_median']:8.3f} {r['timestamp_fps_span']:8.3f} "
            f"{r['timestamps_fps_median']:8.3f} {r['timestamps_fps_span']:8.3f} "
            f"{r['timestamps_dt_std_s']*1e3:7.2f}ms "
            f"{'Y' if r['match_timestamp_median'] else 'N':>4} "
            f"{'Y' if r['match_timestamps_median'] else 'N':>4}"
        )

    print()
    print(f"timestamp   median match: {match_ts}/{n}")
    print(f"timestamp   span   match: {match_ts_span}/{n}")
    print(f"timestamps  median match: {match_abs}/{n}")
    print(f"timestamps  span   match: {match_abs_span}/{n}")
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")

    # summary verdict
    if match_ts == n and match_abs == n:
        print(f"VERDICT: both timestamp and timestamps median FPS match info.fps={declared}")
    else:
        print(
            f"VERDICT: NOT fully matched — "
            f"timestamp {match_ts}/{n}, timestamps {match_abs}/{n} "
            f"(info.fps={declared})"
        )


if __name__ == "__main__":
    main()

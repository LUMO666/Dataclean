#!/usr/bin/env python3
"""Evenly sample episodes and plot gripper width distributions (EgoDex 20d state/action)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    ACTION_KEY,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    STATE_KEY,
    as_lr_gripper,
    even_sample_indices,
    episode_parquet,
    load_grippers,
    load_info,
    resolve_dataset_root,
)


def apply_affine(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return scale * values + offset


def summarize(arr: np.ndarray) -> dict[str, Any]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0}
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "percentiles": {str(q): float(np.percentile(finite, q)) for q in qs},
        "frac_near_zero": float(np.mean(np.abs(finite) < 1e-3)),
        "frac_gt_0_05": float(np.mean(finite > 0.05)),
        "frac_gt_0_09": float(np.mean(finite > 0.09)),
    }


def load_sample(
    dataset_root: Path,
    info: dict[str, Any],
    episode_indices: list[int],
) -> dict[str, Any]:
    action_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    used: list[int] = []
    skipped: list[dict[str, Any]] = []

    for ep in episode_indices:
        path = episode_parquet(dataset_root, info, ep)
        if not path.exists():
            skipped.append({"episode_index": ep, "reason": "missing_parquet"})
            continue
        try:
            action, state = load_grippers(path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"episode_index": ep, "reason": str(exc)})
            continue
        action_chunks.append(action)
        state_chunks.append(state)
        used.append(ep)

    if not used:
        raise RuntimeError("no episodes loaded")

    action = np.concatenate(action_chunks, axis=0)
    state = np.concatenate(state_chunks, axis=0)
    return {
        "used_episodes": used,
        "skipped": skipped,
        "action": action,
        "state": state,
        "frames_per_episode": [int(a.shape[0]) for a in action_chunks],
    }


def _plot_lr_hist(ax_l, ax_r, values, *, title_prefix, xlabel, left_color, right_color, bins=80):
    for ax, col, side, color in (
        (ax_l, 0, "left", left_color),
        (ax_r, 1, "right", right_color),
    ):
        finite = values[:, col][np.isfinite(values[:, col])]
        ax.hist(finite, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(f"{title_prefix} {side}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.axvline(np.mean(finite), color="k", linestyle="--", linewidth=1, label=f"mean={np.mean(finite):.4f}")
        ax.axvline(np.median(finite), color="gray", linestyle=":", linewidth=1, label=f"median={np.median(finite):.4f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)


def plot_distribution(action, state, output_dir, *, n_episodes, transform_label, action_affine, state_affine):
    for name, values, label in (
        ("action", action, transform_label["action"]),
        ("state", state, transform_label["state"]),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        _plot_lr_hist(
            axes[0],
            axes[1],
            values,
            title_prefix=name,
            xlabel=label,
            left_color="#1f77b4" if name == "action" else "#2ca02c",
            right_color="#ff7f0e" if name == "action" else "#d62728",
        )
        fig.suptitle(f"{name} gripper width | episodes={n_episodes} frames={values.shape[0]:,}", fontsize=13)
        fig.savefig(output_dir / f"gripper_{name}_distribution_hist.png", dpi=160)
        plt.close(fig)


def per_episode_means(values: np.ndarray, frames_per_episode: list[int]) -> np.ndarray:
    means: list[np.ndarray] = []
    offset = 0
    for n in frames_per_episode:
        means.append(values[offset : offset + n].mean(axis=0))
        offset += n
    return np.asarray(means, dtype=np.float64)


def resolve_existing_episodes(
    dataset_root: Path,
    info: dict[str, Any],
    candidates: list[int],
    *,
    search_radius: int = 50,
) -> list[int]:
    total = int(info["total_episodes"])
    resolved: list[int] = []
    used: set[int] = set()
    for ep in candidates:
        if ep in used:
            continue
        path = episode_parquet(dataset_root, info, ep)
        if path.exists():
            resolved.append(ep)
            used.add(ep)
            continue
        found = None
        for d in range(1, search_radius + 1):
            for cand in (ep - d, ep + d):
                if cand < 0 or cand >= total or cand in used:
                    continue
                if episode_parquet(dataset_root, info, cand).exists():
                    found = cand
                    break
            if found is not None:
                break
        if found is not None:
            resolved.append(found)
            used.add(found)
    return resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "gripper_check")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--action-offset", type=float, default=0.0)
    p.add_argument("--state-scale", type=float, default=1.0)
    p.add_argument("--state-offset", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    total = int(info["total_episodes"])
    candidates = even_sample_indices(total, args.num_samples)
    episodes = resolve_existing_episodes(dataset_root, info, candidates)

    print(f"dataset={dataset_root}")
    print(f"total_episodes={total}")
    print(f"requested={args.num_samples} resolved={len(episodes)}")

    data = load_sample(dataset_root, info, episodes)
    action = apply_affine(data["action"], args.action_scale, args.action_offset)
    state = apply_affine(data["state"], args.state_scale, args.state_offset)
    used = data["used_episodes"]

    action_label = f"c' = {args.action_scale}*c + {args.action_offset}"
    state_label = f"c' = {args.state_scale}*c + {args.state_offset}"

    summary = {
        "dataset_root": str(dataset_root),
        "declared_fps": info.get("fps"),
        "total_episodes": total,
        "num_loaded": len(used),
        "num_frames": int(action.shape[0]),
        "episode_indices": used,
        "skipped": data["skipped"],
        "action_left": summarize(action[:, 0]),
        "action_right": summarize(action[:, 1]),
        "state_left": summarize(state[:, 0]),
        "state_right": summarize(state[:, 1]),
        "action_left_right_corr": float(np.corrcoef(action[:, 0], action[:, 1])[0, 1]),
        "state_left_right_corr": float(np.corrcoef(state[:, 0], state[:, 1])[0, 1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "gripper_distribution_stats.json"
    csv_path = args.output_dir / "gripper_sampled_episodes.csv"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    action_means = per_episode_means(action, data["frames_per_episode"])
    state_means = per_episode_means(state, data["frames_per_episode"])
    pd.DataFrame(
        {
            "episode_index": used,
            "num_frames": data["frames_per_episode"],
            "action_left_mean": action_means[:, 0],
            "action_right_mean": action_means[:, 1],
            "state_left_mean": state_means[:, 0],
            "state_right_mean": state_means[:, 1],
        }
    ).to_csv(csv_path, index=False)

    plot_distribution(
        action,
        state,
        args.output_dir,
        n_episodes=len(used),
        transform_label={"action": action_label, "state": state_label},
        action_affine=(args.action_scale, args.action_offset),
        state_affine=(args.state_scale, args.state_offset),
    )

    print(f"wrote {stats_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

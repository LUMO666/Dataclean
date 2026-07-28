#!/usr/bin/env python3
"""Evenly sample episodes and plot gripper value distributions.

Optional affine transform before stats/plots:
  c' = scale * c + offset
with separate (scale, offset) for action and state.
"""

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

DEFAULT_DATASET = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/project_rlinf_hs/liuweilin/Dataclean/datasets/humanoid_merged/manual_review/results/gripper_check"
)
ACTION_KEY = "action.effector.position"
STATE_KEY = "observation.state.effector.position"


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def episode_parquet(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_root / info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )


def even_sample_indices(total: int, n: int) -> list[int]:
    """Near-uniform sample over [0, total) without replacement."""
    n = min(n, total)
    if n == total:
        return list(range(total))
    raw = np.linspace(0, total - 1, n)
    idxs = np.unique(np.rint(raw).astype(int))
    if len(idxs) < n:
        unused = np.setdiff1d(np.arange(total), idxs, assume_unique=False)
        need = n - len(idxs)
        fill = unused[np.linspace(0, len(unused) - 1, need).astype(int)]
        idxs = np.sort(np.concatenate([idxs, fill]))
    return idxs.tolist()


def as_lr(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        arr = np.stack(values)
    elif isinstance(first, (list, tuple)):
        arr = np.asarray(values.tolist())
    else:
        arr = np.asarray(values)[:, None]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"unexpected gripper shape {arr.shape}")
    return arr[:, :2]


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
            df = pd.read_parquet(path, columns=[ACTION_KEY, STATE_KEY])
            action = as_lr(df[ACTION_KEY])
            state = as_lr(df[STATE_KEY])
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


def _plot_lr_hist(
    ax_l,
    ax_r,
    values: np.ndarray,
    *,
    title_prefix: str,
    xlabel: str,
    left_color: str,
    right_color: str,
    bins: int = 80,
) -> None:
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


def plot_action_distribution(
    action: np.ndarray,
    output_path: Path,
    *,
    n_episodes: int,
    n_frames: int,
    transform_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    _plot_lr_hist(
        axes[0],
        axes[1],
        action,
        title_prefix="action",
        xlabel=transform_label,
        left_color="#1f77b4",
        right_color="#ff7f0e",
    )
    fig.suptitle(
        f"Action gripper distribution | episodes={n_episodes} frames={n_frames:,}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_state_distribution(
    state: np.ndarray,
    output_path: Path,
    *,
    n_episodes: int,
    n_frames: int,
    transform_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    _plot_lr_hist(
        axes[0],
        axes[1],
        state,
        title_prefix="state",
        xlabel=transform_label,
        left_color="#2ca02c",
        right_color="#d62728",
    )
    fig.suptitle(
        f"State gripper distribution | episodes={n_episodes} frames={n_frames:,}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_lr_scatter(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    rng = np.random.default_rng(0)
    n = values.shape[0]
    take = rng.choice(n, size=min(n, 50000), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.scatter(
        values[take, 0],
        values[take, 1],
        s=2,
        alpha=0.15,
        c="#444444",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_episode_means(
    episode_means: np.ndarray,
    output_path: Path,
    *,
    title_prefix: str,
    xlabel: str,
    left_color: str,
    right_color: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(episode_means[:, 0], bins=40, color=left_color, alpha=0.85)
    axes[0].set_title(f"Per-episode mean {title_prefix} left")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("#episodes")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(episode_means[:, 1], bins=40, color=right_color, alpha=0.85)
    axes[1].set_title(f"Per-episode mean {title_prefix} right")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def per_episode_means(values: np.ndarray, frames_per_episode: list[int]) -> np.ndarray:
    means: list[np.ndarray] = []
    offset = 0
    for n in frames_per_episode:
        means.append(values[offset : offset + n].mean(axis=0))
        offset += n
    return np.asarray(means, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--action-scale", type=float, default=1.0, help="Affine scale for action")
    p.add_argument("--action-offset", type=float, default=0.0, help="Affine offset for action")
    p.add_argument("--state-scale", type=float, default=1.0, help="Affine scale for state")
    p.add_argument("--state-offset", type=float, default=0.0, help="Affine offset for state")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset_root)
    total = int(info["total_episodes"])
    candidates = even_sample_indices(total, args.num_samples)
    episodes = resolve_existing_episodes(args.dataset_root, info, candidates)

    print(f"dataset={args.dataset_root}")
    print(f"total_episodes={total}")
    print(f"requested={args.num_samples} resolved={len(episodes)}")
    print(f"first/last episode={episodes[0]}/{episodes[-1]}")
    print(
        "transform: "
        f"action c'={args.action_scale}*c+{args.action_offset}, "
        f"state c'={args.state_scale}*c+{args.state_offset}"
    )

    data = load_sample(args.dataset_root, info, episodes)
    action_raw = data["action"]
    state_raw = data["state"]
    used = data["used_episodes"]

    action = apply_affine(action_raw, args.action_scale, args.action_offset)
    state = apply_affine(state_raw, args.state_scale, args.state_offset)

    action_means = per_episode_means(action, data["frames_per_episode"])
    state_means = per_episode_means(state, data["frames_per_episode"])

    action_label = f"c' = {args.action_scale}*c + {args.action_offset}"
    state_label = f"c' = {args.state_scale}*c + {args.state_offset}"

    summary = {
        "dataset_root": str(args.dataset_root),
        "declared_fps": info.get("fps"),
        "total_episodes": total,
        "num_requested": args.num_samples,
        "num_loaded": len(used),
        "num_frames": int(action.shape[0]),
        "episode_indices": used,
        "skipped": data["skipped"],
        "transform": {
            "action": {"scale": args.action_scale, "offset": args.action_offset},
            "state": {"scale": args.state_scale, "offset": args.state_offset},
        },
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
    action_hist_path = args.output_dir / "gripper_action_distribution_hist.png"
    state_hist_path = args.output_dir / "gripper_state_distribution_hist.png"
    action_scatter_path = args.output_dir / "gripper_action_lr_scatter.png"
    state_scatter_path = args.output_dir / "gripper_state_lr_scatter.png"
    action_epmean_path = args.output_dir / "gripper_action_per_episode_mean_hist.png"
    state_epmean_path = args.output_dir / "gripper_state_per_episode_mean_hist.png"

    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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

    plot_action_distribution(
        action,
        action_hist_path,
        n_episodes=len(used),
        n_frames=action.shape[0],
        transform_label=action_label,
    )
    plot_state_distribution(
        state,
        state_hist_path,
        n_episodes=len(used),
        n_frames=state.shape[0],
        transform_label=state_label,
    )
    plot_lr_scatter(
        action,
        action_scatter_path,
        title="action left vs right (subsample)",
        xlabel=f"action left ({action_label})",
        ylabel=f"action right ({action_label})",
    )
    plot_lr_scatter(
        state,
        state_scatter_path,
        title="state left vs right (subsample)",
        xlabel=f"state left ({state_label})",
        ylabel=f"state right ({state_label})",
    )
    plot_episode_means(
        action_means,
        action_epmean_path,
        title_prefix="action",
        xlabel=action_label,
        left_color="#1f77b4",
        right_color="#ff7f0e",
    )
    plot_episode_means(
        state_means,
        state_epmean_path,
        title_prefix="state",
        xlabel=state_label,
        left_color="#2ca02c",
        right_color="#d62728",
    )

    def brief(name: str, s: dict[str, Any]) -> str:
        return (
            f"{name}: n={s['count']:,} min={s['min']:.4f} max={s['max']:.4f} "
            f"mean={s['mean']:.4f} std={s['std']:.4f} "
            f"p50={s['percentiles']['50']:.4f} p95={s['percentiles']['95']:.4f}"
        )

    print(brief("action L", summary["action_left"]))
    print(brief("action R", summary["action_right"]))
    print(brief("state  L", summary["state_left"]))
    print(brief("state  R", summary["state_right"]))
    print(
        "corr action L-R={:.3f} state L-R={:.3f}".format(
            summary["action_left_right_corr"],
            summary["state_left_right_corr"],
        )
    )
    print(f"wrote {stats_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {action_hist_path}")
    print(f"wrote {state_hist_path}")
    print(f"wrote {action_scatter_path}")
    print(f"wrote {state_scatter_path}")
    print(f"wrote {action_epmean_path}")
    print(f"wrote {state_epmean_path}")


if __name__ == "__main__":
    main()

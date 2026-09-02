#!/usr/bin/env python3
"""Plot state vs action curves for EgoDex lerobot v21 episodes (20d pose+gripper).

Examples:
  python plot_state_action_curves.py --episodes 0 5
  python plot_state_action_curves.py --dataset-root .../part1/add_remove_lid --episodes 0
"""

from __future__ import annotations

import argparse
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
    STATE_ACTION_NAMES,
    STATE_KEY,
    array_column,
    episode_parquet,
    load_info,
    resolve_dataset_root,
)


def as_array(df: pd.DataFrame, key: str) -> np.ndarray:
    return array_column(df, key).astype(np.float64)


def plot_state_action_grid(
    t: np.ndarray,
    state: np.ndarray,
    action: np.ndarray | None,
    labels: list[str],
    title: str,
    out: Path,
    *,
    ncols: int | None = None,
    ylabel: str = "",
) -> None:
    dims = state.shape[1]
    ncols = ncols or min(dims, 5)
    nrows = int(np.ceil(dims / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for c in range(dims):
        r, col = divmod(c, ncols)
        ax = axes[r][col]
        ax.plot(t, state[:, c], color="#2ca02c", lw=1.2, label="state")
        if action is not None:
            ax.plot(t, action[:, c], color="#1f77b4", lw=1.0, ls="--", label="action")
        ax.set_title(f"[{c}] {labels[c]}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if r == nrows - 1:
            ax.set_xlabel("time (s)")
        if col == 0 and ylabel:
            ax.set_ylabel(ylabel)
        if c == 0:
            ax.legend(fontsize=8, loc="upper right")
    for c in range(dims, nrows * ncols):
        r, col = divmod(c, ncols)
        axes[r][col].axis("off")
    fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_state_action_stacked(
    t: np.ndarray,
    state: np.ndarray,
    action: np.ndarray | None,
    labels: list[str],
    title: str,
    out: Path,
) -> None:
    dims = state.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(12, 1.55 * dims), sharex=True, constrained_layout=True)
    if dims == 1:
        axes = [axes]
    for c, lab in enumerate(labels):
        ax = axes[c]
        ax.plot(t, state[:, c], color="#2ca02c", lw=1.2, label="state")
        if action is not None:
            ax.plot(t, action[:, c], color="#1f77b4", lw=1.0, ls="--", label="action")
        ax.set_ylabel(f"[{c}] {lab}", fontsize=8)
        ax.grid(True, alpha=0.25)
        if c == 0:
            ax.legend(fontsize=8, loc="upper right")
        if c == dims - 1:
            ax.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_episode(dataset_root: Path, info: dict[str, Any], ep: int, out_dir: Path) -> list[Path]:
    pq = episode_parquet(dataset_root, info, ep)
    df = pd.read_parquet(pq, columns=[STATE_KEY, ACTION_KEY, "timestamp"])
    n = len(df)
    fps = float(info.get("fps", 30))
    t = as_array(df, "timestamp").reshape(-1) if "timestamp" in df.columns else np.arange(n) / fps
    state = as_array(df, STATE_KEY)
    action = as_array(df, ACTION_KEY)
    print(f"ep={ep} parquet={pq} frames={n} t=[{t[0]:.3f},{t[-1]:.3f}]s")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    groups = [
        ("left_hand", slice(0, 10), STATE_ACTION_NAMES[:10]),
        ("right_hand", slice(10, 20), STATE_ACTION_NAMES[10:20]),
        ("gripper_width", [9, 19], ["left_width", "right_width"]),
    ]

    for group_name, dim_sel, labels in groups:
        if isinstance(dim_sel, slice):
            st = state[:, dim_sel]
            ac = action[:, dim_sel]
        else:
            st = state[:, dim_sel]
            ac = action[:, dim_sel]
        title = f"ep {ep:06d} | {group_name} | N={n}"
        out = out_dir / f"episode_{ep:06d}_{group_name}_state_action.png"
        plot_state_action_grid(t, st, ac, labels, title, out, ylabel=group_name)
        written.append(out)
        out2 = out_dir / f"episode_{ep:06d}_{group_name}_state_action_stacked.png"
        plot_state_action_stacked(t, st, ac, labels, f"{title} state vs action", out2)
        written.append(out2)

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    for ep in args.episodes:
        out_dir = args.output_dir / f"ep{ep}_all_state_action"
        written = plot_episode(dataset_root, info, ep, out_dir)
        for path in written:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

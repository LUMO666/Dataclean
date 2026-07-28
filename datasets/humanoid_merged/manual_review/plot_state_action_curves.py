#!/usr/bin/env python3
"""Plot state vs action curves for dreamzero humanoid_merged episodes.

Dimension indices in plot titles are 0-based (e.g. fl_joint0 .. fl_joint5).

Examples:
  python plot_state_action_curves.py --episodes 12098 23003
  python plot_state_action_curves.py --dataset-root .../humanoid_merged --episodes 9429
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASET = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged")
DEFAULT_OUTPUT = Path(
    "/mnt/project_rlinf_hs/liuweilin/Dataclean/datasets/humanoid_merged/manual_review/results"
)


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def resolve_parquet(dataset_root: Path, info: dict[str, Any], ep: int) -> Path:
    chunk = ep // int(info["chunks_size"])
    path = dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    if path.exists():
        return path
    matches = list(dataset_root.glob(f"data/**/episode_{ep:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(path)
    return matches[0]


def as_array(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float64)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float64)
    return values.astype(np.float64)[:, None]


def to_zero_based_labels(names: list[str] | None, dims: int, prefix: str = "dim") -> list[str]:
    """Make joint dimension indices 0-based for display (joint1 -> joint0)."""
    if not names:
        return [f"{prefix}{i}" for i in range(dims)]
    out: list[str] = []
    for i, name in enumerate(list(names)[:dims]):
        # fl_joint1 / fr_joint6 / ... -> subtract 1 from the trailing joint index
        def _repl(m: re.Match[str]) -> str:
            return f"{m.group(1)}{int(m.group(2)) - 1}"

        labeled = re.sub(r"(joint)(\d+)", _repl, str(name), flags=re.IGNORECASE)
        # if name had no jointN token, fall back to 0-based dim index suffix
        if labeled == str(name) and re.fullmatch(r".*\d+$", str(name)):
            labeled = re.sub(r"(\d+)$", lambda m: str(int(m.group(1)) - 1), str(name))
        out.append(labeled)
    while len(out) < dims:
        out.append(f"{prefix}{len(out)}")
    return out


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
    ncols = ncols or min(dims, 6)
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
    pq = resolve_parquet(dataset_root, info, ep)
    df = pd.read_parquet(pq)
    n = len(df)
    fps = float(info.get("fps", 30))
    t = as_array(df["timestamp"]).reshape(-1) if "timestamp" in df.columns else np.arange(n) / fps
    print(f"ep={ep} parquet={pq} frames={n} t=[{t[0]:.3f},{t[-1]:.3f}]s")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    groups = [
        (
            "arm_position",
            "observation.state.arm.position",
            "action.arm.position",
            info["features"]["observation.state.arm.position"].get("names"),
        ),
        (
            "effector_position",
            "observation.state.effector.position",
            "action.effector.position",
            ["left_gripper", "right_gripper"],
        ),
    ]

    for group_name, sk, ak, raw_names in groups:
        state = as_array(df[sk])
        action = as_array(df[ak])
        if state.ndim == 1:
            state = state[:, None]
        if action.ndim == 1:
            action = action[:, None]
        dims = state.shape[1]
        labels = to_zero_based_labels(raw_names, dims, prefix=f"{group_name}_")
        print(f"  {group_name}: dims={dims} labels={labels}")

        title = f"ep {ep:06d} | {group_name} | N={n}"
        out = out_dir / f"episode_{ep:06d}_{group_name}_state_action.png"
        plot_state_action_grid(t, state, action, labels, title, out, ylabel=group_name)
        written.append(out)

        out2 = out_dir / f"episode_{ep:06d}_{group_name}_state_action_stacked.png"
        plot_state_action_stacked(
            t, state, action, labels, f"{title} state vs action", out2
        )
        written.append(out2)

    ek = "observation.state.end.position"
    end = as_array(df[ek])
    if end.ndim == 1:
        end = end[:, None]
    end_names = info["features"][ek].get("names")
    end_labels = to_zero_based_labels(end_names, end.shape[1], prefix="end_")
    # keep semantic xyz/quat names (no jointN); only ensure dim index shown as [c]
    if end_names and not any(re.search(r"joint\d+", n, re.I) for n in end_names):
        end_labels = list(end_names)[: end.shape[1]]
    print(f"  end_position: dims={end.shape[1]} labels={end_labels}")

    title = f"ep {ep:06d} | end.position (state only, no action) | N={n}"
    out = out_dir / f"episode_{ep:06d}_end_position_state.png"
    plot_state_action_grid(
        t, end, None, end_labels, title, out, ncols=7, ylabel="end.position"
    )
    written.append(out)

    out2 = out_dir / f"episode_{ep:06d}_end_position_state_stacked.png"
    plot_state_action_stacked(t, end, None, end_labels, f"ep {ep:06d} | end.position state only | N={n}", out2)
    written.append(out2)

    obs = as_array(df["observation.state"])
    act = as_array(df["action"])
    obs14 = obs[:, :14]
    act_names = info["features"]["action"].get("names")
    labels14 = to_zero_based_labels(act_names, 14, prefix="a_")
    # gripper dims: keep readable names
    if len(labels14) >= 14:
        labels14[12] = "left_gripper"
        labels14[13] = "right_gripper"
    title = f"ep {ep:06d} | action vs observation.state[:14] | N={n}"
    out = out_dir / f"episode_{ep:06d}_action_vs_state14.png"
    plot_state_action_grid(t, obs14, act, labels14, title, out, ncols=7)
    written.append(out)

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset_root)
    for ep in args.episodes:
        out_dir = args.output_dir / f"ep{ep}_all_state_action"
        written = plot_episode(args.dataset_root, info, ep, out_dir)
        for path in written:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Overlay left/right gripper width (action + state) onto camera_top episode videos."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from _common import (
    CAMERA_KEY,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    episode_paths,
    load_grippers,
    load_info,
    resolve_dataset_root,
)

PANEL_BG = (24, 24, 24)
PANEL_BORDER = (90, 90, 90)
GRID = (55, 55, 55)
AXIS = (160, 160, 160)
LEFT_ACTION = (220, 180, 40)
RIGHT_ACTION = (60, 140, 255)
LEFT_STATE = (160, 230, 120)
RIGHT_STATE = (180, 100, 255)
CURSOR = (240, 240, 240)
TEXT = (235, 235, 235)


def y_bounds(series: np.ndarray) -> tuple[float, float]:
    finite = series[np.isfinite(series)]
    if len(finite) == 0:
        return 0.0, 1.0
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if abs(y_max - y_min) < 1e-9:
        pad = max(abs(y_min) * 0.05, 1e-3)
        return y_min - pad, y_max + pad
    pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def to_plot_y(values: np.ndarray, y_min: float, y_max: float, top: int, bottom: int) -> np.ndarray:
    if abs(y_max - y_min) < 1e-12:
        return np.full(values.shape, (top + bottom) / 2.0, dtype=np.float32)
    return bottom - (values - y_min) / (y_max - y_min) * (bottom - top)


def draw_dashed_polyline(img, pts, color, *, thickness=1, dash=8, gap=6):
    if len(pts) < 2:
        return
    draw = True
    remain = dash
    for i in range(len(pts) - 1):
        x0, y0 = int(pts[i, 0]), int(pts[i, 1])
        x1, y1 = int(pts[i + 1, 0]), int(pts[i + 1, 1])
        seg_len = float(np.hypot(x1 - x0, y1 - y0))
        if seg_len < 1e-6:
            continue
        ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
        traveled = 0.0
        cx, cy = float(x0), float(y0)
        while traveled < seg_len:
            step = min(remain, seg_len - traveled)
            nx, ny = cx + ux * step, cy + uy * step
            if draw:
                cv2.line(
                    img,
                    (int(round(cx)), int(round(cy))),
                    (int(round(nx)), int(round(ny))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            traveled += step
            cx, cy = nx, ny
            remain -= step
            if remain <= 1e-6:
                draw = not draw
                remain = dash if draw else gap


def draw_curve_panel(frame, action, state, frame_idx, *, panel_height_ratio=0.34, alpha=0.72):
    h, w = frame.shape[:2]
    panel_h = max(150, int(h * panel_height_ratio))
    y0 = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w - 1, h - 1), PANEL_BG, -1)
    cv2.rectangle(overlay, (0, y0), (w - 1, h - 1), PANEL_BORDER, 1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    margin_l, margin_r, margin_t, margin_b = 70, 24, 48, 28
    px0, px1 = margin_l, w - margin_r
    py0, py1 = y0 + margin_t, h - margin_b
    if px1 <= px0 + 10 or py1 <= py0 + 10:
        return frame

    n = min(action.shape[0], state.shape[0])
    action = action[:n]
    state = state[:n]
    y_min, y_max = y_bounds(np.concatenate([action, state], axis=0))
    xs = np.linspace(px0, px1, n, dtype=np.float32)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(py1 - frac * (py1 - py0)))
        cv2.line(frame, (px0, y), (px1, y), GRID, 1, cv2.LINE_AA)
    cv2.line(frame, (px0, py1), (px1, py1), AXIS, 1, cv2.LINE_AA)
    cv2.line(frame, (px0, py0), (px0, py1), AXIS, 1, cv2.LINE_AA)

    act_l_y = to_plot_y(action[:, 0], y_min, y_max, py0, py1)
    act_r_y = to_plot_y(action[:, 1], y_min, y_max, py0, py1)
    st_l_y = to_plot_y(state[:, 0], y_min, y_max, py0, py1)
    st_r_y = to_plot_y(state[:, 1], y_min, y_max, py0, py1)
    act_l_pts = np.round(np.stack([xs, act_l_y], axis=1)).astype(np.int32)
    act_r_pts = np.round(np.stack([xs, act_r_y], axis=1)).astype(np.int32)
    st_l_pts = np.round(np.stack([xs, st_l_y], axis=1)).astype(np.int32)
    st_r_pts = np.round(np.stack([xs, st_r_y], axis=1)).astype(np.int32)

    draw_dashed_polyline(frame, st_l_pts, LEFT_STATE, thickness=2)
    draw_dashed_polyline(frame, st_r_pts, RIGHT_STATE, thickness=2)
    cv2.polylines(frame, [act_l_pts], False, LEFT_ACTION, 2, cv2.LINE_AA)
    cv2.polylines(frame, [act_r_pts], False, RIGHT_ACTION, 2, cv2.LINE_AA)

    t = int(np.clip(frame_idx, 0, n - 1))
    cx = int(round(xs[t]))
    cv2.line(frame, (cx, py0), (cx, py1), CURSOR, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, int(round(st_l_y[t]))), 3, LEFT_STATE, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, int(round(st_r_y[t]))), 3, RIGHT_STATE, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, int(round(act_l_y[t]))), 4, LEFT_ACTION, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, int(round(act_r_y[t]))), 4, RIGHT_ACTION, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "Gripper width: action(solid) + state(dashed)",
        (12, y0 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"frame {t}/{n - 1}",
        (12, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    return frame


def render_episode(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    output_path: Path,
) -> Path:
    parquet_path, video_path = episode_paths(dataset_root, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    action, state = load_grippers(parquet_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or float(info.get("fps", 30))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_video, action.shape[0], state.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open writer: {output_path}")

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        out = draw_curve_panel(frame, action[:n], state[:n], i)
        cv2.putText(
            out,
            f"ep {episode_index:06d} | {CAMERA_KEY}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT,
            2,
            cv2.LINE_AA,
        )
        writer.write(out)

    cap.release()
    writer.release()
    return output_path


def sample_episodes(
    dataset_root: Path,
    info: dict[str, Any],
    n: int,
    seed: int,
    *,
    min_frames: int = 90,
    max_frames: int = 1800,
    min_gripper_range: float = 0.005,
    probe: int = 200,
) -> list[int]:
    total = int(info["total_episodes"])
    rng = random.Random(seed)
    candidates: list[tuple[float, int, int]] = []
    for ep in rng.sample(range(total), min(probe, total)):
        pq, vid = episode_paths(dataset_root, info, ep)
        if not (pq.exists() and vid.exists()):
            continue
        try:
            action, state = load_grippers(pq)
        except Exception:
            continue
        t = action.shape[0]
        if t < min_frames or t > max_frames:
            continue
        span = float(np.nanmax(np.concatenate([action, state])) - np.nanmin(np.concatenate([action, state])))
        if span < min_gripper_range:
            continue
        candidates.append((span, t, ep))
    candidates.sort(reverse=True)
    if len(candidates) < n:
        raise RuntimeError(f"only found {len(candidates)} suitable episodes, need {n}")
    return [ep for _, _, ep in candidates[:n]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    episodes = args.episodes or sample_episodes(dataset_root, info, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset={dataset_root}")
    print(f"episodes={episodes}")
    for ep in episodes:
        out = args.output_dir / f"episode_{ep:06d}_camera_top_gripper.mp4"
        path = render_episode(dataset_root, info, ep, out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Transform humanoid EEF poses into camera_top and overlay pose curves on video.

Source EEF pose (`observation.state.end.position`) is flange-in-arm-base:
  left  = [x,y,z, qx,qy,qz,qw]
  right = [x,y,z, qx,qy,qz,qw]
with independent left/right arm bases.

Calibration extrinsic `camera_front_to_arm_{left,right}.matrix` is
  T_Arm <- CameraTop
so camera-frame pose is:
  T_Cam <- EEF = inv(T_Arm <- Cam) @ T_Arm <- EEF
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DEFAULT_DATASET = Path("/mnt/pfs/datasets/egodex/part1_lerobot")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/pfs/Data/liuweilin/Dataclean/datasets/egodex_lerobot_v21/manual_review/results"
)
CAMERA_KEY = "observation.images.camera_top"
EEF_KEY = "observation.state.end.position"

# BGR
PANEL_BG = (24, 24, 24)
PANEL_BORDER = (90, 90, 90)
GRID = (55, 55, 55)
AXIS = (160, 160, 160)
CURSOR = (240, 240, 240)
TEXT = (235, 235, 235)

# position: L/R x,y,z
POS_COLORS = {
    "L_x": (80, 80, 255),
    "L_y": (80, 200, 80),
    "L_z": (255, 160, 60),
    "R_x": (120, 120, 255),
    "R_y": (120, 230, 120),
    "R_z": (255, 200, 120),
}
# orientation rotvec: L/R rx,ry,rz
ROT_COLORS = {
    "L_rx": (200, 100, 255),
    "L_ry": (255, 180, 80),
    "L_rz": (80, 220, 220),
    "R_rx": (220, 140, 255),
    "R_ry": (255, 210, 140),
    "R_rz": (140, 240, 240),
}


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def episode_chunk(ep: int, chunks_size: int) -> int:
    return ep // chunks_size


def episode_paths(dataset_root: Path, info: dict[str, Any], episode_index: int) -> dict[str, Path]:
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return {
        "parquet": dataset_root
        / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index),
        "video": dataset_root
        / info["video_path"].format(
            episode_chunk=chunk, video_key=CAMERA_KEY, episode_index=episode_index
        ),
        "calibration": (
            dataset_root
            / "parameters"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}"
            / "calibration_bundle_optimized.json"
        ),
    }


def array_column(df: pd.DataFrame, key: str) -> np.ndarray:
    values = df[key].to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float64)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist(), dtype=np.float64)
    return values.astype(np.float64)[:, None]


def load_extrinsics(cal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    ext = cal["extrinsics"]
    t_arm_cam_l = np.asarray(ext["camera_front_to_arm_left"]["matrix"], dtype=np.float64)
    t_arm_cam_r = np.asarray(ext["camera_front_to_arm_right"]["matrix"], dtype=np.float64)
    if t_arm_cam_l.shape != (4, 4) or t_arm_cam_r.shape != (4, 4):
        raise ValueError(f"bad extrinsic shapes in {cal_path}")
    # source matrix is T_Arm <- CameraTop; invert to T_CameraTop <- Arm
    return np.linalg.inv(t_arm_cam_l), np.linalg.inv(t_arm_cam_r)


def pose7_to_matrices(pose7: np.ndarray) -> np.ndarray:
    """(T,7) xyz+quat_xyzw -> (T,4,4) T_Arm <- EEF."""
    t = pose7.shape[0]
    out = np.repeat(np.eye(4, dtype=np.float64)[None, ...], t, axis=0)
    out[:, :3, 3] = pose7[:, :3]
    out[:, :3, :3] = Rotation.from_quat(pose7[:, 3:7]).as_matrix()
    return out


def transform_eef_to_camera(
    end14: np.ndarray, t_cam_arm_l: np.ndarray, t_cam_arm_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right camera-frame poses as (T,7) xyz+quat_xyzw."""
    t_arm_eef_l = pose7_to_matrices(end14[:, 0:7])
    t_arm_eef_r = pose7_to_matrices(end14[:, 7:14])
    t_cam_eef_l = t_cam_arm_l[None, ...] @ t_arm_eef_l
    t_cam_eef_r = t_cam_arm_r[None, ...] @ t_arm_eef_r

    def mats_to_pose7(mats: np.ndarray) -> np.ndarray:
        xyz = mats[:, :3, 3]
        quat = Rotation.from_matrix(mats[:, :3, :3]).as_quat()  # xyzw
        return np.concatenate([xyz, quat], axis=1).astype(np.float32)

    return mats_to_pose7(t_cam_eef_l), mats_to_pose7(t_cam_eef_r)


def y_bounds(series: np.ndarray) -> tuple[float, float]:
    finite = series[np.isfinite(series)]
    if len(finite) == 0:
        return -1.0, 1.0
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
    return (bottom - (values - y_min) / (y_max - y_min) * (bottom - top)).astype(np.float32)


def draw_multi_series_panel(
    frame: np.ndarray,
    series: dict[str, np.ndarray],
    colors: dict[str, tuple[int, int, int]],
    frame_idx: int,
    rect: tuple[int, int, int, int],
    title: str,
    *,
    alpha: float = 0.78,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL_BG, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL_BORDER, 1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    pad_l, pad_r, pad_t, pad_b = 58, 16, 36, 22
    px0, px1 = x0 + pad_l, x1 - pad_r
    py0, py1 = y0 + pad_t, y1 - pad_b
    if px1 <= px0 + 8 or py1 <= py0 + 8:
        return frame

    stacked = np.stack(list(series.values()), axis=1)
    y_min, y_max = y_bounds(stacked)
    n = next(iter(series.values())).shape[0]
    xs = np.linspace(px0, px1, n, dtype=np.float32)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(py1 - frac * (py1 - py0)))
        cv2.line(frame, (px0, y), (px1, y), GRID, 1, cv2.LINE_AA)
    cv2.line(frame, (px0, py1), (px1, py1), AXIS, 1, cv2.LINE_AA)
    cv2.line(frame, (px0, py0), (px0, py1), AXIS, 1, cv2.LINE_AA)

    t = int(np.clip(frame_idx, 0, n - 1))
    cx = int(round(xs[t]))
    cv2.line(frame, (cx, py0), (cx, py1), CURSOR, 1, cv2.LINE_AA)

    legend_x = px0 + 6
    for name, values in series.items():
        color = colors[name]
        ys = to_plot_y(values, y_min, y_max, py0, py1)
        pts = np.round(np.stack([xs, ys], axis=1)).astype(np.int32)
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, int(round(ys[t]))), 3, color, -1, cv2.LINE_AA)
        cv2.putText(frame, name, (legend_x, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        legend_x += 52

    cv2.putText(frame, title, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{y_max:.3f}", (x0 + 4, py0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, AXIS, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{y_min:.3f}", (x0 + 4, py1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, AXIS, 1, cv2.LINE_AA)
    return frame


def draw_pose_panels(
    frame: np.ndarray,
    left_cam: np.ndarray,
    right_cam: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = max(260, int(h * 0.46))
    y0 = h - panel_h
    mid = y0 + panel_h // 2

    left_rot = Rotation.from_quat(left_cam[:, 3:7]).as_rotvec().astype(np.float32)
    right_rot = Rotation.from_quat(right_cam[:, 3:7]).as_rotvec().astype(np.float32)

    pos_series = {
        "L_x": left_cam[:, 0],
        "L_y": left_cam[:, 1],
        "L_z": left_cam[:, 2],
        "R_x": right_cam[:, 0],
        "R_y": right_cam[:, 1],
        "R_z": right_cam[:, 2],
    }
    rot_series = {
        "L_rx": left_rot[:, 0],
        "L_ry": left_rot[:, 1],
        "L_rz": left_rot[:, 2],
        "R_rx": right_rot[:, 0],
        "R_ry": right_rot[:, 1],
        "R_rz": right_rot[:, 2],
    }

    frame = draw_multi_series_panel(
        frame,
        pos_series,
        POS_COLORS,
        frame_idx,
        (0, y0, w - 1, mid),
        "EEF position in camera_top (m)",
    )
    frame = draw_multi_series_panel(
        frame,
        rot_series,
        ROT_COLORS,
        frame_idx,
        (0, mid, w - 1, h - 1),
        "EEF orientation rotvec in camera_top (rad)",
    )

    t = int(np.clip(frame_idx, 0, left_cam.shape[0] - 1))
    cv2.putText(
        frame,
        f"frame {t}/{left_cam.shape[0] - 1}  "
        f"Lxyz=({left_cam[t, 0]:.3f},{left_cam[t, 1]:.3f},{left_cam[t, 2]:.3f})  "
        f"Rxyz=({right_cam[t, 0]:.3f},{right_cam[t, 1]:.3f},{right_cam[t, 2]:.3f})",
        (12, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
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
    paths = episode_paths(dataset_root, info, episode_index)
    for key in ("parquet", "video", "calibration"):
        if not paths[key].exists():
            raise FileNotFoundError(f"missing {key}: {paths[key]}")

    df = pd.read_parquet(paths["parquet"], columns=[EEF_KEY])
    end14 = array_column(df, EEF_KEY)
    if end14.ndim != 2 or end14.shape[1] < 14:
        raise ValueError(f"unexpected {EEF_KEY} shape {end14.shape}")

    t_cam_arm_l, t_cam_arm_r = load_extrinsics(paths["calibration"])
    left_cam, right_cam = transform_eef_to_camera(end14[:, :14], t_cam_arm_l, t_cam_arm_r)

    cap = cv2.VideoCapture(str(paths["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {paths['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(info.get("fps", 30))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), left_cam.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open writer: {output_path}")

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        out = draw_pose_panels(frame, left_cam[:n], right_cam[:n], i)
        cv2.putText(
            out,
            f"ep {episode_index:06d} | EEF -> camera_top",
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

    # also dump transformed poses for inspection
    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        left_xyz_quat_xyzw=left_cam[:n],
        right_xyz_quat_xyzw=right_cam[:n],
        T_cam_arm_left=t_cam_arm_l.astype(np.float32),
        T_cam_arm_right=t_cam_arm_r.astype(np.float32),
        episode_index=np.int32(episode_index),
    )
    print(f"wrote poses {npz_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--episode", type=int, default=32021, help="Episode index to visualize")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = load_info(args.dataset_root)
    out = args.output_dir / f"episode_{args.episode:06d}_camera_top_eef_pose.mp4"
    path = render_episode(args.dataset_root, info, args.episode, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect / sanity-check Dataclean standard-format episode parquet files.

Examples:
  # summary of one episode under a converted dataset root
  python inspect_standard_parquet.py \\
    --dataset-root .../test_result_sample100 --episode 2961

  # print first/mid/last frames for selected columns
  python inspect_standard_parquet.py \\
    --dataset-root .../test_result_sample100 --episode 2961 \\
    --show-frames 0 mid -1 \\
    --columns observation.state.eef.left.pose action.gripper.left.closedness

  # direct parquet path
  python inspect_standard_parquet.py --parquet .../episode_002961.parquet

  # only run checks, compact output
  python inspect_standard_parquet.py --dataset-root ... --episode 2961 --checks-only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(
    "/mnt/project_rlinf_hs/liuweilin/Dataclean/datasets/humanoid_merged/test_result_sample100"
)

# Common standard columns we expect for dual-arm humanoid exports.
EXPECTED_HUMANOID_COLS = [
    "observation.state.eef.left.pose",
    "observation.state.eef.right.pose",
    "observation.geometry.eef.left.rotvec",
    "observation.geometry.eef.right.rotvec",
    "observation.state.arm.left.joint_position",
    "observation.state.arm.right.joint_position",
    "observation.state.gripper.left.closedness",
    "observation.state.gripper.right.closedness",
    "action.eef.left.pose",
    "action.eef.right.pose",
    "action.arm.left.joint_position",
    "action.arm.right.joint_position",
    "action.gripper.left.closedness",
    "action.gripper.right.closedness",
    "extrinsic.camera_top.T_ArmLeft_CameraTop",
    "extrinsic.camera_top.T_ArmRight_CameraTop",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "timestamp",
]

EXPECTED_SHAPES = {
    "observation.state.eef.left.pose": 7,
    "observation.state.eef.right.pose": 7,
    "action.eef.left.pose": 7,
    "action.eef.right.pose": 7,
    "observation.state.arm.left.joint_position": 6,
    "observation.state.arm.right.joint_position": 6,
    "action.arm.left.joint_position": 6,
    "action.arm.right.joint_position": 6,
    "observation.state.gripper.left.closedness": 1,
    "observation.state.gripper.right.closedness": 1,
    "action.gripper.left.closedness": 1,
    "action.gripper.right.closedness": 1,
    "observation.geometry.eef.left.rotvec": 3,
    "observation.geometry.eef.right.rotvec": 3,
    "extrinsic.camera_top.T_ArmLeft_CameraTop": 16,
    "extrinsic.camera_top.T_ArmRight_CameraTop": 16,
    "extrinsic.camera_top.T_ArmLeft_CameraWristLeft": 16,
    "extrinsic.camera_top.T_ArmRight_CameraWristRight": 16,
}


def load_info(dataset_root: Path) -> dict[str, Any] | None:
    path = dataset_root / "meta" / "info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_parquet(
    dataset_root: Path | None,
    episode: int | None,
    parquet: Path | None,
    info: dict[str, Any] | None,
) -> Path:
    if parquet is not None:
        return Path(parquet)
    if dataset_root is None or episode is None:
        raise SystemExit("need --parquet, or --dataset-root + --episode")
    chunks_size = int((info or {}).get("chunks_size", 1000))
    data_path = (info or {}).get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk = episode // chunks_size
    path = dataset_root / data_path.format(episode_chunk=chunk, episode_index=episode)
    if not path.exists():
        # fallback search
        matches = list(dataset_root.glob(f"data/**/episode_{episode:06d}.parquet"))
        if not matches:
            raise FileNotFoundError(path)
        path = matches[0]
    return path


def as_array(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float64)
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values)
    if isinstance(first, (list, tuple)):
        return np.asarray(values.tolist())
    return np.asarray(values)


def fmt_vec(x: np.ndarray, precision: int = 4) -> str:
    arr = np.asarray(x).reshape(-1)
    return np.array2string(arr, precision=precision, suppress_small=True, separator=", ")


def column_summary(name: str, series: pd.Series) -> dict[str, Any]:
    arr = as_array(series)
    out: dict[str, Any] = {
        "name": name,
        "rows": int(arr.shape[0]),
        "shape_per_row": list(arr.shape[1:]) if arr.ndim > 1 else [],
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        return out
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        out["finite_frac"] = float(finite.mean())
        flat = arr[finite]
        if flat.size:
            out["min"] = float(np.min(flat))
            out["max"] = float(np.max(flat))
            out["mean"] = float(np.mean(flat))
            out["std"] = float(np.std(flat))
            # per-dim if vector
            if arr.ndim == 2 and arr.shape[1] > 1:
                out["min_per_dim"] = np.min(arr, axis=0).tolist()
                out["max_per_dim"] = np.max(arr, axis=0).tolist()
                out["mean_per_dim"] = np.mean(arr, axis=0).tolist()
        sample0 = arr[0]
        out["sample0"] = sample0.tolist() if hasattr(sample0, "tolist") else sample0
        if arr.shape[0] > 1:
            mid = arr[arr.shape[0] // 2]
            out["sample_mid"] = mid.tolist() if hasattr(mid, "tolist") else mid
            out["sample_last"] = arr[-1].tolist() if hasattr(arr[-1], "tolist") else arr[-1]
    else:
        out["sample0"] = values_repr(series.iloc[0])
    return out


def values_repr(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def print_schema_vs_info(df: pd.DataFrame, info: dict[str, Any] | None) -> None:
    print("=" * 72)
    print("SCHEMA")
    print("=" * 72)
    print(f"parquet columns ({len(df.columns)}):")
    for c in df.columns:
        arr = as_array(df[c])
        shape = list(arr.shape[1:]) if arr.ndim > 1 else []
        print(f"  - {c}: rows={arr.shape[0]} shape/row={shape} dtype={arr.dtype}")

    if info is None:
        print("\n(no meta/info.json; skip feature schema compare)")
        return

    feats = info.get("features", {})
    # video features are not stored in parquet
    feat_non_video = {
        k: v for k, v in feats.items() if v.get("dtype") != "video"
    }
    missing_in_pq = sorted(set(feat_non_video) - set(df.columns))
    extra_in_pq = sorted(set(df.columns) - set(feat_non_video))
    print(f"\ninfo.json non-video features: {len(feat_non_video)}")
    if missing_in_pq:
        print(f"  MISSING in parquet ({len(missing_in_pq)}):")
        for k in missing_in_pq:
            print(f"    - {k}  expected shape={feat_non_video[k].get('shape')}")
    else:
        print("  all non-video info features present in parquet")
    if extra_in_pq:
        print(f"  EXTRA in parquet (not in info features) ({len(extra_in_pq)}):")
        for k in extra_in_pq:
            print(f"    - {k}")

    # shape checks vs info
    shape_mismatch = []
    for k, meta in feat_non_video.items():
        if k not in df.columns:
            continue
        exp = meta.get("shape")
        if not exp:
            continue
        arr = as_array(df[k])
        got = list(arr.shape[1:]) if arr.ndim > 1 else ([1] if arr.ndim == 1 else [])
        # scalar columns often stored as 1d series -> treat as shape [1]
        if got == [] and arr.ndim == 1:
            got = [1]
        if got != list(exp):
            # allow scalar int/float columns with shape [1] stored as plain 1d
            if list(exp) == [1] and arr.ndim == 1:
                continue
            shape_mismatch.append((k, exp, got))
    if shape_mismatch:
        print(f"  SHAPE mismatches ({len(shape_mismatch)}):")
        for k, exp, got in shape_mismatch:
            print(f"    - {k}: expected {exp}, got {got}")
    else:
        print("  shapes match info.json (within scalar/[1] tolerance)")


def print_expected_humanoid(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("EXPECTED HUMANOID COLUMNS")
    print("=" * 72)
    missing = [c for c in EXPECTED_HUMANOID_COLS if c not in df.columns]
    present = [c for c in EXPECTED_HUMANOID_COLS if c in df.columns]
    print(f"present {len(present)}/{len(EXPECTED_HUMANOID_COLS)}")
    if missing:
        print("missing:")
        for c in missing:
            print(f"  - {c}")
    for c, dim in EXPECTED_SHAPES.items():
        if c not in df.columns:
            continue
        arr = as_array(df[c])
        got = arr.shape[1] if arr.ndim == 2 else (1 if arr.ndim == 1 else None)
        ok = got == dim
        mark = "OK" if ok else "BAD"
        print(f"  [{mark}] {c}: dim={got} expected={dim}")


def print_summaries(df: pd.DataFrame, columns: list[str] | None) -> None:
    print("\n" + "=" * 72)
    print("COLUMN STATS")
    print("=" * 72)
    cols = columns or list(df.columns)
    for c in cols:
        if c not in df.columns:
            print(f"  ! missing column: {c}")
            continue
        s = column_summary(c, df[c])
        shape = s.get("shape_per_row") or [1]
        line = (
            f"{c}: shape/row={shape} dtype={s['dtype']} "
            f"min={s.get('min', float('nan')):.6g} max={s.get('max', float('nan')):.6g} "
            f"mean={s.get('mean', float('nan')):.6g} std={s.get('std', float('nan')):.6g}"
        )
        print(line)
        if "sample0" in s:
            print(f"    [0]   {fmt_vec(np.asarray(s['sample0']))}")
        if "sample_mid" in s:
            print(f"    [mid] {fmt_vec(np.asarray(s['sample_mid']))}")
        if "sample_last" in s:
            print(f"    [-1]  {fmt_vec(np.asarray(s['sample_last']))}")


def resolve_frame_indices(n: int, specs: list[str]) -> list[int]:
    out: list[int] = []
    for s in specs:
        if s == "mid":
            out.append(n // 2)
        else:
            idx = int(s)
            if idx < 0:
                idx = n + idx
            out.append(int(np.clip(idx, 0, n - 1)))
    return out


def print_frames(df: pd.DataFrame, frame_specs: list[str], columns: list[str] | None) -> None:
    n = len(df)
    idxs = resolve_frame_indices(n, frame_specs)
    cols = columns or list(df.columns)
    print("\n" + "=" * 72)
    print(f"FRAMES {idxs} / N={n}")
    print("=" * 72)
    for i in idxs:
        print(f"\n--- frame {i} ---")
        for c in cols:
            if c not in df.columns:
                print(f"  {c}: <missing>")
                continue
            arr = as_array(df[c])
            val = arr[i] if arr.ndim >= 1 else arr
            print(f"  {c}: {fmt_vec(np.asarray(val))}")


def check_se3(mat16: np.ndarray, name: str, tol: float = 1e-2) -> list[str]:
    issues: list[str] = []
    # mat16: (T,16) or (16,)
    mats = mat16.reshape(-1, 16)
    # constancy
    if mats.shape[0] > 1:
        spread = float(np.max(np.ptp(mats, axis=0)))
        if spread > 1e-4:
            issues.append(f"{name}: extrinsic not constant across frames (ptp={spread:.3g})")
    M = mats[0].reshape(4, 4)
    if abs(M[3, 3] - 1.0) > tol or np.max(np.abs(M[3, :3])) > tol:
        issues.append(f"{name}: bottom row not [0,0,0,1]: {M[3]}")
    R = M[:3, :3]
    should_I = R @ R.T
    ortho_err = float(np.max(np.abs(should_I - np.eye(3))))
    det = float(np.linalg.det(R))
    if ortho_err > tol:
        issues.append(f"{name}: R not orthonormal (max|R R^T - I|={ortho_err:.3g})")
    if abs(det - 1.0) > tol:
        issues.append(f"{name}: det(R)={det:.6f} (expected ~1)")
    return issues


def run_checks(df: pd.DataFrame, info: dict[str, Any] | None, episode: int | None) -> list[str]:
    issues: list[str] = []
    n = len(df)
    if n == 0:
        return ["empty parquet"]

    # indices
    if "frame_index" in df.columns:
        fi = as_array(df["frame_index"]).astype(np.int64).reshape(-1)
        if fi[0] != 0:
            issues.append(f"frame_index starts at {fi[0]}, expected 0")
        if not np.array_equal(fi, np.arange(n)):
            issues.append("frame_index is not contiguous 0..N-1")
    if "episode_index" in df.columns:
        ei = as_array(df["episode_index"]).astype(np.int64).reshape(-1)
        if np.unique(ei).size != 1:
            issues.append(f"episode_index not constant: unique={np.unique(ei)[:5]}")
        if episode is not None and int(ei[0]) != int(episode):
            issues.append(f"episode_index={ei[0]} != requested episode={episode}")
    if "timestamp" in df.columns:
        ts = as_array(df["timestamp"]).astype(np.float64).reshape(-1)
        if np.any(np.diff(ts) < -1e-6):
            issues.append("timestamp not non-decreasing")
        if n > 1:
            dt = np.diff(ts)
            fps_med = 1.0 / float(np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")
            declared = float((info or {}).get("fps", 30))
            if np.isfinite(fps_med) and abs(fps_med - declared) > max(0.5, 0.05 * declared):
                issues.append(f"timestamp fps~{fps_med:.3f} != info.fps={declared}")
            print(f"  timestamp span={ts[-1] - ts[0]:.3f}s  fps_median={fps_med:.4f}")

    # grippers in [0,1] (soft bounds)
    for c in [
        "action.gripper.left.closedness",
        "action.gripper.right.closedness",
        "observation.state.gripper.left.closedness",
        "observation.state.gripper.right.closedness",
    ]:
        if c not in df.columns:
            continue
        g = as_array(df[c]).astype(np.float64).reshape(-1)
        if np.any(~np.isfinite(g)):
            issues.append(f"{c}: contains non-finite")
        lo, hi = float(np.min(g)), float(np.max(g))
        if lo < -0.05 or hi > 1.05:
            issues.append(f"{c}: values outside ~[0,1]: [{lo:.4f}, {hi:.4f}]")
        print(f"  {c}: range=[{lo:.4f}, {hi:.4f}] mean={float(np.mean(g)):.4f}")

    # pose quaternion unit norm (wxyz at indices 3:7)
    for c in [
        "observation.state.eef.left.pose",
        "observation.state.eef.right.pose",
        "action.eef.left.pose",
        "action.eef.right.pose",
    ]:
        if c not in df.columns:
            continue
        p = as_array(df[c]).astype(np.float64)
        if p.ndim != 2 or p.shape[1] != 7:
            issues.append(f"{c}: bad shape {p.shape}")
            continue
        q = p[:, 3:7]
        norms = np.linalg.norm(q, axis=1)
        err = float(np.max(np.abs(norms - 1.0)))
        print(f"  {c} quat: |q| err_max={err:.3g} mean={float(np.mean(norms)):.6f}")
        if err > 5e-2:
            issues.append(f"{c}: quaternion not unit (max |||q||-1|={err:.3g})")

    # geometry rotvec finite
    for c in [
        "observation.geometry.eef.left.rotvec",
        "observation.geometry.eef.right.rotvec",
    ]:
        if c not in df.columns:
            continue
        rv = as_array(df[c]).astype(np.float64)
        if rv.ndim != 2 or rv.shape[1] != 3:
            issues.append(f"{c}: bad shape {rv.shape}")
            continue
        if np.any(~np.isfinite(rv)):
            issues.append(f"{c}: contains non-finite")

    # extrinsics
    for c in [
        "extrinsic.camera_top.T_ArmLeft_CameraTop",
        "extrinsic.camera_top.T_ArmRight_CameraTop",
        "extrinsic.camera_top.T_ArmLeft_CameraWristLeft",
        "extrinsic.camera_top.T_ArmRight_CameraWristRight",
    ]:
        if c not in df.columns:
            continue
        mat = as_array(df[c]).astype(np.float64)
        issues.extend(check_se3(mat, c))
        M = mat.reshape(-1, 16)[0].reshape(4, 4)
        print(f"  {c}: t={fmt_vec(M[:3, 3])} det(R)={np.linalg.det(M[:3, :3]):.6f}")

    # eef pose finite
    for c in [
        "observation.state.eef.left.pose",
        "observation.state.eef.right.pose",
        "action.eef.left.pose",
        "action.eef.right.pose",
    ]:
        if c not in df.columns:
            continue
        p = as_array(df[c]).astype(np.float64)
        if p.ndim != 2 or p.shape[1] != 7:
            issues.append(f"{c}: expected (T,7), got {p.shape}")
            continue
        if np.any(~np.isfinite(p)):
            issues.append(f"{c}: contains non-finite")
        xyz = p[:, :3]
        print(
            f"  {c}: xyz range "
            f"x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}] "
            f"y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}] "
            f"z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}]"
        )

    # action.eef vs state.eef identity? (humanoid often copies with lag; just report corr/diff)
    if "action.eef.left.pose" in df.columns and "observation.state.eef.left.pose" in df.columns:
        a = as_array(df["action.eef.left.pose"])
        s = as_array(df["observation.state.eef.left.pose"])
        same = float(np.mean(np.all(np.isclose(a, s, atol=1e-5), axis=1)))
        print(f"  action.eef.left == state.eef.left fraction: {same:.3f}")

    return issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Limit stats/frame dump to these columns",
    )
    p.add_argument(
        "--show-frames",
        nargs="*",
        default=None,
        help="Frame indices to dump, e.g. 0 mid -1 100",
    )
    p.add_argument("--checks-only", action="store_true", help="Only print sanity checks")
    p.add_argument("--no-checks", action="store_true", help="Skip sanity checks")
    p.add_argument(
        "--list-episodes",
        action="store_true",
        help="List available episode parquet indices under dataset-root and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    if dataset_root is None and args.parquet is None:
        dataset_root = DEFAULT_DATASET

    info = load_info(dataset_root) if dataset_root is not None else None

    if args.list_episodes:
        if dataset_root is None:
            raise SystemExit("--list-episodes requires --dataset-root")
        eps = sorted(
            int(p.stem.split("_")[1])
            for p in (dataset_root / "data").glob("chunk-*/episode_*.parquet")
        )
        print(f"dataset={dataset_root}")
        print(f"n_episodes={len(eps)}")
        print("episodes=", eps)
        return

    if args.episode is None and args.parquet is None:
        # default: first available episode
        if dataset_root is not None:
            first = next((dataset_root / "data").glob("chunk-*/episode_*.parquet"), None)
            if first is None:
                raise SystemExit(f"no parquet under {dataset_root}/data")
            args.parquet = first
            print(f"(no --episode/--parquet given; using {first})")

    path = resolve_parquet(dataset_root, args.episode, args.parquet, info)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    print(f"parquet={path}")
    print(f"rows={len(df)} cols={len(df.columns)}")
    if info is not None:
        print(
            f"info: fps={info.get('fps')} total_episodes={info.get('total_episodes')} "
            f"chunks_size={info.get('chunks_size')}"
        )
    if "episode_index" in df.columns:
        print(f"episode_index={int(as_array(df['episode_index']).reshape(-1)[0])}")

    if not args.checks_only:
        print_schema_vs_info(df, info)
        print_expected_humanoid(df)
        print_summaries(df, args.columns)
        if args.show_frames:
            print_frames(df, args.show_frames, args.columns)

    if not args.no_checks:
        print("\n" + "=" * 72)
        print("SANITY CHECKS")
        print("=" * 72)
        ep = args.episode
        if ep is None and "episode_index" in df.columns:
            ep = int(as_array(df["episode_index"]).reshape(-1)[0])
        issues = run_checks(df, info, ep)
        if issues:
            print(f"\nISSUES ({len(issues)}):")
            for msg in issues:
                print(f"  - {msg}")
        else:
            print("\nNo issues found by built-in checks.")


if __name__ == "__main__":
    main()

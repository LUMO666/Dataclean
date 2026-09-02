#!/usr/bin/env python3
"""Inspect / sanity-check Dataclean standard-format episode parquet files for EgoDex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import DEFAULT_DATASET, episode_parquet, load_info, resolve_dataset_root

EXPECTED_EGODEX_COLS = [
    "observation.state.eef.left.position",
    "observation.state.eef.left.rotation_6d",
    "observation.state.eef.right.position",
    "observation.state.eef.right.rotation_6d",
    "observation.state.gripper.left.closedness",
    "observation.state.gripper.right.closedness",
    "action.eef.left.position",
    "action.eef.left.rotation_6d",
    "action.eef.right.position",
    "action.eef.right.rotation_6d",
    "action.gripper.left.closedness",
    "action.gripper.right.closedness",
    "extrinsic.camera_reference.T_Episode_CameraReference",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "timestamp",
]

EXPECTED_SHAPES = {
    "observation.state.eef.left.position": 3,
    "observation.state.eef.left.rotation_6d": 6,
    "observation.state.eef.right.position": 3,
    "observation.state.eef.right.rotation_6d": 6,
    "action.eef.left.position": 3,
    "action.eef.left.rotation_6d": 6,
    "action.eef.right.position": 3,
    "action.eef.right.rotation_6d": 6,
    "observation.state.gripper.left.closedness": 1,
    "observation.state.gripper.right.closedness": 1,
    "action.gripper.left.closedness": 1,
    "action.gripper.right.closedness": 1,
    "extrinsic.camera_reference.T_Episode_CameraReference": 16,
}


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
    return episode_parquet(dataset_root, info or {}, episode)


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
        sample0 = arr[0]
        out["sample0"] = sample0.tolist() if hasattr(sample0, "tolist") else sample0
        if arr.shape[0] > 1:
            mid = arr[arr.shape[0] // 2]
            out["sample_mid"] = mid.tolist() if hasattr(mid, "tolist") else mid
            out["sample_last"] = arr[-1].tolist() if hasattr(arr[-1], "tolist") else arr[-1]
    return out


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
    feat_non_video = {k: v for k, v in feats.items() if v.get("dtype") != "video"}
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
        for k in extra_in_pq[:30]:
            print(f"    - {k}")


def print_expected_egodex(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("EXPECTED EGODEX STANDARD COLUMNS")
    print("=" * 72)
    missing = [c for c in EXPECTED_EGODEX_COLS if c not in df.columns]
    present = [c for c in EXPECTED_EGODEX_COLS if c in df.columns]
    print(f"present {len(present)}/{len(EXPECTED_EGODEX_COLS)}")
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


def run_checks(df: pd.DataFrame, info: dict[str, Any] | None, episode: int | None) -> list[str]:
    issues: list[str] = []
    n = len(df)
    if n == 0:
        return ["empty parquet"]

    if "frame_index" in df.columns:
        fi = as_array(df["frame_index"]).astype(np.int64).reshape(-1)
        if fi[0] != 0:
            issues.append(f"frame_index starts at {fi[0]}, expected 0")
        if not np.array_equal(fi, np.arange(n)):
            issues.append("frame_index is not contiguous 0..N-1")

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
        print(f"  {c}: range=[{lo:.4f}, {hi:.4f}] mean={float(np.mean(g)):.4f}")

    for c in [
        "observation.state.eef.left.rotation_6d",
        "observation.state.eef.right.rotation_6d",
        "action.eef.left.rotation_6d",
        "action.eef.right.rotation_6d",
    ]:
        if c not in df.columns:
            continue
        r = as_array(df[c]).astype(np.float64)
        if r.ndim != 2 or r.shape[1] != 6:
            issues.append(f"{c}: bad shape {r.shape}")
            continue
        # First two matrix rows; check approximate unit row norms.
        n0 = np.linalg.norm(r[:, 0:3], axis=1)
        n1 = np.linalg.norm(r[:, 3:6], axis=1)
        err = float(max(np.max(np.abs(n0 - 1.0)), np.max(np.abs(n1 - 1.0))))
        print(f"  {c}: row-norm err_max={err:.3g}")
        if err > 5e-2:
            issues.append(f"{c}: rotation_6d row norms not ~1 (max={err:.3g})")

    return issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--columns", nargs="*", default=None)
    p.add_argument("--show-frames", nargs="*", default=None)
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--list-episodes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root) if args.dataset_root else None
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

    if args.episode is None and args.parquet is None and dataset_root is not None:
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

    if not args.checks_only:
        print_schema_vs_info(df, info)
        print_expected_egodex(df)
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

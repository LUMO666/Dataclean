#!/usr/bin/env python3
"""Compare pipeline / P6 episode-frame geometry against v2 gold demo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

from robot_data_processing.loader import episode_parquet_path, read_episode_table  # noqa: E402
from robot_data_processing.normalize.review_corrections import apply_review_corrections  # noqa: E402
from robot_data_processing.normalize.standard_types import EpisodeRef  # noqa: E402
from robot_data_processing.phases.orchestrator import _table_to_raw  # noqa: E402
from robot_data_processing.stages.stage2_trend_alignment import (  # noqa: E402
    Stage2Config,
    fill_missing_action_eef,
)
from robot_data_processing.stages.state_action_temporal_alignment import (  # noqa: E402
    apply_p4_to_action_fields,
    parse_temporal_align_config,
    resolve_p4_plan,
)

GEOM_COLS = [
    "observation.state.eef.left.pose",
    "observation.state.eef.right.pose",
    "action.eef.left.pose",
    "action.eef.right.pose",
    "extrinsic.camera_reference.T_Episode_CameraReference",
    "extrinsic.camera_aux_0.T_Episode_CameraAux0",
    "extrinsic.camera_aux_1.T_Episode_CameraAux1",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet_path(root: Path, episode_index: int) -> Path:
    return episode_parquet_path(root, episode_index)


def _table_to_dict(table) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in table.column_names:
        col = np.asarray(table[name].to_pylist(), dtype=np.float32)
        if col.ndim == 1:
            col = col.reshape(-1, 1)
        if name.startswith("extrinsic."):
            if col.ndim == 2 and col.shape[1] == 16:
                col = col.reshape(-1, 4, 4)
            elif col.ndim == 3:
                pass
            else:
                col = np.stack([np.asarray(x, dtype=np.float32).reshape(4, 4) for x in table[name].to_pylist()])
        out[name] = col
    return out


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _build_full_geometry_episode(
    *,
    raw_root: Path,
    episode_index: int,
    package_dir: Path,
    config: dict[str, Any],
    fill_lag: int = 2,
) -> dict[str, np.ndarray]:
    import importlib.util

    adapter_spec = importlib.util.spec_from_file_location(
        "humanoid_adapter_cmp", package_dir / "adapter.py"
    )
    adapter_mod = importlib.util.module_from_spec(adapter_spec)
    assert adapter_spec.loader is not None
    adapter_spec.loader.exec_module(adapter_mod)
    adapter = adapter_mod.HumanoidAdapter(package_dir)

    geom_spec = importlib.util.spec_from_file_location(
        "humanoid_episode_geometry_cmp", package_dir / "episode_geometry.py"
    )
    geom_mod = importlib.util.module_from_spec(geom_spec)
    assert geom_spec.loader is not None
    geom_spec.loader.exec_module(geom_mod)
    geometry = geom_mod.HumanoidEpisodeGeometry(package_dir)
    path = _parquet_path(raw_root, episode_index)
    table = read_episode_table(path)
    raw = _table_to_raw(table)
    ref = EpisodeRef(episode_index=episode_index, dataset_root=raw_root)
    meta = {"fps": float(config.get("dataset", {}).get("fps", 30)), "embodiment": "humanoid", "tasks": []}
    standard = adapter.to_standard_episode(ref, raw, meta)
    standard = apply_review_corrections(standard, config)
    s2_cfg = Stage2Config(fill_missing_action_eef=True, fill_lag=int(fill_lag))
    _, _, _ = fill_missing_action_eef(standard.fields, s2_cfg, fill_lag=int(fill_lag))
    temporal = config.get("temporal_align") or {}
    p4_plan = resolve_p4_plan(parse_temporal_align_config(temporal, {}))
    standard.fields = apply_p4_to_action_fields(
        standard.fields, p4_plan, lag=int(fill_lag), target_delay=1
    )
    geom_cfg = config.get("episode_geometry") or {}
    keymap = _load_json(package_dir / "keymap.json")
    result = geometry.apply_episode_frame_geometry(
        standard,
        ref=ref,
        geometry_cfg=geom_cfg,
        keymap=keymap,
    )
    if result.discard:
        raise RuntimeError(f"P6 discard ep{episode_index}: {result.discard_reasons}")

    fields = dict(result.fields)
    for k, v in result.extrinsic.items():
        col = k if k.startswith("extrinsic.") else f"extrinsic.{k}"
        fields[col] = np.asarray(v, dtype=np.float32)
    return fields


def _compare_episode(
    *,
    episode_index: int,
    gold_root: Path,
    raw_root: Path,
    pipeline_root: Path | None,
    package_dir: Path,
    config: dict[str, Any],
    fill_lag: int,
    atol: float,
) -> dict[str, Any]:
    gold_path = _parquet_path(gold_root, episode_index)
    gold = _table_to_dict(pq.read_table(gold_path))
    pred = _build_full_geometry_episode(
        raw_root=raw_root,
        episode_index=episode_index,
        package_dir=package_dir,
        config=config,
        fill_lag=fill_lag,
    )

    report: dict[str, Any] = {
        "episode_index": episode_index,
        "gold_rows": int(next(iter(gold.values())).shape[0]),
        "pred_rows": int(next(iter(pred.values())).shape[0]),
        "column_diffs": {},
        "pipeline_rows": None,
        "pipeline_column_diffs": {},
    }

    for col in GEOM_COLS:
        if col not in gold or col not in pred:
            report["column_diffs"][col] = {"status": "missing", "gold": col in gold, "pred": col in pred}
            continue
        g = gold[col]
        p = pred[col]
        n = min(g.shape[0], p.shape[0])
        diff = _max_abs(g[:n], p[:n])
        ref_ok = None
        if col.startswith("extrinsic.camera_reference"):
            ref_ok = _max_abs(g[:n], np.broadcast_to(np.eye(4), g[:n].shape))
        report["column_diffs"][col] = {
            "max_abs_diff": diff,
            "ok": diff <= atol,
            "reference_identity_max_abs_error": ref_ok,
        }

    if pipeline_root is not None:
        pipe_path = _parquet_path(pipeline_root, episode_index)
        if pipe_path.exists():
            pipe = _table_to_dict(pq.read_table(pipe_path))
            report["pipeline_rows"] = int(next(iter(pipe.values())).shape[0])
            for col in GEOM_COLS:
                if col not in gold or col not in pipe:
                    continue
                # pipeline export reindexes frame_index after quality; compare prefix length only
                n = min(gold[col].shape[0], pipe[col].shape[0])
                diff = _max_abs(gold[col][:n], pipe[col][:n])
                report["pipeline_column_diffs"][col] = {
                    "max_abs_diff_prefix": diff,
                    "ok_prefix": diff <= atol,
                    "note": "prefix compare; lengths may differ due to quality/static crop",
                }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PACKAGE_DIR / "humanoid_demo_11_raw",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=PACKAGE_DIR
        / "humanoid_demo_11_camera_geometry_v2_bundle_20260826"
        / "dataset"
        / "humanoid_demo_11_camera_geometry_v2",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=None,
        help="Optional pipeline export root for secondary prefix comparison",
    )
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "config.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fill-lag", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or (args.raw_root / "demo_manifest.json")
    manifest = _load_json(manifest_path)
    episodes = [int(x) for x in manifest["episodes"]]

    reports = []
    all_ok = True
    for ep in episodes:
        rep = _compare_episode(
            episode_index=ep,
            gold_root=args.gold_root,
            raw_root=args.raw_root,
            pipeline_root=args.pipeline_root,
            package_dir=PACKAGE_DIR,
            config=config,
            fill_lag=args.fill_lag,
            atol=args.atol,
        )
        reports.append(rep)
        col_ok = all(v.get("ok", False) for v in rep["column_diffs"].values() if "ok" in v)
        if not col_ok:
            all_ok = False
        print(
            f"ep{ep}: gold={rep['gold_rows']} pred={rep['pred_rows']} "
            f"pipe={rep['pipeline_rows']} geom_ok={col_ok}",
            flush=True,
        )
        for col, info in rep["column_diffs"].items():
            if "max_abs_diff" in info:
                print(f"  {col}: max_diff={info['max_abs_diff']:.3e} ok={info['ok']}", flush=True)

    summary = {
        "episodes": len(reports),
        "all_geometry_ok": all_ok,
        "fill_lag": args.fill_lag,
        "atol": args.atol,
        "raw_root": str(args.raw_root),
        "gold_root": str(args.gold_root),
        "pipeline_root": str(args.pipeline_root) if args.pipeline_root else None,
        "reports": reports,
    }
    out_path = args.output or (args.pipeline_root or PACKAGE_DIR) / "geometry_vs_gold_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

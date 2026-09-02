#!/usr/bin/env python3
"""Compare pipeline adapter output against a golden standard-format export."""

from __future__ import annotations

import argparse
import importlib.util
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
    "observation.state.eef.left.position",
    "observation.state.eef.left.rotation_6d",
    "observation.state.eef.right.position",
    "observation.state.eef.right.rotation_6d",
    "action.eef.left.position",
    "action.eef.left.rotation_6d",
    "action.eef.right.position",
    "action.eef.right.rotation_6d",
    "observation.state.gripper.left.closedness",
    "observation.state.gripper.right.closedness",
    "action.gripper.left.closedness",
    "action.gripper.right.closedness",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_to_dict(table) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in table.column_names:
        col = np.asarray(table[name].to_pylist(), dtype=np.float32)
        if col.ndim == 1:
            col = col.reshape(-1, 1)
        out[name] = col
    return out


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _load_adapter(package_dir: Path):
    spec = importlib.util.spec_from_file_location("egodex_adapter_cmp", package_dir / "adapter.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_adapter(package_dir)


def _build_standard_episode(
    *,
    raw_root: Path,
    episode_index: int,
    package_dir: Path,
    config: dict[str, Any],
    fill_lag: int = 2,
) -> dict[str, np.ndarray]:
    adapter = _load_adapter(package_dir)
    path = episode_parquet_path(raw_root, episode_index)
    table = read_episode_table(path)
    raw = _table_to_raw(table)
    ref = EpisodeRef(episode_index=episode_index, dataset_root=raw_root)
    meta = {"fps": float(config.get("dataset", {}).get("fps", 30)), "embodiment": "egodex", "tasks": []}
    standard = adapter.to_standard_episode(ref, raw, meta)
    standard = apply_review_corrections(standard, config)
    s2_cfg = Stage2Config(fill_missing_action_eef=True, fill_lag=int(fill_lag))
    _, _, _ = fill_missing_action_eef(standard.fields, s2_cfg, fill_lag=int(fill_lag))
    temporal = config.get("temporal_align") or {}
    p4_plan = resolve_p4_plan(parse_temporal_align_config(temporal, {}))
    standard.fields = apply_p4_to_action_fields(
        standard.fields, p4_plan, lag=int(fill_lag), target_delay=1
    )
    return {k: np.asarray(v, dtype=np.float32) for k, v in standard.fields.items()}


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
    gold_path = episode_parquet_path(gold_root, episode_index)
    gold = _table_to_dict(pq.read_table(gold_path))
    pred = _build_standard_episode(
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
            report["column_diffs"][col] = {
                "status": "missing",
                "gold": col in gold,
                "pred": col in pred,
            }
            continue
        g = gold[col]
        p = pred[col]
        n = min(g.shape[0], p.shape[0])
        diff = _max_abs(g[:n], p[:n])
        report["column_diffs"][col] = {"max_abs_diff": diff, "ok": diff <= atol}

    if pipeline_root is not None:
        pipe_path = episode_parquet_path(pipeline_root, episode_index)
        if pipe_path.exists():
            pipe = _table_to_dict(pq.read_table(pipe_path))
            report["pipeline_rows"] = int(next(iter(pipe.values())).shape[0])
            for col in GEOM_COLS:
                if col not in gold or col not in pipe:
                    continue
                n = min(gold[col].shape[0], pipe[col].shape[0])
                diff = _max_abs(gold[col][:n], pipe[col][:n])
                report["pipeline_column_diffs"][col] = {
                    "max_abs_diff_prefix": diff,
                    "ok_prefix": diff <= atol,
                }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Raw LeRobot task root (part/task dir with meta/info.json)",
    )
    parser.add_argument("--gold-root", type=Path, required=True, help="Golden standard export root")
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=None,
        help="Optional pipeline export root for secondary prefix comparison",
    )
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "config.yaml")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to compare (within the raw task root)",
    )
    parser.add_argument("--fill-lag", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reports = []
    all_ok = True
    for ep in args.episodes:
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
            f"pipe={rep['pipeline_rows']} ok={col_ok}",
            flush=True,
        )
        for col, info in rep["column_diffs"].items():
            if "max_abs_diff" in info:
                print(f"  {col}: max_diff={info['max_abs_diff']:.3e} ok={info['ok']}", flush=True)

    summary = {
        "episodes": len(reports),
        "all_ok": all_ok,
        "fill_lag": args.fill_lag,
        "atol": args.atol,
        "raw_root": str(args.raw_root),
        "gold_root": str(args.gold_root),
        "pipeline_root": str(args.pipeline_root) if args.pipeline_root else None,
        "reports": reports,
    }
    out_path = args.output or (PACKAGE_DIR / "manual_review" / "results" / "standard_vs_gold_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

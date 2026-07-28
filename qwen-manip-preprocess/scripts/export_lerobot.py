#!/usr/bin/env python3
"""Export a LeRobot dataset by running the pipeline in filter mode (single-pass, no mask parquet)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.lerobot_export import verify_lerobot_alignment
from robot_data_processing.loader import list_episode_indices
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline


def main(argv: list[str] | None = None) -> int:
    default_config = ROOT / "config" / "humanoid_merged.yaml"
    parser = argparse.ArgumentParser(
        description="Run pipeline with LeRobot export (filter mode; cropped parquet + videos, no mask column)"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--source-root", type=Path, default=None, help="Original LeRobot dataset root")
    parser.add_argument("--output-root", type=Path, required=True, help="Exported LeRobot dataset root")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Reuse stats cache from a prior pipeline run (directory containing cache/)",
    )
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--episode-indices", type=str, default=None, help="Comma-separated episode indices")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-video-stats", action="store_true", help="Skip ffmpeg video stats sampling")
    parser.add_argument("--verify-only", action="store_true", help="Only run alignment verification")
    args = parser.parse_args(argv)

    if args.episode_indices:
        indices = [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
    else:
        yaml_cfg = load_config(args.config)
        source_root = Path(args.source_root or yaml_cfg["dataset"]["root"])
        all_indices = list_episode_indices(source_root, yaml_cfg["dataset"].get("total_episodes"))
        limit = args.episode_limit or len(all_indices)
        indices = all_indices[:limit]

    if args.verify_only:
        report = verify_lerobot_alignment(args.output_root, indices)
        print(json.dumps(report, indent=2))
        return 0 if report["aligned"] else 1

    yaml_cfg = load_config(args.config)
    overrides: dict = {
        "output_dir": str(args.output_root),
        "output_mode": "filter",
        "stats_recompute": False,
        "stage1_stats_recompute": False,
        "state_action_lag_recompute": False,
        "recompute_video_stats": not args.skip_video_stats,
    }
    if args.source_root:
        overrides["dataset_root"] = str(args.source_root)
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.cache_dir:
        cache = Path(args.cache_dir) / "cache"
        overrides["stats_cache_path"] = str(cache / "global_stats.npz")
        overrides["stage1_stats_cache_path"] = str(cache / "stage1_global_stats.npz")
        overrides["state_action_lag_cache_path"] = str(cache / "state_action_lag.npz")

    cfg = pipeline_config_from_yaml(yaml_cfg, overrides)
    print(f"Exporting {len(indices)} episodes")
    print(f"  source: {cfg.dataset_root}")
    print(f"  output: {cfg.output_dir}")

    run_pipeline(cfg, indices, stats_episode_indices=indices, show_progress=True)
    report_path = cfg.output_dir / "alignment_report.json"
    if report_path.exists():
        print(f"Alignment report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

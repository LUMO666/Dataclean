#!/usr/bin/env python3
"""Run full humanoid_merged pipeline (all episodes).

Preferred entry (Dataclean phases P0–P7):
  python datasets/humanoid_merged/run.py --output-dir <out> [--force-skip-gate]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import list_episode_indices
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "humanoid_merged.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged_qwenmanip_processed"),
    )
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--export-workers", type=int, default=None)
    parser.add_argument("--output-mode", choices=["report", "filter", "both"], default="both")
    parser.add_argument("--enable-stage5", action="store_true")
    parser.add_argument("--stage2-diff-epsilon", type=float, default=None)
    parser.add_argument("--skip-alignment-verify", action="store_true")
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    dataset_root = Path(yaml_cfg["dataset"]["root"])
    total = yaml_cfg["dataset"]["total_episodes"]
    all_indices = list_episode_indices(dataset_root, total)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    overrides: dict = {
        "output_dir": str(out),
        "output_mode": args.output_mode,
        "num_workers": args.num_workers,
        "stage5_enabled": bool(args.enable_stage5),
    }
    if args.export_workers is not None:
        overrides["export_workers"] = args.export_workers
    if args.stage2_diff_epsilon is not None:
        overrides["stage2_diff_epsilon"] = args.stage2_diff_epsilon
    if args.skip_alignment_verify:
        overrides["skip_alignment_verify"] = True

    cfg = pipeline_config_from_yaml(yaml_cfg, overrides)

    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(out),
        "total_episodes": len(all_indices),
        "process_episodes": None,
        "ignored_episodes": None,
        "ignore_episodes_path": str(cfg.ignore_episodes_path) if cfg.ignore_episodes_path else None,
        "ignore_episodes_paths": [str(p) for p in cfg.ignore_episodes_paths],
        "output_mode": args.output_mode,
        "num_workers": args.num_workers,
        "export_workers": cfg.export_workers,
        "stage5_enabled": cfg.stage5.enabled if cfg.stage5 else False,
        "stage2_diff_epsilon": cfg.stage2.diff_epsilon if cfg.stage2 else None,
        "parallel_videos": cfg.parallel_videos,
        "recompute_video_stats": cfg.recompute_video_stats,
        "skip_alignment_verify": cfg.skip_alignment_verify,
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    print(f"Dataset episodes: {len(all_indices)}")
    print(f"Output: {out}")
    print(f"Mode: {args.output_mode}, workers: {args.num_workers}, export_workers: {cfg.export_workers}")

    t0 = time.perf_counter()
    results = run_pipeline(cfg, all_indices, stats_episode_indices=all_indices)
    elapsed = time.perf_counter() - t0

    ignored = 0
    if cfg.ignore_episodes_paths or cfg.ignore_episodes_path:
        from robot_data_processing.ignore_list import load_ignore_episode_lists

        paths = list(cfg.ignore_episodes_paths)
        if not paths and cfg.ignore_episodes_path is not None:
            paths = [cfg.ignore_episodes_path]
        ignored = len(load_ignore_episode_lists(paths))

    discarded = sum(1 for r in results if r.discard)
    kept_frames = sum(r.kept_frames for r in results)
    total_frames = sum(r.num_frames for r in results)

    run_meta.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "processed": len(results),
            "ignored_episodes": ignored,
            "process_episodes": len(all_indices) - ignored,
            "discarded": discarded,
            "kept_frames": kept_frames,
            "total_frames": total_frames,
            "keep_ratio": round(kept_frames / max(total_frames, 1), 4),
        }
    )
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    print("\n=== Done ===")
    print(f"Elapsed: {elapsed/3600:.2f} h")
    print(f"Processed: {len(results)}")
    print(f"Discarded: {discarded} ({100*discarded/len(results):.2f}%)")
    print(f"Kept frames: {kept_frames}/{total_frames} ({100*kept_frames/max(total_frames,1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

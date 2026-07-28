#!/usr/bin/env python3
"""Run 1000-episode humanoid filter export with timing and validation."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import list_episode_indices  # noqa: E402
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline  # noqa: E402


def main() -> int:
    out = ROOT / "output" / "humanoid_merged_1000_eps1e-3_filter"
    src_cache = ROOT / "output" / "humanoid_merged_1000_eps1e-3" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    if src_cache.exists():
        dst_cache = out / "cache"
        if not dst_cache.exists():
            import shutil
            shutil.copytree(src_cache, dst_cache)

    sample_path = ROOT / "output" / "humanoid_merged_1000_eps1e-3" / "sampled_episode_indices.json"
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())
    else:
        import numpy as np
        yaml_cfg = load_config(ROOT / "config" / "humanoid_merged.yaml")
        dataset_root = Path(yaml_cfg["dataset"]["root"])
        all_indices = list_episode_indices(dataset_root, yaml_cfg["dataset"]["total_episodes"])
        rng = np.random.default_rng(42)
        sample = sorted(rng.choice(all_indices, size=1000, replace=False).tolist())
    (out / "sampled_episode_indices.json").write_text(json.dumps(sample, indent=2))

    yaml_cfg = load_config(ROOT / "config" / "humanoid_merged.yaml")
    cfg = pipeline_config_from_yaml(
        yaml_cfg,
        {
            "output_dir": str(out),
            "output_mode": "filter",
            "num_workers": 64,
            "stage5_enabled": False,
            "stage2_diff_epsilon": 1e-3,
            "stats_recompute": False,
            "stage1_stats_recompute": False,
            "state_action_lag_recompute": False,
            "recompute_video_stats": True,
        },
    )

    timing: dict[str, float | dict] = {"sample_size": len(sample), "output_dir": str(out)}
    t0 = time.perf_counter()
    results = run_pipeline(cfg, sample, stats_episode_indices=sample, show_progress=True)
    timing["pipeline_total_sec"] = round(time.perf_counter() - t0, 2)

    discarded = sum(1 for r in results if r.discard)
    kept_frames = sum(r.kept_frames for r in results)
    total_frames = sum(r.num_frames for r in results)
    timing["processed"] = len(results)
    timing["discarded"] = discarded
    timing["kept_frames"] = kept_frames
    timing["total_frames"] = total_frames
    timing["keep_ratio"] = round(kept_frames / max(total_frames, 1), 4)

    t1 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_export.py"), str(out)],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    timing["validation_sec"] = round(time.perf_counter() - t1, 2)
    timing["validation_exit_code"] = proc.returncode
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    validation_path = out / "validation_report.json"
    if validation_path.exists():
        timing["validation"] = json.loads(validation_path.read_text())

    timing_path = out / "timing_report.json"
    timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"\nTiming report: {timing_path}")
    print(json.dumps({k: v for k, v in timing.items() if k != "validation"}, indent=2, ensure_ascii=False))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

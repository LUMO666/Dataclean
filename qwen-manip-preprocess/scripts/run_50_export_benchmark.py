#!/usr/bin/env python3
"""Benchmark optimized export (parallel workers + parallel videos) on 50 episodes."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline  # noqa: E402

CACHE_SRC = ROOT / "output" / "humanoid_merged_1000_eps1e-3" / "cache"
SAMPLE_SRC = ROOT / "output" / "humanoid_merged_1000_eps1e-3" / "sampled_episode_indices.json"


def _prepare_out(out: Path) -> list[int]:
    out.mkdir(parents=True, exist_ok=True)
    if CACHE_SRC.exists():
        dst_cache = out / "cache"
        if dst_cache.exists():
            shutil.rmtree(dst_cache)
        shutil.copytree(CACHE_SRC, dst_cache)
    sample = json.loads(SAMPLE_SRC.read_text())
    subset = sample[:50]
    (out / "sampled_episode_indices.json").write_text(json.dumps(subset, indent=2))
    return subset


def _run_export_benchmark(
    out: Path,
    sample: list[int],
    *,
    export_workers: int,
    parallel_videos: bool,
    recompute_video_stats: bool,
    skip_alignment_verify: bool,
) -> dict:
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
            "export_workers": export_workers,
            "parallel_videos": parallel_videos,
            "recompute_video_stats": recompute_video_stats,
            "skip_alignment_verify": skip_alignment_verify,
        },
    )
    t0 = time.perf_counter()
    results = run_pipeline(cfg, sample, stats_episode_indices=sample, show_progress=True)
    elapsed = round(time.perf_counter() - t0, 2)
    export_report = json.loads((out / "export_report.json").read_text()) if (out / "export_report.json").exists() else {}
    return {
        "output_dir": str(out),
        "export_workers": export_workers,
        "parallel_videos": parallel_videos,
        "recompute_video_stats": recompute_video_stats,
        "skip_alignment_verify": skip_alignment_verify,
        "elapsed_sec": elapsed,
        "processed": len(results),
        "exported_episodes": export_report.get("total_episodes"),
        "exported_frames": export_report.get("total_frames"),
        "skipped_episodes": export_report.get("skipped_episodes"),
    }


def _validate(out: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_export.py"), str(out)],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads((out / "validation_report.json").read_text()) if (out / "validation_report.json").exists() else {}
    return {
        "exit_code": proc.returncode,
        "aligned": report.get("alignment", {}).get("aligned"),
        "meta_ok": report.get("meta", {}).get("meta_ok"),
        "issues": report.get("alignment", {}).get("issues", []) + report.get("meta", {}).get("issues", []),
    }


def main() -> int:
    baseline_out = ROOT / "output" / "benchmark_50ep_baseline"
    optimized_out = ROOT / "output" / "benchmark_50ep_optimized"

    print("=== Baseline: export_workers=1, sequential videos ===")
    sample = _prepare_out(baseline_out)
    baseline = _run_export_benchmark(
        baseline_out,
        sample,
        export_workers=1,
        parallel_videos=False,
        recompute_video_stats=True,
        skip_alignment_verify=True,
    )
    print(json.dumps(baseline, indent=2))

    print("\n=== Optimized: export_workers=16, parallel videos, no video stats ===")
    _prepare_out(optimized_out)
    optimized = _run_export_benchmark(
        optimized_out,
        sample,
        export_workers=16,
        parallel_videos=True,
        recompute_video_stats=False,
        skip_alignment_verify=True,
    )
    print(json.dumps(optimized, indent=2))

    speedup = baseline["elapsed_sec"] / max(optimized["elapsed_sec"], 0.01)
    print(f"\nSpeedup (wall): {speedup:.2f}x")

    print("\n=== Validation (optimized output) ===")
    validation = _validate(optimized_out)
    print(json.dumps(validation, indent=2, ensure_ascii=False))

    summary = {
        "sample_size": len(sample),
        "baseline": baseline,
        "optimized": optimized,
        "speedup": round(speedup, 2),
        "validation": validation,
    }
    report_path = ROOT / "output" / "benchmark_50ep_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path}")
    ok = validation.get("exit_code") == 0
    print(f"Overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sample and process episodes from egodex_lerobot_v21 (multi-part layout)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline
from robot_data_processing.report import build_quality_report, write_exclusion_log, write_quality_report
from robot_data_processing.types import EpisodeResult


def _load_task_results(task_out: Path) -> list[EpisodeResult]:
    log_path = task_out / "reports" / "exclusion_log.jsonl"
    if not log_path.exists():
        return []
    results: list[EpisodeResult] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(
            EpisodeResult(
                episode_index=row["episode_index"],
                num_frames=row["num_frames"],
                discard=row["discard"],
                discard_reasons=row.get("discard_reasons") or [],
                first_abnormal_frame=row.get("first_abnormal_frame"),
                stage1_flagged_frames=row.get("stage1_flagged_frames", 0),
                stage2_da_mean=row.get("stage2_da_mean"),
                stage2_da_per_dim=row.get("stage2_da_per_dim"),
                stage2_lags=row.get("stage2_lags"),
                stage3_excluded_frames=row.get("stage3_excluded_frames", 0),
                stage4_removed_frames=row.get("stage4_removed_frames", 0),
                metadata=row.get("metadata") or {},
            )
        )
    return results


def discover_egodex_episodes(root: Path) -> list[dict]:
    refs: list[dict] = []
    for part in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in part.iterdir() if p.is_dir()):
            info_path = task_dir / "meta" / "info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            total = int(info.get("total_episodes", 0))
            for ep in range(total):
                refs.append(
                    {
                        "part": part.name,
                        "task": task_dir.name,
                        "dataset_root": str(task_dir),
                        "episode_index": ep,
                    }
                )
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "egodex_22T.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--output-mode", choices=["report", "filter", "both"], default="report")
    parser.add_argument("--resume", action="store_true", help="Skip task groups with existing reports")
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    dataset_root = Path(yaml_cfg["dataset"]["root"])
    all_refs = discover_egodex_episodes(dataset_root)
    print(f"Discovered {len(all_refs)} episodes across egodex 22T layout")

    rng = np.random.default_rng(args.seed)
    n = min(args.sample_size, len(all_refs))
    pick = sorted(rng.choice(len(all_refs), size=n, replace=False).tolist())
    sample = [all_refs[i] for i in pick]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "sampled_episodes.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for ref in sample:
        by_task[ref["dataset_root"]].append(ref)

    print(f"Sample size: {n}, seed: {args.seed}, task groups: {len(by_task)}")

    all_results = []
    t0 = time.time()
    for i, (task_root, refs) in enumerate(sorted(by_task.items()), start=1):
        eps = sorted({r["episode_index"] for r in refs})
        part = refs[0]["part"]
        task = refs[0]["task"]
        task_out = out / "_task_runs" / part / task
        if args.resume and (task_out / "reports" / "quality_report.json").exists():
            print(f"[{i}/{len(by_task)}] {part}/{task}: skip (cached)", flush=True)
            cached = _load_task_results(task_out)
            ref_by_ep = {r["episode_index"]: r for r in refs}
            for result in cached:
                ref = ref_by_ep.get(result.episode_index, {})
                result.metadata.update(
                    {
                        "part": ref.get("part"),
                        "task": ref.get("task"),
                        "dataset_root": ref.get("dataset_root"),
                    }
                )
            all_results.extend(cached)
            continue
        task_out.mkdir(parents=True, exist_ok=True)

        cfg = pipeline_config_from_yaml(
            yaml_cfg,
            {
                "dataset_root": task_root,
                "output_dir": str(task_out),
                "output_mode": args.output_mode,
                "num_workers": args.num_workers,
            },
        )
        print(f"[{i}/{len(by_task)}] {part}/{task}: {len(eps)} episodes", flush=True)
        results = run_pipeline(cfg, eps, stats_episode_indices=eps, show_progress=False)
        ref_by_ep = {r["episode_index"]: r for r in refs}
        for result in results:
            ref = ref_by_ep.get(result.episode_index, {})
            result.metadata.update(
                {
                    "part": ref.get("part"),
                    "task": ref.get("task"),
                    "dataset_root": ref.get("dataset_root"),
                }
            )
            all_results.append(result)

    elapsed = time.time() - t0
    discarded = sum(1 for r in all_results if r.discard)
    kept_frames = sum(r.kept_frames for r in all_results)
    total_frames = sum(r.num_frames for r in all_results)

    report = build_quality_report(
        all_results,
        stats_meta={
            "mode": "per_task_stats",
            "sample_size": n,
            "task_groups": len(by_task),
            "total_catalog_episodes": len(all_refs),
            "embodiment": yaml_cfg["schema"]["embodiment"],
        },
        config_summary={
            "dataset_root": str(dataset_root),
            "output_mode": args.output_mode,
            "num_workers": args.num_workers,
            "sample_seed": args.seed,
            "elapsed_seconds": round(elapsed, 1),
        },
    )
    reports_dir = out / "reports"
    write_quality_report(reports_dir / "quality_report.json", report)
    write_exclusion_log(reports_dir / "exclusion_log.jsonl", all_results)

    run_meta = {
        "sample_size": n,
        "seed": args.seed,
        "total_catalog_episodes": len(all_refs),
        "task_groups": len(by_task),
        "elapsed_seconds": round(elapsed, 1),
        "discarded_episodes": discarded,
        "kept_frames": kept_frames,
        "total_frames": total_frames,
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"Processed: {len(all_results)}")
    print(f"Discarded: {discarded} ({100 * discarded / max(len(all_results), 1):.2f}%)")
    print(
        f"Kept frames: {kept_frames}/{total_frames} "
        f"({100 * kept_frames / max(total_frames, 1):.2f}%)"
    )
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print(f"Report: {reports_dir / 'quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

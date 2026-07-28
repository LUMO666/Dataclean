#!/usr/bin/env python3
"""Run full egodex_lerobot_v21 pipeline (all tasks), mirroring source layout.

Preferred entry (Dataclean phases P0–P7):
  python datasets/egodex_lerobot_v21/run.py --output-dir <out> [--force-skip-gate]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.loader import list_episode_indices
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline
from robot_data_processing.report import build_quality_report, write_exclusion_log, write_quality_report
from robot_data_processing.types import EpisodeResult


def discover_tasks(root: Path) -> list[dict]:
    tasks: list[dict] = []
    for part in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in part.iterdir() if p.is_dir()):
            info_path = task_dir / "meta" / "info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            total = int(info.get("total_episodes", 0))
            tasks.append(
                {
                    "part": part.name,
                    "task": task_dir.name,
                    "dataset_root": task_dir,
                    "total_episodes": total,
                    "rel_path": f"{part.name}/{task_dir.name}",
                }
            )
    return tasks


def _task_done(task_out: Path, output_mode: str) -> bool:
    if not (task_out / "reports" / "quality_report.json").exists():
        return False
    if output_mode in ("filter", "both"):
        return (task_out / "export_report.json").exists() and (task_out / "meta" / "episodes.jsonl").exists()
    return True


def _load_task_results(task_out: Path, part: str, task: str, dataset_root: str) -> list[EpisodeResult]:
    log_path = task_out / "reports" / "exclusion_log.jsonl"
    if not log_path.exists():
        return []
    results: list[EpisodeResult] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        meta.update({"part": part, "task": task, "dataset_root": dataset_root})
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
                metadata=meta,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "egodex_22T.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_lerobot_v21_qwenmanip_processed"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--export-workers", type=int, default=None)
    parser.add_argument("--output-mode", choices=["report", "filter", "both"], default="both")
    parser.add_argument("--skip-alignment-verify", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks for smoke testing")
    args = parser.parse_args()
    resume = args.resume and not args.no_resume

    yaml_cfg = load_config(args.config)
    dataset_root = Path(yaml_cfg["dataset"]["root"])
    tasks = discover_tasks(dataset_root)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    catalog_eps = sum(t["total_episodes"] for t in tasks)
    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(out),
        "output_layout": "mirror_source_part_task",
        "total_tasks": len(tasks),
        "total_catalog_episodes": catalog_eps,
        "output_mode": args.output_mode,
        "num_workers": args.num_workers,
        "resume": resume,
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    (out / "task_catalog.json").write_text(
        json.dumps(
            [
                {
                    "part": t["part"],
                    "task": t["task"],
                    "rel_path": t["rel_path"],
                    "dataset_root": str(t["dataset_root"]),
                    "total_episodes": t["total_episodes"],
                }
                for t in tasks
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Tasks: {len(tasks)}, catalog episodes: {catalog_eps}", flush=True)
    print(f"Output: {out}", flush=True)
    print(f"Mode: {args.output_mode}, workers: {args.num_workers}, resume={resume}", flush=True)

    all_results: list[EpisodeResult] = []
    task_status: list[dict] = []
    t0 = time.perf_counter()

    for i, task_info in enumerate(tasks, start=1):
        part = task_info["part"]
        task = task_info["task"]
        task_root = task_info["dataset_root"]
        task_out = out / part / task
        rel = task_info["rel_path"]

        if resume and _task_done(task_out, args.output_mode):
            print(f"[{i}/{len(tasks)}] {rel}: skip (done)", flush=True)
            results = _load_task_results(task_out, part, task, str(task_root))
            all_results.extend(results)
            task_status.append(
                {
                    "rel_path": rel,
                    "status": "skipped",
                    "episodes": len(results),
                    "elapsed_sec": 0.0,
                }
            )
            continue

        eps = list_episode_indices(task_root, task_info["total_episodes"])
        overrides: dict = {
            "dataset_root": str(task_root),
            "output_dir": str(task_out),
            "output_mode": args.output_mode,
            "num_workers": args.num_workers,
        }
        if args.export_workers is not None:
            overrides["export_workers"] = args.export_workers
        if args.skip_alignment_verify:
            overrides["skip_alignment_verify"] = True

        cfg = pipeline_config_from_yaml(yaml_cfg, overrides)
        print(
            f"[{i}/{len(tasks)}] {rel}: {len(eps)} episodes "
            f"(export_workers={cfg.export_workers})",
            flush=True,
        )

        task_t0 = time.perf_counter()
        try:
            results = run_pipeline(cfg, eps, stats_episode_indices=eps, show_progress=True)
            for result in results:
                result.metadata.update(
                    {"part": part, "task": task, "dataset_root": str(task_root)}
                )
            all_results.extend(results)
            task_elapsed = time.perf_counter() - task_t0
            discarded = sum(1 for r in results if r.discard)
            kept = sum(r.kept_frames for r in results)
            total_frames = sum(r.num_frames for r in results)
            status = {
                "rel_path": rel,
                "status": "ok",
                "episodes": len(results),
                "discarded": discarded,
                "kept_frames": kept,
                "total_frames": total_frames,
                "elapsed_sec": round(task_elapsed, 1),
            }
            task_status.append(status)
            print(
                f"  -> done in {task_elapsed/60:.1f} min, "
                f"discard={discarded}, kept={kept}/{total_frames}",
                flush=True,
            )
        except Exception as exc:
            task_elapsed = time.perf_counter() - task_t0
            err = f"{type(exc).__name__}: {exc}"
            print(f"  -> FAILED after {task_elapsed/60:.1f} min: {err}", flush=True)
            traceback.print_exc()
            task_status.append(
                {
                    "rel_path": rel,
                    "status": "failed",
                    "error": err,
                    "elapsed_sec": round(task_elapsed, 1),
                }
            )
            (task_out / "task_error.json").write_text(
                json.dumps(
                    {"error": err, "traceback": traceback.format_exc()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # Persist progress after each task
        (out / "task_status.json").write_text(
            json.dumps(task_status, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    elapsed = time.perf_counter() - t0
    discarded = sum(1 for r in all_results if r.discard)
    kept_frames = sum(r.kept_frames for r in all_results)
    total_frames = sum(r.num_frames for r in all_results)
    failed = [t for t in task_status if t.get("status") == "failed"]

    report = build_quality_report(
        all_results,
        stats_meta={
            "mode": "per_task_stats",
            "task_groups": len(tasks),
            "total_catalog_episodes": catalog_eps,
            "embodiment": yaml_cfg["schema"]["embodiment"],
        },
        config_summary={
            "dataset_root": str(dataset_root),
            "output_dir": str(out),
            "output_mode": args.output_mode,
            "num_workers": args.num_workers,
            "elapsed_seconds": round(elapsed, 1),
            "failed_tasks": len(failed),
        },
    )
    reports_dir = out / "reports"
    write_quality_report(reports_dir / "quality_report.json", report)
    write_exclusion_log(reports_dir / "exclusion_log.jsonl", all_results)

    run_meta.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "processed_episodes": len(all_results),
            "discarded": discarded,
            "kept_frames": kept_frames,
            "total_frames": total_frames,
            "keep_ratio": round(kept_frames / max(total_frames, 1), 4),
            "failed_tasks": len(failed),
            "ok_tasks": sum(1 for t in task_status if t.get("status") == "ok"),
            "skipped_tasks": sum(1 for t in task_status if t.get("status") == "skipped"),
        }
    )
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    (out / "task_status.json").write_text(
        json.dumps(task_status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Done ===", flush=True)
    print(f"Elapsed: {elapsed/3600:.2f} h", flush=True)
    print(f"Processed: {len(all_results)} episodes", flush=True)
    print(f"Discarded: {discarded} ({100*discarded/max(len(all_results),1):.2f}%)", flush=True)
    print(
        f"Kept frames: {kept_frames}/{total_frames} "
        f"({100*kept_frames/max(total_frames,1):.2f}%)",
        flush=True,
    )
    print(f"Failed tasks: {len(failed)}", flush=True)
    print(f"Report: {reports_dir / 'quality_report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

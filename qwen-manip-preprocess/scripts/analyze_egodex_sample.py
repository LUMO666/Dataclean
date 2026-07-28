#!/usr/bin/env python3
"""Analyze egodex 22T sample pipeline results."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    quality = _load_json(output_dir / "reports" / "quality_report.json") or {}
    run_meta = _load_json(output_dir / "run_meta.json") or {}
    sample = _load_json(output_dir / "sampled_episodes.json") or []
    exclusion = _load_jsonl(output_dir / "reports" / "exclusion_log.jsonl")

    summary = quality.get("summary") or {}
    stage2 = quality.get("stage2_da") or {}

    by_part: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "discarded": 0, "frames": 0, "kept": 0})
    by_task: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "discarded": 0, "frames": 0, "kept": 0})
    discard_reasons: Counter = Counter()
    frame_lengths = []

    for row in exclusion:
        meta = row.get("metadata") or {}
        part = meta.get("part") or "unknown"
        task = meta.get("task") or "unknown"
        key = f"{part}/{task}"
        for bucket in (by_part[part], by_task[key]):
            bucket["episodes"] += 1
            bucket["frames"] += row.get("num_frames", 0)
            bucket["kept"] += row.get("kept_frames", 0)
            if row.get("discard"):
                bucket["discarded"] += 1
        frame_lengths.append(row.get("num_frames", 0))
        for reason in row.get("discard_reasons") or []:
            discard_reasons[reason.split("=")[0]] += 1

    part_rows = []
    for part, stats in sorted(by_part.items()):
        fr = stats["kept"] / stats["frames"] if stats["frames"] else 0.0
        part_rows.append(
            {
                "part": part,
                "episodes": stats["episodes"],
                "discarded": stats["discarded"],
                "valid_frame_rate": round(fr, 4),
            }
        )

    task_rows = []
    for task, stats in sorted(by_task.items(), key=lambda x: -x[1]["episodes"]):
        fr = stats["kept"] / stats["frames"] if stats["frames"] else 0.0
        task_rows.append(
            {
                "task": task,
                "episodes": stats["episodes"],
                "discarded": stats["discarded"],
                "valid_frame_rate": round(fr, 4),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "run_meta": run_meta,
        "catalog": {
            "sample_size": len(sample),
            "unique_tasks_in_sample": len(by_task),
            "unique_parts_in_sample": len(by_part),
        },
        "highlights": {
            "total_episodes": summary.get("total_episodes"),
            "discarded_episodes": summary.get("discarded_episodes"),
            "discard_rate": summary.get("discard_rate"),
            "total_frames": summary.get("total_frames"),
            "kept_frames": summary.get("kept_frames"),
            "valid_frame_rate": summary.get("valid_prefix_frame_rate"),
            "stage1_flagged_frames": summary.get("stage1_flagged_frames"),
            "stage3_excluded_frames": summary.get("stage3_excluded_frames"),
            "stage4_removed_frames": summary.get("stage4_removed_frames"),
            "stage2_da_mean": stage2.get("mean"),
            "stage2_da_below_0.7": stage2.get("below_0.7"),
            "elapsed_minutes": round((run_meta.get("elapsed_seconds") or 0) / 60, 1),
        },
        "frame_length": {
            "mean": round(sum(frame_lengths) / len(frame_lengths), 1) if frame_lengths else None,
            "min": min(frame_lengths) if frame_lengths else None,
            "max": max(frame_lengths) if frame_lengths else None,
        },
        "by_part": part_rows,
        "top_tasks_by_episode_count": task_rows[:20],
        "discard_reasons": dict(discard_reasons.most_common(20)),
        "quality_report": quality,
    }
    return report


def write_markdown(report: dict, path: Path) -> None:
    h = report["highlights"]
    lines = [
        "# EgoDex 22T Sample Analysis",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Output: `{report['output_dir']}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sample episodes | {h.get('total_episodes')} |",
        f"| Discarded episodes | {h.get('discarded_episodes')} ({100 * (h.get('discard_rate') or 0):.2f}%) |",
        f"| Valid frame rate | {100 * (h.get('valid_frame_rate') or 0):.2f}% |",
        f"| Stage2 DA mean | {h.get('stage2_da_mean')} |",
        f"| Stage2 DA < 0.7 | {h.get('stage2_da_below_0.7')} |",
        f"| Stage1 flagged frames | {h.get('stage1_flagged_frames')} |",
        f"| Stage3 excluded frames | {h.get('stage3_excluded_frames')} |",
        f"| Stage4 removed frames | {h.get('stage4_removed_frames')} |",
        f"| Elapsed | {h.get('elapsed_minutes')} min |",
        "",
        "## By Partition",
        "",
        "| Part | Episodes | Discarded | Valid Frame Rate |",
        "|------|----------|-----------|------------------|",
    ]
    for row in report.get("by_part") or []:
        lines.append(
            f"| {row['part']} | {row['episodes']} | {row['discarded']} | "
            f"{100 * row['valid_frame_rate']:.2f}% |"
        )
    lines.extend(["", "## Top Tasks in Sample", ""])
    for row in report.get("top_tasks_by_episode_count") or []:
        lines.append(
            f"- `{row['task']}`: {row['episodes']} ep, discard={row['discarded']}, "
            f"valid={100 * row['valid_frame_rate']:.1f}%"
        )
    if report.get("discard_reasons"):
        lines.extend(["", "## Discard Reasons", ""])
        for reason, count in report["discard_reasons"].items():
            lines.append(f"- `{reason}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.output_dir)
    out_json = args.output_dir / "analysis_report.json"
    out_md = args.output_dir / "analysis_report.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, out_md)
    print(json.dumps(report["highlights"], indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

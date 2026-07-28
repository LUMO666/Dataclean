#!/usr/bin/env python3
"""Generate post-run analysis report for full humanoid_merged pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _dir_size_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], stderr=subprocess.DEVNULL, text=True)
        return round(int(out.split()[0]) / 1024**3, 2)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(output_dir: Path, *, run_validation: bool = True) -> dict:
    output_dir = Path(output_dir)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
    }

    report["run_meta"] = _load_json(output_dir / "run_meta.json")
    report["quality"] = _load_json(output_dir / "reports" / "quality_report.json")
    report["export"] = _load_json(output_dir / "export_report.json")
    report["alignment"] = _load_json(output_dir / "alignment_report.json")

    sizes = {}
    for name in ("data", "videos", "meta", "parameters", "cache", "reports"):
        sizes[name] = _dir_size_gb(output_dir / name)
    sizes["total_gb"] = _dir_size_gb(output_dir)
    report["storage_gb"] = sizes

    q = report.get("quality") or {}
    summary = q.get("summary") or {}
    exp = report.get("export") or {}

    report["highlights"] = {
        "total_episodes_processed": summary.get("total_episodes"),
        "discarded_episodes": summary.get("discarded_episodes"),
        "exported_episodes": exp.get("total_episodes"),
        "skipped_export_episodes": exp.get("skipped_episodes"),
        "original_frames": summary.get("total_frames"),
        "kept_frames": summary.get("kept_frames"),
        "exported_frames": exp.get("total_frames"),
        "valid_frame_rate": summary.get("valid_prefix_frame_rate"),
        "stage1_flagged_frames": summary.get("stage1_flagged_frames"),
        "stage3_excluded_frames": summary.get("stage3_excluded_frames"),
        "stage4_removed_frames": summary.get("stage4_removed_frames"),
        "stage2_da_mean": (q.get("stage2_da") or {}).get("mean"),
        "stage2_da_below_0.7": (q.get("stage2_da") or {}).get("below_0.7"),
        "truncated_export_episodes": exp.get("truncated_episodes"),
        "total_output_gb": sizes.get("total_gb"),
        "video_output_gb": sizes.get("videos"),
    }

    if run_validation and (output_dir / "meta" / "episodes.jsonl").exists():
        from robot_data_processing.lerobot_export import verify_lerobot_alignment

        from robot_data_processing.lerobot_export import _load_jsonl

        episodes = _load_jsonl(output_dir / "meta" / "episodes.jsonl")
        indices = [row["episode_index"] for row in episodes]
        report["validation"] = verify_lerobot_alignment(output_dir, indices)
    else:
        report["validation"] = report.get("alignment")

    # Stage2 discard sample from exclusion log (first 20)
    excl_path = output_dir / "reports" / "exclusion_log.jsonl"
    stage2_discard_eps: list[int] = []
    empty_mask_eps = 0
    if excl_path.exists():
        with excl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                reasons = row.get("discard_reasons") or []
                if any(str(r).startswith("stage2:") for r in reasons):
                    stage2_discard_eps.append(int(row["episode_index"]))
                kept = row.get("metadata", {}).get("kept_frames", row.get("kept_frames"))
                if kept == 0:
                    empty_mask_eps += 1
    report["stage2_discard_episodes_count"] = len(stage2_discard_eps)
    report["stage2_discard_episodes_sample"] = sorted(stage2_discard_eps)[:20]
    report["zero_kept_frames_episodes"] = empty_mask_eps

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged_qwenmanip_processed"),
    )
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = analyze(args.output_dir, run_validation=not args.no_validation)
    out_json = args.output_dir / "analysis_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = args.markdown or (args.output_dir / "analysis_report.md")
    h = report.get("highlights") or {}
    rm = report.get("run_meta") or {}
    lines = [
        "# Humanoid Merged 全量处理分析报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 输出目录: `{report['output_dir']}`",
        "",
        "## 运行配置",
        "",
        f"- 开始: {rm.get('started_at', 'N/A')}",
        f"- 结束: {rm.get('finished_at', '进行中/未知')}",
        f"- 耗时: {rm.get('elapsed_sec', 'N/A')} s",
        f"- Workers: {rm.get('num_workers')} (export: {rm.get('export_workers')})",
        f"- Mode: {rm.get('output_mode')}",
        f"- Stage2 diff_epsilon: {rm.get('stage2_diff_epsilon')}",
        "",
        "## 数据质量摘要",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 处理 episode 数 | {h.get('total_episodes_processed', 'N/A')} |",
        f"| 标记 discard 数 | {h.get('discarded_episodes', 'N/A')} |",
        f"| 成功导出 episode | {h.get('exported_episodes', 'N/A')} |",
        f"| 导出跳过 (无有效帧) | {h.get('skipped_export_episodes', 'N/A')} |",
        f"| 原始总帧数 | {h.get('original_frames', 'N/A'):,} |" if h.get("original_frames") else "| 原始总帧数 | N/A |",
        f"| 保留帧数 | {h.get('kept_frames', 'N/A'):,} |" if h.get("kept_frames") else "| 保留帧数 | N/A |",
        f"| 导出帧数 | {h.get('exported_frames', 'N/A'):,} |" if h.get("exported_frames") else "| 导出帧数 | N/A |",
        f"| 有效帧率 | {100*(h.get('valid_frame_rate') or 0):.2f}% |",
        f"| Stage2 DA 均值 | {h.get('stage2_da_mean', 'N/A')} |",
        f"| Stage2 DA<0.7 ep 数 | {h.get('stage2_da_below_0.7', 'N/A')} |",
        f"| Stage1 标记帧 | {h.get('stage1_flagged_frames', 'N/A'):,} |" if h.get("stage1_flagged_frames") is not None else "| Stage1 标记帧 | N/A |",
        f"| Stage3 剔除帧 | {h.get('stage3_excluded_frames', 'N/A'):,} |" if h.get("stage3_excluded_frames") is not None else "| Stage3 剔除帧 | N/A |",
        f"| Stage4 静态删帧 | {h.get('stage4_removed_frames', 'N/A'):,} |" if h.get("stage4_removed_frames") is not None else "| Stage4 静态删帧 | N/A |",
        f"| 导出 truncated ep | {h.get('truncated_export_episodes', 'N/A')} |",
        "",
        "## 存储占用 (GB)",
        "",
    ]
    for k, v in (report.get("storage_gb") or {}).items():
        lines.append(f"- {k}: {v}")
    val = report.get("validation") or {}
    lines.extend(
        [
            "",
            "## 对齐校验",
            "",
            f"- aligned: {val.get('aligned', 'N/A')}",
            f"- episodes_checked: {val.get('episodes_checked', 'N/A')}",
            f"- issues: {len(val.get('issues') or [])}",
            "",
            f"详细 JSON: `{out_json}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"json": str(out_json), "markdown": str(md_path), "highlights": h}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

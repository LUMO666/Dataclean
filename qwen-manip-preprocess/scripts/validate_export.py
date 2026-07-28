#!/usr/bin/env python3
"""Validate LeRobot export alignment and meta consistency."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_data_processing.lerobot_export import (  # noqa: E402
    _load_jsonl,
    load_info,
    verify_lerobot_alignment,
)


def _weighted_mean(episode_stats: list[dict], feature: str, stat: str) -> list[float]:
    total = 0
    acc: list[float] = []
    for ep in episode_stats:
        count = ep["stats"][feature]["count"][0]
        vals = ep["stats"][feature][stat]
        total += count
        if not acc:
            acc = [0.0] * len(vals)
        for i, v in enumerate(vals):
            acc[i] += float(v) * count
    return [v / max(total, 1) for v in acc]


def verify_meta(output_root: Path) -> dict:
    output_root = Path(output_root)
    info = load_info(output_root)
    episodes = _load_jsonl(output_root / "meta" / "episodes.jsonl")
    ep_stats = _load_jsonl(output_root / "meta" / "episodes_stats.jsonl")
    source_map = _load_jsonl(output_root / "meta" / "episode_source_map.jsonl")
    stats_path = output_root / "meta" / "stats.json"

    issues: list[str] = []
    summed_len = sum(int(r["length"]) for r in episodes)
    if int(info.get("total_episodes", -1)) != len(episodes):
        issues.append(
            f"info.total_episodes={info.get('total_episodes')} != episodes.jsonl rows={len(episodes)}"
        )
    if int(info.get("total_frames", -1)) != summed_len:
        issues.append(f"info.total_frames={info.get('total_frames')} != sum(length)={summed_len}")

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    expected_videos = len(episodes) * len(video_keys)
    if int(info.get("total_videos", -1)) != expected_videos:
        issues.append(f"info.total_videos={info.get('total_videos')} != {expected_videos}")

    ep_len_map = {r["episode_index"]: int(r["length"]) for r in episodes}
    stats_map = {r["episode_index"]: r for r in ep_stats}
    if set(ep_len_map) != set(stats_map):
        issues.append("episodes.jsonl and episodes_stats.jsonl episode_index mismatch")

    expected_global = 0
    for row in sorted(source_map, key=lambda r: r["merged_episode_index"]):
        ep = int(row["merged_episode_index"])
        length = ep_len_map.get(ep)
        start = int(row["merged_global_index_start"])
        end = int(row["merged_global_index_end"])
        num = int(row["merged_num_frames"])
        if length is None:
            issues.append(f"source_map ep {ep} missing from episodes.jsonl")
            continue
        if num != length:
            issues.append(f"ep {ep}: merged_num_frames={num} != length={length}")
        if start != expected_global:
            issues.append(f"ep {ep}: global start={start} != expected {expected_global}")
        if end != start + length - 1:
            issues.append(f"ep {ep}: global end={end} != start+len-1={start + length - 1}")
        expected_global = end + 1

    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            global_stats = json.load(f)
        state_key = "observation.state"
        action_key = "action"
        ep_stat_rows = [stats_map[ep] for ep in sorted(stats_map)]
        for feature, key in [("state", state_key), ("action", action_key)]:
            for stat in ("mean", "std"):
                expected = _weighted_mean(ep_stat_rows, key, stat)
                actual = global_stats[feature][stat]
                if not np.allclose(expected, actual, rtol=0, atol=1e-5):
                    max_diff = max(abs(a - b) for a, b in zip(actual, expected))
                    issues.append(f"stats.json {feature}.{stat} mismatch, max_diff={max_diff:.2e}")
            exp_min = [min(r["stats"][key]["min"][d] for r in ep_stat_rows) for d in range(len(global_stats[feature]["min"]))]
            exp_max = [max(r["stats"][key]["max"][d] for r in ep_stat_rows) for d in range(len(global_stats[feature]["max"]))]
            if not np.allclose(global_stats[feature]["min"], exp_min, rtol=0, atol=1e-6):
                issues.append(f"stats.json {feature}.min mismatch")
            if not np.allclose(global_stats[feature]["max"], exp_max, rtol=0, atol=1e-6):
                issues.append(f"stats.json {feature}.max mismatch")
    else:
        issues.append("missing meta/stats.json")

    export_report = output_root / "export_report.json"
    export_summary = {}
    if export_report.exists():
        with export_report.open("r", encoding="utf-8") as f:
            export_summary = json.load(f)
        if export_summary.get("total_frames") != summed_len:
            issues.append("export_report.total_frames != sum(episodes.length)")

    return {
        "meta_ok": len(issues) == 0,
        "total_episodes": len(episodes),
        "total_frames": summed_len,
        "total_videos": expected_videos,
        "export_report": {
            "total_episodes": export_summary.get("total_episodes"),
            "total_frames": export_summary.get("total_frames"),
            "truncated_episodes": export_summary.get("truncated_episodes"),
            "skipped_episodes": export_summary.get("skipped_episodes"),
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    alignment = verify_lerobot_alignment(args.output_root)
    meta = verify_meta(args.output_root)
    report = {"alignment": alignment, "meta": meta}
    out_path = args.output_root / "validation_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = alignment["aligned"] and meta["meta_ok"]
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run Dataclean P0–P7 for egodex_lerobot_v21."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

from robot_data_processing.phases.orchestrator import run_dataset_phases  # noqa: E402


def _parse_episode_range(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if ":" not in text:
        raise SystemExit("--episode-range must be START:END (END exclusive, source episode_index)")
    start_s, end_s = text.split(":", 1)
    start, end = int(start_s), int(end_s)
    if end <= start:
        raise SystemExit(f"--episode-range invalid: {raw!r} (need END > START)")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-skip-gate", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-mode", choices=["report", "filter", "both"], default=None)
    parser.add_argument(
        "--episode-range",
        type=str,
        default=None,
        help="After sample: keep source episode_index in [START, END).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="0-based shard id; requires --num-shards. Splits by list position after sample/range.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total shards; pair with --shard-id.",
    )
    parser.add_argument(
        "--stats-cache-dir",
        type=Path,
        default=None,
        help="Shared PFS dir for global_stats / stage1 / lag npz (read or write).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute global stats on the full sample (before shard) into --stats-cache-dir, then exit.",
    )
    parser.add_argument(
        "--export-shard-only",
        action="store_true",
        help="Export LeRobot v2.1 per-episode files only (no renumber / v3 / relative_stats).",
    )
    args = parser.parse_args()

    if (args.shard_id is None) ^ (args.num_shards is None):
        raise SystemExit("Use --shard-id and --num-shards together")
    if args.num_shards is not None and args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")
    if args.stats_only and args.stats_cache_dir is None:
        raise SystemExit("--stats-only requires --stats-cache-dir")

    episode_range = _parse_episode_range(args.episode_range)

    if args.output_mode is not None:
        import yaml

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cfg.setdefault("pipeline", {})["output_mode"] = args.output_mode
        tmp = args.output_dir / "_runtime_config.yaml"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        config_path = tmp
    else:
        config_path = args.config

    return run_dataset_phases(
        package_dir=PACKAGE_DIR,
        config_path=config_path,
        output_dir=args.output_dir,
        force_skip_gate=args.force_skip_gate,
        sample_size=args.sample_size,
        seed=args.seed,
        episode_range=episode_range,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        stats_cache_dir=args.stats_cache_dir,
        stats_only=bool(args.stats_only),
        export_shard_only=bool(args.export_shard_only),
    )


if __name__ == "__main__":
    raise SystemExit(main())

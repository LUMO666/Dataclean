#!/usr/bin/env python3
"""Generate ignore list for episodes missing calibration_bundle_optimized.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor

from robot_data_processing.ignore_list import (
    CALIBRATION_BUNDLE,
    scan_missing_calibration_episodes,
    write_ignore_episode_list,
)


def scan_missing_calibration_parallel(dataset_root: Path, total_episodes: int, workers: int = 64) -> list[int]:
    root = Path(dataset_root) / "parameters"

    def check(ep: int) -> int | None:
        chunk = ep // 1000
        cal = root / f"chunk-{chunk:03d}" / f"episode_{ep:06d}" / CALIBRATION_BUNDLE
        return ep if not cal.is_file() else None

    missing: list[int] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for ep in pool.map(check, range(total_episodes), chunksize=256):
            if ep is not None:
                missing.append(ep)
    missing.sort()
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged"),
    )
    parser.add_argument("--total-episodes", type=int, default=46475)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "humanoid_merged_ignore_no_calibration.json",
    )
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    missing = scan_missing_calibration_parallel(args.dataset_root, args.total_episodes, workers=args.workers)
    write_ignore_episode_list(
        args.output,
        missing,
        reason="missing calibration_bundle_optimized.json",
        dataset_root=str(args.dataset_root),
    )
    print(f"Wrote {len(missing)} ignored episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

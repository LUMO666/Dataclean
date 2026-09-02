#!/usr/bin/env python3
"""Compare meta/statistics_relative.json against a golden reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

from robot_data_processing.relative_statistics import compare_relative_statistics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("actual_root", type=Path, help="Dataset with computed statistics_relative.json")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Golden dataset root (must contain meta/statistics_relative.json)",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.golden is None:
        raise SystemExit(
            "EgoDex has no bundled golden reference; pass --golden <pipeline_output_dir> "
            "from an approved baseline run."
        )

    actual_path = args.actual_root / "meta" / "statistics_relative.json"
    golden_path = args.golden / "meta" / "statistics_relative.json"
    if not actual_path.is_file():
        raise SystemExit(f"Missing {actual_path}")
    if not golden_path.is_file():
        raise SystemExit(f"Missing {golden_path}")

    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    mismatches = compare_relative_statistics(
        actual,
        golden,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not mismatches:
        print("OK: statistics_relative.json matches golden.")
        print(
            f"  anchors={actual['anchor_count']} frames={actual['source_frame_count']} "
            f"episodes={actual['episode_count']}"
        )
        return 0

    print(f"FAIL: {len(mismatches)} mismatch(es):")
    for msg in mismatches[:50]:
        print(f"  - {msg}")
    if len(mismatches) > 50:
        print(f"  ... and {len(mismatches) - 50} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

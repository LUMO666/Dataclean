#!/usr/bin/env python3
"""Run Dataclean P0–P7 for robomind_ur."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATACLEAN_ROOT = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(DATACLEAN_ROOT / "engine"))

from robot_data_processing.phases.orchestrator import run_dataset_phases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-skip-gate", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run_dataset_phases(
        package_dir=PACKAGE_DIR,
        config_path=args.config,
        output_dir=args.output_dir,
        force_skip_gate=args.force_skip_gate,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())

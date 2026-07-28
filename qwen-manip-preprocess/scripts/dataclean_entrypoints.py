#!/usr/bin/env python3
"""Thin wrappers kept for backwards compatibility.

Prefer Dataclean dataset packages:
  python datasets/humanoid_merged/run.py ...
  python datasets/egodex_lerobot_v21/run.py ...
  python datasets/robomind_ur/run.py ...
"""
from __future__ import annotations

import sys
from pathlib import Path

print(
    "Note: prefer Dataclean datasets/*/run.py (P0–P7). "
    "Legacy runners remain under scripts/run_*_full.py.",
    file=sys.stderr,
)
ROOT = Path(__file__).resolve().parents[1]
print(f"  humanoid: {ROOT.parents[0] / 'datasets' / 'humanoid_merged' / 'run.py'}", file=sys.stderr)
print(f"  egodex:    {ROOT.parents[0] / 'datasets' / 'egodex_lerobot_v21' / 'run.py'}", file=sys.stderr)
print(f"  robomind:  {ROOT.parents[0] / 'datasets' / 'robomind_ur' / 'run.py'}", file=sys.stderr)

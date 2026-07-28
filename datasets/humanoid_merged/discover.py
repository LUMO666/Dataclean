"""Enumerate humanoid_merged episodes."""
from __future__ import annotations

from pathlib import Path

from robot_data_processing.phases.ingest import discover_single_root
from robot_data_processing.normalize.standard_types import EpisodeRef


def discover(dataset_root: Path, layout: str = "single_root", total_episodes: int | None = None) -> list[EpisodeRef]:
    return discover_single_root(Path(dataset_root), total_episodes)

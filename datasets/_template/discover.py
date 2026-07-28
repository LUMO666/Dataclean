"""Enumerate episodes for this dataset layout."""
from __future__ import annotations

from pathlib import Path

from robot_data_processing.normalize.standard_types import EpisodeRef
from robot_data_processing.phases.ingest import discover_single_root, discover_part_task


def discover(dataset_root: Path, layout: str = "single_root") -> list[EpisodeRef]:
    root = Path(dataset_root)
    if layout == "part_task":
        return discover_part_task(root)
    return discover_single_root(root)

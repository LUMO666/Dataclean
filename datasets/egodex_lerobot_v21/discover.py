from __future__ import annotations

from pathlib import Path

from robot_data_processing.normalize.standard_types import EpisodeRef
from robot_data_processing.phases.ingest import discover_part_task


def discover(dataset_root: Path, layout: str = "part_task") -> list[EpisodeRef]:
    return discover_part_task(Path(dataset_root))

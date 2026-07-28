"""Dataset adapter: source episode → standard / quality-space arrays.

Replace stubs with real mappings from keymap.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Engine on PYTHONPATH (see run.py)
from robot_data_processing.normalize.standard_types import StandardEpisode, EpisodeRef


def load_keymap(package_dir: Path) -> dict[str, Any]:
    import json

    path = package_dir / "keymap.json"
    return json.loads(path.read_text(encoding="utf-8"))


def to_standard_episode(ref: EpisodeRef, raw: dict[str, np.ndarray], meta: dict[str, Any]) -> StandardEpisode:
    """Map raw source columns into Dataclean standard field dicts."""
    raise NotImplementedError("Implement source → standard mapping for this dataset")


def to_quality_arrays(standard: StandardEpisode) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical (state, action) arrays for Stage1–5 quality filter."""
    raise NotImplementedError("Implement standard → quality-space arrays")

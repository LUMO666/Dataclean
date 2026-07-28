from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_BUNDLE = "calibration_bundle_optimized.json"


def load_ignore_episode_list(path: Path | str | None) -> set[int]:
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ignore list not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {int(x) for x in data}
    if isinstance(data, dict):
        if "episode_indices" in data:
            return {int(x) for x in data["episode_indices"]}
        if "episodes" in data:
            return {int(x) for x in data["episodes"]}
    raise ValueError(f"Unsupported ignore list format: {path}")


def filter_episode_indices(indices: list[int], ignore: set[int]) -> list[int]:
    if not ignore:
        return list(indices)
    return [idx for idx in indices if idx not in ignore]


def scan_missing_calibration_episodes(dataset_root: Path, total_episodes: int) -> list[int]:
    """Episodes without parameters/.../calibration_bundle_optimized.json."""
    root = Path(dataset_root) / "parameters"
    missing: list[int] = []
    for ep in range(total_episodes):
        chunk = ep // 1000
        cal = root / f"chunk-{chunk:03d}" / f"episode_{ep:06d}" / CALIBRATION_BUNDLE
        if not cal.is_file():
            missing.append(ep)
    return missing


def write_ignore_episode_list(
    path: Path,
    episode_indices: list[int],
    *,
    reason: str,
    dataset_root: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": reason,
        "count": len(episode_indices),
        "dataset_root": dataset_root,
        "episode_indices": sorted(int(x) for x in episode_indices),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

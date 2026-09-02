"""Relative-action statistics for LeRobot v3 Episode Camera Geometry datasets.

Hard-requires per-role eef position/rotation_6d, gripper closedness, and
``extrinsic.camera_reference.T_Episode_CameraReference``. Arm
``joint_position`` fields are optional: when both state and action joints exist
for a role they are included; otherwise they are skipped.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_processing.normalize.transforms_standard import matrices_to_rotation_6d

STATISTICS_FILE = "statistics_relative.json"
SCHEMA_VERSION = "episode_camera_geometry_state_action_statistics_v1"
DEFAULT_STATISTICS_HORIZON_SECONDS = 2.0
REFERENCE_TRANSFORM_KEY = "extrinsic.camera_reference.T_Episode_CameraReference"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _column_as_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    shape = [len(values)]
    while pa.types.is_fixed_size_list(values.type):
        shape.append(values.type.list_size)
        values = values.values
    if pa.types.is_floating(values.type) or pa.types.is_integer(values.type):
        output = np.asarray(values.to_numpy(zero_copy_only=False)).reshape(shape)
    else:
        output = np.asarray(column.to_pylist())
    if not np.isfinite(output).all():
        raise ValueError("Statistics input columns must contain only finite values.")
    return output


class _LeadMoments:
    def __init__(self, horizon: int, dimension: int) -> None:
        shape = (horizon, dimension)
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.minimum = np.full(shape, np.inf, dtype=np.float64)
        self.maximum = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, lead: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[1]:
            raise ValueError(f"Unexpected statistics batch shape {values.shape}.")
        if len(values) == 0:
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        old_count = self.count[lead].astype(np.float64)
        total_count = old_count + batch_count
        delta = batch_mean - self.mean[lead]
        self.mean[lead] += delta * batch_count / total_count
        self.m2[lead] += batch_m2 + delta**2 * old_count * batch_count / total_count
        self.count[lead] = total_count.astype(np.int64)
        self.minimum[lead] = np.minimum(self.minimum[lead], values.min(axis=0))
        self.maximum[lead] = np.maximum(self.maximum[lead], values.max(axis=0))

    def to_json(self) -> dict[str, list]:
        if np.any(self.count == 0):
            raise ValueError("Every relative-action lead and dimension must have samples.")
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


class _Moments:
    def __init__(self, dimension: int) -> None:
        self.count = np.zeros(dimension, dtype=np.int64)
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError(f"Unexpected state statistics batch shape {values.shape}.")
        if len(values) == 0:
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        old_count = self.count.astype(np.float64)
        total_count = old_count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total_count
        self.m2 += batch_m2 + delta**2 * old_count * batch_count / total_count
        self.count = total_count.astype(np.int64)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def to_json(self) -> dict[str, list]:
        if np.any(self.count == 0):
            raise ValueError("Every state dimension must have samples.")
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def _rotation_6d_to_matrices(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"{name} must have finite shape [N,6], got {values.shape}.")
    rows = values.reshape(-1, 2, 3)
    first = rows[:, 0]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm <= 1e-8):
        raise ValueError(f"{name} contains a zero first row.")
    first = first / first_norm
    second = rows[:, 1] - np.sum(first * rows[:, 1], axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm <= 1e-8):
        raise ValueError(f"{name} contains collinear Rotation 6D rows.")
    second = second / second_norm
    matrices = np.stack([first, second, np.cross(first, second)], axis=1)
    if not np.allclose(matrices[:, :2], rows, atol=1e-4, rtol=0):
        raise ValueError(f"{name} is not an unnormalized raw Rotation 6D field.")
    return matrices


def _arm_roles(info: dict[str, Any]) -> list[str]:
    """Return eef roles. Hard-requires eef pose + gripper + camera reference; joints optional."""
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json must declare a features object.")
    roles = sorted(
        key.removeprefix("action.eef.").removesuffix(".position")
        for key in features
        if key.startswith("action.eef.") and key.endswith(".position")
    )
    if not roles:
        raise ValueError("Dataset has no action.eef.<role>.position fields.")
    required = {REFERENCE_TRANSFORM_KEY}
    for role in roles:
        required.update(
            {
                f"observation.state.eef.{role}.position",
                f"observation.state.eef.{role}.rotation_6d",
                f"action.eef.{role}.position",
                f"action.eef.{role}.rotation_6d",
                f"action.gripper.{role}.closedness",
            }
        )
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(
            f"Episode Camera Geometry statistics fields are missing: {missing}."
        )
    return roles


def _joint_roles(info: dict[str, Any], roles: list[str]) -> list[str]:
    """Roles that have both state and action arm joint_position features."""
    features = info.get("features") or {}
    out: list[str] = []
    for role in roles:
        state_key = f"observation.state.arm.{role}.joint_position"
        action_key = f"action.arm.{role}.joint_position"
        has_state = state_key in features
        has_action = action_key in features
        if has_state and has_action:
            out.append(role)
        elif has_state or has_action:
            present = state_key if has_state else action_key
            missing = action_key if has_state else state_key
            raise ValueError(
                f"Incomplete arm joint fields for role={role!r}: found {present}, "
                f"missing {missing}. Provide both or neither."
            )
    return out


def _joint_dim(info: dict[str, Any], role: str) -> int:
    key = f"action.arm.{role}.joint_position"
    shape = (info.get("features") or {}).get(key, {}).get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return 6


def _episode_records(dataset_path: Path) -> list[dict[str, int]]:
    paths = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise ValueError("LeRobot v3 episode metadata parquet files are missing.")
    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "length",
    ]
    rows: list[dict[str, int]] = []
    for path in paths:
        for raw in pq.read_table(path, columns=columns).to_pylist():
            rows.append({key: int(value) for key, value in raw.items()})
    rows.sort(key=lambda row: row["episode_index"])
    if not rows:
        raise ValueError("Dataset contains no episodes.")
    indices = [row["episode_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Episode indices must be contiguous from zero.")
    for row in rows:
        if row["dataset_to_index"] - row["dataset_from_index"] != row["length"]:
            raise ValueError(
                f"Episode {row['episode_index']} has inconsistent global bounds and length."
            )
    return rows


def _data_path(dataset_path: Path, pattern: str, shard: tuple[int, int]) -> Path:
    chunk_index, file_index = shard
    path = dataset_path / pattern.format(
        chunk_index=chunk_index,
        file_index=file_index,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset parquet shard is missing: {path}")
    return path


def _shard_anchors(
    records: list[dict[str, int]],
    horizon: int,
) -> tuple[np.ndarray, int]:
    file_start = min(row["dataset_from_index"] for row in records)
    ranges = []
    for row in records:
        count = max(0, row["length"] - horizon)
        local_start = row["dataset_from_index"] - file_start
        if count:
            ranges.append(np.arange(local_start, local_start + count, dtype=np.int64))
    if not ranges:
        return np.empty(0, dtype=np.int64), file_start
    return np.concatenate(ranges), file_start


def prepare_relative_action_statistics(
    dataset_path: Path | str,
    *,
    statistics_horizon_seconds: float = DEFAULT_STATISTICS_HORIZON_SECONDS,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Write per-lead anchor-reference-camera action moments for one v3 dataset."""
    dataset_path = Path(dataset_path)
    output_path = dataset_path / "meta" / STATISTICS_FILE
    if output_path.exists() and not rebuild:
        raise FileExistsError(
            f"Statistics already exist at {output_path}; pass rebuild=True to replace them."
        )
    info = _read_json(dataset_path / "meta" / "info.json")
    fps = float(info["fps"])
    exact_horizon = statistics_horizon_seconds * fps
    horizon = int(round(exact_horizon))
    if (
        fps <= 0
        or statistics_horizon_seconds <= 0
        or horizon <= 0
        or not np.isclose(exact_horizon, horizon, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "statistics_horizon_seconds * dataset FPS must be a positive integer."
        )
    roles = _arm_roles(info)
    joint_roles = _joint_roles(info, roles)
    joint_dims = {role: _joint_dim(info, role) for role in joint_roles}
    records = _episode_records(dataset_path)
    by_shard: dict[tuple[int, int], list[dict[str, int]]] = {}
    for row in records:
        shard = (row["data/chunk_index"], row["data/file_index"])
        by_shard.setdefault(shard, []).append(row)

    state_fields: dict[str, _Moments] = {}
    relative_fields: dict[str, _LeadMoments] = {}
    for role in roles:
        state_fields[f"eef.{role}.position"] = _Moments(3)
        state_fields[f"eef.{role}.rotation_6d"] = _Moments(6)
        relative_fields[f"eef.{role}.position"] = _LeadMoments(horizon, 3)
        relative_fields[f"eef.{role}.rotation_6d"] = _LeadMoments(horizon, 6)
        relative_fields[f"gripper.{role}.closedness"] = _LeadMoments(horizon, 1)
    for role in joint_roles:
        relative_fields[f"arm.{role}.joint_position"] = _LeadMoments(
            horizon, joint_dims[role]
        )

    data_pattern = str(info["data_path"])
    required_columns = [REFERENCE_TRANSFORM_KEY]
    for role in roles:
        required_columns.extend(
            [
                f"observation.state.eef.{role}.position",
                f"observation.state.eef.{role}.rotation_6d",
                f"action.eef.{role}.position",
                f"action.eef.{role}.rotation_6d",
                f"action.gripper.{role}.closedness",
            ]
        )
    for role in joint_roles:
        required_columns.extend(
            [
                f"observation.state.arm.{role}.joint_position",
                f"action.arm.{role}.joint_position",
            ]
        )

    joint_role_set = set(joint_roles)
    anchor_count = 0
    contributing_episodes = 0
    total_frames = 0
    for shard, shard_records in sorted(by_shard.items()):
        path = _data_path(dataset_path, data_pattern, shard)
        table = pq.read_table(path, columns=required_columns)
        cached = {
            column: _column_as_numpy(table.column(column))
            for column in required_columns
        }
        anchors, file_start = _shard_anchors(shard_records, horizon)
        expected_rows = max(row["dataset_to_index"] for row in shard_records) - file_start
        if table.num_rows != expected_rows:
            raise ValueError(
                f"Shard {path} has {table.num_rows} rows, expected {expected_rows}."
            )
        total_frames += sum(row["length"] for row in shard_records)
        contributing_episodes += sum(row["length"] > horizon for row in shard_records)
        if len(anchors) == 0:
            continue
        anchor_count += len(anchors)

        reference = cached[REFERENCE_TRANSFORM_KEY]
        if reference.shape[1:] != (4, 4):
            raise ValueError(
                f"{REFERENCE_TRANSFORM_KEY} must have shape [N,4,4], got {reference.shape}."
            )
        rotation_anchor_episode = reference[anchors, :3, :3].astype(np.float64)
        rotation_anchor_from_episode = np.swapaxes(rotation_anchor_episode, -1, -2)
        camera_origin_episode = reference[anchors, :3, 3].astype(np.float64)

        for role in roles:
            state_position_key = f"observation.state.eef.{role}.position"
            state_rotation_key = f"observation.state.eef.{role}.rotation_6d"
            action_position_key = f"action.eef.{role}.position"
            action_rotation_6d_key = f"action.eef.{role}.rotation_6d"
            gripper_key = f"action.gripper.{role}.closedness"
            state_position = cached[state_position_key].astype(np.float64)
            action_position = cached[action_position_key].astype(np.float64)
            gripper = cached[gripper_key]
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            if gripper.ndim != 2 or gripper.shape[1] != 1:
                raise ValueError(
                    f"{gripper_key} must have shape [N,1], got {gripper.shape}."
                )
            if np.any((gripper < 0.0) | (gripper > 1.0)):
                raise ValueError(f"{gripper_key} must use closedness in [0,1].")

            state_rotation_all = _rotation_6d_to_matrices(
                cached[state_rotation_key], name=state_rotation_key
            )
            action_rotation_6d = _rotation_6d_to_matrices(
                cached[action_rotation_6d_key], name=action_rotation_6d_key
            )
            state_rotation = state_rotation_all[anchors]
            anchor_state_position = state_position[anchors]
            state_position_anchor = np.einsum(
                "nij,nj->ni",
                rotation_anchor_from_episode,
                anchor_state_position - camera_origin_episode,
            )
            state_rotation_anchor = rotation_anchor_from_episode @ state_rotation
            state_fields[f"eef.{role}.position"].update(state_position_anchor)
            state_fields[f"eef.{role}.rotation_6d"].update(
                matrices_to_rotation_6d(
                    _pad_rotation_mats(state_rotation_anchor), name=state_rotation_key
                )
            )

            state_joint = None
            action_joint = None
            if role in joint_role_set:
                state_joint_key = f"observation.state.arm.{role}.joint_position"
                action_joint_key = f"action.arm.{role}.joint_position"
                state_joint = cached[state_joint_key].astype(np.float64)
                action_joint = cached[action_joint_key].astype(np.float64)
                if state_joint.ndim != 2 or state_joint.shape[1] != joint_dims[role]:
                    raise ValueError(
                        f"{state_joint_key} must have shape [N,{joint_dims[role]}], "
                        f"got {state_joint.shape}."
                    )
                if action_joint.ndim != 2 or action_joint.shape[1] != joint_dims[role]:
                    raise ValueError(
                        f"{action_joint_key} must have shape [N,{joint_dims[role]}], "
                        f"got {action_joint.shape}."
                    )
                anchor_state_joint = state_joint[anchors]
            else:
                anchor_state_joint = None

            for lead in range(horizon):
                action_indices = anchors + lead
                delta_position_episode = (
                    action_position[action_indices] - anchor_state_position
                )
                delta_position_anchor = np.einsum(
                    "nij,nj->ni", rotation_anchor_from_episode, delta_position_episode
                )
                delta_rotation_episode_6d = action_rotation_6d[
                    action_indices
                ] @ np.swapaxes(state_rotation, -1, -2)
                delta_rotation_anchor_6d = (
                    rotation_anchor_from_episode
                    @ delta_rotation_episode_6d
                    @ rotation_anchor_episode
                )
                relative_fields[f"eef.{role}.position"].update(
                    lead, delta_position_anchor
                )
                relative_fields[f"eef.{role}.rotation_6d"].update(
                    lead,
                    matrices_to_rotation_6d(
                        _pad_rotation_mats(delta_rotation_anchor_6d),
                        name=action_rotation_6d_key,
                    ),
                )
                relative_fields[f"gripper.{role}.closedness"].update(
                    lead, gripper[action_indices]
                )
                if action_joint is not None and anchor_state_joint is not None:
                    relative_fields[f"arm.{role}.joint_position"].update(
                        lead, action_joint[action_indices] - anchor_state_joint
                    )

    if total_frames != int(info.get("total_frames", total_frames)):
        raise ValueError(
            f"Episode metadata totals {total_frames} frames, info.json declares "
            f"{info.get('total_frames')}."
        )
    if anchor_count == 0:
        raise ValueError("Dataset has no complete statistics windows.")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_fps": fps,
        "statistics_horizon_seconds": float(statistics_horizon_seconds),
        "statistics_horizon_frames": horizon,
        "anchor_frame": "camera_reference_at_anchor",
        "pose_storage_frame": "episode_initial_reference_camera",
        "reference_camera_pose_key": REFERENCE_TRANSFORM_KEY,
        "anchor_population": "all_complete_chunks_with_boundary_frame",
        "required_terminal_frame_offset": horizon,
        "incomplete_tail_policy": "exclude",
        "position_delta_convention": "p_action_in_anchor - p_state_in_anchor",
        "rotation_delta_convention": "R_action_in_anchor @ inverse(R_state_in_anchor)",
        "state_representation": "camera_frame_absolute_xyz_plus_rotation_6d",
        "rotation_6d_convention": "first_two_rotation_matrix_rows_flattened",
        "relative_action_rotation_representation": "rotation_6d",
        "rotation_source_fields": {
            role: {
                "state": f"observation.state.eef.{role}.rotation_6d",
                "action": f"action.eef.{role}.rotation_6d",
            }
            for role in roles
        },
        "gripper_semantics": "absolute_target_closedness",
        "joint_roles": list(joint_roles),
        "episode_count": len(records),
        "contributing_episode_count": contributing_episodes,
        "source_frame_count": total_frames,
        "anchor_count": anchor_count,
        "state": {key: value.to_json() for key, value in state_fields.items()},
        "relative_action": {
            key: value.to_json() for key, value in relative_fields.items()
        },
    }
    if joint_roles:
        payload["joint_delta_convention"] = "q_action - q_state_at_anchor"
        payload["joint_source_fields"] = {
            role: {
                "state": f"observation.state.arm.{role}.joint_position",
                "action": f"action.arm.{role}.joint_position",
            }
            for role in joint_roles
        }
    else:
        payload["joint_delta_convention"] = None
        payload["joint_source_fields"] = {}
    _atomic_json_dump(output_path, payload)
    return payload


def _pad_rotation_mats(rot: np.ndarray) -> np.ndarray:
    n = len(rot)
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = rot
    return out


def compare_relative_statistics(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> list[str]:
    """Return human-readable mismatch messages; empty list means match."""
    mismatches: list[str] = []
    scalar_keys = [
        "schema_version",
        "source_fps",
        "statistics_horizon_seconds",
        "statistics_horizon_frames",
        "anchor_frame",
        "pose_storage_frame",
        "reference_camera_pose_key",
        "anchor_population",
        "required_terminal_frame_offset",
        "incomplete_tail_policy",
        "position_delta_convention",
        "rotation_delta_convention",
        "joint_delta_convention",
        "state_representation",
        "rotation_6d_convention",
        "relative_action_rotation_representation",
        "gripper_semantics",
        "episode_count",
        "contributing_episode_count",
        "source_frame_count",
        "anchor_count",
    ]
    for key in scalar_keys:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"scalar {key}: {actual.get(key)!r} != {expected.get(key)!r}")

    if actual.get("rotation_source_fields") != expected.get("rotation_source_fields"):
        mismatches.append("rotation_source_fields mismatch")
    if actual.get("joint_source_fields") != expected.get("joint_source_fields"):
        mismatches.append("joint_source_fields mismatch")

    for section in ("state", "relative_action"):
        actual_section = actual.get(section) or {}
        expected_section = expected.get(section) or {}
        if set(actual_section) != set(expected_section):
            mismatches.append(
                f"{section} keys differ: {sorted(actual_section)} vs {sorted(expected_section)}"
            )
            continue
        for field, actual_stats in actual_section.items():
            expected_stats = expected_section[field]
            for stat_name in ("count", "mean", "m2", "min", "max"):
                a = np.asarray(actual_stats[stat_name], dtype=np.float64)
                e = np.asarray(expected_stats[stat_name], dtype=np.float64)
                if a.shape != e.shape:
                    mismatches.append(f"{section}.{field}.{stat_name} shape {a.shape} != {e.shape}")
                    continue
                if not np.allclose(a, e, rtol=rtol, atol=atol):
                    err = float(np.max(np.abs(a - e)))
                    mismatches.append(
                        f"{section}.{field}.{stat_name} max_abs_err={err:.6e}"
                    )
    return mismatches

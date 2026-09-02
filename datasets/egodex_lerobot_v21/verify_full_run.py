#!/usr/bin/env python3
"""Post-run verification for egodex_lerobot_v21 v3.0 export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

EXTRINSIC_KEY = "extrinsic.camera_reference.T_Episode_CameraReference"


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/mnt/pfs/Data/liuweilin/Dataclean/datasets/egodex_lerobot_v21/egodex_v30_smoke"
    )
    errors: list[str] = []

    export_report = root / "export_report.json"
    if not export_report.exists():
        errors.append(f"missing {export_report}")
    else:
        rep = json.loads(export_report.read_text())
        print(
            f"export: episodes={rep.get('total_episodes')} "
            f"frames={rep.get('total_frames')} "
            f"skipped={rep.get('skipped_episodes')} "
            f"version={rep.get('lerobot_version') or rep.get('codebase_version')}"
        )

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        errors.append(f"missing {info_path}")
    else:
        info = json.loads(info_path.read_text())
        print(
            f"info: codebase={info.get('codebase_version')} "
            f"episodes={info.get('total_episodes')} "
            f"embodiment={info.get('embodiment') or info.get('robot_type')}"
        )
        if str(info.get("codebase_version")) != "v3.0":
            errors.append(f"expected codebase_version v3.0, got {info.get('codebase_version')!r}")

    rel_path = root / "meta" / "statistics_relative.json"
    if rel_path.exists():
        rel = json.loads(rel_path.read_text())
        print(
            f"relative_stats: anchors={rel.get('anchor_count')} "
            f"episodes={rel.get('episode_count')} "
            f"frames={rel.get('source_frame_count')}"
        )
    else:
        print("relative_stats: absent (ok when relative_statistics.enabled=false)")

    data_dir = root / "data"
    if not data_dir.exists():
        errors.append(f"missing data dir {data_dir}")
    else:
        nulls = 0
        rows = 0
        missing_col = 0
        for path in sorted(data_dir.rglob("*.parquet")):
            names = pq.ParquetFile(path).schema_arrow.names
            if EXTRINSIC_KEY not in names:
                missing_col += 1
                continue
            table = pq.read_table(path, columns=[EXTRINSIC_KEY])
            vals = table.column(0).to_pylist()
            rows += len(vals)
            nulls += sum(v is None for v in vals)
        print(f"extrinsic: rows={rows} null_frames={nulls} shards_missing_col={missing_col}")
        if nulls:
            errors.append(f"found {nulls} null extrinsic frames")
        if missing_col:
            errors.append(f"{missing_col} parquet shards missing extrinsic column")

    log = root / "pipeline.log"
    if log.exists():
        text = log.read_text(errors="replace")
        if "Traceback" in text or "=== Phases done" not in text:
            if "=== Phases done" not in text:
                errors.append("pipeline.log does not show completion (Phases done)")
            if "Traceback" in text:
                errors.append("pipeline.log contains Traceback")

    if errors:
        print("FAIL:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK: egodex run verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

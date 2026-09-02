#!/usr/bin/env python3
"""Post-run verification for humanoid_merged full v3.0 export."""
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
        else "/mnt/pfs/Data/liuweilin/Dataclean/datasets/humanoid_merged/humanoid_v30_full"
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
            f"skipped={rep.get('skipped_episodes')}"
        )

    rel_path = root / "meta" / "statistics_relative.json"
    if not rel_path.exists():
        errors.append(f"missing {rel_path}")
    else:
        rel = json.loads(rel_path.read_text())
        print(
            f"relative_stats: anchors={rel.get('anchor_count')} "
            f"episodes={rel.get('episode_count')} "
            f"frames={rel.get('source_frame_count')}"
        )

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

    print("OK: full run verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

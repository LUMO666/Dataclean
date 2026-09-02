"""P0–P7 phase orchestrator for Dataclean dataset packages."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from robot_data_processing.frame_alignment import check_import_frame_alignment, summarize_alignment_failure
from robot_data_processing.gates.manual_gate import (
    assert_eef_direction_or_raise,
    assert_gate_or_raise,
)
from robot_data_processing.loader import episode_parquet_path, read_episode_table
from robot_data_processing.meta_tasks import load_episode_task_meta, resolve_episode_tasks
from robot_data_processing.normalize.review_corrections import apply_review_corrections
from robot_data_processing.normalize.standard_types import EpisodeRef, StandardEpisode
from robot_data_processing.phases.ingest import discover_part_task, discover_single_root
from robot_data_processing.phases.params import (
    DEFAULT_HUMANOID_GOLDEN_EE_SPEED,
    check_timestamp_fps,
    compute_speed_scale,
    params_meta_from_speed,
)
from robot_data_processing.pipeline import load_config, pipeline_config_from_yaml, run_pipeline
from robot_data_processing.report import (
    build_quality_report,
    write_exclusion_log,
    write_optimal_lags,
    write_quality_report,
)
from robot_data_processing.schema import stack_column
from robot_data_processing.stages.state_action_temporal_alignment import (
    StateActionAlignStats,
    apply_p4_to_action_fields,
    parse_temporal_align_config,
    resolve_alignment_lag,
    resolve_p4_plan,
    resolve_state_action_delay,
)
from robot_data_processing.stages.stage2_trend_alignment import (
    Stage2Config,
    fill_missing_action_eef,
    parse_stage2_fill_config,
)
from robot_data_processing.types import EpisodeResult


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _table_to_raw(table) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    for name in table.column_names:
        col = table.column(name)
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        try:
            values = col.to_numpy(zero_copy_only=False)
            if isinstance(values, np.ndarray) and values.dtype == object:
                raw[name] = stack_column(values)
            else:
                arr = np.asarray(values)
                if arr.ndim == 1 and name in (
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ):
                    raw[name] = arr
                else:
                    raw[name] = arr if arr.ndim >= 1 else stack_column(values)
        except Exception:
            continue
    return raw


def _group_refs_by_root(refs: list[EpisodeRef]) -> dict[Path, list[EpisodeRef]]:
    groups: dict[Path, list[EpisodeRef]] = {}
    for ref in refs:
        groups.setdefault(ref.dataset_root, []).append(ref)
    return groups


def _resolve_quality_config(package_dir: Path, cfg: dict[str, Any], embodiment: str) -> Path:
    quality_path = cfg.get("quality_config_path")
    if quality_path:
        qpath = Path(quality_path)
        if qpath.is_absolute() and qpath.exists():
            return qpath
        dataclean_root = package_dir.parents[1]
        for cand in (dataclean_root / qpath, package_dir / quality_path, qpath):
            if cand.exists():
                return cand
    local = package_dir / "quality.yaml"
    if local.exists():
        return local
    cfg_dir = package_dir.parents[1] / "qwen-manip-preprocess" / "config"
    mapping = {
        "humanoid": cfg_dir / "humanoid_merged.yaml",
        "egodex": cfg_dir / "egodex_22T.yaml",
        "robomind_ur": cfg_dir / "robomind_ur_lerobot.yaml",
    }
    path = mapping.get(embodiment)
    if path and path.exists():
        return path
    raise FileNotFoundError(f"No quality config for embodiment={embodiment}")


def _load_episode_geometry(package_dir: Path, cfg: dict[str, Any]):
    geom_cfg = cfg.get("episode_geometry") or {}
    if not geom_cfg.get("enabled", False):
        return None, geom_cfg
    mod_name = str(geom_cfg.get("module", "episode_geometry"))
    mod_path = package_dir / f"{mod_name}.py"
    if not mod_path.exists():
        print(f"[P6] episode_geometry.enabled but {mod_path} missing — skip", flush=True)
        return None, geom_cfg
    mod = _load_module(mod_path, f"dataclean_geom_{package_dir.name}")
    for cls_name in (
        "HumanoidEpisodeGeometry",
        "EgoDexEpisodeGeometry",
        "EpisodeGeometry",
        "TemplateEpisodeGeometry",
    ):
        if hasattr(mod, cls_name):
            return getattr(mod, cls_name)(package_dir), geom_cfg
    if hasattr(mod, "build_episode_geometry"):
        return mod.build_episode_geometry(package_dir), geom_cfg
    raise RuntimeError(f"No episode geometry class in {mod_path}")


def _mark_export_discard(
    all_pairs: list[tuple[StandardEpisode, EpisodeResult | None]],
    index: int,
    standard: StandardEpisode,
    result: EpisodeResult | None,
    reasons: list[str],
) -> None:
    """Ensure P7 skips this episode even when quality never produced a result."""
    if result is None:
        result = EpisodeResult(
            episode_index=int(standard.episode_index),
            num_frames=int(standard.num_frames),
            discard=True,
            discard_reasons=list(reasons),
        )
        all_pairs[index] = (standard, result)
        return
    result.discard = True
    result.discard_reasons.extend(reasons)


def _apply_p6_geometry(
    all_pairs: list[tuple[StandardEpisode, EpisodeResult | None]],
    *,
    episode_geometry,
    geometry_cfg: dict[str, Any],
    keymap: dict[str, Any],
    refs_by_ep: dict[int, EpisodeRef],
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Run P6 on all episodes; return (video_export_map, info_extras, wrist_view)."""
    video_export_map: dict[str, str] = {}
    info_extras: dict[str, Any] = {}
    wrist_view: list[str] = []
    geom_discards = 0

    for idx, (standard, result) in enumerate(all_pairs):
        ref = refs_by_ep.get(standard.episode_index)
        if ref is None:
            ref = EpisodeRef(
                episode_index=standard.episode_index,
                dataset_root=Path(standard.meta.get("dataset_root", ".")),
            )
        try:
            geom = episode_geometry.apply_episode_frame_geometry(
                standard,
                ref=ref,
                geometry_cfg=geometry_cfg,
                keymap=keymap,
            )
        except Exception as exc:
            geom_discards += 1
            print(f"  [P6] discard ep{standard.episode_index}: {exc}", flush=True)
            _mark_export_discard(
                all_pairs,
                idx,
                standard,
                result,
                [f"p6_geometry_error:{exc}"],
            )
            continue

        if geom.discard:
            geom_discards += 1
            print(
                f"  [P6] discard ep{standard.episode_index}: {geom.discard_reasons}",
                flush=True,
            )
            _mark_export_discard(
                all_pairs,
                idx,
                standard,
                result,
                list(geom.discard_reasons),
            )
            continue

        standard.fields = geom.fields
        standard.extrinsic = geom.extrinsic
        standard.camera_keys = geom.camera_keys
        standard.intrinsic = {}
        standard.meta = {
            **standard.meta,
            "camera_intrinsics": geom.camera_intrinsics,
            "episode_frame_definition": geom.episode_frame_definition,
            "geometry_meta": geom.geometry_meta,
        }
        if not video_export_map:
            video_export_map = dict(geom.video_export_map)
        if not info_extras and geom.info_extras:
            info_extras = dict(geom.info_extras)
        if not wrist_view and geom.wrist_view_cameras:
            wrist_view = list(geom.wrist_view_cameras)

    print(
        f"[P6] episode geometry applied; discards={geom_discards} "
        f"video_map={video_export_map}",
        flush=True,
    )
    return video_export_map, info_extras, wrist_view


def _apply_episode_range(
    refs: list[EpisodeRef], episode_range: tuple[int, int] | None
) -> list[EpisodeRef]:
    if episode_range is None:
        return refs
    start, end = episode_range
    out = [r for r in refs if start <= int(r.episode_index) < end]
    print(
        f"[P0] episode-range [{start}, {end}): {len(refs)} -> {len(out)}",
        flush=True,
    )
    return out


def _apply_shard(
    refs: list[EpisodeRef],
    *,
    shard_id: int | None,
    num_shards: int | None,
) -> list[EpisodeRef]:
    if shard_id is None or num_shards is None:
        return refs
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    out = [r for i, r in enumerate(refs) if i % num_shards == shard_id]
    print(
        f"[P0] shard {shard_id}/{num_shards}: {len(refs)} -> {len(out)} "
        f"(by list position after sample/range)",
        flush=True,
    )
    return out


def _stats_cache_paths(stats_cache_dir: Path) -> dict[str, Path]:
    root = Path(stats_cache_dir)
    return {
        "stats_cache_path": root / "global_stats.npz",
        "stage1_stats_cache_path": root / "stage1_global_stats.npz",
        "state_action_lag_cache_path": root / "state_action_lag.npz",
    }


def run_dataset_phases(
    *,
    package_dir: Path,
    config_path: Path,
    output_dir: Path,
    force_skip_gate: bool = False,
    sample_size: int | None = None,
    seed: int = 42,
    episode_range: tuple[int, int] | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    stats_cache_dir: Path | None = None,
    stats_only: bool = False,
    export_shard_only: bool = False,
    adapter_factory: Callable[[], Any] | None = None,
) -> int:
    package_dir = Path(package_dir)
    cfg = _load_yaml(config_path)
    ds = cfg.get("dataset") or {}
    pipe = cfg.get("pipeline") or {}
    media_cfg = cfg.get("media") or {}
    speed_cfg = cfg.get("speed_align") or {}

    dataset_root = Path(ds["root"])
    layout = ds.get("layout", "single_root")
    embodiment = ds.get("embodiment", "unknown")
    fps = float(ds.get("fps", 30))
    output_mode = pipe.get("output_mode", "both")
    force_skip = bool(force_skip_gate or pipe.get("force_skip_gate", False))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checklist = package_dir / "review_checklist.yaml"
    # Front-loaded: eef direction human review before any pipeline work / eef transforms
    eef_gate = assert_eef_direction_or_raise(checklist, force_skip=force_skip)
    print(f"[P2/eef] {eef_gate.message}", flush=True)

    require_export = output_mode in ("filter", "both")
    gate = assert_gate_or_raise(
        checklist, force_skip=force_skip, require_for_export=require_export
    )
    print(f"[P2] {gate.message}", flush=True)

    # P0 Ingest
    if layout == "part_task":
        refs = discover_part_task(dataset_root)
    else:
        total = ds.get("total_episodes")
        refs = discover_single_root(dataset_root, int(total) if total is not None else None)

    sample_mode = str(pipe.get("sample_mode", "random")).lower()
    if sample_size is not None and sample_size < len(refs):
        if sample_mode in ("sequential", "seq", "head", "first"):
            refs = refs[: int(sample_size)]
            print(
                f"[P0] sequential sample: first {len(refs)} episodes "
                f"(indices {refs[0].episode_index}..{refs[-1].episode_index})",
                flush=True,
            )
        else:
            rng = np.random.default_rng(seed)
            pick = sorted(rng.choice(len(refs), size=sample_size, replace=False).tolist())
            refs = [refs[i] for i in pick]
            print(f"[P0] random sample: {len(refs)} episodes (seed={seed})", flush=True)

    refs = _apply_episode_range(refs, episode_range)
    # Full sample (after range, before shard) — used for shared global stats.
    stats_refs = list(refs)
    refs = _apply_shard(refs, shard_id=shard_id, num_shards=num_shards)
    print(f"[P0] discovered {len(refs)} episodes (layout={layout})", flush=True)
    if not refs and not stats_only:
        raise RuntimeError("No episodes left after sample/range/shard selection")
    if stats_cache_dir is not None:
        Path(stats_cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[P0] stats-cache-dir={Path(stats_cache_dir).resolve()}", flush=True)
    if stats_only:
        print(
            f"[P0] stats-only: will compute stats on {len(stats_refs)} episodes "
            f"(process/export skipped)",
            flush=True,
        )
    if export_shard_only:
        print("[P0] export-shard-only: v2.1 per-episode export, no renumber/v3/relative", flush=True)

    if adapter_factory is not None:
        adapter = adapter_factory()
    else:
        adapter_mod = _load_module(package_dir / "adapter.py", f"dataclean_adapter_{package_dir.name}")
        for cls_name in ("HumanoidAdapter", "EgoDexAdapter", "RobomindAdapter"):
            if hasattr(adapter_mod, cls_name):
                adapter = getattr(adapter_mod, cls_name)(package_dir)
                break
        else:
            if hasattr(adapter_mod, "build_adapter"):
                adapter = adapter_mod.build_adapter(package_dir)
            else:
                raise RuntimeError(f"No adapter class in {package_dir / 'adapter.py'}")

    camera_map = adapter.camera_name_map()
    require_top = bool(media_cfg.get("require_camera_top", True))
    keymap_path = package_dir / "keymap.json"
    if keymap_path.exists():
        keymap = json.loads(keymap_path.read_text(encoding="utf-8"))
    else:
        keymap = getattr(adapter, "keymap", {}) or {}
    episode_geometry, geometry_cfg = _load_episode_geometry(package_dir, cfg)
    geometry_enabled = bool(geometry_cfg.get("enabled", False)) and episode_geometry is not None
    qpath = _resolve_quality_config(package_dir, cfg, embodiment)
    yaml_cfg = load_config(qpath)
    if layout != "part_task":
        yaml_cfg.setdefault("dataset", {})["root"] = str(dataset_root)

    temporal_align_cfg = cfg.get("temporal_align") or {}
    align_cfg = parse_temporal_align_config(
        temporal_align_cfg, yaml_cfg.get("state_action_alignment") or {}
    )
    p4_plan = resolve_p4_plan(align_cfg)
    fill_missing_action_eef_flag = parse_stage2_fill_config(
        cfg, (yaml_cfg.get("stage2") or {})
    )
    print(
        f"[P4] plan={p4_plan.name} apply_delay={align_cfg.apply_delay} "
        f"mode={align_cfg.mode} manual_delay={align_cfg.manual_delay}",
        flush=True,
    )
    print(
        f"[P3/stage2] fill_missing_action_eef={fill_missing_action_eef_flag}",
        flush=True,
    )

    run_meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": str(package_dir),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "layout": layout,
        "output_mode": output_mode,
        "num_episodes": len(refs),
        "num_stats_episodes": len(stats_refs),
        "quality_config": str(qpath),
        "sharding": {
            "episode_range": list(episode_range) if episode_range is not None else None,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "stats_cache_dir": str(stats_cache_dir) if stats_cache_dir is not None else None,
            "stats_only": bool(stats_only),
            "export_shard_only": bool(export_shard_only),
        },
        "pipeline": {
            "num_workers": int(pipe.get("num_workers", 64)),
            "export_workers": pipe.get("export_workers"),
            "parallel_videos": pipe.get("parallel_videos", True),
        },
        "temporal_align": {
            "plan": p4_plan.name,
            "apply_delay": align_cfg.apply_delay,
            "mode": align_cfg.mode,
            "manual_delay": align_cfg.manual_delay,
        },
        "stage2": {"fill_missing_action_eef": fill_missing_action_eef_flag},
        "gate": {"ok": gate.ok, "force_skip": force_skip, "message": gate.message},
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    timings: dict[str, float] = {}

    def _tick(name: str, t_start: float) -> float:
        dt = time.perf_counter() - t_start
        timings[name] = round(dt, 3)
        print(f"[timing] {name}: {dt:.2f}s ({dt / 60:.2f} min)", flush=True)
        return time.perf_counter()

    t0 = time.perf_counter()
    all_results: list[EpisodeResult] = []
    fps_failures = 0
    speed_meta_samples: list[dict[str, Any]] = []
    total_standard = 0
    all_pairs: list[tuple[StandardEpisode, EpisodeResult | None]] = []
    delay_samples: list[int] = []
    export_workers = int(pipe["export_workers"]) if pipe.get("export_workers") is not None else None
    parallel_videos = bool(pipe.get("parallel_videos", True))
    # Flat LeRobot root: renumber for part_task, or when explicitly requested.
    # Relative-statistics needs contiguous 0..N-1; enable via export.renumber_episodes
    # or relative_statistics.enabled=true.
    export_cfg_early = cfg.get("export") or {}
    rel_cfg_early = cfg.get("relative_statistics") or {}
    renumber_episodes = layout == "part_task" or bool(
        export_cfg_early.get("renumber_episodes", False)
    )
    if bool(rel_cfg_early.get("enabled", False)):
        renumber_episodes = True
    if export_shard_only:
        # Keep source episode_index as on-disk id so shards never collide.
        renumber_episodes = False
    merged_task_name_to_index: dict[str, int] = {}

    # Stats-only: compute shared caches on the pre-shard sample, then exit.
    if stats_only:
        if stats_cache_dir is None:
            raise RuntimeError("stats_only requires stats_cache_dir")
        cache_paths = _stats_cache_paths(Path(stats_cache_dir))
        groups_stats = _group_refs_by_root(stats_refs)
        for gi, (root, group_refs) in enumerate(sorted(groups_stats.items(), key=lambda x: str(x[0])), start=1):
            eps = sorted({r.episode_index for r in group_refs})
            quality_out = output_dir / "_quality"
            if layout == "part_task":
                part = group_refs[0].part or "part"
                task = group_refs[0].task or "task"
                quality_out = output_dir / "_quality" / part / task
            quality_out.mkdir(parents=True, exist_ok=True)
            print(
                f"[P3/stats-only] [{gi}/{len(groups_stats)}] root={root} n={len(eps)} "
                f"→ {stats_cache_dir}",
                flush=True,
            )
            overrides = {
                "dataset_root": str(root),
                "output_dir": str(quality_out),
                "output_mode": "report",
                "num_workers": int(pipe.get("num_workers", 64)),
                "temporal_align": temporal_align_cfg,
                "stage2_fill_missing_action_eef": fill_missing_action_eef_flag,
                "stats_recompute": True,
                "stage1_stats_recompute": True,
                "state_action_lag_recompute": True,
                **{k: str(v) for k, v in cache_paths.items()},
            }
            if pipe.get("ignore_episodes_paths") is not None:
                overrides["ignore_episodes_paths"] = pipe.get("ignore_episodes_paths")
            elif pipe.get("ignore_episodes_path") is not None:
                overrides["ignore_episodes_path"] = pipe.get("ignore_episodes_path")
            pcfg = pipeline_config_from_yaml(yaml_cfg, overrides)
            p3_timings: dict[str, float] = {}
            t_p3 = time.perf_counter()
            # Empty process list: only stats / lag caches are written.
            run_pipeline(
                pcfg,
                [],
                stats_episode_indices=eps,
                show_progress=True,
                timings=p3_timings,
            )
            _tick(f"p3_stats_only_group_{gi}", t_p3)
            for k, v in p3_timings.items():
                timings[f"group{gi}_{k}"] = v
        elapsed = time.perf_counter() - t0
        timings["total"] = round(elapsed, 3)
        run_meta.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": round(elapsed, 2),
                "stats_only": True,
                "stats_cache_files": {k: str(v) for k, v in cache_paths.items()},
                "phase_timing": {"sec": timings, "total_min": round(elapsed / 60.0, 3)},
            }
        )
        (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        (output_dir / "phase_timing.json").write_text(
            json.dumps(run_meta["phase_timing"], indent=2), encoding="utf-8"
        )
        print(f"=== Stats-only done in {elapsed/60:.1f} min → {stats_cache_dir} ===", flush=True)
        return 0

    groups = _group_refs_by_root(refs)
    for gi, (root, group_refs) in enumerate(sorted(groups.items(), key=lambda x: str(x[0])), start=1):
        eps = sorted({r.episode_index for r in group_refs})
        episode_tasks_map, task_name_to_index, task_index_to_name = load_episode_task_meta(root)
        for name, ti in task_name_to_index.items():
            merged_task_name_to_index.setdefault(name, ti)
        # Quality caches stay under _quality/; final export is always flat at output_dir.
        if layout == "part_task":
            part = group_refs[0].part or "part"
            task = group_refs[0].task or "task"
            quality_out = output_dir / "_quality" / part / task
        else:
            quality_out = output_dir / "_quality"
        quality_out.mkdir(parents=True, exist_ok=True)

        print(f"[P3] [{gi}/{len(groups)}] quality filter root={root} n={len(eps)}", flush=True)
        overrides = {
            "dataset_root": str(root),
            "output_dir": str(quality_out),
            "output_mode": "report",
            "num_workers": int(pipe.get("num_workers", 64)),
            "temporal_align": temporal_align_cfg,
            "stage2_fill_missing_action_eef": fill_missing_action_eef_flag,
        }
        if export_workers is not None:
            overrides["export_workers"] = export_workers
        # Prefer Dataclean package config ignore lists when present.
        if pipe.get("ignore_episodes_paths") is not None:
            overrides["ignore_episodes_paths"] = pipe.get("ignore_episodes_paths")
        elif pipe.get("ignore_episodes_path") is not None:
            overrides["ignore_episodes_path"] = pipe.get("ignore_episodes_path")
        if stats_cache_dir is not None:
            cache_paths = _stats_cache_paths(Path(stats_cache_dir))
            overrides.update({k: str(v) for k, v in cache_paths.items()})
            have_all = all(p.exists() for p in cache_paths.values())
            if not have_all:
                missing = [str(p) for p in cache_paths.values() if not p.exists()]
                raise FileNotFoundError(
                    "Shared stats cache incomplete. Run once with --stats-only "
                    f"--stats-cache-dir {stats_cache_dir} first. Missing: {missing}"
                )
            overrides["stats_recompute"] = False
            overrides["stage1_stats_recompute"] = False
            overrides["state_action_lag_recompute"] = False
            print(f"[P3] using shared stats cache: {stats_cache_dir}", flush=True)
        pcfg = pipeline_config_from_yaml(yaml_cfg, overrides)
        p3_timings: dict[str, float] = {}
        t_p3 = time.perf_counter()
        results = run_pipeline(
            pcfg, eps, stats_episode_indices=eps, show_progress=True, timings=p3_timings
        )
        _tick(f"p3_quality_group_{gi}", t_p3)
        for k, v in p3_timings.items():
            timings[f"group{gi}_{k}"] = v
        result_by_ep = {r.episode_index: r for r in results}
        all_results.extend(results)

        align_stats: StateActionAlignStats | None = None
        lag_cache = quality_out / "cache" / "state_action_lag.npz"
        if stats_cache_dir is not None:
            shared_lag = Path(stats_cache_dir) / "state_action_lag.npz"
            if shared_lag.exists():
                lag_cache = shared_lag
        if p4_plan.compute_stats and lag_cache.exists():
            try:
                align_stats = StateActionAlignStats.load(str(lag_cache))
            except Exception:
                align_stats = None
        align_lag = resolve_alignment_lag(align_cfg, align_stats)
        state_action_delay = resolve_state_action_delay(align_cfg, align_stats, p4_plan)
        delay_samples.append(int(state_action_delay))
        print(
            f"[P4] lag_mean={align_stats.lag_mean if align_stats else None} "
            f"align_lag={align_lag} state_action_delay={state_action_delay}",
            flush=True,
        )

        write_quality_report(
            quality_out / "reports" / "quality_report.json",
            build_quality_report(
                results,
                stats_meta={"root": str(root)},
                config_summary={"phase": "P3", "output_mode": "report"},
            ),
        )
        write_exclusion_log(quality_out / "reports" / "exclusion_log.jsonl", results)
        write_optimal_lags(quality_out / "reports" / "optimal_lags.json", results)

        group_pairs: list[tuple[StandardEpisode, EpisodeResult | None]] = []
        t_std = time.perf_counter()
        for ref in group_refs:
            r = result_by_ep.get(ref.episode_index)
            path = episode_parquet_path(ref.dataset_root, ref.episode_index)
            if not path.exists():
                continue
            table = read_episode_table(path)
            raw = _table_to_raw(table)
            parquet_task_index = None
            if "task_index" in raw:
                try:
                    parquet_task_index = int(np.asarray(raw["task_index"]).reshape(-1)[0])
                except Exception:
                    parquet_task_index = None
            tasks_list = resolve_episode_tasks(
                episode_index=ref.episode_index,
                episode_tasks=episode_tasks_map,
                task_index_to_name=task_index_to_name,
                fallback_task=ref.task,
                parquet_task_index=parquet_task_index,
            )
            meta = {
                "fps": fps,
                "embodiment": embodiment,
                "dataset_root": str(ref.dataset_root),
                "part": ref.part,
                "task": ref.task or (tasks_list[0] if tasks_list else None),
                "tasks": tasks_list,
            }
            try:
                standard = adapter.to_standard_episode(ref, raw, meta)
            except Exception as exc:
                print(f"  [P1] skip ep{ref.episode_index}: {exc}", flush=True)
                continue

            standard = apply_review_corrections(standard, cfg)
            import_align = check_import_frame_alignment(
                raw=raw,
                standard_num_frames=int(standard.num_frames),
                fields=standard.fields,
                extrinsic=standard.extrinsic,
                dataset_root=ref.dataset_root,
                episode_index=ref.episode_index,
                camera_keys=list(standard.camera_keys),
                require_videos=bool(require_top or standard.camera_keys),
            )
            if not import_align.ok:
                reason = summarize_alignment_failure(import_align, stage="import")
                print(f"  [P1] discard ep{ref.episode_index}: {reason}", flush=True)
                if r is not None:
                    r.discard = True
                    r.discard_reasons.append(reason)
                continue
            # Stage2: fill missing action.eef from state.eef using joint lag (same policy as P3)
            s2_fill_cfg = Stage2Config(fill_missing_action_eef=fill_missing_action_eef_flag)
            # Copy DA-related thresholds from quality yaml when present
            q_s2 = yaml_cfg.get("stage2") or {}
            if q_s2:
                align = q_s2.get("alignment") or {}
                smooth = q_s2.get("smoothing") or {}
                s2_fill_cfg.max_lag_frames = int(align.get("max_lag_frames", s2_fill_cfg.max_lag_frames))
                s2_fill_cfg.diff_epsilon = float(align.get("diff_epsilon", s2_fill_cfg.diff_epsilon))
                s2_fill_cfg.min_active_samples = int(
                    align.get("min_active_samples", s2_fill_cfg.min_active_samples)
                )
                s2_fill_cfg.median_kernel = int(smooth.get("median_kernel", s2_fill_cfg.median_kernel))
                s2_fill_cfg.savgol_window = int(smooth.get("savgol_window", s2_fill_cfg.savgol_window))
                s2_fill_cfg.savgol_polyorder = int(
                    smooth.get("savgol_polyorder", s2_fill_cfg.savgol_polyorder)
                )
                s2_fill_cfg.action_type = str(q_s2.get("action_type", s2_fill_cfg.action_type))
            s2_fill_cfg.fill_lag = int(align_lag)
            _, filled_eef, fill_lags = fill_missing_action_eef(
                standard.fields, s2_fill_cfg, fill_lag=align_lag
            )
            standard.meta = {
                **standard.meta,
                "stage2_filled_action_eef": filled_eef,
                "stage2_fill_lags": fill_lags,
            }
            standard.fields = apply_p4_to_action_fields(
                standard.fields, p4_plan, lag=align_lag, target_delay=1
            )
            standard.meta = {
                **standard.meta,
                "state_action_delay": state_action_delay,
                "p4_plan": p4_plan.name,
                "source_episode_index": standard.meta.get("source_episode_index", ref.episode_index),
            }
            if require_top and not standard.require_camera_top():
                print(f"  [P1] discard ep{ref.episode_index}: missing camera_top", flush=True)
                if r is not None:
                    r.discard = True
                    r.discard_reasons.append("missing_camera_top")
                continue

            ts = raw.get("timestamp")
            fps_check = check_timestamp_fps(ts, expected_fps=fps)
            if not fps_check.ok:
                fps_failures += 1
                if r is not None:
                    r.discard = True
                    r.discard_reasons.append(f"fps_unstable:{fps_check.message}")
                continue

            if speed_cfg.get("enabled") and "action.eef.left.position" in standard.fields:
                xyz = standard.fields["action.eef.left.position"]
                sp = compute_speed_scale(
                    xyz,
                    fps,
                    golden_speed_mps=float(
                        speed_cfg.get("golden_speed_mps", DEFAULT_HUMANOID_GOLDEN_EE_SPEED)
                    ),
                    target_fps=float(speed_cfg.get("target_fps", 30)),
                )
                standard.meta.update(params_meta_from_speed(sp))
                if len(speed_meta_samples) < 20:
                    speed_meta_samples.append(params_meta_from_speed(sp))

            group_pairs.append((standard, r))

        _tick(f"p1_standardize_group_{gi}", t_std)
        total_standard += len(group_pairs)
        all_pairs.extend(group_pairs)

    # P6 EpisodeFrameGeometry (before P7 export)
    geom_info_extras: dict[str, Any] = {}
    geom_wrist_view: list[str] = []
    if geometry_enabled and all_pairs:
        refs_by_ep = {r.episode_index: r for r in refs}
        video_export_map, geom_info_extras, geom_wrist_view = _apply_p6_geometry(
            all_pairs,
            episode_geometry=episode_geometry,
            geometry_cfg=geometry_cfg,
            keymap=keymap,
            refs_by_ep=refs_by_ep,
        )
        if video_export_map:
            camera_map = dict(video_export_map)

    # P7: one flat LeRobot dataset at output_dir (no per-task subfolders)
    export_cfg = cfg.get("export") or {}
    lerobot_version = str(export_cfg.get("lerobot_version", "v2.1")).lower()
    if export_shard_only:
        lerobot_version = "v2.1"
    v30_cfg = export_cfg.get("v30") or {}
    if output_mode in ("filter", "both") and all_pairs:
        from robot_data_processing.export_standard import export_standard_dataset

        dataclean_root = package_dir.parents[1]
        std_info = dataclean_root / "standard_info.json"
        # Prefer majority delay across groups; fall back to last sample / 1
        if delay_samples:
            state_action_delay = max(set(delay_samples), key=delay_samples.count)
        else:
            state_action_delay = 1

        v30_output_root = None
        if lerobot_version == "both":
            v30_output_root = output_dir.parent / f"{output_dir.name}_v30"

        print(
            f"[P7] export → {output_dir} ({len(all_pairs)} candidates); "
            f"state_action_delay={state_action_delay} "
            f"export_workers={export_workers} renumber={renumber_episodes} "
            f"v2_geometry={geometry_enabled} lerobot_version={lerobot_version}"
            f"{' shard_only' if export_shard_only else ''}",
            flush=True,
        )
        t_p7 = time.perf_counter()
        short_side = geometry_cfg.get("video_short_side") if geometry_enabled else None
        if short_side is None and geometry_enabled:
            short_side = int(media_cfg.get("target_short_side", 384))

        export_summary = export_standard_dataset(
            output_root=output_dir,
            episodes=all_pairs,
            camera_map=camera_map,
            source_root=dataset_root,
            fps=fps,
            embodiment=embodiment,
            state_action_delay=int(state_action_delay),
            target_height=int(
                media_cfg.get(
                    "target_height",
                    media_cfg.get("target_width", 384),  # legacy alias
                )
            ),
            target_short_side=int(short_side) if short_side is not None else None,
            max_keyframe_interval=int(media_cfg.get("max_keyframe_interval", 10)),
            standard_info_path=std_info if std_info.exists() else None,
            normalize_videos=True,
            export_workers=export_workers,
            parallel_videos=parallel_videos,
            renumber_episodes=renumber_episodes,
            task_name_to_index=merged_task_name_to_index or None,
            show_progress=True,
            skip_parameters=geometry_enabled,
            info_extras=geom_info_extras or None,
            wrist_view_cameras=geom_wrist_view or None,
            lerobot_version=lerobot_version,
            v30_output_root=v30_output_root,
            v30_data_file_size_in_mb=int(v30_cfg.get("data_file_size_in_mb", 100)),
            v30_video_file_size_in_mb=int(v30_cfg.get("video_file_size_in_mb", 200)),
            keep_all_frames=bool(export_cfg.get("keep_all_frames", False)),
        )
        _tick("p7_standard_export", t_p7)
        if export_shard_only:
            manifest = {
                "export_shard_only": True,
                "shard_id": shard_id,
                "num_shards": num_shards,
                "output_root": str(output_dir),
                "total_episodes": export_summary.get("total_episodes"),
                "total_frames": export_summary.get("total_frames"),
                "lerobot_version": "v2.1",
                "renumber_episodes": False,
            }
            map_path = output_dir / "meta" / "episode_source_map.jsonl"
            if map_path.exists():
                src_ids = []
                for line in map_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    src_ids.append(int(row.get("source_episode_index", row["episode_index"])))
                manifest["source_episode_indices"] = sorted(src_ids)
            (output_dir / "shard_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(
                f"[P7/shard] wrote shard_manifest.json episodes={manifest.get('total_episodes')}",
                flush=True,
            )
        if export_summary.get("lerobot_v30"):
            v30_summary = export_summary["lerobot_v30"]
            print(
                f"[P7/v30] done: {v30_summary.get('codebase_version')} "
                f"episodes={v30_summary.get('total_episodes')} → {v30_summary.get('output_root')}",
                flush=True,
            )
            rel_cfg = cfg.get("relative_statistics") or {}
            if (
                not export_shard_only
                and geometry_enabled
                and bool(rel_cfg.get("enabled", False))
            ):
                from robot_data_processing.relative_statistics import (
                    prepare_relative_action_statistics,
                )

                v30_root = Path(
                    v30_summary.get("output_root")
                    or (v30_output_root if lerobot_version == "both" else output_dir)
                )
                horizon_s = float(
                    rel_cfg.get(
                        "statistics_horizon_seconds",
                        2.0,
                    )
                )
                t_rel = time.perf_counter()
                rel_payload = prepare_relative_action_statistics(
                    v30_root,
                    statistics_horizon_seconds=horizon_s,
                    rebuild=True,
                )
                _tick("p7_relative_statistics", t_rel)
                print(
                    f"[P7/rel] statistics_relative.json: anchors={rel_payload['anchor_count']} "
                    f"frames={rel_payload['source_frame_count']} → {v30_root / 'meta' / 'statistics_relative.json'}",
                    flush=True,
                )
                export_summary["relative_statistics"] = {
                    "anchor_count": rel_payload["anchor_count"],
                    "source_frame_count": rel_payload["source_frame_count"],
                    "output_path": str(v30_root / "meta" / "statistics_relative.json"),
                }
        del export_summary

    elapsed = time.perf_counter() - t0
    timings["total"] = round(elapsed, 3)
    # Aggregate top-level phase sums for quick reading
    summary_keys = {
        "p3_quality": [k for k in timings if k.startswith("p3_quality_group_")],
        "p1_standardize": [k for k in timings if k.startswith("p1_standardize_group_")],
        "p7_standard_export": [k for k in timings if k == "p7_standard_export" or k.startswith("p7_standard_export_")],
        "p7_relative_statistics": [k for k in timings if k == "p7_relative_statistics"],
        "p3_global_stats": [k for k in timings if k.endswith("p3_global_stats")],
        "p3_stage1_stats": [k for k in timings if k.endswith("p3_stage1_stats")],
        "p3_state_action_lag_stats": [k for k in timings if k.endswith("p3_state_action_lag_stats")],
        "p3_process_episodes": [k for k in timings if k.endswith("p3_process_episodes")],
    }
    phase_summary = {
        name: round(sum(timings[k] for k in keys), 3) for name, keys in summary_keys.items() if keys
    }
    timing_payload = {
        "sec": timings,
        "summary_sec": phase_summary,
        "summary_min": {k: round(v / 60.0, 3) for k, v in phase_summary.items()},
        "total_min": round(elapsed / 60.0, 3),
    }
    (output_dir / "phase_timing.json").write_text(
        json.dumps(timing_payload, indent=2), encoding="utf-8"
    )
    print("[timing] === phase summary (min) ===", flush=True)
    for k, v in sorted(phase_summary.items(), key=lambda kv: -kv[1]):
        print(f"[timing]   {k}: {v / 60:.2f} min ({v:.1f}s)", flush=True)
    print(f"[timing]   TOTAL: {elapsed / 60:.2f} min", flush=True)

    run_meta.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "processed_quality": len(all_results),
            "standard_episodes": total_standard,
            "fps_failures": fps_failures,
            "speed_align_samples": speed_meta_samples,
            "phase_timing": timing_payload,
        }
    )
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if all_results:
        write_quality_report(
            output_dir / "reports" / "quality_report.json",
            build_quality_report(
                all_results,
                stats_meta={"phases": "P0-P7"},
                config_summary=run_meta,
            ),
        )
        write_exclusion_log(output_dir / "reports" / "exclusion_log.jsonl", all_results)
        write_optimal_lags(output_dir / "reports" / "optimal_lags.json", all_results)

    print(f"=== Phases done in {elapsed/60:.1f} min ===", flush=True)
    return 0

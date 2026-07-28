# Dataclean datasets

Each subdirectory is a **dataset package** (custom adapter + checklist + entrypoint).

| Package | Layout | Entry |
|---------|--------|-------|
| `_template/` | copy-me | — |
| `humanoid_merged/` | single_root | `python datasets/humanoid_merged/run.py --output-dir ...` |
| `egodex_lerobot_v21/` | part_task | `python datasets/egodex_lerobot_v21/run.py --output-dir ...` |
| `robomind_ur/` | single_root | `python datasets/robomind_ur/run.py --output-dir ...` |

Shared engine: `../engine` → `qwen-manip-preprocess/src`.

Phases: see [`../specs/PIPELINE_PHASES.md`](../specs/PIPELINE_PHASES.md).

Before full `filter`/`both` export, approve `review_checklist.yaml` or pass `--force-skip-gate`.

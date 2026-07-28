# Dataset package template

Copy this folder to `datasets/<dataset_name>/` and fill in:

| File | Purpose |
|------|---------|
| `config.yaml` | Paths, quality stages, export mode |
| `keymap.md` / `keymap.json` | Source → Dataclean field mapping |
| `review_checklist.yaml` | Manual ⚠ gates (P2) |
| `adapter.py` | Source episode → standard / canonical arrays |
| `discover.py` | Episode enumeration (single root or part/task) |
| `run.py` | Thin CLI: discover → phases → engine |

Review artifacts (plots, sample frames) go under `review_artifacts/` (gitignored locally if large).

See [`specs/PIPELINE_PHASES.md`](../../specs/PIPELINE_PHASES.md).

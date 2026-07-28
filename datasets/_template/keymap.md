# Review corrections — `datasets/*/config.yaml`

After human review (`review_checklist.yaml`), write numeric fixes into this package's
**`config.yaml`** under `review_corrections` (not into the checklist or keymap).

The engine applies them in P1 after the adapter (`apply_review_corrections`).

## `extrinsic_rotation_correction` (from `eef_direction` review)

```yaml
review_corrections:
  extrinsic_rotation_correction:
    camera_top.T_ArmLeft_CameraTop: null
    camera_top.T_ArmRight_CameraTop: null
```

- Each value: `null` (no-op) or a 3×3 / 4×4 matrix `R`
- **Apply:** only for that key: `T_corrected = R_4x4 @ T`
- Other extrinsics are never touched

## `gripper_closedness_correction` (from `gripper_closedness` review)

```yaml
review_corrections:
  gripper_closedness_correction:
    action: null
    state: null
```

Each side: `null` or `{scale: a, offset: b, clip: true}`

- `action` → `action.gripper.*.closedness`
- `state` → `observation.state.gripper.*.closedness`

**Example** (invert action only):

```yaml
review_corrections:
  gripper_closedness_correction:
    action: {scale: -1.0, offset: 1.0}
    state: null
```

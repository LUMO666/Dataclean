# egodex_lerobot_v21 → Dataclean 标准格式键值对照表

- **源数据**: `/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_lerobot_v21`（`part/task`）
- **目标规范**: `standard_info.json` + `processing.md`
- **内外参**: parquet 列 `observation.camera_intrinsics` / `observation.camera_extrinsics_world`
- **约定**: 双手 EgoDex；仅 `camera_top`（导出为 `camera_reference`）；无腕相机 / 无臂关节

---

## 0. 全局 / meta 字段

| 目标键 | 源键 / 取值 | 转换说明 |
|---|---|---|
| `fps` | `fps` | `30` |
| `embodiment` | `robot_type`=`egodex` | 写 `"egodex"` |
| `camera_view_direction` | — | `"ego"` |
| `state_action_delay` | — | 见 `temporal_align` |
| `splits` | — | **不写入**目标 `info.json` |

---

## 1. 视频 / 图像

| 目标键 | 源键 | 转换说明 |
|---|---|---|
| `observation.images.camera_top` | `observation.images.camera_top` | 原生约 1920×1080；P6 后导出名为 `camera_reference` |
| `observation.images.camera_wrist_*` | — | **源无，省略** |

短边缩放到 384，保持宽高比；MP4 h264，关键帧间隔 ≤10。

---

## 2. Action / State（逐帧）

源 20d：`[L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)]`

| 目标键 | shape | 源切片 |
|---|---|---|
| `action.eef.left.position` | 3 | `action[0:3]` |
| `action.eef.left.rotation_6d` | 6 | `action[3:9]` |
| `action.gripper.left.closedness` | 1 | `action[9]` |
| `action.eef.right.position` | 3 | `action[10:13]` |
| `action.eef.right.rotation_6d` | 6 | `action[13:19]` |
| `action.gripper.right.closedness` | 1 | `action[19]` |
| `observation.state.eef.*` | 同左 | `observation.state` 对应切片 |
| `observation.state.hand_features` | * | 可选 |

> 不再导出 `*.eef.*.pose` / `*.geometry.*.rotvec`（与 humanoid v2 一致，统一 position + rotation_6d）。

---

## 3. 内外参 / P6

| 目标键 | 源 | 说明 |
|---|---|---|
| `intrinsic.camera_top.matrix` | `observation.camera_intrinsics` frame0 | 3×3，按短边 384 缩放后写入 episodes |
| `extrinsic.camera_reference.T_Episode_CameraReference` | `observation.camera_extrinsics_world` | `inv(T_world_cam0) @ T_world_cam[t]` |
| EEF | 通常已在 `episode_first_head_camera` | `poses_in_episode_frame: true` 时 P6 不二次变换位姿 |

---

## 4. Layout

- 发现：`root/{part}/{task}/meta/info.json`（`layout: part_task`）
- 导出名：全局 renumber `0..N-1`；`meta.source_episode_index` + `episode_source_map.jsonl` 保留源 id / part / task

---

## 5. 相对统计

`relative_statistics` **硬要求** eef `position` / `rotation_6d`（及 gripper、`T_Episode_CameraReference`）。
`arm.*.joint_position` **有则算、无则跳过**（egodex 无关节时只输出 eef/gripper 相对统计）。

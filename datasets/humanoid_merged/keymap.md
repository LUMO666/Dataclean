# humanoid_merged → Dataclean 标准格式键值对照表

- **源数据**: `/mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged`
- **目标规范**: `standard_info.json` + `processing.md`
- **内外参参考**: `parameters/chunk-XXX/episode_YYYYYY/calibration_bundle_optimized.json`
- **约定**: 双臂 humanoid；`*.primary.*` 不适用；overhead 相机源数据不存在则省略

---

## 0. 全局 / meta 字段

| 目标键 (Dataclean) | 源键 / 取值 (humanoid_merged) | 转换说明 |
|---|---|---|
| `codebase_version` | `codebase_version` | 直接拷贝，`v2.1` |
| `data_path` | `data_path` | 路径模板不变 |
| `video_path` | `video_path` | 路径模板不变；`{video_key}` 用目标相机名 |
| `fps` | `fps` | `30` |
| `total_episodes` / `total_frames` / `total_tasks` / `total_videos` / `total_chunks` | 对应 meta 字段 | 导出后重算 |
| `embodiment` | `robot_type`=`humanoid` | 目标写 `"humanoid"`（示例里的 `agilex` 仅为样例） |
| `camera_view_direction` | — | 默认 `"arm_side"` |
| `prompt_template` | — | 用 `standard_info.json` 模板 |
| `horizon` | — | 按轨迹长度或配置填写 |
| `state_action_delay` | — | 见 `temporal_align`：默认填统计 lag；`apply_delay` 时为 1；manual 时为 `manual_delay` |
| `splits` | — | **不写入**目标 `info.json` |

---

## 1. 视频 / 图像

| 目标键 | 源键 | 转换说明 |
|---|---|---|
| `observation.images.camera_top` | `observation.images.camera_top` | 直接映射；源相机标定名 `camera_front` |
| `observation.images.camera_wrist_left` | `observation.images.camera_wrist_left` | 直接映射；源标定名 `camera_left` / `camera_hand_left` |
| `observation.images.camera_wrist_right` | `observation.images.camera_wrist_right` | 直接映射；源标定名 `camera_right` / `camera_hand_right` |
| `observation.images.camera_chest` | — | **源无，省略** |
| `observation.images.camera_overhead_*` | — | **源无，省略** |

分辨率/编码按 `processing.md`：宽统一到 384，保持原生宽高比；MP4 h264，关键帧间隔 ≤10。

---

## 2. Action（逐帧）

> **`action.eef.*`：** humanoid 源 `action` 列无独立 EE。默认由 Stage2 在键缺失时按关节 lag
> 从 `observation.state.eef` 补全（`action.eef[t]=state.eef[t+L]`；见 `config.yaml` → `stage2.fill_missing_action_eef`）。

| 目标键 | shape | 源键 / 切片 | 转换说明 |
|---|---|---|---|
| `action.eef.left.pose` | 6 | （缺失时）`observation.state.eef.left.pose` + 关节 lag | Stage2 补全 |
| `action.eef.right.pose` | 6 | （缺失时）`observation.state.eef.right.pose` + 关节 lag | Stage2 补全 |
| `action.eef.primary.pose` | 6 | — | 双臂场景 **不适用 / 省略** |
| `action.arm.left.joint_position` | 6 | `action.arm.position[0:6]` 或 `action[0:6]` | `fl_joint1..6`，单位 rad，直接拷贝 |
| `action.arm.right.joint_position` | 6 | `action.arm.position[6:12]` 或 `action[6:12]` | `fr_joint1..6` |
| `action.arm.primary.joint_position` | 6 | — | **省略** |
| `action.gripper.left.closedness` | 1 | `action.effector.position[0]` 或 `action[12]` | `fl_joint7/8`；约定 0=全开、1=全闭（⚠ 需人工确认量纲是否已是 closedness） |
| `action.gripper.right.closedness` | 1 | `action.effector.position[1]` 或 `action[13]` | 同上，右爪 |
| `action.gripper.primary.closedness` | 1 | — | **省略** |
| `action.geometry.eef.left.quaternion_wxyz` | 4 | （可选）与补全后的 eef 一致 | 可由 state geometry 导出；非必须 |
| `action.geometry.eef.right.quaternion_wxyz` | 4 | （可选）与补全后的 eef 一致 | 同上 |
| `action.geometry.eef.primary.quaternion_wxyz` | 4 | — | **省略** |

### 2.1 `observation.state.end.position` 分量对照（源 14d）

| 源 index | 源 name | 用于目标 |
|---|---|---|
| 0–2 | `l_x,l_y,l_z` | `*.eef.left.pose[0:3]` |
| 3–6 | `l_qx,l_qy,l_qz,l_qw` | left rotvec / left quaternion_wxyz |
| 7–9 | `r_x,r_y,r_z` | `*.eef.right.pose[0:3]` |
| 10–13 | `r_qx,r_qy,r_qz,r_qw` | right rotvec / right quaternion_wxyz |

旋转转换（建议）：
```python
from scipy.spatial.transform import Rotation as R
# pose rotvec
rotvec = R.from_quat([qx, qy, qz, qw]).as_rotvec()  # scipy 默认 xyzw
# geometry quaternion_wxyz
wxyz = [qw, qx, qy, qz]
```

---

## 3. Observation state（逐帧）

| 目标键 | shape | 源键 / 切片 | 转换说明 |
|---|---|---|---|
| `observation.state.eef.left.pose` | 6 | `observation.state.end.position[0:7]` | 与 action.eef.left 相同转换 |
| `observation.state.eef.right.pose` | 6 | `observation.state.end.position[7:14]` | 与 action.eef.right 相同 |
| `observation.state.eef.primary.pose` | 6 | — | **省略** |
| `observation.state.arm.left.joint_position` | 6 | `observation.state.arm.position[0:6]` 或 `observation.state[0:6]` | `fl_joint1..6` |
| `observation.state.arm.right.joint_position` | 6 | `observation.state.arm.position[6:12]` 或 `observation.state[6:12]` | `fr_joint1..6` |
| `observation.state.arm.primary.joint_position` | 6 | — | **省略** |
| `observation.state.gripper.left.closedness` | 1 | `observation.state.effector.position[0]` 或 `observation.state[12]` | ⚠ 同 action gripper |
| `observation.state.gripper.right.closedness` | 1 | `observation.state.effector.position[1]` 或 `observation.state[13]` | ⚠ 同 action gripper |
| `observation.state.gripper.primary.closedness` | 1 | — | **省略** |
| `observation.geometry.eef.left.quaternion_arm_wxyz` | 4 | `observation.state.end.position[3:7]` | xyzw → wxyz |
| `observation.geometry.eef.right.quaternion_arm_wxyz` | 4 | `observation.state.end.position[10:14]` | xyzw → wxyz |
| `observation.geometry.eef.primary.quaternion_wxyz` | 4 | — | **省略** |
| `observation.state.hand_features` | — | — | ego 专用；humanoid **省略** |

### 3.1 源合并列 `observation.state` (28d) 拆分备忘

| 源切片 | 内容 | 目标 |
|---|---|---|
| `[0:12]` | arm joints | `observation.state.arm.{left,right}.joint_position` |
| `[12:14]` | effector | `observation.state.gripper.{left,right}.closedness` |
| `[14:28]` | end pose (同 `end.position`) | eef pose + quaternion |

---

## 4. 索引 / 时间戳（逐帧）

| 目标键 | 源键 | 转换说明 |
|---|---|---|
| `episode_index` | `episode_index` | 直接拷贝（导出时可重编号） |
| `frame_index` | `frame_index` | 直接拷贝，从 0 起 |
| `index` | `index` | 全局帧索引；导出后重算 |
| `task_index` | `task_index` | 直接拷贝 |
| `timestamp` | `timestamp` | episode 内相对秒；按 fps 校验稳定性 |
| — | `timestamps` (int64) | 目标格式无此字段，**丢弃** |

---

## 5. Intrinsic（每条轨迹一组，非逐帧）

源文件：`parameters/.../calibration_bundle_optimized.json` → `intrinsics`

| 目标键 | 源键 | 转换说明 |
|---|---|---|
| `intrinsic.camera_top.matrix` | `intrinsics.camera_front.matrix` | 3×3，直接拷贝 |
| `intrinsic.camera_top.dist_coeffs` | `intrinsics.camera_front.dist_coeffs` | 5d，直接拷贝 |
| `intrinsic.camera_wrist_left.matrix` | `intrinsics.camera_left.matrix` | 3×3 |
| `intrinsic.camera_wrist_left.dist_coeffs` | `intrinsics.camera_left.dist_coeffs` | 5d |
| `intrinsic.camera_wrist_right.matrix` | `intrinsics.camera_right.matrix` | 3×3 |
| `intrinsic.camera_wrist_right.dist_coeffs` | `intrinsics.camera_right.dist_coeffs` | 5d |

> `standard_info.json` 样例只列了 `camera_top`；按 `processing.md` 应对齐所有存在的相机，故补齐双腕内参。

---

## 6. Extrinsic（逐帧列；humanoid 标定通常为常量，可逐帧广播）

源文件：`calibration_bundle_optimized.json` → `extrinsics`  
命名约定（`processing.md`）：`T_{dst←src}`，键名 `T_{Dst}_{Src}`，将 src 系点变到 dst 系。

| 目标键 (`extrinsic.*`) | 源键 | 语义 / 转换 |
|---|---|---|
| `camera_top.T_ArmLeft_CameraTop` | `extrinsics.camera_front_to_arm_left.matrix` | \(T_{\text{ArmLeft} \leftarrow \text{CameraTop}}\)；源名 `camera_front_to_arm_left` |
| `camera_top.T_ArmRight_CameraTop` | `extrinsics.camera_front_to_arm_right.matrix` | \(T_{\text{ArmRight} \leftarrow \text{CameraTop}}\) |
| `camera_top.T_ArmLeft_CameraWristLeft` | `extrinsics.camera_hand_left_to_arm_left.matrix` | \(T_{\text{ArmLeft} \leftarrow \text{CameraWristLeft}}\) |
| `camera_top.T_ArmRight_CameraWristRight` | `extrinsics.camera_hand_right_to_arm_right.matrix` | \(T_{\text{ArmRight} \leftarrow \text{CameraWristRight}}\) |

相机名对照：

| 目标相机名 | 源标定 / 视频名 |
|---|---|
| `camera_top` | `camera_front` / `camera_top` |
| `camera_wrist_left` | `camera_left` / `camera_hand_left` / `camera_wrist_left` |
| `camera_wrist_right` | `camera_right` / `camera_hand_right` / `camera_wrist_right` |

矩阵：取源 `.matrix`（4×4 float32）。`pose_xyzrpy` 仅作调试参考，不写入目标字段。

⚠ 外参运动方向并入 `eef_direction`；仅对 `camera_top.T_ArmLeft_CameraTop` / `T_ArmRight_CameraTop` 在 `config.yaml` → `review_corrections.extrinsic_rotation_correction` 写 `R`（`T'=R@T`）。其他外参不参与。统一到「回相机系后 +Z 朝前」的右手系。

---

## 7. 源有而目标不保留 / 仅作中间量

| 源键 | 处理 |
|---|---|
| `observation.state` (28d 合并) | 拆分写入各子字段后可不保留合并列 |
| `action` (14d 合并) | 拆分为 arm/gripper 后可不保留 |
| `observation.state.arm.position` | 拆到 left/right joint |
| `observation.state.effector.position` | 拆到 gripper closedness |
| `observation.state.end.position` | 拆到 eef pose + geometry quaternion；并作为 **action.eef** 来源 |
| `action.arm.position` / `action.effector.position` | 拆到目标 action.* |
| `timestamps` | 丢弃 |

---

## 8. 目标有而源无（省略）

| 目标键族 | 原因 |
|---|---|
| `*.primary.*` | 单臂字段，双臂 humanoid 不用 |
| `observation.images.camera_overhead_*` / `camera_chest` | 源无对应视频 |
| `observation.state.hand_features` | ego 专用 |

---

## 9. 转换伪代码（单帧）

```python
end = row["observation.state.end.position"]  # 14
# action.eef / observation.state.eef 同源
left_xyz, left_q_xyzw = end[0:3], end[3:7]
right_xyz, right_q_xyzw = end[7:10], end[10:14]

action_eef_left = concat(left_xyz, quat_xyzw_to_rotvec(left_q_xyzw))   # 6
action_eef_right = concat(right_xyz, quat_xyzw_to_rotvec(right_q_xyzw))

action_arm_left  = row["action.arm.position"][0:6]
action_arm_right = row["action.arm.position"][6:12]
action_grip_left  = row["action.effector.position"][0:1]
action_grip_right = row["action.effector.position"][1:2]

# observation 同理，关节/夹爪取 observation.state.*
# intrinsic/extrinsic 从同 episode 的 calibration_bundle_optimized.json 读取并广播
```

---

## 10. 待人工确认项（processing.md ⚠）

1. gripper 是否已是 `0=开 / 1=闭`；不符则在 `config.yaml` → `review_corrections.gripper_closedness_correction.action` / `.state` 分别写 scale/offset
2. eef / 外参方向是否与「相机系 +Z 朝前」一致（`eef_direction`）；不符则在 `review_corrections.extrinsic_rotation_correction` 的左右臂键中写 `R`
3. 是否需要对视频做 width→384 重编码（本对照表只定键映射，不含重采样实现）

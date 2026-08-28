# Standard Robot Data Pipeline — Processing Steps

## 符号约定

| 符号 | 含义 |
|------|------|
| `Nd` | N 维向量或 N 维逐帧列 |
| `[x, y, z, w, x, y, z]` | 末端位姿 7 维向量：位置 (m) + 四元数 (wxyz) |
| `T_{dst←src}` | 4×4 SE(3) 齐次变换矩阵，将 src 坐标系下的点变换到 dst 坐标系 |
| `⚠ 人工查看` | 需人工介入确认或校验的步骤 |

**统一坐标系约定：** 根据外参变换回相机系以后，正前方为+Z的右手系

---

## 1. Action

### 1.1 字段定义

| 字段 | 维度 | 单位 / 取值 | 说明 |
|------|------|-------------|------|
| `eef.{left/right/primary}.pose` | 7d | m, wxyz | `[x, y, z, w, x, y, z]` |
| `gripper.{left/right/primary}.closedness` | 1d | 无量纲 | 0 = 全开，1 = 全闭 |
| `arm.{left/right/primary}.joint_position` | 6d | rad | 关节角 |

### 1.2 处理流程

1. ⚠ 人工查看并确认坐标系方向和统一单位，并目视核对外参运动方向，查看四元数wxyz顺序
   - 若 `camera_top.T_ArmLeft_CameraTop` / `camera_top.T_ArmRight_CameraTop` 需改向：在 `config.yaml` → `review_corrections.extrinsic_rotation_correction` 中为对应键填入旋转矩阵 `R`（3×3 或 4×4），流水线仅对该键执行 `T' = R @ T`（其他外参不参与）
   - 旋转表示：若非四元数或 rotvec，先转为四元数；pose 主字段存 **wxyz 四元数**，geometry 可选存 **3d rotvec**
   - 如果是双臂本体，则分别记录 <role> = right,left ；如果是单臂本体则记录 <role> = primary
2. **gripper.<role>.closedness** — ⚠ 人工查看 统一开合表示（0=全开，1=全闭）
   - 若源数据不符合该约定：在 `config.yaml` 的 `review_corrections.gripper_closedness_correction` 中分别填入 `action` / `state` 的 `{scale: a, offset: b}`，流水线分别对 `action.gripper.*.closedness` 与 `observation.state.gripper.*.closedness` 执行 `c' = a*c + b`（默认 clip 到 [0,1]）
3. **arm.<role>.joint_position** — 统一关节表示
4. **state–action delay（P4）**
   - 默认：只统计 lag，不改写 action；`info.state_action_delay` = 统计 `lag_mean`
   - `temporal_align.apply_delay: true`：按统计 lag 对齐到 delay=1，并写入 `state_action_delay=1`
   - `temporal_align.mode: manual` + `manual_delay: k`：不统计；`+k` 延迟 / `-k` 提前 action，并写入该 `k`
5. 以上字段均为**逐帧列**

---

## 2. Observation

### 2.1 字段定义

**图像（`images`）**

所有相机图像按以下约定命名：

| 相机位置 | 字段名 |
|----------|--------|
| 顶部 / 头戴 | `camera_top` |
| 左侧机身 / 侧视 | `camera_left` |
| 右侧机身 / 侧视 | `camera_right` |
| 左腕部（手部） | `camera_wrist_left` |
| 右腕部（手部） | `camera_wrist_right` |

- 存储格式：LeRobot 2.1 规范，MP4；通道顺序遵循 MP4 默认顺序
- 分辨率： 保持数据中视频的原生比例，统一将width变换为384；MP4视频编码关键帧间隔不超过10帧

**状态量**

`eef.pose`、`gripper`、`joint_position` 的定义与 Action 章节一致。
`state.hand_features` 用于储存ego数据的手部关节信息，保持原始信息

### 2.2 处理流程

1. 对照 Action 章节，检查对应字段的一致性
2. ⚠ 人工查看并对齐所有图像的定义与命名，如果没有camera_top或对应的上方视角camera则直接丢弃数据集
3. 以上字段均为**逐帧列**

---

## 3. Parameters

### 3.1 字段定义

| 字段 | 说明 |
|------|------|
| `extrinsic` | 相机外参；命名与 Observation 中相机一致；按 `T_{dst}_{src}` 命名，4×4 SE(3) 矩阵，**逐帧列** |
| `intrinsic` | 相机内参；命名与 Observation 中相机一致；每组含 `matrix` 与 `dist_coeffs`；**每条轨迹仅一组** |
| `state_action_delay` | 见 Action 处理流程 / P4 `temporal_align`；Stage2 可在 `action.eef` 缺失时按关节 lag 从 `state.eef` 补全 |
| `fps` | 数据帧率 |
| `embodiment` | 本体类型 |
| `camera_view_direction` | 相机朝向；默认取值：`arm_side` ，如果有 `opposite_side`单独标（⚠ 待定，暂难统一） |

### 3.2 处理流程

1. ⚠ 人工查看并对齐内外参的相机命名；左右臂相对 `camera_top` 的外参方向在 `eef_direction` 中确认（校正键：`extrinsic_rotation_correction` 下的两条 `T_Arm*_CameraTop`）
2. 依据 `fps` 校验数据中的 timestamp，确认整体帧率稳定且标注无误
3. fps不稳定的丢弃
4. 预处理之后，统计action中的 ee pose 的数值分布范围，根据数据自身的fps换算出其在30fps下的 ee pose 的数值范围，并与 golden ee pose 范围做对比，决定是否要快放/慢放以同一数据中的真实动作速度，以及缩放的比例。（目前暂时以humanoid数据为golden标准）

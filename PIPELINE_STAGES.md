# Dataclean 流水线阶段说明（算法与 I/O）

本文描述 `Dataclean` 定制–通用交叉流水线 **P0–P7** 各阶段的操作、核心算法，以及输入/输出格式。规范字段见 [`processing.md`](processing.md)、[`standard_info.json`](standard_info.json)；相位总览见 [`specs/PIPELINE_PHASES.md`](specs/PIPELINE_PHASES.md)。

**代码入口：** `datasets/<name>/run.py` → `engine/robot_data_processing/phases/orchestrator.py`  
**共享引擎：** `engine/` → `qwen-manip-preprocess/src`

---

## 0. 端到端数据流

```text
源 LeRobot (parquet / videos / parameters)
    │
    ▼
P0 Ingest ──────────────► EpisodeRef[]
    │
    ▼
P1 SchemaNormalize ─────► StandardEpisode (标准键字段)
    │
    ▼
P2 ManualGate ──────────► pass / block（checklist）
    │
    ▼
P3 QualityFilter ───────► EpisodeResult (validity mask, discard)
    │   Stage1–5 + 全局 stats
    ▼
P4 TemporalAlign ───────► 默认只统计 lag；可选 apply_delay / manual；写入 state_action_delay
    │
    ▼
P5 MediaNormalize ──────► 标准相机名视频 (width=384, GOP≤10)
    │
    ▼
P6 ParamsAndSpeed ──────► fps 校验 / 内外参 / ee 速度缩放元数据
    │
    ▼
P7 StandardExport ──────► 标准 LeRobot 输出根
```

### 通用符号

| 符号 | 含义 |
|------|------|
| \(T\) | 单 episode 帧数 |
| \(D_s, D_a\) | 质量空间 state / action 维数（随 embodiment 变化） |
| `EpisodeRef` | `{episode_index, dataset_root, part?, task?}` |
| `StandardEpisode` | 标准键 → `(T, d)` 数组 + camera / intrinsic / extrinsic / meta |
| `EpisodeResult` | 质量结果：`step_validity_mask`、`discard`、各 stage 统计 |

### 质量空间（P3 内部）维数约定

| Embodiment | state | action | 说明 |
|------------|-------|--------|------|
| humanoid | 28 | 14 | 关节+夹爪+EE；对齐维 14 |
| egodex | 14 | 14 | xyz+rpy+gripper ×2（由 20d rot6d 转来） |
| robomind_ur | 26 | 52 | 对齐用 compact 14d teleop |

---

## P0 — Ingest（数据发现与装载）

### 操作

1. 按数据集布局枚举 episode：
   - `single_root`：读 `meta/info.json` 的 `total_episodes`，检查 `data/chunk-***/episode_******.parquet`
   - `part_task`（EgoDex）：遍历 `{part}/{task}/meta/info.json`，每 task 独立 LeRobot 根
2. 可选：随机采样 `--sample-size`
3. 后续阶段按需 `read_episode_table` / `read_episode_canonical`

### 算法

无数值算法；纯文件系统发现。

### 输入

| 项 | 格式 |
|----|------|
| 数据集根 | 目录路径（配置 `dataset.root`） |
| 布局 | `single_root` \| `part_task` |

### 输出

| 项 | 格式 |
|----|------|
| `EpisodeRef[]` | Python dataclass 列表 |
| 日志 | `[P0] discovered N episodes` |

**实现：** `phases/ingest.py` · 定制：`datasets/<name>/discover.py`

---

## P1 — SchemaNormalize（字段标准化）

### 操作

1. 用数据集 `adapter.py` 将源 parquet 列映射为 Dataclean 标准键
2. 旋转统一为 **rotvec**（四元数 xyzw→rotvec；EgoDex rot6d→rotvec）
3. 夹爪统一为 **closedness**（0=开，1=闭，⚠ 极性由人工确认）
4. 相机键映射到 `observation.images.camera_*`
5. 若配置 `require_camera_top` 且无上方视角 → 丢弃该 episode

### 算法（示例）

**Humanoid EE：** 源 `observation.state.end.position` 每臂 7d = xyz(3)+quat_xyzw(4)

\[
\text{pose}_6 = \big[\,x,y,z,\; \mathrm{RotVec}(q_{xyzw})\,\big]
\]

**EgoDex：** 源 20d = 左(xyz+rot6d+g) + 右(…) → 标准 eef.pose(6) + gripper(1)

### 输入

| 项 | 格式 |
|----|------|
| 源 parquet 列 | 数据集原始键（见 `keymap.json`） |
| `EpisodeRef` | 见上 |

### 输出

| 项 | 格式 |
|----|------|
| `StandardEpisode.fields` | `dict[str, float32 ndarray (T,d)]`，键对齐 `standard_info.json` |
| `camera_keys` | 如 `observation.images.camera_top` |
| `has_camera_top` | bool |

典型标准键（双臂）：

- `action.eef.{left,right}.pose` — `(T,6)` m + rad rotvec  
- `action.gripper.{left,right}.closedness` — `(T,1)`  
- `action.arm.{left,right}.joint_position` — `(T,6)` rad  
- `observation.state.*` 对称字段  
- `action.geometry.eef.*.quaternion_wxyz` — `(T,4)`（可选）

**实现：** `datasets/<name>/adapter.py` · 几何：`normalize/transforms_standard.py`

---

## P2 — ManualGate（人工查看门禁）

### 操作

**最先**确认 `eef_direction`（eef 坐标系 / +Z 朝前），再检查其余槽位。  
流水线**默认不执行 Stage5**，不对 eef 做 SE(3) / `rotation_correction` 变换；方向对齐仅靠人工 review。

`output_mode` 为 `filter`/`both` 时未通过全部门禁则抛错；实验可用 `--force-skip-gate`。

### 槽位（顺序固定）

| 槽位 | 含义 |
|------|------|
| `eef_direction` | **首项**：eef 坐标系 / +Z 朝前，**并**核对外参运动方向；兼容旧键 `coord_frame`。校正写入 `config.yaml` → `review_corrections.extrinsic_rotation_correction`（仅 `camera_top.T_ArmLeft_CameraTop` / `T_ArmRight_CameraTop`） |
| `gripper_closedness` | 0=开 / 1=闭；校正写入 `config.yaml` → `review_corrections.gripper_closedness_correction.action` / `.state`（默认 clip `[0,1]`） |
| `camera_naming` | 标准相机名；存在 `camera_top` |
| `fps_stable` | 抽样 timestamp 稳定 |

状态：`pending` \| `approved` \| `blocked`

### 输入 / 输出

| 方向 | 格式 |
|------|------|
| 输入 | YAML checklist |
| 输出 | `GateResult{ok, pending[], blocked[], message}` |

**实现：** `gates/manual_gate.py`（`assert_eef_direction_or_raise` 在编排最前调用）

---

## P3 — QualityFilter（质量过滤 Stage1–5）

在 **质量空间**（canonical state/action）上运行。编排复用 `pipeline.run_pipeline`：先算全局 stats / Stage1 stats /（可选）lag stats，再逐 episode 跑 Stage1–5，产出 `EpisodeResult`。

有效性合成：

1. Stage1 异常帧 ∪（Stage2 discard → 全帧异常）∪ Stage3 剔除帧  
2. **前缀截断**：第一个异常帧之后全部丢弃  
3. 在有效前缀上跑 Stage4 稀疏删静止帧  
4. `step_validity_mask ∈ {0,1}^T`（1=保留）

---

### P3 / Stage1 — 突变检测

**文件：** `stages/stage1_sudden_change.py` · 全局阈：`stage1_stats.py`

#### 算法

1. **全局统计（全库或采样集）：** 对各维平滑后算 residual / \|accel\| / \|jerk\| 分布，取  
   \(\mathrm{thr} = \max(\mathrm{percentile\_floor},\; \mu + k\cdot\sigma)\)（配置 `k_residual/accel/jerk`）
2. **逐 episode：** median + savgol 平滑 → 超阈记突发；再并硬限幅（关节角、EE、RPY、夹爪等）
3. **簇规则：** 短簇（长度 ≤ `frame_max_cluster`）标为异常帧；过长簇可整集 discard

#### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | `state (T,Ds)`, `action (T,Da)`；可选启动段 exclude mask |
| 输出 | `abnormal_frames (T,) bool`；`flagged_count`；可选整集 discard |

---

### P3 / Stage2 — 趋势一致性（Direction Agreement）

**文件：** `stages/stage2_trend_alignment.py`

> **注意：** Stage2 在 **标准键** 上做质检（不再用 packed `observation.state`/`action` 匿名维）。  
> 真正的 state–action 时序改写仍在 **P4**。夹爪 closedness **不参与** DA。

#### 配对字段（物理对应）

双臂只看 `left`+`right`；单臂只看 `primary`：

| state | action |
|-------|--------|
| `observation.state.eef.<role>.pose` | `action.eef.<role>.pose` |
| `observation.state.arm.<role>.joint_position` | `action.arm.<role>.joint_position` |

#### 算法

1. （可选）若 `action.eef.<role>` **缺失/空** 且 `stage2.fill_missing_action_eef: true`（默认）：  
   用 **joint_position** 估计最优 lag_mean \(L\)，再  
   \(\mathrm{action.eef}[t]=\mathrm{state.eef}[t+L]\)（末尾 \(L\) 帧重复 `state.eef[-1]`）  
2. 对上述存在的配对维：平滑 → \(\Delta\) → 互相关取 lag → DA  
3. 任维 \(\mathrm{DA} < \mathrm{da\_per\_dim}\) → 整集 discard（异常全帧）

配置：`datasets/*/config.yaml` → `stage2.fill_missing_action_eef`

#### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | 标准字段 dict（或 quality 空间经 `canonical_to_stage2_fields` 构造） |
| 输出 | `discard`, `da_mean`, `da_per_dim[]`, `dim_names[]`, `lags[]`, `filled_action_eef`, `fill_lags` |

---

### P3 / Stage3 — 极值检测

**文件：** `stages/stage3_extreme_value.py` · 全局：`stats.py`

#### 算法

1. 全库直方图估计各维分位数 \(q_{01}, q_{99}\)（及 min/max）  
2. 带宽：\([q_{01}-\alpha\cdot\mathrm{range},\; q_{99}+\alpha\cdot\mathrm{range}]\)（夹爪/RPY 可豁免）  
3. 超带或硬限幅 → `remove_frames`  
4. 参与后续前缀截断；剩余过短可 discard

#### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | `state`, `action`, `GlobalStats` |
| 输出 | `remove_frames (T,) bool`；`excluded_count` |

---

### P3 / Stage4 — 静止区间缩短

**文件：** `stages/stage4_static_interval.py`

#### 算法

在**有效前缀**内扫描：若连续帧满足 state、action 均不变（\(\|\Delta\|\le\epsilon\)），则每个静止 run 只保留前 `max_static_steps` 帧，其余 `remove=True`（稀疏删帧，非前缀截断）。

#### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | 前缀内 `state[:valid_end]`, `action[:valid_end]` |
| 输出 | `remove_frames`；`static_frames_removed` |

---

### P3 / Stage5 — 相机系对齐（**默认关闭**）

**文件：** `stages/stage5_frame_alignment.py`  
**默认：** `enabled: false`（`Stage5Config` 与各数据集 yaml）

> eef 方向 / +Z 朝前由 **P2 `eef_direction`** 人工确认；流水线默认**不对 eef 做**相机系重定系或 `rotation_correction`。仅在显式开启 Stage5 时才计算对齐元数据。

#### 算法（仅 `enabled: true` 时）

- 以 `camera_top` 第 0 帧为原点，将 EE 位姿变换到相机系  
- Humanoid：读 calibration JSON；EgoDex：用外参列  
- 可叠加固定欧拉修正

#### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | 原始 pose + 外参 |
| 输出 | metadata（`stage5_applied`）；默认路径不改写导出 eef |

---

### P3 汇总结论格式

```text
EpisodeResult:
  episode_index: int
  num_frames: T
  discard: bool
  discard_reasons: [str, ...]
  step_validity_mask: int8[T]   # 1=keep
  kept_frames: int
  stage1_flagged_frames / stage2_da_mean / stage3_excluded_frames / stage4_removed_frames
  metadata: {...}
```

落盘：`reports/quality_report.json`、`reports/exclusion_log.jsonl`

---

## P4 — TemporalAlign（State–Action 时序对齐）

**文件：** `stages/state_action_temporal_alignment.py` · 统计：`state_action_lag_stats.py` · 应用：标准导出 / `lerobot_export.py`  
**配置：** 优先读数据集包 `config.yaml` 的 `temporal_align`（可覆盖 quality YAML 的 `state_action_alignment`）

### 模式（默认：只统计、不改写）

| 配置 | 行为 | `state_action_delay` |
|------|------|----------------------|
| `mode: stats` + `apply_delay: false`（**默认**） | 只做全局 lag 统计，**不**平移 action | 统计得到的 `lag_mean` |
| `mode: stats` + `apply_delay: true` | 统计 lag，再把 action 对齐到名义 delay=1 | `1` |
| `mode: manual` + `manual_delay: k` | **不**统计 lag；按约定平移 action | 填入的 `k` |

手动约定：`+k` = 将 action **延迟** k 步；`-k` = 将 action **提前** k 步。

### 操作

1. **全局统计 lag（`mode=stats`）：** 每条 episode、每个对齐维互相关，lag ∈ [0, max_lag]；聚合成 `lag_mean`（可被 `fixed_lag` 覆盖）  
2. **是否改写 action：** 由 `apply_delay` / `mode=manual` 决定  
   - **apply_stats：** 已有 action → \(\mathrm{action}'[t]=\mathrm{action}[t+(1-L)]\)；缺维才用 state 填  
   - **manual：** \(\mathrm{action}'[t]=\mathrm{action}[t-k]\)（edge padding；\(k=\) `manual_delay`）  
3. **meta：** 写入上表对应的 `state_action_delay`

### 输入 / 输出

| 方向 | 内容 |
|------|------|
| 输入 | 质量/标准空间 action；`datasets/*/config.yaml` → `temporal_align` |
| 中间 | `cache/state_action_lag.npz`（仅 stats 模式） |
| 输出 | （可选）平移后的 action；`info.json` 中 `state_action_delay` |

---

## P5 — MediaNormalize（视频规范化）

**文件：** `phases/media.py`

### 操作

1. 按 adapter `camera_name_map`：源视频键 → `observation.images.camera_*`  
2. ffmpeg 重编码：宽度缩放到 **384**（高按比例、偶数），关键帧间隔 **≤10**  
3. 编码器探测顺序：`libx264` → 其它可用 H.264/mpeg4；失败则 copy 回退

### 输入 / 输出

| 方向 | 路径模板 |
|------|----------|
| 输入 | `{root}/videos/chunk-{c:03d}/{src_key}/episode_{i:06d}.mp4` |
| 输出 | `{out}/videos/chunk-{c:03d}/observation.images.{cam}/episode_{i:06d}.mp4` |

---

## P6 — ParamsAndSpeed（参数与速度）

**文件：** `phases/params.py` · 外参写入：`export_standard.write_parameters`

### 操作

1. **fps / timestamp：** \(\widehat{\mathrm{fps}} = 1/\mathrm{median}(\Delta t)\)；相对期望 fps 超出相对容差 → 丢弃  
2. **内外参：** 按 keymap 写入标准名；外参键约定 `T_{Dst}_{Src}`（\(T_{\mathrm{dst}\leftarrow\mathrm{src}}\)）  
3. **ee 速度对齐（可选）：**  
   \(v = \mathrm{mean}(\|\Delta\mathrm{xyz}\|)\cdot\mathrm{fps}_{\mathrm{target}}\)  
   与 humanoid golden（默认约 \(0.08\,\mathrm{m/s}@30\mathrm{fps}\)）比，得到 `speed_align_scale` 写入 meta（缩放策略可后续用于重采样）

### 输入 / 输出

| 方向 | 格式 |
|------|------|
| 输入 | `timestamp (T,)`；标准 `action.eef.*.pose`；标定 JSON |
| 输出 | fps 判定；`parameters/.../calibration_standard.json`；speed meta |

---

## P7 — StandardExport（标准导出）

**文件：** `export_standard.py`

### 操作

1. 对每个未 discard 且 `kept_frames>0` 的 episode：按 `step_validity_mask` 过滤标准字段  
2. 写 parquet（向量列 list\<float32\>）+ 索引列  
3. 写 `meta/info.json`（对齐 `standard_info.json` 特征模板 + `embodiment` / `fps` / `state_action_delay`）  
4. 写 `episodes.jsonl` / `tasks.jsonl` / `export_report.json`  
5. 调用 P5 写视频；写 `parameters/`

### 输出目录

```text
<output_root>/                 # EgoDex: <out>/<part>/<task>/
├── data/chunk-***/episode_******.parquet
├── videos/chunk-***/observation.images.camera_*/...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── parameters/chunk-***/episode_*******/calibration_standard.json
├── reports/                   # P3 质量报告
├── export_report.json
└── run_meta.json
```

### Parquet 逐帧列（核心）

| 列 | dtype / shape | 说明 |
|----|---------------|------|
| `action.eef.*.pose` | list f32, 6 | xyz + rotvec |
| `action.gripper.*.closedness` | list f32, 1 | |
| `action.arm.*.joint_position` | list f32, 6 | 若有 |
| `observation.state.*` | 同上 | |
| `timestamp` | f32 | 秒，相对 episode |
| `frame_index` / `episode_index` / `index` / `task_index` | i64 | |

`info.json` 额外字段：`state_action_delay`（对齐后为 **1**）、`embodiment`、`fps`、`camera_view_direction`。

---

## 数据集定制 vs 共享对照

| 内容 | 位置 |
|------|------|
| Stage1–5、stats、lag、标准导出、视频转码、gate runner | `engine/robot_data_processing/` |
| keymap、adapter、checklist、discover、config、ignore list | `datasets/<name>/` |

新建数据集：复制 `datasets/_template/` → 填 keymap / checklist → 实现 `adapter.py` → `run.py`。

---

## 相关配置片段

```yaml
# datasets/<name>/config.yaml（示意）
dataset:
  root: ...
  embodiment: humanoid   # egodex | robomind_ur
  layout: single_root    # part_task
  fps: 30
pipeline:
  output_mode: both      # report | filter | both
  force_skip_gate: false
media:
  target_width: 384
  max_keyframe_interval: 10
  require_camera_top: true
speed_align:
  enabled: true
  golden_speed_mps: 0.08
quality_config_path: qwen-manip-preprocess/config/humanoid_merged.yaml
```

质量 YAML 中的 `stage1`–`stage5`、`state_action_alignment` 原样驱动 P3/P4。

---

## 参考实现路径速查

| 阶段 | 路径 |
|------|------|
| 编排 | `phases/orchestrator.py` |
| P0 | `phases/ingest.py` |
| P1 | `datasets/*/adapter.py`, `normalize/` |
| P2 | `gates/manual_gate.py` |
| P3 S1–S5 | `stages/stage*.py`, `pipeline.py` |
| P4 | `stages/state_action_temporal_alignment.py`, `state_action_lag_stats.py` |
| P5 | `phases/media.py` |
| P6 | `phases/params.py` |
| P7 | `export_standard.py` |

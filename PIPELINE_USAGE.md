# Dataclean Pipeline 使用流程

本文说明**当前**整条数据处理流水线怎么用：从环境准备、人工 review、跑批，到看结果与接入新数据集。  
算法细节见 [`PIPELINE_STAGES.md`](PIPELINE_STAGES.md)；相位总览见 [`specs/PIPELINE_PHASES.md`](specs/PIPELINE_PHASES.md)。

**仓库根目录：** `/mnt/project_rlinf_hs/liuweilin/Dataclean`

---

## 1. 整体结构（先建立心智模型）

```text
Dataclean/
├── processing.md / standard_info.json   # 规范与输出字段契约
├── PIPELINE_STAGES.md                   # 各阶段算法与 I/O
├── engine/ → qwen-manip-preprocess/src  # 共享引擎（P0–P7）
├── datasets/
│   ├── _template/                       # 新数据集模板
│   ├── humanoid_merged/
│   ├── egodex_lerobot_v21/
│   └── robomind_ur/
└── qwen-manip-preprocess/config/        # 质量 Stage1–4 等 YAML（被 quality_config_path 引用）
```

| 放哪里 | 放什么 |
|--------|--------|
| `engine/` | Stage1–4、lag 对齐、标准导出、视频转码、ManualGate runner |
| `datasets/<name>/` | keymap、adapter、checklist、discover、config、`run.py` |

**相位顺序（固定）：**

```text
人工 eef_direction（最先）
  → P2 全部门禁
  → P0 发现 episode
  → P3 质量过滤（Stage1–4；Stage5 默认关）
  → P1 标准字段映射
  → P4/P5/P6/P7（filter/both 时导出）
```

要点：
- **不对 eef 做 Stage5 重定系**（默认关）；方向靠人工 `eef_direction`
- Stage2 **不算**左右夹爪维的 DA
- P4：默认**只统计 lag、不改写**；`apply_delay: true` 才对齐到 delay=1；`mode: manual` 用给定 `±k` 平移

---

## 2. 环境准备

```bash
cd /mnt/project_rlinf_hs/liuweilin/Dataclean

# 依赖：与 qwen-manip-preprocess 相同（numpy / pyarrow / scipy / pyyaml / tqdm / ffmpeg 等）
pip install -r requirements.txt
# 或
pip install -r qwen-manip-preprocess/requirements.txt
```

`datasets/*/run.py` 会把 `engine/` 加入 `PYTHONPATH`，一般**不必**手动 export。

确认数据源路径在对应包的 `config.yaml` → `dataset.root` 下可访问。

---

## 3. 推荐工作流（已有数据集）

以 **humanoid_merged** 为例（egodex / robomind 同理，只换目录）。

### 步骤 A — 人工 Review（必须先做）

编辑：

```text
datasets/humanoid_merged/review_checklist.yaml
```

槽位顺序：

| 顺序 | 槽位 | 做什么 |
|------|------|--------|
| 1 | `eef_direction` | 确认 eef 坐标系 / +Z 朝前，**并**目视外参运动方向。若左右臂相对 camera_top 的外参需改向，在 `config.yaml` → `review_corrections.extrinsic_rotation_correction` 两键中写 `R`（见下） |
| 2 | `gripper_closedness` | 确认 0=开、1=闭；不符则在 `config.yaml` → `review_corrections.gripper_closedness_correction` 写校正 |
| 3 | `camera_naming` | 标准相机名；有 `camera_top` |
| 4 | `fps_stable` | 帧率/timestamp 抽样 |

将确认通过的项改为：

```yaml
  eef_direction:
    status: approved   # pending | approved | blocked
```

### Review 校正写入 `config.yaml`（数值接口）

人工审查**通过后**，若需要数值校正，写入同包 `config.yaml` 的 `review_corrections`（不要写进 checklist / keymap）：

**外参（对应 `eef_direction`）**

```yaml
review_corrections:
  extrinsic_rotation_correction:
    camera_top.T_ArmLeft_CameraTop: null
    camera_top.T_ArmRight_CameraTop: null
```

需要改向时，在对应键填 3×3 或 4×4 矩阵 `R`，流水线**仅对该外参**：`T' = R @ T`。腕部外参等其他矩阵不参与。

**夹爪（对应 `gripper_closedness`）**

```yaml
review_corrections:
  gripper_closedness_correction:
    action: null
    state: null
```

需要校正时，在对应组填写（可只改一侧）：

```yaml
review_corrections:
  gripper_closedness_correction:
    action: {scale: -1.0, offset: 1.0}
    state: null
```

- `action` → 仅 `action.gripper.*.closedness`
- `state` → 仅 `observation.state.gripper.*.closedness`

每组 `c' = scale * c + offset`；pipeline 默认再 **clip 到 `[0,1]`**（`clip` 默认为 `true`）。

引擎在 P1 adapter 之后调用 `apply_review_corrections`。细则见 `datasets/_template/keymap.md`。

- **全量导出**（`filter` / `both`）：五项都需 `approved`，否则报错  
- **仅报告**（`report`）：编排仍会先检查 `eef_direction`；其余项在 report 模式下可未全过（导出门禁不拦）  
- **临时跳过**（仅实验）：`--force-skip-gate`

可视化草稿可放：`datasets/<name>/review_artifacts/`。

### 步骤 B — 核对配置

```text
datasets/humanoid_merged/config.yaml
```

常用项：

| 字段 | 含义 |
|------|------|
| `dataset.root` | 源 LeRobot 根 |
| `dataset.layout` | `single_root` 或 `part_task`（EgoDex） |
| `pipeline.output_mode` | `report` \| `filter` \| `both` |
| `quality_config_path` | 指向 `qwen-manip-preprocess/config/*.yaml`（Stage1–4 阈值） |
| `media.target_width` | 默认 384 |
| `speed_align.enabled` | ee 速度相对 golden 的元数据 |
| `temporal_align.apply_delay` | 默认 `false`：只统计 lag；`true` 时对齐到 delay=1 |
| `temporal_align.mode` | `stats`（默认）或 `manual` |
| `temporal_align.manual_delay` | manual 时必填：`+k` 延迟 action / `-k` 提前 |
| `stage2.fill_missing_action_eef` | 默认 `true`：`action.eef` 缺失时用关节 lag 从 `state.eef` 补全 |
| `review_corrections.*` | P2 数值校正（外参旋转 / gripper affine）；见上文 |

质量 YAML 里 **`stage5.enabled` 保持 `false`**（默认）。若要开 Stage5，需显式改 yaml 或 legacy CLI `--enable-stage5`。

### 步骤 C — 小样本冒烟

```bash
cd /mnt/project_rlinf_hs/liuweilin/Dataclean

# 仅质量报告（快）
python datasets/humanoid_merged/run.py \
  --output-dir /tmp/humanoid_smoke_report \
  --sample-size 8 \
  --seed 42 \
  --output-mode report \
  --force-skip-gate

# 标准导出试跑（含视频规范化；需 ffmpeg）
python datasets/humanoid_merged/run.py \
  --output-dir /tmp/humanoid_smoke_filter \
  --sample-size 4 \
  --seed 42 \
  --output-mode filter
  # 正式跑请去掉 --force-skip-gate，并先 approve checklist
```

### 步骤 D — 全量跑批

```bash
python datasets/humanoid_merged/run.py \
  --output-dir /mnt/project_rlinf_hs/dreamzero_pretrain_data/humanoid_merged_dataclean_processed \
  --output-mode both
```

EgoDex（多 part/task，输出镜像源布局）：

```bash
python datasets/egodex_lerobot_v21/run.py \
  --output-dir /mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_lerobot_v21_dataclean_processed \
  --output-mode both
```

RoboMind：

```bash
python datasets/robomind_ur/run.py \
  --output-dir /path/to/robomind_dataclean_processed \
  --output-mode report   # 或 both
```

长时间任务建议 `nohup` / `tmux`，日志重定向到输出目录下的 `run.log`。

### 步骤 E — 查看结果

单根输出（humanoid）示意：

```text
<output_dir>/
├── run_meta.json
├── reports/
│   ├── quality_report.json      # P3 汇总
│   └── exclusion_log.jsonl      # 每集 discard / kept_frames
├── data/chunk-***/episode_******.parquet   # 标准键（filter/both）
├── videos/.../observation.images.camera_*/...
├── meta/info.json               # 含 state_action_delay（统计 lag / 1 / manual_k）等
├── parameters/.../calibration_standard.json
├── export_report.json
└── _quality/                    # 中间质量 cache / 子报告
```

EgoDex：`<output_dir>/<part>/<task>/` 各自一套上述结构。

---

## 4. CLI 参数速查

| 参数 | 说明 |
|------|------|
| `--output-dir` | **必填**，输出根目录 |
| `--config` | 默认 `datasets/<name>/config.yaml` |
| `--output-mode` | 覆盖配置：`report` / `filter` / `both` |
| `--sample-size` | 随机采样 episode 数 |
| `--seed` | 采样随机种子（默认 42） |
| `--force-skip-gate` | 跳过 ManualGate（含 `eef_direction`） |

`output_mode` 含义：

| 模式 | 行为 |
|------|------|
| `report` | 只跑质量过滤 + 报告，不写标准 data/videos |
| `filter` | 质量 + **标准导出**（P7） |
| `both` | 报告 + 标准导出 |

---

## 5. 跑批时日志里会看到什么

```text
[P2/eef] eef_direction approved          # 或 skipped
[P2] ManualGate OK
[P0] discovered N episodes (layout=...)
[P3] [i/M] quality filter root=... n=...
     stats / stage1 stats / state-action lag / process episodes
[P7] standard export → ...               # filter/both 时
=== Phases done in ... min ===
```

---

## 6. 接入新数据集（定制包流程）

```bash
cd /mnt/project_rlinf_hs/liuweilin/Dataclean
cp -r datasets/_template datasets/my_new_dataset
```

按顺序填写：

1. **`keymap.md` / `keymap.json`** — 源字段 → `standard_info.json` 键  
2. **`adapter.py`** — 实现 `to_standard_episode`、`camera_name_map`（可提供 `HumanoidAdapter` 同类名或 `build_adapter`）  
3. **`discover.py`** — `single_root` 或 `part_task`  
4. **`config.yaml`** — `dataset.root`、`embodiment`、`quality_config_path`  
5. **`review_checklist.yaml`** — 先做 **`eef_direction`**  
6. **`run.py`** — 模板已可用，一般不用改  

然后从小样本 `report` → `filter` → 全量 `both`。

---

## 7. 与旧入口的关系

| 旧脚本 | 建议 |
|--------|------|
| `qwen-manip-preprocess/scripts/run_humanoid_full.py` | 改用 `datasets/humanoid_merged/run.py` |
| `scripts/run_egodex_22t_full.py` | 改用 `datasets/egodex_lerobot_v21/run.py` |
| `scripts/run_pipeline.py` | 仍可直接跑质量引擎；**不含**完整 P0–P7 标准导出编排 |

旧 CLI 若需 Stage5：`--enable-stage5`。新默认路径下 Stage5 **关闭**。

---

## 8. 常见问题

**Q: 一跑就报 `eef_direction must be approved`？**  
A: 先在 checklist 里把 `eef_direction` 设为 `approved`，或实验时加 `--force-skip-gate`。

**Q: `filter` 报 ManualGate pending？**  
A: 五项都要 `approved` 才能全量导出。

**Q: 为什么 eef 朝向没变？**  
A: 设计如此——方向由人工确认，默认不做 Stage5 变换。

**Q: 导出 0 个 episode？**  
A: 看 `exclusion_log.jsonl`：可能 Stage2 DA 过低、kept_frames=0、缺 `camera_top`、或 fps 不稳定。

**Q: 视频编码失败？**  
A: 安装可用 ffmpeg 编码器；引擎会探测 `libx264`/mpeg4 等，失败时可能 copy 回退。

**Q: EgoDex 输出怎么组织？**  
A: 按源数据镜像 `part/task`，每个 task 独立标准 LeRobot 根。

---

## 9. 最小检查清单（上线前）

- [ ] `dataset.root` 可读，布局与 `layout` 一致  
- [ ] `eef_direction`（及导出所需其余槽位）已 `approved`  
- [ ] `quality_config_path` 存在，且 `stage5.enabled: false`（除非有意开启）  
- [ ] 小样本 `report` 通过，`exclusion_log` 合理  
- [ ] 小样本 `filter` 写出标准 parquet + `meta/info.json`（含 `state_action_delay`）  
- [ ] 磁盘空间足够（`both` 含重编码视频）  
- [ ] 全量任务用持久会话 / nohup，并监控 `run_meta.json`

---

## 10. 相关文档索引

| 文档 | 内容 |
|------|------|
| [`PIPELINE_STAGES.md`](PIPELINE_STAGES.md) | 各阶段算法与 I/O |
| [`specs/PIPELINE_PHASES.md`](specs/PIPELINE_PHASES.md) | 相位与门禁 |
| [`specs/ENGINE.md`](specs/ENGINE.md) | 引擎挂载说明 |
| [`processing.md`](processing.md) | 字段与 ⚠ 人工项规范 |
| [`standard_info.json`](standard_info.json) | 输出 schema |
| [`datasets/README.md`](datasets/README.md) | 数据集包入口列表 |

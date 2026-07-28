# Egodata preprocess / Dataclean

## Dataclean unified pipeline (P0–P7)

**使用流程（推荐先读）：** [`PIPELINE_USAGE.md`](PIPELINE_USAGE.md)

另见 [`specs/PIPELINE_PHASES.md`](specs/PIPELINE_PHASES.md)、[`PIPELINE_STAGES.md`](PIPELINE_STAGES.md)、[`specs/ENGINE.md`](specs/ENGINE.md)、[`processing.md`](processing.md)。

- Shared engine: `engine/` → `qwen-manip-preprocess/src`
- Dataset packages: `datasets/<name>/` (`_template`, `humanoid_merged`, `egodex_lerobot_v21`, `robomind_ur`)
- Standard output contract: [`standard_info.json`](standard_info.json)

```bash
# Example: humanoid sample (skip manual gate for experiments)
python datasets/humanoid_merged/run.py \
  --output-dir /tmp/humanoid_phases_smoke \
  --sample-size 2 \
  --force-skip-gate \
  --output-mode report
```

## Installation and Code Structure
```
conda create --name ego python==3.11
conda activate ego
conda install -c conda-forge ffmpeg=7.1.1
pip install -r requirements.txt
```
### EgoDex:
```
python convert/build_egodex_prefilter_step1.py 
#把人手映射成夹爪，存成tensorflow文件

python convert/visualize_script/visualize_gripper_axes_egodex.py
#可视化夹爪坐标系，附带arkit confidence

python convert/egodex_tensorflow_to_lerobot_v21_step2.py
#把tensorflow文件转成lerobotv2.1格式
```

### EgoVerse:
```
python /mnt/project_rlinf/runze/ml-egodex/convert/build_egodex_prefilter_step1.py 
#把人手映射成夹爪，存成tensorflow文件

python convert/visualize_script/visualize_gripper_axes_egoverse.py
#可视化夹爪坐标系

python convert/egoverse_tensorflow_to_lerobot_v21_step2.py
#把tensorflow文件转成lerobotv2.1格式
```

## Lerobot v2.1 Feature
```
observation.state: ( position(3) + rotation(6) + gripper(1) ) * 2, all in first-frame camera frame, same with action  
action: ( position(3) + rotation(6) + gripper(1) ) * 2, all in first-frame camera frame  
observation.confidence(egodex only): confidence of arkit joint of "left_wrist", "left_thumb", "left_index", "left_middle", "right_wrist", "right_thumb", "right_index", "right_middle"   
observation.camera_extrinsics_world  
observation.camera_intrinsics  
observation.images.camera_top  
```

# Qwen Data Pipeline (Stage1 - Stage 3)

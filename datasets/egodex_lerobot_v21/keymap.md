# EgoDex lerobot v21 → Dataclean

| Target | Source |
|--------|--------|
| `observation.images.camera_top` | `observation.images.camera_top` |
| `*.eef.*.pose` | 20d xyz+rot6d → xyz+wxyz quaternion |
| `*.geometry.eef.*.rotvec` | rot6d → rotvec（可选） |
| `*.gripper.*.closedness` | width dims |
| `observation.state.hand_features` | optional ego hand joints |

Layout: `part/task` mirrored on export.

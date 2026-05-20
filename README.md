# RL Training & Validation

Train / validate the Gymnasium environments registered by
[rl_environments](https://github.com/ncbdrck/rl_environments) using
Stable Baselines3 + [sb3_ros_support](https://github.com/ncbdrck/sb3_ros_support).

Pure registry-based: every env that `rl_environments` registers is
runnable here. There is no separate "implementation-status table" —
if `gym.envs.registry` contains it, the scripts can drive it.

## Currently registered envs (30)

| Robot | Task | Std (sim) | Goal (sim) | Std (real) | Goal (real) |
|---|---|---|---|---|---|
| RX200 (kinect) | Reach | `RX200ReacherSim-v0` | `RX200ReacherGoalSim-v0` | `RX200ReacherReal-v0` | `RX200ReacherGoalReal-v0` |
| RX200 (kinect) | Push  | `RX200PushSim-v0` | `RX200PushGoalSim-v0` | `RX200PushReal-v0` | `RX200PushGoalReal-v0` |
| RX200 (kinect) | PnP   | `RX200PnPSim-v0` | `RX200PnPGoalSim-v0` | `RX200PnPReal-v0` | `RX200PnPGoalReal-v0` |
| RX200 (zed2)   | Reach | `RX200Zed2ReacherSim-v0` | `RX200Zed2ReacherGoalSim-v0` | — | — |
| RX200 (zed2)   | Push  | `RX200Zed2PushSim-v0` | `RX200Zed2PushGoalSim-v0` | — | — |
| RX200 (zed2)   | PnP   | `RX200Zed2PnPSim-v0` | `RX200Zed2PnPGoalSim-v0` | — | — |
| Ned2 (kinect)  | Reach | `NED2ReacherSim-v0` | `NED2ReacherGoalSim-v0` | `NED2ReacherReal-v0` | `NED2ReacherGoalReal-v0` |
| Ned2 (kinect)  | Push  | `NED2PushSim-v0` | `NED2PushGoalSim-v0` | `NED2PushReal-v0` | `NED2PushGoalReal-v0` |
| Ned2 (kinect)  | PnP   | `NED2PnPSim-v0` | `NED2PnPGoalSim-v0` | `NED2PnPReal-v0` | `NED2PnPGoalReal-v0` |

PnP envs include `is_grasped` derived obs, grasp-aware layered dense
reward, and a `multi_goal` flag for intermediate-lift curriculum.

Real-side push and PnP track the cube via an externally-published
`geometry_msgs/PoseStamped` on `/cube_pose` (configurable via
`--cube-pose-topic`). When no message is received within
`cube_pose_timeout_s`, the env falls back to the YAML `cube_init_pos`
and emits a throttled warning — runnable for dry-run / code-review
even when no vision pipeline is wired up. Wire up aruco_ros,
AprilTag, mocap, or your detector of choice for actual cube tracking.

NED2 push / pnp and UR5e (all tasks) are referenced in stub train/validate
scripts but not yet registered in `rl_environments`. `check_env_constructable`
will surface a clear error if you try to run them.

## Prerequisites

1. **`rl_environments`** — sibling catkin package. Provides the env
   classes + Gymnasium registrations.
2. **`sb3_ros_support`** — <https://github.com/ncbdrck/sb3_ros_support>
   (`gymnasium` branch). Provides SB3 wrappers with ROS integration.
3. **`UniROS`** (`multiros` + `realros`) — Gazebo / real-robot base envs.

## Smoke tests

Run these BEFORE Gazebo. Pure introspection.

```bash
# 1) Enumerate the registry from the training repo's perspective.
python3 scripts/list_available_envs.py

# 2) Verify env_safety / config wiring (no Gazebo).
python3 scripts/smoke_test_training_config.py

# 3) Verify every env id referenced by a non-audit script is registered.
python3 scripts/check_env_availability.py

# 4) Verify every Goal env exposes the GoalEnv hooks (compute_reward / _terminated / _truncated).
python3 scripts/check_goal_training_setup.py
```

### Live smoke (requires Gazebo)

`scripts/live_smoke_envs.py` actually does `gym.make`, `reset`, one
`step`, and `close` for each env id. Each Gazebo bring-up takes
~30–60 s, so a full sweep is 5–10 minutes for the 14 sim envs.

```bash
# Source the workspace first.
source devel/setup.bash

# Smoke every RX200 / NED2 sim env (skips real envs by default):
python3 scripts/live_smoke_envs.py

# Subset by substring match:
python3 scripts/live_smoke_envs.py --filter PnP
python3 scripts/live_smoke_envs.py --filter Goal
python3 scripts/live_smoke_envs.py --filter Zed2

# Include real envs (requires hardware + double-gating):
ALLOW_REAL_ROBOT_MOTION=1 rosparam set /allow_real_robot_motion true
python3 scripts/live_smoke_envs.py --include-real --allow-real-robot-motion --filter Real
```

Each env id is gated with a SIGALRM-based timeout (default 120 s for
`gym.make`, 60 s for `reset`/`step`); a hung env doesn't stall the run.

## Training

In separate terminals:

```bash
# Terminal 1: ROS master
roscore

# Terminal 2: trainer. The env launches its own Gazebo subprocess and
# init_node, so you don't need a pre-running gazebo_ros.
rosrun rl_training_validation rx200_reach_train_sim.py
```

### RX200 sim tasks

```bash
# Reach (TD3 default; --algo sac switches to SAC):
rosrun rl_training_validation rx200_reach_train_sim.py
rosrun rl_training_validation rx200_reach_train_sim.py --goal       # TD3+HER

# Push (TD3 only in this repo):
rosrun rl_training_validation rx200_push_train_sim.py
rosrun rl_training_validation rx200_push_train_sim.py --goal        # TD3+HER

# Pick-and-Place:
rosrun rl_training_validation rx200_pnp_train_sim.py
rosrun rl_training_validation rx200_pnp_train_sim.py --goal         # TD3+HER
```

### NED2 sim Reach

```bash
rosrun rl_training_validation ned2_reach_train_sim.py
rosrun rl_training_validation ned2_reach_train_sim.py --goal
```

### Validate a trained policy

```bash
# Sim
rosrun rl_training_validation rx200_reach_validate_sim.py --episodes 20

# Real (double-gated; see safety section)
rosrun rl_training_validation rx200_reach_validate_real.py --episodes 10 \
    --allow-real-robot-motion
```

### RX200 real tasks

Real-robot motion is double-gated:

1. CLI flag `--allow-real-robot-motion` on the script.
2. EITHER ROS param `/allow_real_robot_motion=true` OR env var
   `ALLOW_REAL_ROBOT_MOTION=1`.

Without both, the script aborts and no interbotix driver is launched.

```bash
# Reach (no cube needed):
rosparam set /allow_real_robot_motion true
rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion
rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion --goal   # HER

# Push (needs cube pose on /cube_pose, default topic):
rosrun rl_training_validation rx200_push_train_real.py --allow-real-robot-motion
rosrun rl_training_validation rx200_push_train_real.py --allow-real-robot-motion --goal
# Override the cube-pose topic:
rosrun rl_training_validation rx200_push_train_real.py --allow-real-robot-motion --cube-pose-topic /aruco/cube_pose

# Pick-and-Place (also needs cube tracking + gripper):
rosrun rl_training_validation rx200_pnp_train_real.py --allow-real-robot-motion
rosrun rl_training_validation rx200_pnp_train_real.py --allow-real-robot-motion --goal --multi-goal
```

#### Cube tracking on real

Real push and PnP envs subscribe to `/cube_pose` (overridable via
`--cube-pose-topic`) expecting `geometry_msgs/PoseStamped`. If no
message arrives within `cube_pose_timeout_s` (default 1.0 s, set in
`rx200_{push,pnp}_task_config.yaml`), the env falls back to
`cube_init_pos` (default `[0.25, 0.0, 0.015]`) and emits a throttled
warning. Cube-tracking pipeline is **deliberately external** to the
env — use whichever you prefer: `aruco_ros`, AprilTag (`apriltag_ros`),
mocap (OptiTrack / Vicon driver), or a deep-learning detector. Wire it
up to publish `PoseStamped` and the env Just Works.

##### Sibling package: `rl_envs_cube_tracker`

For a turnkey AprilTag pipeline, use
[`rl_envs_cube_tracker`](../rl_envs_cube_tracker/README.md). Print
`tag36h11` ID 0 at 30 mm, stick it on the cube, and:

```bash
# Separate terminal (the documented "external" pattern):
roslaunch rl_envs_cube_tracker kinect2.launch \
    target_frame:=rx200/base_link
```

Or have the env auto-launch it for you (opt-in via CLI):

```bash
rosrun rl_training_validation rx200_push_train_real.py \
    --allow-real-robot-motion \
    --cube-tracker auto \
    --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame rx200/base_link
```

`--cube-tracker auto` makes the env roslaunch the tracker (registered
with the same managed-process registry as roscore + interbotix_driver,
so `env.close` reaps it). Default is `none` to preserve the "vision is
external" contract for mocap / YOLO / custom-detector users.

Calibrate the camera extrinsic before relying on `--cube-tracker-target-frame`:
see [`rl_envs_cube_tracker/config/extrinsics/README.md`](../rl_envs_cube_tracker/config/extrinsics/README.md).

#### Validate a trained policy on real

```bash
rosrun rl_training_validation rx200_push_validate_real.py --episodes 10 --allow-real-robot-motion
rosrun rl_training_validation rx200_pnp_validate_real.py  --episodes 10 --allow-real-robot-motion --goal --multi-goal
```

## Repository layout

```
src/rl_training_validation/
  utils/
    env_safety.py            # registry-based env classification + real-motion gate
    multi_task_env.py        # multi-task wrapper used by multi_train_sim
    multi_task_goal_env.py
  _blocked_stub.py           # shared 'blocked env id' bail-out (legacy; rarely used now)
  rx200/  ned2/  ur5e/
    reach/  push/  pnp/      # per-task train + validate scripts
  multi_task_learning/
    multi_train_sim.py
config/
  rx200_*.yaml               # SB3 hyperparams per algo / env
  ned2_*.yaml
  ur5e_*.yaml

scripts/
  list_available_envs.py     # enumerate registry by robot
  check_env_availability.py  # cross-check script refs vs registry
  check_goal_training_setup.py  # GoalEnv hook contract
  smoke_test_training_config.py # full no-Gazebo audit pass
  live_smoke_envs.py         # gym.make / reset / step / close per env (needs Gazebo)
```

## Safety contract

* Real-robot trainers require `--allow-real-robot-motion` (CLI) AND a
  process-visible consent flag (`ALLOW_REAL_ROBOT_MOTION=1` env var or
  `/allow_real_robot_motion=true` rosparam). The flag is propagated
  down to env-side checks.
* Goal-conditioned env ids (`...Goal{Sim,Real}-v0`) are routed to HER
  algorithms (TD3_GOAL / SAC_GOAL) automatically by the train scripts.
  `check_goal_training_setup.py` verifies the GoalEnv hook contract
  (`compute_reward`, `compute_terminated`, `compute_truncated`) holds
  for every registered goal env.
* Sim envs run with per-link FK safety in `execute_action` — every joint
  trajectory target is checked link-by-link against the table floor
  before publishing (`_check_action_links_safe` in the RX200/NED2 robot
  envs). See `rl_environments/config/rx200_reach_task_config.yaml` for
  the safety params.

## What's new (May 2026)

* Reach / push / PnP now have **kinect AND zed2 sim variants**, each in
  std + goal flavour.
* Push obs extended with **cube linear / angular velocity** (finite-diff)
  and **cube position relative to EE**.
* PnP adds **6-DOF action** (5 arm joints + 1 gripper scalar), `is_grasped`
  **derived obs**, **grasp-aware layered dense reward**, and the **`multi_goal`**
  flag for intermediate-lift curriculum.
* `realtime_mode: bool = True` kwarg on every task env — `True` runs the
  UniROS paper §7 real-time loop (rospy.Timer-driven, matches the real
  env), `False` runs a standard MDP pause-step-resume loop for clean
  RL-algorithm benchmarking.

## Contact

[j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie)

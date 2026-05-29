# RL Training & Validation

Train / validate the Gymnasium environments registered by
[rl_environments](https://github.com/ncbdrck/rl_environments) using
Stable Baselines3 + [sb3_ros_support](https://github.com/ncbdrck/sb3_ros_support).

Every env that `rl_environments` registers is runnable here — the
scripts dispatch off the Gymnasium registry.

## Registered envs

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
| VX300S (kinect) | Reach | `VX300SReacherSim-v0` | `VX300SReacherGoalSim-v0` | `VX300SReacherReal-v0` | `VX300SReacherGoalReal-v0` |
| VX300S (kinect) | Push  | `VX300SPushSim-v0` | `VX300SPushGoalSim-v0` | `VX300SPushReal-v0` | `VX300SPushGoalReal-v0` |
| VX300S (kinect) | PnP   | `VX300SPnPSim-v0` | `VX300SPnPGoalSim-v0` | `VX300SPnPReal-v0` | `VX300SPnPGoalReal-v0` |
| UR5e + 2F-85   | Reach | `UR5eReacherSim-v0` | `UR5eReacherGoalSim-v0` | `UR5eReacherReal-v0` | `UR5eReacherGoalReal-v0` |
| UR5e + 2F-85   | Push  | `UR5ePushSim-v0` | `UR5ePushGoalSim-v0` | `UR5ePushReal-v0` | `UR5ePushGoalReal-v0` |
| UR5e + 2F-85   | PnP   | `UR5ePnPSim-v0` | `UR5ePnPGoalSim-v0` | `UR5ePnPReal-v0` | `UR5ePnPGoalReal-v0` |

PnP envs include `is_grasped` derived obs, grasp-aware layered dense
reward, and a `multi_goal` flag for an intermediate-lift curriculum.

Real-side push and PnP track the cube via an externally-published
`geometry_msgs/PoseStamped` on `/cube_pose` (configurable via
`--cube-pose-topic`). When no message is received within
`cube_pose_timeout_s`, the env falls back to the YAML `cube_init_pos`
and emits a throttled warning — runnable for dry-runs even without
a vision pipeline. Wire up `aruco_ros`, AprilTag, mocap, or your
detector of choice for actual cube tracking; see
[`rl_envs_cube_tracker`](https://github.com/ncbdrck/rl_envs_cube_tracker)
for a turnkey AprilTag pipeline.

## Easiest install: one-shot script

If you're starting fresh on Ubuntu 20.04, run the bootstrap installer
that ships in this repo (and identically in every other ecosystem
repo: UniROS, MultiROS, RealROS, sb3_ros_support, rl_environments).
It installs ROS Noetic, UniROS (MultiROS + RealROS), sb3_ros_support,
rl_environments (with all 4 robots' vendor packages + supporting
description-extras + cube tracker), and this package.

```bash
git clone https://github.com/ncbdrck/rl_training_validation.git /tmp/uniros_bootstrap
bash /tmp/uniros_bootstrap/install_uniros_stack.sh                # interactive
bash /tmp/uniros_bootstrap/install_uniros_stack.sh -y             # unattended
bash /tmp/uniros_bootstrap/install_uniros_stack.sh -p ~/my_ws -y  # custom path
```

The script asks once whether to install all components or pick per-
component; refuses to run on non-Ubuntu-20.04 (Noetic requirement).
The manual prerequisites below still work for piece-by-piece installs.

**Don't have Ubuntu 20.04?** A Docker image is shipped under
[`docker/`](docker/) — see [`docker/README.md`](docker/README.md)
for build / run / GPU / hardware-passthrough instructions. Works on
Ubuntu 22.04 / 24.04 hosts, on WSL2, and on machines with GPUs that
have no Ubuntu 20.04 driver.

## Prerequisites (manual)

1. `rl_environments` — provides the env classes + Gymnasium registrations.
2. `sb3_ros_support` — <https://github.com/ncbdrck/sb3_ros_support>
   (`gymnasium` branch). SB3 wrappers with ROS integration.
3. `UniROS` (`multiros` + `realros`) — Gazebo / real-robot base envs.
4. Robot description-extras packages (sim only):
   - RX200 sim → `reactorx200_description`.
   - VX300S sim → `viperx300s_description`.
   - Ned2 sim → `niryo_ned2_description_extras`.
   - UR5e sim → `ur5e_description_extras` (wraps upstream UR5e +
     Robotiq 2F-85, mounts on a 4-legged base next to a cafe-table).

   Real envs don't need these — `niryo_robot_bringup` / Interbotix
   driver handle hardware bring-up.

## Smoke tests (no Gazebo)

```bash
python3 scripts/list_available_envs.py             # enumerate the registry
python3 scripts/smoke_test_training_config.py      # verify env_safety / config wiring
python3 scripts/check_env_availability.py          # cross-check script refs vs registry
python3 scripts/check_goal_training_setup.py       # GoalEnv hook contract
```

### Live smoke (requires Gazebo)

`scripts/live_smoke_envs.py` does `gym.make` → `reset` → one `step` →
`close` per env id. Each Gazebo bring-up takes ~30–60 s.

```bash
source devel/setup.bash

python3 scripts/live_smoke_envs.py                   # all sim envs
python3 scripts/live_smoke_envs.py --filter VX300S   # subset by substring
python3 scripts/live_smoke_envs.py --filter PnP

# Include real envs (requires hardware + --allow-real-robot-motion):
python3 scripts/live_smoke_envs.py --include-real --allow-real-robot-motion --filter Real
```

Each env id is gated with a SIGALRM-based timeout (default 120 s for
`gym.make`, 60 s for `reset`/`step`); a hung env doesn't stall the run.

## Training

```bash
# Terminal 1: ROS master
roscore

# Terminal 2: trainer. The env launches its own Gazebo and init_node.
rosrun rl_training_validation rx200_reach_train_sim.py
```

### Seeds (training + evaluation)

Every train and validate script accepts `--seed N` (default `10`).
On the train side this is now wired all the way through:

* the env's `np_random` (goal / cube / init sampling),
* the SB3 learner's RNG (network init, exploration noise, replay /
  HER minibatch sampling), and
* the on-disk checkpoint + TensorBoard log directories, which get a
  `seed_<N>/` suffix plus a `_s<N>_<timestamp>` stamp on
  `save_prefix` / `trained_model_name` / `log_folder`.

So `--seed 1`, `--seed 2`, `--seed 3` produces three genuinely
independent runs that land in their own directories without
clobbering each other. (Earlier revisions only seeded the env; the
SB3 learner was pinned to the YAML default of 10 and re-runs aborted
on the duplicate log folder.)

Validate scripts additionally accept `--eval-seed N` (default `1000`).
`--seed` continues to pick which trained-policy directory to load
(`seed_<N>/`), and `--eval-seed` drives the *evaluation env's* RNG —
so the rollout goals are sampled from a stream disjoint from the one
the policy was trained on, making the reported success rate a
generalization estimate rather than a memorization check.

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

### NED2 sim tasks

Sim bring-up (separate terminal):

```bash
roscore
roslaunch niryo_ned2_description_extras ned2_gazebo.launch                 # reach / push
roslaunch niryo_ned2_description_extras ned2_gazebo.launch gripper:=true   # pnp
```

Then:

```bash
rosrun rl_training_validation ned2_reach_train_sim.py
rosrun rl_training_validation ned2_reach_train_sim.py --goal
rosrun rl_training_validation ned2_push_train_sim.py
rosrun rl_training_validation ned2_push_train_sim.py --goal
rosrun rl_training_validation ned2_pnp_train_sim.py
rosrun rl_training_validation ned2_pnp_train_sim.py --goal
```

Pass `--wrist-camera` on any NED2 train/validate script to enable the
built-in wrist camera subscriber (off by default).

### VX300S sim tasks

Optional scene check (table + red cube, no RL env):

```bash
roslaunch viperx300s_description vx300s_gazebo.launch load_cube:=true
```

Training:

```bash
rosrun rl_training_validation vx300s_reach_train_sim.py
rosrun rl_training_validation vx300s_reach_train_sim.py --goal
rosrun rl_training_validation vx300s_push_train_sim.py
rosrun rl_training_validation vx300s_push_train_sim.py --goal
rosrun rl_training_validation vx300s_pnp_train_sim.py
rosrun rl_training_validation vx300s_pnp_train_sim.py --goal
```

### UR5e sim tasks

UR5e + Robotiq 2F-85 mounted on a 4-legged base next to a cafe-table.
Optional standalone scene check (no RL env):

```bash
roslaunch ur5e_description_extras ur5e_gazebo.launch
```

Training:

```bash
rosrun rl_training_validation ur5e_reach_train_sim.py
rosrun rl_training_validation ur5e_reach_train_sim.py --goal
rosrun rl_training_validation ur5e_push_train_sim.py
rosrun rl_training_validation ur5e_push_train_sim.py --goal
rosrun rl_training_validation ur5e_pnp_train_sim.py
rosrun rl_training_validation ur5e_pnp_train_sim.py --goal
```

### Real-robot tasks (RX200 / NED2 / VX300S / UR5e)

Real motion is gated by `--allow-real-robot-motion`.
`check_env_constructable` refuses to build any `...Real-v0` env
without it, so the script bails out before any driver is launched.
When the flag is passed, the helper exports
`ALLOW_REAL_ROBOT_MOTION=1` so subprocess workers can read the same
consent.

Every real task env additionally defaults `reset_env_prompt=True`,
so the first `env.reset()` (which homes the arm via MoveIt, and for
PnP closes the gripper) pauses for an `input()` confirmation before
moving the hardware. Pressing Enter continues the run; Ctrl-C cleanly
aborts. For unattended runs (CI, overnight sweeps, headless boxes)
pass `reset_env_prompt=False` through the env constructor — e.g. via
your own train script that builds `env_kwargs` — to opt out of the
prompt while keeping the rest of the safety gating in place.

```bash
# RX200 — reach (no cube), push, pnp
rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion
rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion --goal
rosrun rl_training_validation rx200_push_train_real.py  --allow-real-robot-motion
rosrun rl_training_validation rx200_push_train_real.py  --allow-real-robot-motion --goal
rosrun rl_training_validation rx200_pnp_train_real.py   --allow-real-robot-motion
rosrun rl_training_validation rx200_pnp_train_real.py   --allow-real-robot-motion --goal --multi-goal

# NED2 — bring up niryo_robot_bringup first
rosrun rl_training_validation ned2_reach_train_real.py --allow-real-robot-motion
rosrun rl_training_validation ned2_push_train_real.py  --allow-real-robot-motion \
    --cube-tracker auto --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame base_link
rosrun rl_training_validation ned2_pnp_train_real.py   --allow-real-robot-motion \
    --cube-tracker auto --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame base_link --multi-goal

# VX300S — bring up the Interbotix hardware driver first
rosrun rl_training_validation vx300s_reach_train_real.py --allow-real-robot-motion
rosrun rl_training_validation vx300s_push_train_real.py  --allow-real-robot-motion
rosrun rl_training_validation vx300s_pnp_train_real.py   --allow-real-robot-motion

# UR5e — bring up ur_robot_driver + Robotiq driver + MoveIt first
# (ur5e_description_extras/launch/ur5e_real.launch is the expected
# wrapper; create it at the lab to suit your network IP + calibration).
rosrun rl_training_validation ur5e_reach_train_real.py --allow-real-robot-motion
rosrun rl_training_validation ur5e_push_train_real.py  --allow-real-robot-motion \
    --cube-tracker auto --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame base_link
rosrun rl_training_validation ur5e_pnp_train_real.py   --allow-real-robot-motion \
    --cube-tracker auto --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame base_link --multi-goal
```

NED2 + UR5e real envs use BARE URDF link names (no `ned2/` / `ur5e/`
prefix — that's sim-only). Validate scripts mirror train scripts;
swap `train_real` for `validate_real`.

### Cube tracking on real

Real push and PnP envs subscribe to `/cube_pose` (overridable via
`--cube-pose-topic`). Cube-tracking pipeline is deliberately external:
use `aruco_ros`, AprilTag, mocap, or a deep detector — wire it up to
publish `PoseStamped` and the env Just Works.

Turnkey AprilTag pipeline via
[`rl_envs_cube_tracker`](https://github.com/ncbdrck/rl_envs_cube_tracker):

```bash
# Separate terminal:
roslaunch rl_envs_cube_tracker kinect2.launch target_frame:=rx200/base_link
```

Or auto-launch from the env:

```bash
rosrun rl_training_validation rx200_push_train_real.py \
    --allow-real-robot-motion \
    --cube-tracker auto \
    --cube-tracker-camera kinect2 \
    --cube-tracker-target-frame rx200/base_link
```

`--cube-tracker auto` registers the tracker with the same managed-process
registry as roscore + interbotix_driver, so `env.close` reaps it.
Default is `none` so mocap / YOLO / custom-detector users keep control.

Calibrate the camera extrinsic before relying on
`--cube-tracker-target-frame`: see
[`rl_envs_cube_tracker/config/extrinsics/README.md`](https://github.com/ncbdrck/rl_envs_cube_tracker/blob/main/config/extrinsics/README.md).

### Validate

```bash
# Sim — held-out eval seed disjoint from the training --seed
rosrun rl_training_validation rx200_reach_validate_sim.py --episodes 20
rosrun rl_training_validation vx300s_reach_validate_sim.py --episodes 20 --seed 10 --eval-seed 1000

# Sim pnp — pass the same task-variant flags you trained with so the
# evaluator runs the matching curriculum, not a sibling variant
rosrun rl_training_validation rx200_pnp_validate_sim.py --episodes 20 --goal --multi-goal
rosrun rl_training_validation vx300s_pnp_validate_sim.py --episodes 20 --goal --multi-goal --no-realtime

# Real
rosrun rl_training_validation rx200_push_validate_real.py --episodes 10 --allow-real-robot-motion
rosrun rl_training_validation rx200_pnp_validate_real.py  --episodes 10 --allow-real-robot-motion --goal --multi-goal
```

The pnp sim validate scripts (`rx200`, `ur5e`, `vx300s`) accept the
same `--multi-goal` / `--no-realtime` flags as their train
counterparts; without them a policy trained with the intermediate-
lift curriculum would be evaluated on the simpler single-goal
variant and the reported success rate would not correspond to the
trained task.

## Repository layout

```
src/rl_training_validation/
  utils/
    env_safety.py            # registry-based env classification + real-motion gate
    multi_task_env.py        # multi-task wrapper used by multi_train_sim
    multi_task_goal_env.py
  rx200/  ned2/  vx300s/  ur5e/
    reach/  push/  pnp/      # per-task train + validate scripts
  multi_task_learning/
    multi_train_sim.py

config/
  rx200_*.yaml  ned2_*.yaml  vx300s_*.yaml  ur5e_*.yaml  # SB3 hyperparams

scripts/
  list_available_envs.py
  check_env_availability.py
  check_goal_training_setup.py
  smoke_test_training_config.py
  live_smoke_envs.py
```

## Safety contract

- Real-robot trainers require the explicit CLI flag
  `--allow-real-robot-motion`. `check_env_constructable` refuses to
  build any `...Real-v0` env without it.
- Goal-conditioned env ids (`...Goal{Sim,Real}-v0`) are routed to HER
  algorithms (TD3_GOAL / SAC_GOAL) automatically by the train scripts.
- Sim envs run with per-link FK safety in `execute_action` — every
  joint trajectory target is checked link-by-link against the table
  floor before publishing (`_check_action_links_safe` in the
  per-robot robot envs).

## Contact

[j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie)

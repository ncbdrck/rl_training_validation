# RL Training & Validation — UniROS Prebuilt Environments

[![Documentation Status](https://readthedocs.org/projects/uniros/badge/?version=latest)](https://uniros.readthedocs.io/en/latest/?badge=latest)

📚 **Full documentation**: [uniros.readthedocs.io](https://uniros.readthedocs.io/)

Train / validate the Gymnasium prebuilt tasks from
[rl_environments](https://github.com/ncbdrck/rl_environments) using
Stable Baselines3 + [sb3_ros_support](https://github.com/ncbdrck/sb3_ros_support).

> **Audited implementation status** (kept in lockstep with
> `rl_environments`):
> **14 / 48** env ids in the UniROS matrix are actually implemented today;
> the rest are registered with `UnimplementedRLEnv` and refuse to construct.
> See `python3 scripts/list_available_envs.py`.
>
> | Robot | Sim Reach | Sim Push/PnP/Slide | Real Reach | Real Push/PnP/Slide |
> |-------|-----------|--------------------|------------|---------------------|
> | RX200 | ✅ std + goal | ✅ std + goal | ✅ std + goal (gated) | ❌ stub |
> | Ned2  | ✅ std + goal | ❌ stub | ❌ stub | ❌ stub |
> | UR5   | ✅ std + goal | ❌ stub | ❌ stub | ❌ stub |
>
> "Stub" means the script prints a clear message and exits without
> constructing the env. Lift a stub only after the corresponding env
> class is properly implemented and `env_status.is_implemented` is
> flipped to True.

## Prerequisites

1. **`rl_environments`** — install from the audited
   `feature/audit-robot-limits-object-spawning-safety` branch (or
   later) in the same `catkin_ws`. The training scripts import
   `rl_environments.common.env_status` and `.common.safety`.
2. **`sb3_ros_support`** —
   <https://github.com/ncbdrck/sb3_ros_support> (`gymnasium` branch).
3. **`UniROS`** (`multiros` + `realros`) — provides the Gazebo /
   real-robot base envs and the action / observation wrappers used by
   every training script in this repo.

## Quick reference

### What can I train right now?

```bash
# List every env id in the registry and its implementation status:
python3 scripts/list_available_envs.py

# Only the ones that work today:
python3 scripts/list_available_envs.py --only-implemented
```

Currently:

```
UniROS-RX200ReachSim-v0          UniROS-RX200ReachGoalSim-v0
UniROS-RX200PushSim-v0           UniROS-RX200PushGoalSim-v0
UniROS-RX200PnPSim-v0            UniROS-RX200PnPGoalSim-v0
UniROS-RX200SlideSim-v0          UniROS-RX200SlideGoalSim-v0
UniROS-Ned2ReachSim-v0           UniROS-Ned2ReachGoalSim-v0
UniROS-UR5ReachSim-v0            UniROS-UR5ReachGoalSim-v0
UniROS-RX200ReachReal-v0         UniROS-RX200ReachGoalReal-v0   (real, gated)
```

### Smoke-check the repo

```bash
# Pure introspection, no Gazebo, no hardware:
python3 scripts/smoke_test_training_config.py
python3 scripts/check_env_availability.py
python3 scripts/check_goal_training_setup.py
```

### Train: RX200 sim Reach

In separate terminals:

```bash
# 1. ROS master
roscore

# 2. Gazebo (empty world is fine; the env spawns the RX200 URDF itself)
roslaunch gazebo_ros empty_world.launch

# 3. Trainer (TD3 by default; Reach also supports --algo sac)
rosrun rl_training_validation rx200_reach_train_sim.py
```

### Train: RX200 sim object tasks

RX200 Push, Pick-and-Place, and Slide are wired for TD3 only in this
repo. Use `--goal` to switch to the goal-conditioned env and TD3+HER.

```bash
rosrun rl_training_validation rx200_push_train_sim.py
rosrun rl_training_validation rx200_pnp_train_sim.py --goal
rosrun rl_training_validation rx200_slide_train_sim.py
```

### Train: goal envs

```bash
rosrun rl_training_validation rx200_reach_train_sim.py --goal
rosrun rl_training_validation ned2_reach_train_sim.py --goal
rosrun rl_training_validation ur5_reach_train_sim.py --goal
```

HER is only wired into the `*_GOAL` algorithms in `sb3_ros_support`.
Adding `--goal` switches the env id, the wrapper kwargs (so the dict
observation is normalised correctly), and the algorithm class.

### Train: RX200 real Reach

Real-robot motion is **double-gated**:

1. The CLI flag ``--allow-real-robot-motion`` (this script).
2. Either ROS param ``/allow_real_robot_motion=true`` OR env var
   ``ALLOW_REAL_ROBOT_MOTION=1`` (checked again inside the env's
   constructor and again before every joint trajectory publish).

Without **both**, the script aborts and the real interbotix driver is
**not** launched.

```bash
# Set the rosparam ONCE before running:
rosparam set /allow_real_robot_motion true

# Now run the trainer with the CLI flag:
rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion
```

### Validate a trained policy

```bash
# Sim
rosrun rl_training_validation rx200_reach_validate_sim.py --episodes 20

# Real (with both gates set as above)
rosrun rl_training_validation rx200_reach_validate_real.py --episodes 10 \
    --allow-real-robot-motion
```

## Repository layout

```
src/rl_training_validation/
  utils/
    env_safety.py            # implementation-status + real-motion CLI helpers
    multi_task_env.py        # multi-task wrapper used by multi_train_sim
  _blocked_stub.py           # shared 'blocked env id' bail-out
  rx200/ned2/ur5/
    reach/  push/  pnp/  slide/  # per-task train + validate scripts/stubs
  multi_task_learning/
    multi_train_sim.py
config/
  rx200_*.yaml               # SB3 hyperparams for each algo / env

scripts/
  list_available_envs.py     # iterate the registry
  check_env_availability.py  # cross-check every referenced env id is implemented
  check_goal_training_setup.py  # API contract for goal envs (HER)
  smoke_test_training_config.py # full audit pass
```

## Safety contract

* Real-robot trainers require both `--allow-real-robot-motion` AND a
  `/allow_real_robot_motion`-style consent flag visible to the env-side
  safety module. See
  [`rl_environments/src/rl_environments/common/safety.py`](https://github.com/ncbdrck/rl_environments).
* Goal-conditioned env ids must only be passed to `*_GOAL` algorithms
  (HER). The training scripts in this repo route them automatically;
  scripts/check_goal_training_setup.py audits that the API contract
  holds for every implemented goal env.
* Blocked env ids (Ned2 push, UR5 pnp, etc.) refuse to construct via
  `UnimplementedRLEnv`. The corresponding `*_train/validate_*.py`
  scripts are intentional stubs that print a clear message and exit.
  They should be replaced with real scripts after the underlying env
  is implemented and `env_status.is_implemented` is flipped to True.
* Validation scripts surface every `info["sensor_timeout"]=True` (cube
  perception timeout) and `info["is_success"]` so success rates are
  tracked correctly.
* The smoke test checks that every `CFG_* = "*.yaml"` referenced by a
  non-stub train/validate script exists in `config/`.

## Documentation

The ecosystem documentation lives in
[`UniROS/docs/`](https://github.com/ncbdrck/UniROS/tree/main/docs).

## Contact

For questions, suggestions, or collaborations:
[j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie).

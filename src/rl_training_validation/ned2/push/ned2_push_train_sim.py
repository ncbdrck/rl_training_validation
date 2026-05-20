#!/usr/bin/env python3
"""
Train an SB3 policy on the Ned2 sim Push task.

Standard env id:  ``UniROS-Ned2PushSim-v0``
Goal env id:      ``UniROS-Ned2PushGoalSim-v0``

Requires Gazebo + roscore to already be running, with a world that
contains the Niryo workspace_1 pad. Recommended:

    export GAZEBO_MODEL_PATH=$(rospack find niryo_robot_gazebo)/models:$GAZEBO_MODEL_PATH
    roslaunch gazebo_ros empty_world.launch \\
        world_name:=$(rospack find rl_environments)/worlds/ned2_workspace_only.world

That world ships with rl_environments and contains workspace_1 +
workspace_grille and NOTHING else, so the env's
``spawn_cube_in_gazebo("red_cube")`` call has no name collision.
The bundled ``niryo_cube_world.world`` would conflict on the first
spawn because it pre-bakes cube_blue / cube_red / cube_green.

Mirrors the RX200 push train script — same TD3 / TD3_GOAL pipeline,
same wrappers (normalise action, normalise obs incl. goal spaces for
the goal variant, time limit).
"""
from __future__ import annotations

import argparse
import sys

import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # paper §6.1: subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_real_motion_cli, check_env_constructable, is_goal_env,
)

from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD  = "NED2PushSim-v0"
ENV_GOAL = "NED2PushGoalSim-v0"
CFG_STD  = "ned2_push_td3.yaml"
CFG_GOAL = "ned2_push_td3_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true",
                   help="Use the goal-conditioned env + HER.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--gazebo-gui", action="store_true")
    p.add_argument("--reward-type", default=None)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = ENV_GOAL if args.goal else ENV_STD
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    env_kwargs = dict(
        seed=args.seed,
        gazebo_gui=args.gazebo_gui,
        ee_action_type=False,
        delta_action=True,
        environment_loop_rate=10.0,
        action_cycle_time=0.500,
        use_smoothing=False,
        action_speed=0.100,
        log_internal_state=False,
    )
    if args.reward_type:
        env_kwargs["reward_type"] = args.reward_type
    elif is_goal_env(env_id):
        env_kwargs["reward_type"] = "Sparse"
    else:
        env_kwargs["reward_type"] = "Dense"

    env = gym.make(env_id, **env_kwargs)
    env = NormalizeActionWrapper(env)
    if is_goal_env(env_id):
        env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)
    else:
        env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)
    env.reset()

    pkg_path = "rl_training_validation"
    if args.goal:
        cfg = CFG_GOAL
        save_path = "/models/sim/td3_goal/ned2/push/"
        log_path  = "/logs/sim/td3_goal/ned2/push/"
        ModelCls = TD3_GOAL
    else:
        cfg = CFG_STD
        save_path = "/models/sim/td3/ned2/push/"
        log_path  = "/logs/sim/td3/ned2/push/"
        ModelCls = TD3

    model = ModelCls(env, save_path, log_path, model_pkg_path=pkg_path,
                     config_file_pkg=pkg_path, config_filename=cfg)
    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

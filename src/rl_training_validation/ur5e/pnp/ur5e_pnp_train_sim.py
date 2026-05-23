#!/usr/bin/env python3
"""
Train an SB3 policy on the UR5e sim Pick-and-Place task.

Standard env id:  ``UR5ePnPSim-v0``
Goal env id:      ``UR5ePnPGoalSim-v0``

Requires Gazebo + roscore to already be running. The env class
launches the appropriate MoveIt stack itself.
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


ENV_STD  = "UR5ePnPSim-v0"
ENV_GOAL = "UR5ePnPGoalSim-v0"
CFG_STD  = "ur5e_pnp_td3.yaml"
CFG_GOAL = "ur5e_pnp_td3_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true",
                   help="Use the goal-conditioned env + HER.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--gazebo-gui", action="store_true")
    p.add_argument("--reward-type", default=None)
    p.add_argument("--multi-goal", action="store_true",
                   help="Enable the intermediate-lift goal curriculum.")
    p.add_argument("--no-realtime", action="store_true",
                   help="Use the standard MDP pause-step-resume loop instead "
                        "of the paper §7 real-time loop.")
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
        multi_goal=args.multi_goal,
        realtime_mode=not args.no_realtime,
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
        save_path = "/models/sim/td3_goal/ur5e/pnp/"
        log_path  = "/logs/sim/td3_goal/ur5e/pnp/"
        ModelCls = TD3_GOAL
    else:
        cfg = CFG_STD
        save_path = "/models/sim/td3/ur5e/pnp/"
        log_path  = "/logs/sim/td3/ur5e/pnp/"
        ModelCls = TD3

    model = ModelCls(env, save_path, log_path, model_pkg_path=pkg_path,
                     config_file_pkg=pkg_path, config_filename=cfg)
    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Train an SB3 policy on the UR5e *real* Reach task.

This is the explicitly-opt-in real-robot trainer. Real motion is
gated by ``check_env_constructable``: it refuses to construct any
``...Real`` env unless ``--allow-real-robot-motion`` is passed on the
command line. The helper also exports ``ALLOW_REAL_ROBOT_MOTION=1``
so downstream code can read consent from a single source — that env
var is a propagation of the same gate, not an independent channel.

You MUST also have:
  * the actual UR5e connected and powered up,
  * the interbotix MoveIt / driver dependencies installed,
  * (optional) ``rosparam set /allow_real_robot_motion true`` if your
    launch chain prefers to query the parameter server.

Default behaviour without ``--allow-real-robot-motion`` is a clear
SystemExit with no motion.
"""
from __future__ import annotations

import argparse
import sys

import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # paper §6.1: subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_real_motion_cli, check_env_constructable, is_goal_env, with_seed_suffix,
)

from sb3_ros_support.sac import SAC
from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL
from sb3_ros_support.sac_goal import SAC_GOAL

from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true")
    p.add_argument("--algo", default="td3", choices=("td3", "sac"))
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--reward-type", default=None)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = "UR5eReacherGoalReal-v0" if args.goal else "UR5eReacherReal-v0"
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    env_kwargs = dict(
        seed=args.seed,
        delta_action=True,
        ee_action_type=False,
        environment_loop_rate=10.0,
        action_cycle_time=0.500,
        use_smoothing=False,
        action_speed=0.100,
        log_internal_state=False,
    )
    if args.reward_type:
        env_kwargs["reward_type"] = args.reward_type
    elif args.goal:
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
        if args.algo == "td3":
            config_file_name = "ur5e_reacher_td3_goal.yaml"
            save_path = "/models/real/td3_goal/ur5e/reach/"
            log_path = "/logs/real/td3_goal/ur5e/reach/"
            save_path = with_seed_suffix(save_path, args.seed)
            log_path = with_seed_suffix(log_path, args.seed)
            model = TD3_GOAL(env, save_path, log_path, model_pkg_path=pkg_path,
                             config_file_pkg=pkg_path, config_filename=config_file_name,
                             seed=args.seed)
        else:
            config_file_name = "ur5e_reacher_sac_goal.yaml"
            save_path = "/models/real/sac_goal/ur5e/reach/"
            log_path = "/logs/real/sac_goal/ur5e/reach/"
            save_path = with_seed_suffix(save_path, args.seed)
            log_path = with_seed_suffix(log_path, args.seed)
            model = SAC_GOAL(env, save_path, log_path, model_pkg_path=pkg_path,
                             config_file_pkg=pkg_path, config_filename=config_file_name,
                             seed=args.seed)
    else:
        if args.algo == "td3":
            config_file_name = "ur5e_reacher_td3.yaml"
            save_path = "/models/real/td3/ur5e/reach/"
            log_path = "/logs/real/td3/ur5e/reach/"
            save_path = with_seed_suffix(save_path, args.seed)
            log_path = with_seed_suffix(log_path, args.seed)
            model = TD3(env, save_path, log_path, model_pkg_path=pkg_path,
                        config_file_pkg=pkg_path, config_filename=config_file_name,
                        seed=args.seed)
        else:
            config_file_name = "ur5e_reacher_sac.yaml"
            save_path = "/models/real/sac/ur5e/reach/"
            log_path = "/logs/real/sac/ur5e/reach/"
            save_path = with_seed_suffix(save_path, args.seed)
            log_path = with_seed_suffix(log_path, args.seed)
            model = SAC(env, save_path, log_path, model_pkg_path=pkg_path,
                        config_file_pkg=pkg_path, config_filename=config_file_name,
                        seed=args.seed)

    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

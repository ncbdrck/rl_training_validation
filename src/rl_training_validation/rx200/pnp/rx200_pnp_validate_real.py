#!/usr/bin/env python3
"""
Validate a trained SB3 policy on the RX200 *real* Pick-and-Place task.

Loads a previously-trained policy (TD3 or TD3+HER for --goal) and
runs the requested number of episodes on the real robot.

See rx200_pnp_train_real.py for the safety gating, cube-tracking,
and ``--multi-goal`` prerequisites.
"""
from __future__ import annotations

import argparse
import sys

import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # paper §6.1: subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_cube_tracker_cli, add_real_motion_cli, apply_cube_tracker_kwargs,
    check_env_constructable, is_goal_env,
)

from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL

from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD = "RX200PnPReal-v0"
ENV_GOAL = "RX200PnPGoalReal-v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--multi-goal", action="store_true")
    p.add_argument("--cube-pose-topic", default="/cube_pose")
    add_cube_tracker_cli(p)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = ENV_GOAL if args.goal else ENV_STD
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    env_kwargs = dict(
        delta_action=True,
        ee_action_type=False,
        environment_loop_rate=10.0,
        action_cycle_time=0.500,
        use_smoothing=False,
        action_speed=0.100,
        log_internal_state=False,
        multi_goal=args.multi_goal,
        cube_pose_topic=args.cube_pose_topic,
    )
    apply_cube_tracker_kwargs(env_kwargs, args)

    env = gym.make(env_id, **env_kwargs)
    env = NormalizeActionWrapper(env)
    if is_goal_env(env_id):
        env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)
    else:
        env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)

    pkg_path = "rl_training_validation"
    if args.goal:
        save_path = "/models/real/td3_goal/rx200/pnp/"
        log_path = "/logs/real/td3_goal/rx200/pnp/"
        ModelCls = TD3_GOAL
        cfg = "rx200_pnp_td3_goal.yaml"
    else:
        save_path = "/models/real/td3/rx200/pnp/"
        log_path = "/logs/real/td3/rx200/pnp/"
        ModelCls = TD3
        cfg = "rx200_pnp_td3.yaml"

    model = ModelCls(env, save_path, log_path, model_pkg_path=pkg_path,
                     config_file_pkg=pkg_path, config_filename=cfg)

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
        ok = bool(info.get("is_success", False))
        successes += int(ok)
        rospy.loginfo(f"[validate] ep {ep + 1}/{args.episodes} success={ok} reward={ep_reward:.3f}")

    env.close()
    rospy.loginfo(f"[validate] success rate: {successes}/{args.episodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Train an SB3 policy on the VX300S *real* Push task.

Real motion is gated by ``check_env_constructable``: it refuses to
construct any ``...Real`` env unless ``--allow-real-robot-motion`` is
passed on the command line. The helper also exports
``ALLOW_REAL_ROBOT_MOTION=1`` so downstream code can read consent
from a single source — that env var is a propagation of the same
gate, not an independent channel.

Real push additionally requires a vision pipeline publishing the cube
pose on ``geometry_msgs/PoseStamped`` topic (default ``/cube_pose``).
Without one the env falls back to the YAML ``cube_init_pos`` and
prints a throttled warning — usable for dry-runs but obviously not
for training. Wire up aruco_ros / AprilTag / mocap / deep detector
of your choice before training for real.

Default behaviour without ``--allow-real-robot-motion`` is a clear
SystemExit with no motion.
"""
from __future__ import annotations

import argparse
import sys

import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_cube_tracker_cli, add_real_motion_cli, apply_cube_tracker_kwargs,
    check_env_constructable, is_goal_env, with_seed_suffix,
)

from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL

from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD = "VX300SPushReal-v0"
ENV_GOAL = "VX300SPushGoalReal-v0"
CFG_STD = "vx300s_push_td3.yaml"
CFG_GOAL = "vx300s_push_td3_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true",
                   help="Use the goal-conditioned env + HER.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--reward-type", default=None)
    p.add_argument("--cube-pose-topic", default="/cube_pose",
                   help="Topic publishing the cube's geometry_msgs/PoseStamped (default /cube_pose).")
    add_cube_tracker_cli(p)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = ENV_GOAL if args.goal else ENV_STD
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
        cube_pose_topic=args.cube_pose_topic,
    )
    apply_cube_tracker_kwargs(env_kwargs, args)
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
        save_path = "/models/real/td3_goal/vx300s/push/"
        log_path = "/logs/real/td3_goal/vx300s/push/"
        ModelCls = TD3_GOAL
    else:
        cfg = CFG_STD
        save_path = "/models/real/td3/vx300s/push/"
        log_path = "/logs/real/td3/vx300s/push/"
        ModelCls = TD3

    save_path = with_seed_suffix(save_path, args.seed)
    log_path = with_seed_suffix(log_path, args.seed)
    model = ModelCls(env, save_path, log_path, model_pkg_path=pkg_path,
                     config_file_pkg=pkg_path, config_filename=cfg,
                     seed=args.seed)
    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Validate a trained SB3 policy on the Ned2 *real* Push task.

Loads a previously-trained policy (TD3 or TD3+HER for --goal) and
runs the requested number of episodes on the real Ned2 robot.

See ned2_push_train_real.py for the safety gating + cube tracking
prerequisites — both also apply here.
"""
from __future__ import annotations

import argparse
import os
import sys

import rospkg
import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # paper §6.1: subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_cube_tracker_cli, add_goal_pose_cli, add_real_motion_cli,
    add_wrist_camera_cli, apply_cube_tracker_kwargs, apply_goal_pose_kwargs,
    apply_wrist_camera_kwargs,
    check_env_constructable, is_goal_env, with_seed_suffix,
)

from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL

from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD = "NED2PushReal-v0"
ENV_GOAL = "NED2PushGoalReal-v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--cube-pose-topic", default="/cube_pose")
    p.add_argument("--model-tag", default="trained_model",
                   help="Filename stem under the model directory (no .zip).")
    add_cube_tracker_cli(p)
    add_goal_pose_cli(p)
    add_wrist_camera_cli(p)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = ENV_GOAL if args.goal else ENV_STD
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    # Verify the model exists BEFORE bringing up the real robot.
    pkg_path = "rl_training_validation"
    if args.goal:
        base = "/models/real/td3_goal/ned2/push/"
        ModelCls = TD3_GOAL
        cfg = "ned2_push_td3_goal.yaml"
    else:
        base = "/models/real/td3/ned2/push/"
        ModelCls = TD3
        cfg = "ned2_push_td3.yaml"
    base = with_seed_suffix(base, args.seed)
    rel_model_path = base + args.model_tag
    abs_model_path = rospkg.RosPack().get_path(pkg_path) + rel_model_path
    if not os.path.exists(abs_model_path + ".zip"):
        raise SystemExit(
            f"[validate] trained model not found at {abs_model_path}.zip. "
            "Either pass --model-tag <name> matching a file you trained, "
            "or run ned2_push_train_real.py first."
        )

    env_kwargs = dict(
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
    apply_goal_pose_kwargs(env_kwargs, args)
    apply_wrist_camera_kwargs(env_kwargs, args)

    env = gym.make(env_id, **env_kwargs)
    env = NormalizeActionWrapper(env)
    if is_goal_env(env_id):
        env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)
    else:
        env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)

    # load_trained_model() routes through TD3(load_trained=True), which
    # calls stable_baselines3.TD3.load(...) instead of constructing a
    # fresh untrained policy.
    model = ModelCls.load_trained_model(
        model_path=rel_model_path,
        model_pkg=pkg_path,
        config_filename=cfg,
        env=env,
    )

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
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

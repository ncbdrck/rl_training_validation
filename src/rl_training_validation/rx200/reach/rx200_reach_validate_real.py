#!/usr/bin/env python3
"""
Validate a trained policy against the RX200 *real* Reach task.

Same real-robot double-gating as ``rx200_reach_train_real``.
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
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model-tag", default="trained_model")
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = "RX200ReacherGoalReal-v0" if args.goal else "RX200ReacherReal-v0"
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
        reward_type="Sparse" if args.goal else "Dense",
    )
    env = gym.make(env_id, **env_kwargs)
    env = NormalizeActionWrapper(env)
    if is_goal_env(env_id):
        env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)
    else:
        env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)

    pkg_path = "rl_training_validation"
    if args.goal:
        if args.algo == "td3":
            cfg = "rx200_reacher_td3_goal.yaml"
            model_path = "/models/real/td3_goal/rx200/reach/" + args.model_tag
            model = TD3_GOAL.load_trained_model(model_path=model_path, model_pkg=pkg_path,
                                                config_filename=cfg, env=env)
        else:
            cfg = "rx200_reacher_sac_goal.yaml"
            model_path = "/models/real/sac_goal/rx200/reach/" + args.model_tag
            model = SAC_GOAL.load_trained_model(model_path=model_path, model_pkg=pkg_path,
                                                config_filename=cfg, env=env)
    else:
        if args.algo == "td3":
            cfg = "rx200_reacher_td3.yaml"
            model_path = "/models/real/td3/rx200/reach/" + args.model_tag
            model = TD3.load_trained_model(model_path=model_path, model_pkg=pkg_path,
                                           config_filename=cfg, env=env)
        else:
            cfg = "rx200_reacher_sac.yaml"
            model_path = "/models/real/sac/rx200/reach/" + args.model_tag
            model = SAC.load_trained_model(model_path=model_path, model_pkg=pkg_path,
                                           config_filename=cfg, env=env)

    obs, _ = env.reset()
    successes = 0
    truncs = 0
    timeouts = 0
    for ep in range(args.episodes):
        ep_done = False
        ep_success = False
        while not ep_done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("sensor_timeout"):
                timeouts += 1
            if info.get("is_success"):
                ep_success = True
            if terminated or truncated:
                ep_done = True
                if truncated and not terminated:
                    truncs += 1
        if ep_success:
            successes += 1
        rospy.loginfo(f"Episode {ep + 1}/{args.episodes} success={ep_success}")
        obs, _ = env.reset()

    print(f"\nResults over {args.episodes} episodes:")
    print(f"  success rate:        {successes}/{args.episodes} = {100*successes/args.episodes:.1f}%")
    print(f"  truncated (no term): {truncs}")
    print(f"  sensor_timeout flags: {timeouts}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

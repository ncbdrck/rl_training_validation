#!/usr/bin/env python3
"""
Validate a trained policy against the Ned2 *real* Reach task.

Same single-channel CLI gate as ``ned2_reach_train_real``.
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
    add_real_motion_cli, add_wrist_camera_cli, apply_wrist_camera_kwargs,
    check_env_constructable, is_goal_env,
)

from sb3_ros_support.sac import SAC
from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL
from sb3_ros_support.sac_goal import SAC_GOAL

from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD = "NED2ReacherReal-v0"
ENV_GOAL = "NED2ReacherGoalReal-v0"
CFG_STD_TD3 = "rx200_reacher_td3.yaml"
CFG_STD_SAC = "rx200_reacher_sac.yaml"
CFG_GOAL_TD3 = "rx200_reacher_td3_goal.yaml"
CFG_GOAL_SAC = "rx200_reacher_sac_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true")
    p.add_argument("--algo", default="td3", choices=("td3", "sac"))
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model-tag", default="trained_model")
    add_wrist_camera_cli(p)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env_id = ENV_GOAL if args.goal else ENV_STD
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    # Resolve the trained-model path BEFORE bringing up the real robot —
    # mirrors the push/pnp validate_real pattern.
    pkg_path = "rl_training_validation"
    if args.goal:
        cfg = CFG_GOAL_TD3 if args.algo == "td3" else CFG_GOAL_SAC
        base = "/models/real/td3_goal/ned2/reach/" if args.algo == "td3" else "/models/real/sac_goal/ned2/reach/"
        ModelCls = TD3_GOAL if args.algo == "td3" else SAC_GOAL
    else:
        cfg = CFG_STD_TD3 if args.algo == "td3" else CFG_STD_SAC
        base = "/models/real/td3/ned2/reach/" if args.algo == "td3" else "/models/real/sac/ned2/reach/"
        ModelCls = TD3 if args.algo == "td3" else SAC
    rel_model_path = base + args.model_tag
    abs_model_path = rospkg.RosPack().get_path(pkg_path) + rel_model_path
    if not os.path.exists(abs_model_path + ".zip"):
        raise SystemExit(
            f"[validate] trained model not found at {abs_model_path}.zip. "
            "Either pass --model-tag <name> matching a file you trained, "
            "or run ned2_reach_train_real.py first."
        )

    env_kwargs = dict(
        seed=args.seed,
        delta_action=True,
        ee_action_type=False,
        use_smoothing=False,
        action_speed=0.100,
        log_internal_state=False,
        reward_type="Sparse" if args.goal else "Dense",
    )
    apply_wrist_camera_kwargs(env_kwargs, args)

    env = gym.make(env_id, **env_kwargs)
    env = NormalizeActionWrapper(env)
    if is_goal_env(env_id):
        env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)
    else:
        env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)

    model = ModelCls.load_trained_model(
        model_path=rel_model_path,
        model_pkg=pkg_path,
        config_filename=cfg,
        env=env,
    )

    obs, _ = env.reset()
    successes, truncs, timeouts = 0, 0, 0
    for ep in range(args.episodes):
        done = False
        ep_success = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("sensor_timeout"):
                timeouts += 1
            if info.get("is_success"):
                ep_success = True
            if terminated or truncated:
                done = True
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

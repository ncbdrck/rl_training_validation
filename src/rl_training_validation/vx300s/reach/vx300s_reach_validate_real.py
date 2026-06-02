#!/usr/bin/env python3
"""
Validate a trained policy against the VX300S *real* Reach task.

Same single-channel CLI gate as ``vx300s_reach_train_real``.
"""
from __future__ import annotations

import argparse
import os
import sys

import rospkg
import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # subprocess-isolated env proxy; drop-in for gym.Env

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
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="RNG seed for the evaluation env, independent of --seed "
                        "(which selects the trained-policy directory). Picking a "
                        "value far from the training --seed ensures evaluation "
                        "goals are sampled from a held-out stream rather than the "
                        "same distribution the policy was trained on.")
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model-tag", default="trained_model")
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rospy.loginfo(f"[validate] model_seed={args.seed} eval_seed={args.eval_seed}")
    env_id = "VX300SReacherGoalReal-v0" if args.goal else "VX300SReacherReal-v0"
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    # Resolve the trained-model path BEFORE bringing up the real robot —
    # mirrors the push/pnp validate_real pattern. Failing fast on a
    # missing file saves a 20+ second driver bring-up just to hit
    # FileNotFoundError mid-run.
    pkg_path = "rl_training_validation"
    if args.goal:
        if args.algo == "td3":
            cfg = "vx300s_reacher_td3_goal.yaml"
            base = "/models/real/td3_goal/vx300s/reach/"
            ModelCls = TD3_GOAL
        else:
            cfg = "vx300s_reacher_sac_goal.yaml"
            base = "/models/real/sac_goal/vx300s/reach/"
            ModelCls = SAC_GOAL
    else:
        if args.algo == "td3":
            cfg = "vx300s_reacher_td3.yaml"
            base = "/models/real/td3/vx300s/reach/"
            ModelCls = TD3
        else:
            cfg = "vx300s_reacher_sac.yaml"
            base = "/models/real/sac/vx300s/reach/"
            ModelCls = SAC
    base = with_seed_suffix(base, args.seed)
    rel_model_path = base + args.model_tag
    abs_model_path = rospkg.RosPack().get_path(pkg_path) + rel_model_path
    if not os.path.exists(abs_model_path + ".zip"):
        raise SystemExit(
            f"[validate] trained model not found at {abs_model_path}.zip. "
            "Either pass --model-tag <name> matching a file you trained, "
            "or run vx300s_reach_train_real.py first."
        )

    env_kwargs = dict(
        seed=args.eval_seed,
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

    model = ModelCls.load_trained_model(
        model_path=rel_model_path,
        model_pkg=pkg_path,
        config_filename=cfg,
        env=env,
    )

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

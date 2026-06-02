#!/usr/bin/env python3
"""
Validate a trained policy against the Ned2 sim Pick-and-Place task.

Mirrors the train script for env construction; loads a saved model
and rolls it out for ``--episodes`` episodes, logging success rate,
truncations, and sensor timeouts from ``info``.
"""
from __future__ import annotations

import argparse
import sys

import rospy
# import gymnasium as gym  # uncomment + comment uniros below to test against vanilla Gymnasium
import uniros as gym  # subprocess-isolated env proxy; drop-in for gym.Env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_real_motion_cli, add_wrist_camera_cli, apply_wrist_camera_kwargs,
    check_env_constructable, is_goal_env, with_seed_suffix,
)

from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD  = "NED2PnPSim-v0"
ENV_GOAL = "NED2PnPGoalSim-v0"
CFG_STD  = "ned2_pnp_td3.yaml"
CFG_GOAL = "ned2_pnp_td3_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="RNG seed for the evaluation env, independent of --seed "
                        "(which selects the trained-policy directory). Picking a "
                        "value far from the training --seed ensures evaluation "
                        "goals are sampled from a held-out stream rather than the "
                        "same distribution the policy was trained on.")
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--gazebo-gui", action="store_true")
    p.add_argument("--model-tag", default="trained_model")
    add_wrist_camera_cli(p)
    add_real_motion_cli(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rospy.loginfo(f"[validate] model_seed={args.seed} eval_seed={args.eval_seed}")
    env_id = ENV_GOAL if args.goal else ENV_STD
    check_env_constructable(env_id, allow_real_flag=args.allow_real_robot_motion)

    env_kwargs = dict(
        seed=args.eval_seed,
        gazebo_gui=args.gazebo_gui,
        ee_action_type=False,
        delta_action=True,
        environment_loop_rate=10.0,
        action_cycle_time=0.500,
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

    pkg_path = "rl_training_validation"
    if args.goal:
        cfg = CFG_GOAL
        base = "/models/sim/td3_goal/ned2/pnp/"
        ModelCls = TD3_GOAL
    else:
        cfg = CFG_STD
        base = "/models/sim/td3/ned2/pnp/"
        ModelCls = TD3
    base = with_seed_suffix(base, args.seed)
    model_path = base + args.model_tag
    model = ModelCls.load_trained_model(model_path=model_path, model_pkg=pkg_path,
                                        config_filename=cfg, env=env)

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

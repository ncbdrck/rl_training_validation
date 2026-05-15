#!/usr/bin/env python3
"""
Multi-task TD3 training across several UniROS sim envs simultaneously.

By default this uses the two implemented RX200 sim envs that share
similar dynamics: Reach and Push. The user can switch in any other
implemented env via the CLI; the script refuses to construct an
unimplemented env id.

This script is sim-only. Real envs cannot be added to the multi-task
mix from here — they require explicit per-env consent and are not
covered by the multi-task wrapper's resampling logic.
"""
from __future__ import annotations

import argparse
import sys

import rospy
from stable_baselines3.common.env_util import make_vec_env

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    check_env_constructable, is_real, list_implemented,
)
from rl_training_validation.utils.multi_task_env import MultiTaskEnv

from sb3_ros_support.td3 import TD3


DEFAULT_ENVS = [
    "UniROS-RX200ReachSim-v0",
    "UniROS-RX200PushSim-v0",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS,
                   help="Env ids to mix. Must all be sim envs. Default: "
                        f"{DEFAULT_ENVS}")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--config", default="multi_task_td3.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # All envs must be implemented and not real.
    for eid in args.envs:
        if is_real(eid):
            print(f"[multi_train_sim] refusing to add real env {eid} to the "
                  "multi-task mix. Use per-env real trainers instead.",
                  file=sys.stderr)
            return 1
        check_env_constructable(eid, allow_real_flag=False)

    env_args = {
        eid: dict(
            gazebo_gui=False,
            ee_action_type=False,
            seed=args.seed,
            delta_action=True,
            environment_loop_rate=10.0,
            action_cycle_time=0.600,
            use_smoothing=False,
            action_speed=0.100,
            reward_type="Dense",
        )
        for eid in args.envs
    }
    wrapper_list = ["NormalizeActionWrapper", "NormalizeObservationWrapper", "TimeLimitWrapper"]
    wrapper_args = {
        "NormalizeActionWrapper": {},
        "NormalizeObservationWrapper": {"normalize_goal_spaces": True},
        "TimeLimitWrapper": {"max_episode_steps": args.max_episode_steps},
    }

    multi_task_env = MultiTaskEnv(args.envs, env_args, wrapper_list, wrapper_args)
    vec_env = make_vec_env(lambda: multi_task_env, n_envs=1)

    pkg_path = "rl_training_validation"
    save_path = "/models/sim/td3/multi/"
    log_path = "/logs/sim/td3/multi/"

    model = TD3(vec_env, save_path, log_path, model_pkg_path=pkg_path,
                config_file_pkg=pkg_path, config_filename=args.config)
    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

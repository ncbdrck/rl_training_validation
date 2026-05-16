#!/usr/bin/env python3
"""
Train an SB3 policy on the UR5e sim Reach task.

Standard env id:  ``UniROS-UR5eReachSim-v0``
Goal env id:      ``UniROS-UR5eReachGoalSim-v0``

Requires Gazebo + roscore to already be running, e.g.:

    roslaunch gazebo_ros empty_world.launch \
        world_name:=$(rospack find rl_environments)/worlds/ur5_workspace.world

The env class spawns the UR5e + Robotiq 85 gripper from
``rl_environments/urdf/ur5e_robotiq85_grasp.urdf.xacro``
(adds a virtual ``robotiq_grasp_link`` so MoveIt / IK target the
gripper-tip). For Reach the gripper sits closed and is treated as
a fixed end-effector.
"""
from __future__ import annotations

import argparse
import sys

import rospy
import gymnasium as gym

import rl_environments  # noqa: F401  trigger registration

from rl_training_validation.utils.env_safety import (
    add_real_motion_cli, check_env_constructable, is_goal_env,
)

from sb3_ros_support.sac import SAC
from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL
from sb3_ros_support.sac_goal import SAC_GOAL

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_STD  = "UniROS-UR5eReachSim-v0"
ENV_GOAL = "UniROS-UR5eReachGoalSim-v0"
CFG_STD_TD3 = "ur5e_reach_td3.yaml"
CFG_STD_SAC = "ur5e_reach_sac.yaml"
CFG_GOAL_TD3 = "ur5e_reach_td3_goal.yaml"
CFG_GOAL_SAC = "ur5e_reach_sac_goal.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", action="store_true",
                   help="Use the goal-conditioned env + HER.")
    p.add_argument("--algo", default="td3", choices=("td3", "sac"))
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--gazebo-gui", action="store_true")
    p.add_argument("--reward-type", default=None)
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
        cfg = CFG_GOAL_TD3 if args.algo == "td3" else CFG_GOAL_SAC
        save_path = "/models/sim/td3_goal/ur5e/reach/" if args.algo == "td3" else "/models/sim/sac_goal/ur5e/reach/"
        log_path  = "/logs/sim/td3_goal/ur5e/reach/"   if args.algo == "td3" else "/logs/sim/sac_goal/ur5e/reach/"
        ModelCls = TD3_GOAL if args.algo == "td3" else SAC_GOAL
    else:
        cfg = CFG_STD_TD3 if args.algo == "td3" else CFG_STD_SAC
        save_path = "/models/sim/td3/ur5e/reach/" if args.algo == "td3" else "/models/sim/sac/ur5e/reach/"
        log_path  = "/logs/sim/td3/ur5e/reach/"   if args.algo == "td3" else "/logs/sim/sac/ur5e/reach/"
        ModelCls = TD3 if args.algo == "td3" else SAC

    model = ModelCls(env, save_path, log_path, model_pkg_path=pkg_path,
                     config_file_pkg=pkg_path, config_filename=cfg)
    model.train()
    model.save_model()
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

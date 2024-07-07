#!/bin/python3
import sys

# ROS packages required
import rospy

# gym
import gymnasium as gym
import numpy as np

# We can use the following import statement if we want to use the multiros package
from multiros.utils import ros_common

# Models
from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL
from stable_baselines3.common.env_util import make_vec_env

# wrappers
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper

# multi-task environments
from rl_training_validation.utils.multi_task_env import MultiTaskEnv

# import the environment
import rl_environments

"""
Environments are registered inside the main __init__.py of the rl_environments package
- RX200ReacherSim-v0  # RX200 Reacher Multiros Default Environment
- RX200ReacherGoalSim-v0  # RX200 Reacher Goal Multiros Default Environment
- RX200PushSim-v0  # RX200 Push Multiros Default Environment
"""

if __name__ == '__main__':
    # Kill all processes related to previous runs
    # ros_common.kill_all_ros_and_gazebo()

    # Clear ROS logs
    # ros_common.clean_ros_logs()

    # List of environments to train on
    env_list = ['RX200ReacherSim-v0', 'RX200PushSim-v0']
    env_args_dict = {
        'RX200ReacherSim-v0': {"gazebo_gui": False, "ee_action_type": False, "seed": 10, "delta_action": True,
                               "environment_loop_rate": 10.0, "action_cycle_time": 0.600, "use_smoothing": False,
                               "action_speed": 0.100, "reward_type": "dense"},
        'RX200PushSim-v0': {"gazebo_gui": False, "ee_action_type": False, "seed": 10, "delta_action": True,
                            "environment_loop_rate": 10.0, "action_cycle_time": 0.600, "use_smoothing": False,
                            "action_speed": 0.100, "load_table": True, "random_goal": False, "random_cube_spawn": True,
                            "reward_type": "dense"}
    }
    wrapper_list = ['NormalizeActionWrapper', 'NormalizeObservationWrapper', 'TimeLimitWrapper']
    wrapper_args_dict = {
        'NormalizeActionWrapper': {},
        'NormalizeObservationWrapper': {"normalize_goal_spaces": True},
        'TimeLimitWrapper': {'max_episode_steps': 100}
    }

    # Create the multi-task environment
    multi_task_env = MultiTaskEnv(env_list, env_args_dict, wrapper_list, wrapper_args_dict)

    # Wrap the environment for vectorized processing
    vec_env = make_vec_env(lambda: multi_task_env, n_envs=1)

    # path to the package
    pkg_path = "rl_training_validation"

    # Default base environments - TD3
    config_file_name = "multi_task_td3.yaml"
    save_path = "/models/sim/td3/multi/"
    log_path = "/logs/sim/td3/multi/"

    # create the model - TD3
    model = TD3(vec_env, save_path, log_path, model_pkg_path=pkg_path,
                config_file_pkg=pkg_path, config_filename=config_file_name)

    # # Goal-conditioned environments - TD3+HER
    # config_file_name = "rx200_reacher_td3_goal.yaml"
    # save_path = "/models/sim/td3_goal/multi/"
    # log_path = "/logs/sim/td3_goal/multi/"
    #
    # # create the model
    # model = TD3_GOAL(env, save_path, log_path, model_pkg_path=pkg_path,
    #                  config_file_pkg=pkg_path, config_filename=config_file_name)

    # train the models
    model.train()
    model.save_model()
    model.close_env()

    sys.exit()

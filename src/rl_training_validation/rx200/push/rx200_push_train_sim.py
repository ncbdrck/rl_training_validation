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

# wrappers
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper

# import the environment
import rl_environments

"""
Environments are registered inside the main __init__.py of the rl_environments package
- RX200PushSim-v0  # RX200 Push Multiros Default Environment
"""

if __name__ == '__main__':
    # Kill all processes related to previous runs
    # ros_common.kill_all_ros_and_gazebo()

    # Clear ROS logs
    # ros_common.clean_ros_logs()

    # # --- normal environments
    env = gym.make('RX200PushSim-v0', gazebo_gui=False, ee_action_type=False, seed=10,
                   delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.600,
                   use_smoothing=False, action_speed=0.100, load_table = True,
                   random_goal=False, random_cube_spawn=True)

    # # --- goal environments
    # env = gym.make('RX200PushGoalSim-v0', gazebo_gui=False, ee_action_type=False, seed=10,
    #                delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.600,
    #                use_smoothing=False, action_speed=0.100, load_table = True,
    #                random_goal = False, random_cube_spawn = True)

    # Normalize action space
    env = NormalizeActionWrapper(env)

    # Normalize observation space
    # env = NormalizeObservationWrapper(env)
    env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)  # goal-conditioned environments

    # Set max steps
    env = TimeLimitWrapper(env, max_episode_steps=100)

    # reset the environment
    env.reset()

    # path to the package
    pkg_path = "rl_training_validation"

    # Default base environments - TD3
    config_file_name = "rx200_push_td3.yaml"
    save_path = "/models/sim/td3/rx200/push/"
    log_path = "/logs/sim/td3/rx200/push/"

    # create the model - TD3
    model = TD3(env, save_path, log_path, model_pkg_path=pkg_path,
                config_file_pkg=pkg_path, config_filename=config_file_name)

    # # Goal-conditioned environments - TD3+HER
    # config_file_name = "rx200_push_td3_goal.yaml"
    # save_path = "/models/sim/td3_goal/rx200/push/"
    # log_path = "/logs/sim/td3_goal/rx200/push/"
    #
    # # create the model
    # model = TD3_GOAL(env, save_path, log_path, model_pkg_path=pkg_path,
    #                  config_file_pkg=pkg_path, config_filename=config_file_name)

    # train the models
    model.train()
    model.save_model()
    model.close_env()

    sys.exit()

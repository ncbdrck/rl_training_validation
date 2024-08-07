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
from sb3_ros_support.sac import SAC
from sb3_ros_support.td3 import TD3
from sb3_ros_support.td3_goal import TD3_GOAL
from sb3_ros_support.sac_goal import SAC_GOAL

# wrappers
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper

# import the environment
import rl_environments

"""
Environments are registered inside the main __init__.py of the rl_environments package
- RX200ReacherSim-v0  # RX200 Reacher Multiros Default Environment
- RX200ReacherGoalSim-v0  # RX200 Reacher Goal Multiros Default Environment
"""

if __name__ == '__main__':
    # Kill all processes related to previous runs
    # ros_common.kill_all_ros_processes()


    # Clear ROS logs
    # ros_common.clean_ros_logs()

    # --- normal environments
    env = gym.make('RX200ReacherSim-v0', gazebo_gui=True, ee_action_type=False, seed=10,
                   delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.500,
                   use_smoothing=False, action_speed=0.100, reward_type="dense", log_internal_state=False)

    # # --- goal environments
    # env = gym.make('RX200ReacherGoalSim-v0', gazebo_gui=True, ee_action_type=False, seed=10,
    #                delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.500,
    #                use_smoothing=False, action_speed=0.100, reward_type="sparse", log_internal_state=False)

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

    # # Default base environments - SAC
    # model_path = "/models/sim/sac/rx200/reach/" + "trained_model_mar10"
    # config_file_name = "rx200_reacher_sac.yaml"
    # # Load the model
    # model = SAC.load_trained_model(model_path=model_path, model_pkg=pkg_path, config_filename=config_file_name,
    #                                    env=env)

    # # Goal-conditioned environments - SAC+HER
    # model_path = "/models/sim/sac_goal/rx200/reach/" + "trained_model_mar10"
    # config_file_name = "rx200_reacher_sac_goal.yaml"
    # # Load the model
    # model = SAC_GOAL.load_trained_model(model_path=model_path, model_pkg=pkg_path, config_filename=config_file_name,
    #                                    env=env)

    # Default base environments - TD3
    model_path = "/models/sim/td3/rx200/reach/" + "trained_model_mar10"
    config_file_name = "rx200_reacher_td3.yaml"
    # create the model
    model = TD3.load_trained_model(model_path=model_path, model_pkg=pkg_path, config_filename=config_file_name,
                                   env=env)

    # # Goal-conditioned environments - TD3+HER
    # model_path = "/models/sim/td3_goal/rx200/reach/" + "trained_model_mar10"
    # config_file_name = "rx200_reacher_td3_goal.yaml"
    # # Load the model
    # model = TD3_GOAL.load_trained_model(model_path=model_path, model_pkg=pkg_path, config_filename=config_file_name,
    #                                    env=env)

    obs, _ = env.reset()
    episodes = 1000
    epi_count = 0
    while epi_count < episodes:
        action, _states = model.predict(observation=obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            epi_count += 1
            rospy.logwarn("Episode: " + str(epi_count))
            obs, _ = env.reset()

    env.close()
    sys.exit()
from copy import deepcopy
import gymnasium as gym
from typing import Optional, Union
from gymnasium import spaces
import numpy as np

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


class MultiTaskEnv(gym.Env):
    """
    This class is a wrapper for multiple environments. It is used to train multiple tasks in parallel.
    Please note that this is not MPI parallelism, but rather a way to train multiple tasks with one agent.
    - Made for UniROS based environments
    - Not tested with other environments
    """

    def __init__(self, env_list: list[str], env_args_dict: Optional[dict] = None,
                 wrapper_list: Optional[list] = None, wrapper_args_dict: Optional[dict] = None):
        """
        Args:
            env_list: list of environment names - must be registered in gym
            env_args_dict: dictionary with environment arguments (dict of dicts)
            wrapper_list: list of wrapper names
            wrapper_args_dict: dictionary with wrapper arguments (dict of dicts)
        """
        super(MultiTaskEnv, self).__init__()

        # Initialize optional dictionaries
        if env_args_dict is None:
            env_args_dict = {}
        if wrapper_args_dict is None:
            wrapper_args_dict = {}

        # Create the environments
        self.env_list = []
        for env_name in env_list:
            env = gym.make(env_name, **env_args_dict.get(env_name, {}))
            self.env_list.append(env)

        # Apply wrappers
        if wrapper_list is not None:
            for i in range(len(self.env_list)):
                # Get the environment
                env = self.env_list[i]

                # Apply the wrappers
                for wrapper_name in wrapper_list:
                    if wrapper_name == 'NormalizeActionWrapper':
                        env = NormalizeActionWrapper(env)
                    elif wrapper_name == 'NormalizeObservationWrapper':
                        env = NormalizeObservationWrapper(env, **wrapper_args_dict.get(wrapper_name, {}))
                    elif wrapper_name == 'TimeLimitWrapper':
                        env = TimeLimitWrapper(env, **wrapper_args_dict.get(wrapper_name, {}))
                    else:
                        raise ValueError(f"Wrapper {wrapper_name} not implemented")

                # Update the environment in the list
                self.env_list[i] = env

        # Find maximum observation and action space dimensions
        max_obs_dim = max(env.observation_space.shape[0] for env in self.env_list)
        max_act_dim = max(env.action_space.shape[0] for env in self.env_list)

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max_act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_obs_dim,), dtype=np.float32)

        self.current_env = None

    def step(self, action):
        """
        Step through the environment with the given action
        """
        # Unpad the action
        action = self._unpad_action(action)

        # Step through the environment
        observation, reward, terminated, truncated, info = self.current_env.step(action)

        # Pad the observation to match the maximum observation space dimension
        observation = self._pad_obs(observation)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment by selecting a new environment and resetting it.
        """
        # Reset the current environment - For ROS-based environments, this is important to stop the current environment
        if self.current_env is not None:
            self.current_env.reset()  # we don't need to close it since we may choose it again

        # Randomly select a new environment from the list
        rand_choice = np.random.choice(len(self.env_list))
        self.current_env = self.env_list[rand_choice]

        # Reset the new environment
        obs, info = self.current_env.reset(**kwargs)

        # Pad the observation to match the maximum observation space dimension
        padded_obs = self._pad_obs(obs)

        return padded_obs, info

    def close(self):
        for env in self.env_list:
            env.close()

    def _pad_obs(self, obs):
        """
        Pad the observation with zeros to match the maximum observation space dimension
        """
        padded_obs = np.zeros(self.observation_space.shape)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def _unpad_action(self, action):
        """
        Remove the padding from the action
        """
        unpadded_action = action[:self.current_env.action_space.shape[0]]
        return unpadded_action

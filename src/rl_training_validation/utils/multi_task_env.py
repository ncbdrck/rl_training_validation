from copy import deepcopy
import gymnasium as gym
from typing import Optional, List
from gymnasium import spaces
import numpy as np

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper

import uniros as uniros_gym

class MultiTaskEnv(gym.Env):
    """
    A wrapper that trains multiple UniROS-based Gym environments in one agent.
    Not MPI-parallel; it simply samples a different task each reset.
    """

    def __init__(
        self,
        env_list: List[str],
        env_args_list: Optional[List[dict]] = None,
        wrapper_list: Optional[List[str]] = None,
        wrapper_args_dict: Optional[dict] = None,
    ):
        """
        Args:
            env_list: list of registered gym env names (e.g. UniROS names)
            env_args_list: list of dicts, one per env in env_list, passed to make()
            wrapper_list: list of wrapper class names (strings) to apply to every env
            wrapper_args_dict: mapping from wrapper name → kwargs dict
        """
        super().__init__()

        # Default to empty-kwargs for each env
        if env_args_list is None:
            env_args_list = [{} for _ in env_list]
        if len(env_args_list) != len(env_list):
            raise ValueError("env_args_list must have same length as env_list")

        wrapper_args_dict = wrapper_args_dict or {}

        # Create and wrap each env
        self.env_list = []
        for env_name, args in zip(env_list, env_args_list):
            env = uniros_gym.make(env_name, **args)

            if wrapper_list:
                for wr in wrapper_list:
                    if wr == "NormalizeActionWrapper":
                        env = NormalizeActionWrapper(env)
                    elif wr == "NormalizeObservationWrapper":
                        env = NormalizeObservationWrapper(
                            env, **wrapper_args_dict.get(wr, {})
                        )
                    elif wr == "TimeLimitWrapper":
                        env = TimeLimitWrapper(
                            env, **wrapper_args_dict.get(wr, {})
                        )
                    else:
                        raise ValueError(f"Wrapper {wr} not implemented")

            self.env_list.append(env)

        # Determine unified action / obs dims
        max_obs_dim = max(e.observation_space.shape[0] for e in self.env_list)
        max_act_dim = max(e.action_space.shape[0] for e in self.env_list)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(max_act_dim,), dtype=np.float32
        )

        self.current_env = None

    def step(self, action):
        # trim to current env’s action dim
        raw_act = action[: self.current_env.action_space.shape[0]]
        obs, rew, term, trunc, info = self.current_env.step(raw_act)
        # pad obs up to max_obs_dim
        padded = np.zeros(self.observation_space.shape, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded, rew, term, trunc, info

    def reset(self, **kwargs):
        # optionally reset previously used env
        if self.current_env is not None:
            self.current_env.reset()

        # pick a new one
        idx = np.random.choice(len(self.env_list))
        self.current_env = self.env_list[idx]

        obs, info = self.current_env.reset(**kwargs)
        padded = np.zeros(self.observation_space.shape, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded, info

    def close(self):
        for e in self.env_list:
            e.close()

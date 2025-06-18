from copy import deepcopy
import gymnasium as gym
import gymnasium_robotics
from typing import Optional, List
from gymnasium import spaces
import numpy as np

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper

import uniros as uniros_gym

class MultiTaskGoalEnv(gymnasium_robotics.GoalEnv):
    """
    A wrapper that trains multiple UniROS-based GoalEnv gymnasium environments in one agent.
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
            env_list: list of registered gym env names (must be GoalEnv)
            env_args_list: list of dicts, one per env in env_list, passed to make()
            wrapper_list: list of wrapper class names (strings) to apply to every env
            wrapper_args_dict: mapping from wrapper name → kwargs dict
        """
        super().__init__()

        # Default to empty kwargs for each env
        if env_args_list is None:
            env_args_list = [{} for _ in env_list]
        if len(env_args_list) != len(env_list):
            raise ValueError("env_args_list must have same length as env_list")

        wrapper_args_dict = wrapper_args_dict or {}

        # Create & wrap each env
        self.env_list = []
        for name, args in zip(env_list, env_args_list):
            env = uniros_gym.make(name, **args)
            if wrapper_list:
                for wr in wrapper_list:
                    if wr == "NormalizeActionWrapper":
                        env = NormalizeActionWrapper(env)
                    elif wr == "NormalizeObservationWrapper":
                        env = NormalizeObservationWrapper(env, **wrapper_args_dict.get(wr, {}))
                    elif wr == "TimeLimitWrapper":
                        env = TimeLimitWrapper(env, **wrapper_args_dict.get(wr, {}))
                    else:
                        raise ValueError(f"Wrapper {wr} not implemented")
            self.env_list.append(env)

        # Compute maximum dims for each component of the dict-obs
        max_obs_dim = max(e.observation_space["observation"].shape[0] for e in self.env_list)
        max_ag_dim  = max(e.observation_space["achieved_goal"].shape[0] for e in self.env_list)
        max_dg_dim  = max(e.observation_space["desired_goal"].shape[0] for e in self.env_list)
        max_act_dim = max(e.action_space.shape[0] for e in self.env_list)

        # Build unified spaces
        self.observation_space = spaces.Dict({
            "observation":    spaces.Box(-np.inf, np.inf, (max_obs_dim,), dtype=np.float32),
            "achieved_goal":  spaces.Box(-np.inf, np.inf, (max_ag_dim,),  dtype=np.float32),
            "desired_goal":   spaces.Box(-np.inf, np.inf, (max_dg_dim,),  dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (max_act_dim,), dtype=np.float32)

        self.current_env = None

    def step(self, action):
        # trim to current env’s action dim
        raw_a = action[: self.current_env.action_space.shape[0]]
        obs_dict, reward, terminated, truncated, info = self.current_env.step(raw_a)

        # pad each piece of the dict
        obs_padded = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)
        obs_padded[: obs_dict["observation"].shape[0]] = obs_dict["observation"]

        ag_padded  = np.zeros(self.observation_space["achieved_goal"].shape, dtype=np.float32)
        ag_padded[: obs_dict["achieved_goal"].shape[0]] = obs_dict["achieved_goal"]

        dg_padded  = np.zeros(self.observation_space["desired_goal"].shape, dtype=np.float32)
        dg_padded[: obs_dict["desired_goal"].shape[0]] = obs_dict["desired_goal"]

        new_obs = {
            "observation":    obs_padded,
            "achieved_goal":  ag_padded,
            "desired_goal":   dg_padded,
        }
        return new_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # optionally reset old env
        if self.current_env is not None:
            self.current_env.reset()

        # pick a new one
        idx = np.random.choice(len(self.env_list))
        self.current_env = self.env_list[idx]

        obs_dict, info = self.current_env.reset(**kwargs)

        # pad each piece
        obs_padded = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)
        obs_padded[: obs_dict["observation"].shape[0]] = obs_dict["observation"]

        ag_padded  = np.zeros(self.observation_space["achieved_goal"].shape, dtype=np.float32)
        ag_padded[: obs_dict["achieved_goal"].shape[0]] = obs_dict["achieved_goal"]

        dg_padded  = np.zeros(self.observation_space["desired_goal"].shape, dtype=np.float32)
        dg_padded[: obs_dict["desired_goal"].shape[0]] = obs_dict["desired_goal"]

        new_obs = {
            "observation":    obs_padded,
            "achieved_goal":  ag_padded,
            "desired_goal":   dg_padded,
        }
        return new_obs, info

    def close(self):
        for env in self.env_list:
            env.close()

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict
    ) -> float:
        """
        Called by HER to recompute rewards when relabeling.
        Forward to the currently active environment.
        """
        return self.current_env.compute_reward(achieved_goal, desired_goal, info)

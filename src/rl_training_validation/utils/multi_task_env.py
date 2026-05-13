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
        self.current_env_idx = None  # set on each reset()

    def step(self, action):
        # trim to current env’s action dim
        raw_act = action[: self.current_env.action_space.shape[0]]
        obs, rew, term, trunc, info = self.current_env.step(raw_act)
        # pad obs up to max_obs_dim
        padded = np.zeros(self.observation_space.shape, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        # Tag every step's info with the currently-active sub-env index.
        # Consumers (HER replay, evaluation logs, etc.) can attribute
        # transitions back to a specific task.
        info["task_id"] = self.current_env_idx
        return padded, rew, term, trunc, info

    def reset(self, *, seed=None, options=None, **kwargs):
        # Seed our OWN np_random (Gymnasium semantics). Used below to
        # pick which sub-env runs this episode. The previous implementation
        # called np.random.choice() against numpy's global state, so the
        # per-reset task choice was not seedable and not reproducible.
        super().reset(seed=seed)

        # Note: a previous implementation called ``self.current_env.reset()``
        # here before picking the next task. That extra reset stepped the
        # sub-env's RNG and produced an observation that was thrown away,
        # which (a) wasted a real-world / Gazebo cycle and (b) shifted
        # reproducibility against any seed plan that assumes one reset
        # per episode. The chosen sub-env is reset below — that single
        # reset is sufficient.

        # Pick a new sub-env using our seeded np_random.
        idx = int(self.np_random.integers(0, len(self.env_list)))
        self.current_env_idx = idx
        self.current_env = self.env_list[idx]

        # Reset the chosen sub-env. We forward `options` and any extra
        # kwargs, but NOT `seed`: sub-envs maintain their own RNG state
        # across episodes, and re-seeding them on every reset would
        # defeat that. If you need cross-run reproducibility of sub-env
        # randomness, seed each sub-env at construction time via env_args.
        sub_kwargs = dict(kwargs)
        if options is not None:
            sub_kwargs["options"] = options
        obs, info = self.current_env.reset(**sub_kwargs)

        # Record the active task in info so downstream consumers (in
        # particular the MultiTaskGoalEnv reward-routing logic) can
        # identify which sub-env produced this transition.
        info["task_id"] = idx

        padded = np.zeros(self.observation_space.shape, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded, info

    def close(self):
        for e in self.env_list:
            e.close()

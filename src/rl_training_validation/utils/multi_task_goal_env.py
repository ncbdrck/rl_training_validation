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
        self.current_env_idx = None  # set on each reset()

    def step(self, action):
        """
        Step the currently-active sub-env with ``action`` trimmed to
        its action dimension, then pad each piece of the returned
        observation dict (``observation`` / ``achieved_goal`` /
        ``desired_goal``) to the unified maximum dims and stamp
        ``info["task_id"]``.
        """
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
        # Tag every step's info with the currently-active sub-env index.
        # HER's replay buffer preserves info dicts; compute_reward()
        # (below) reads info["task_id"] to route reward recomputation
        # to the *correct* sub-env, not whichever one happens to be
        # active when the replay batch is being relabelled.
        info["task_id"] = self.current_env_idx
        return new_obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None, **kwargs):
        # Seed our OWN np_random for task choice (Gymnasium semantics).
        # The previous implementation called np.random.choice() against
        # numpy's global state, so the per-reset task choice was not
        # seedable and not reproducible.
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

        # Reset the chosen sub-env. We forward `options` and extra
        # kwargs but NOT `seed`: sub-envs maintain their own RNG state.
        # If you need cross-run reproducibility of sub-env randomness,
        # seed each sub-env at construction via env_args.
        sub_kwargs = dict(kwargs)
        if options is not None:
            sub_kwargs["options"] = options
        obs_dict, info = self.current_env.reset(**sub_kwargs)

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
        # Record the active task in info so HER and other consumers can
        # attribute this transition to a specific sub-env.
        info["task_id"] = idx
        return new_obs, info

    def close(self):
        for env in self.env_list:
            env.close()

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Called by HER to recompute rewards when relabelling experience.

        HER batches transitions from possibly-different sub-envs; the
        "currently active" sub-env at recompute time may not be the
        same as the sub-env that produced the transition. We therefore
        route by ``info["task_id"]`` (stamped at reset() and step()
        time) rather than ``self.current_env``.

        Falls back to ``self.current_env`` if task_id is missing from
        ``info`` (e.g. info dict was constructed by hand for a
        single-task test). This is a best-effort fallback; for HER
        correctness, ``info["task_id"]`` should be present.
        """
        task_id = info.get("task_id") if isinstance(info, dict) else None
        if task_id is None:
            # Backwards-compat fallback for callers who don't preserve
            # task_id through replay. May silently misroute under HER
            # if the active sub-env has changed since the transition
            # was collected.
            return self.current_env.compute_reward(achieved_goal, desired_goal, info)
        return self.env_list[task_id].compute_reward(achieved_goal, desired_goal, info)

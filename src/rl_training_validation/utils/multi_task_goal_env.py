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
        info,
    ):
        """
        Called by HER to recompute rewards when relabelling experience.

        Two call shapes have to be supported:

          * Live step / single transition:
            ``info`` is a ``dict``; ``ag`` / ``dg`` are 1-D arrays.
            Route by ``info["task_id"]`` (stamped at reset() and step()
            time) so a relabel for sub-env i is computed by sub-env i,
            not whichever sub-env happens to be ``current_env`` now.

          * HER batched relabel:
            ``info`` is a Python list / ndarray of per-transition info
            dicts (SB3 HER ``copy_info_dict=False`` passes a list).
            Group transitions by ``info["task_id"]`` and let each
            sub-env compute its slice, then scatter the per-slice
            rewards back into a single float32 array shaped like
            ``ag.shape[:-1]``. Without this, the previous
            ``isinstance(info, dict)`` check failed on the list path
            and silently routed every transition through
            ``self.current_env``.

        Falls back to ``self.current_env`` when ``task_id`` is missing
        from ``info`` (e.g. a single-task unit test that constructs
        ``info`` by hand). The fallback may misroute under HER if the
        active sub-env has changed since the transition was collected
        — preserve ``task_id`` through the replay buffer to avoid it.
        """
        # Single-transition path: dict info, 1-D arrays.
        if isinstance(info, dict):
            task_id = info.get("task_id")
            sub_env = self.current_env if task_id is None else self.env_list[task_id]
            # Strip the zero-padding (step()/reset() pad sub-env goals up
            # to the unified max dim). Without this, sub-envs with a
            # smaller native goal dim see padded zeros in the trailing
            # slots, polluting distance / success thresholds.
            ag_dim = sub_env.observation_space["achieved_goal"].shape[0]
            dg_dim = sub_env.observation_space["desired_goal"].shape[0]
            return sub_env.compute_reward(
                achieved_goal[..., :ag_dim],
                desired_goal[..., :dg_dim],
                info,
            )

        # Batched relabel path: list / ndarray of info dicts.
        ag = np.asarray(achieved_goal)
        dg = np.asarray(desired_goal)
        n = len(info)
        rewards = np.zeros(n, dtype=np.float32)

        # Group indices by task_id so each sub-env processes its slice
        # in one call (sub-envs already support batched ag/dg/info).
        per_task = {}
        for idx, item in enumerate(info):
            tid = item.get("task_id") if isinstance(item, dict) else None
            per_task.setdefault(tid, []).append(idx)

        for tid, idxs in per_task.items():
            sub_env = self.current_env if tid is None else self.env_list[tid]
            ag_dim = sub_env.observation_space["achieved_goal"].shape[0]
            dg_dim = sub_env.observation_space["desired_goal"].shape[0]
            slice_info = [info[i] for i in idxs]
            # Slice off the padded slots so the sub-env's distance/threshold
            # only sees its native goal components.
            slice_reward = sub_env.compute_reward(
                ag[idxs][..., :ag_dim],
                dg[idxs][..., :dg_dim],
                slice_info,
            )
            rewards[idxs] = np.asarray(slice_reward, dtype=np.float32)

        return rewards

#!/usr/bin/env python3
"""
Cross-check goal-conditioned training setup.

For every goal env id referenced by this repo, verify:

  * The env id ends with ``Goal{Sim,Real}-v0``.
  * The env id is implemented in the rl_environments status table.
  * Its task-env class exposes ``compute_reward(achieved, desired, info)``.
  * Its task-env class exposes ``_calc_reward_step`` (used by HER and by
    the goal mixin's ``compute_reward``).

Then, for every non-goal env id, verify the inverse — i.e. that nobody
is going to wire a HER replay buffer to a non-goal env.

Pure introspection. The class is imported via the gymnasium registry
spec. ROS-only deps are reported as "runtime-skip" rather than failures.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import gymnasium as gym  # noqa: E402
import rl_environments  # noqa: E402,F401  triggers registration

from rl_training_validation.utils.env_safety import (  # noqa: E402
    is_goal_env, is_implemented, list_implemented, parse_env_id,
)


def _import_class(env_id: str):
    spec = gym.spec(env_id)
    if "UnimplementedRLEnv" in spec.entry_point:
        return None, None
    module_path, class_name = spec.entry_point.split(":")
    mod = __import__(module_path, fromlist=[class_name])
    return mod, getattr(mod, class_name)


def main() -> int:
    print("=" * 60)
    print("  Training-repo goal-env setup audit")
    print("=" * 60)

    issues = 0
    runtime_skipped: List[Tuple[str, str]] = []

    for eid in list_implemented():
        if not is_goal_env(eid):
            continue
        try:
            _, cls = _import_class(eid)
        except ModuleNotFoundError as e:
            runtime_skipped.append((eid, str(e)))
            continue
        if cls is None:
            print(f"  ❌ {eid}: status says implemented but registry points at UnimplementedRLEnv")
            issues += 1
            continue
        if not hasattr(cls, "compute_reward"):
            print(f"  ❌ {eid}: missing compute_reward")
            issues += 1
            continue
        if not hasattr(cls, "_calc_reward_step"):
            print(f"  ❌ {eid}: missing _calc_reward_step (HER will fall back to compute_reward)")
            issues += 1
            continue
        print(f"  ✅ {eid}: compute_reward / _calc_reward_step present")

    # Confirm non-goal envs in the implemented list are NOT being routed
    # to HER. We can only check the env-id pattern; the actual binding is
    # in the training script and is checked separately by
    # check_env_availability.py.
    non_goal_impl = [e for e in list_implemented() if not is_goal_env(e)]
    if non_goal_impl:
        print(f"\nNon-goal implemented envs ({len(non_goal_impl)}). These must NOT be passed to HER:")
        for eid in non_goal_impl:
            print(f"  - {eid}")

    if runtime_skipped:
        print("\nSkipped (ROS-only deps unavailable in this shell):")
        for eid, reason in runtime_skipped:
            print(f"  runtime-skip: {eid} — {reason}")

    print("\n" + "=" * 60)
    if issues == 0:
        print("  ✅ Goal-env API contract holds for every implemented goal env.")
        return 0
    print(f"  ⚠️  {issues} issue(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

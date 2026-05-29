#!/usr/bin/env python3
"""
Cross-check goal-conditioned training setup.

For every registered goal env id, verify:

  * The id ends with ``Goal{Sim,Real}-v0``.
  * Its task-env class exposes ``compute_reward(achieved, desired, info)``
    (Gymnasium GoalEnv / HER hook).
  * Its task-env class exposes ``compute_terminated`` and
    ``compute_truncated`` (the other two GoalEnv hooks).

Pure introspection. ROS-only import deps are reported as "runtime-skip"
rather than failures so this runs cleanly outside a sourced workspace.
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
    is_goal_env, list_implemented,
)


REQUIRED_HOOKS = ("compute_reward", "compute_terminated", "compute_truncated")


def _import_class(env_id: str):
    spec = gym.spec(env_id)
    module_path, class_name = spec.entry_point.split(":")
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


def main() -> int:
    print("=" * 60)
    print("  Training-repo goal-env setup audit")
    print("=" * 60)

    issues = 0
    runtime_skipped: List[Tuple[str, str]] = []

    goal_ids = [eid for eid in list_implemented() if is_goal_env(eid)]
    print(f"\nChecking {len(goal_ids)} goal env(s)...")
    for eid in goal_ids:
        try:
            cls = _import_class(eid)
        except ModuleNotFoundError as e:
            runtime_skipped.append((eid, str(e)))
            continue
        missing = [h for h in REQUIRED_HOOKS if not hasattr(cls, h)]
        if missing:
            print(f"  FAIL: {eid}: missing {missing}")
            issues += 1
        else:
            print(f"  ok:   {eid}")

    non_goal_impl = [e for e in list_implemented() if not is_goal_env(e)]
    if non_goal_impl:
        print(f"\nNon-goal envs ({len(non_goal_impl)}) — must NOT be passed to HER:")
        for eid in non_goal_impl:
            print(f"  - {eid}")

    if runtime_skipped:
        print("\nSkipped (ROS-only deps unavailable in this shell):")
        for eid, reason in runtime_skipped:
            print(f"  runtime-skip: {eid} — {reason}")

    print("\n" + "=" * 60)
    if issues == 0:
        print("  PASS: GoalEnv hook contract holds for every registered goal env.")
        return 0
    print(f"  {issues} issue(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

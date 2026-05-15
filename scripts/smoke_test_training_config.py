#!/usr/bin/env python3
"""
Smoke-test the training/validation repo's relationship with the audited
rl_environments registry. Does NOT start Gazebo, ROS, or hardware.

Verifies:
  1. rl_environments imports and registers all 48 UniROS-... ids.
  2. The implementation-status table is consistent with the registry
     (implemented -> real entry point, blocked -> UnimplementedRLEnv).
  3. Blocked real envs raise NotImplementedError at gym.make(); they
     should never reach the construction path.
  4. The env-safety helper correctly classifies a sample of env ids.
  5. Algorithm-config YAML files in ``config/`` look well-formed.

Run it like::

    python3 scripts/smoke_test_training_config.py
"""
from __future__ import annotations

import os
import sys
import yaml

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def _import_gym():
    import gymnasium as gym
    import rl_environments  # noqa: F401  triggers registration
    return gym


def test_imports() -> int:
    print("\n[1] Imports + registration...")
    try:
        gym = _import_gym()
    except Exception as e:
        print(f"  ❌ import failed: {e}")
        return 1
    n = sum(1 for s in gym.envs.registry if s.startswith("UniROS-"))
    if n != 48:
        print(f"  ❌ expected 48 UniROS-... ids, got {n}")
        return 1
    print(f"  ✅ rl_environments importable, {n} UniROS ids registered")
    return 0


def test_env_safety_helpers() -> int:
    print("\n[2] env_safety helpers...")
    from rl_training_validation.utils.env_safety import (
        is_implemented, is_real, is_goal_env, list_implemented, parse_env_id,
    )
    issues = 0
    # Implemented today (per the audit):
    expected_impl = {
        "UniROS-RX200ReachSim-v0", "UniROS-RX200ReachGoalSim-v0",
        "UniROS-RX200PushSim-v0", "UniROS-RX200PushGoalSim-v0",
        "UniROS-RX200PnPSim-v0", "UniROS-RX200PnPGoalSim-v0",
        "UniROS-RX200SlideSim-v0", "UniROS-RX200SlideGoalSim-v0",
        "UniROS-Ned2ReachSim-v0", "UniROS-Ned2ReachGoalSim-v0",
        "UniROS-UR5ReachSim-v0", "UniROS-UR5ReachGoalSim-v0",
        "UniROS-RX200ReachReal-v0", "UniROS-RX200ReachGoalReal-v0",
    }
    actual_impl = set(list_implemented())
    extra = actual_impl - expected_impl
    missing = expected_impl - actual_impl
    if extra or missing:
        print(f"  ❌ implemented set mismatch — extra={sorted(extra)} missing={sorted(missing)}")
        issues += 1
    else:
        print(f"  ✅ {len(actual_impl)} implemented ids match the audit")

    # Spot-check classifiers.
    for eid, want in [
        ("UniROS-RX200ReachSim-v0", ("rx200", "sim", "reach", False)),
        ("UniROS-Ned2ReachGoalSim-v0", ("ned2", "sim", "reach", True)),
        ("UniROS-UR5PnPGoalReal-v0", ("ur5", "real", "pnp", True)),
    ]:
        got = parse_env_id(eid)
        if got != want:
            print(f"  ❌ parse_env_id({eid}) = {got}, expected {want}")
            issues += 1

    if is_real("UniROS-RX200ReachSim-v0"):
        print("  ❌ is_real('...Sim-v0') wrongly True")
        issues += 1
    if not is_real("UniROS-RX200ReachReal-v0"):
        print("  ❌ is_real('...Real-v0') wrongly False")
        issues += 1
    if not is_goal_env("UniROS-RX200ReachGoalSim-v0"):
        print("  ❌ is_goal_env wrongly False on a Goal id")
        issues += 1
    if is_goal_env("UniROS-RX200ReachSim-v0"):
        print("  ❌ is_goal_env wrongly True on a non-Goal id")
        issues += 1
    if is_implemented("UniROS-Ned2PushSim-v0"):
        print("  ❌ is_implemented wrongly True for a blocked env")
        issues += 1

    if issues == 0:
        print("  ✅ helpers behave as documented")
    return issues


def test_blocked_envs_raise() -> int:
    print("\n[3] Blocked envs raise NotImplementedError...")
    gym = _import_gym()
    issues = 0
    for eid in ("UniROS-Ned2PushSim-v0", "UniROS-UR5PnPGoalSim-v0",
                "UniROS-Ned2ReachReal-v0", "UniROS-UR5ReachGoalReal-v0"):
        try:
            gym.make(eid)
        except NotImplementedError:
            print(f"  ✅ {eid} blocked")
        except Exception as e:
            print(f"  ❌ {eid} raised {type(e).__name__} (expected NotImplementedError): {e}")
            issues += 1
        else:
            print(f"  ❌ {eid} unexpectedly constructed")
            issues += 1
    return issues


def test_yaml_configs() -> int:
    print("\n[4] config/*.yaml are valid YAML...")
    issues = 0
    cfg_dir = os.path.join(REPO_ROOT, "config")
    if not os.path.isdir(cfg_dir):
        print(f"  ⚠️ no config/ directory at {cfg_dir}")
        return 0
    for f in sorted(os.listdir(cfg_dir)):
        if not f.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(cfg_dir, f)
        try:
            with open(path) as fh:
                yaml.safe_load(fh)
            print(f"  ✅ {f}")
        except yaml.YAMLError as e:
            print(f"  ❌ {f}: {e}")
            issues += 1
    return issues


def main() -> int:
    print("=" * 60)
    print("  Training-repo smoke test (no Gazebo, no hardware)")
    print("=" * 60)
    total = 0
    total += test_imports()
    total += test_env_safety_helpers()
    total += test_blocked_envs_raise()
    total += test_yaml_configs()
    print("\n" + "=" * 60)
    if total == 0:
        print("  ✅ ALL SMOKE TESTS PASSED")
        return 0
    print(f"  ⚠️  {total} issue(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

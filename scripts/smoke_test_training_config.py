#!/usr/bin/env python3
"""
Smoke-test the training/validation repo's relationship with the
``rl_environments`` registry. Pure introspection — does NOT start
Gazebo, ROS, or hardware.

Verifies:
  1. ``rl_environments`` imports and at least the expected RX200 + NED2 + VX300S
     ids land in the registry.
  2. The env_safety classifiers (is_real / is_goal_env) behave as
     documented.
  3. Every CFG_*.yaml referenced by a train/validate script exists.
  4. Every YAML in ``config/`` parses cleanly.

Run it like::

    python3 scripts/smoke_test_training_config.py
"""
from __future__ import annotations

import os
import re
import sys

import yaml

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def _import_gym():
    import gymnasium as gym
    import rl_environments  # noqa: F401  triggers registration
    return gym


def _expected_ids() -> set:
    """Derive the expected task-env id set from the authoritative
    ``ALL_*_NAMES`` manifests in ``rl_environments/__init__.py``.

    Reading the manifest directly (instead of duplicating it here) keeps
    the smoke test in lock-step when ids are added or renamed there, so
    a drift between the manifest and the registry shows up as a real
    test failure rather than being masked by a stale local copy.
    """
    import rl_environments  # noqa: F401  triggers registration + exports manifests
    ids = set()
    for attr in ("ALL_REACH_SIM_NAMES", "ALL_PUSH_SIM_NAMES", "ALL_PNP_SIM_NAMES",
                 "ALL_REACH_REAL_NAMES", "ALL_PUSH_REAL_NAMES", "ALL_PNP_REAL_NAMES"):
        ids.update(getattr(rl_environments, attr))
    return ids


def _registered_task_ids(gym) -> set:
    """Registry ids that look like our task ids (one of the four robot
    prefixes + ``-v0``). Filters out the abstract robot-base ids
    (e.g. ``RX200RobotGoalEnv-v0``) so the symmetry check below
    compares like-for-like task ids only."""
    robot_prefixes = ("RX200", "NED2", "VX300S", "UR5e")
    ours = set()
    for eid in gym.envs.registry.keys():
        if not eid.endswith("-v0"):
            continue
        if not any(eid.startswith(p) for p in robot_prefixes):
            continue
        # Drop the abstract robot-base registrations.
        if "Robot" in eid:
            continue
        ours.add(eid)
    return ours


def test_imports() -> int:
    print("\n[1] Imports + registration...")
    try:
        gym = _import_gym()
    except Exception as e:
        print(f"  FAIL: import error: {e}")
        return 1
    expected = _expected_ids()
    registered = _registered_task_ids(gym)
    missing = expected - registered
    extra = registered - expected
    if missing or extra:
        if missing:
            print(f"  FAIL: missing from registry: {sorted(missing)}")
        if extra:
            print(f"  FAIL: registered but not in ALL_*_NAMES: {sorted(extra)}")
        return 1
    print(f"  ok: all {len(expected)} task ids registered (and no extras)")
    return 0


def test_env_safety_helpers() -> int:
    print("\n[2] env_safety helpers...")
    from rl_training_validation.utils.env_safety import (
        is_registered, is_real, is_goal_env, parse_env_id,
    )
    issues = 0

    # Spot-check classifiers.
    for eid, want in [
        ("RX200ReacherSim-v0",   ("rx200", "sim",  "reach", False)),
        ("RX200ReacherGoalSim-v0", ("rx200", "sim",  "reach", True)),
        ("NED2ReacherGoalSim-v0", ("ned2", "sim",  "reach", True)),
        ("RX200PnPGoalSim-v0",   ("rx200", "sim",  "pnp",   True)),
        ("RX200Zed2PushSim-v0",  ("rx200", "sim",  "push",  False)),
        ("RX200ReacherReal-v0",  ("rx200", "real", "reach", False)),
        ("VX300SReacherGoalSim-v0", ("vx300s", "sim", "reach", True)),
        ("VX300SReacherReal-v0", ("vx300s", "real", "reach", False)),
    ]:
        got = parse_env_id(eid)
        if got != want:
            print(f"  FAIL: parse_env_id({eid}) = {got}, expected {want}")
            issues += 1

    checks = [
        (not is_real("RX200ReacherSim-v0"),  "is_real Sim==False"),
        (is_real("RX200ReacherReal-v0"),     "is_real Real==True"),
        (is_goal_env("RX200ReacherGoalSim-v0"), "is_goal_env Goal==True"),
        (is_goal_env("VX300SReacherGoalReal-v0"), "is_goal_env VX300S Real Goal==True"),
        (not is_goal_env("RX200ReacherSim-v0"), "is_goal_env non-Goal==False"),
        (is_registered("RX200PnPSim-v0"),    "is_registered PnP==True"),
        (is_registered("VX300SReacherSim-v0"), "is_registered VX300S reach==True"),
        (not is_registered("NotARealId-v0"), "is_registered fake==False"),
    ]
    for ok, label in checks:
        if not ok:
            print(f"  FAIL: {label}")
            issues += 1

    if issues == 0:
        print("  ok: helpers behave as documented")
    return issues


def test_yaml_configs() -> int:
    print("\n[3] config/*.yaml are valid YAML...")
    issues = 0
    cfg_dir = os.path.join(REPO_ROOT, "config")
    if not os.path.isdir(cfg_dir):
        print(f"  warn: no config/ directory at {cfg_dir}")
        return 0
    for f in sorted(os.listdir(cfg_dir)):
        if not f.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(cfg_dir, f)
        try:
            with open(path) as fh:
                yaml.safe_load(fh)
        except yaml.YAMLError as e:
            print(f"  FAIL: {f}: {e}")
            issues += 1
    if issues == 0:
        print(f"  ok: {len([f for f in os.listdir(cfg_dir) if f.endswith(('.yaml', '.yml'))])} config files parse")
    return issues


def test_script_config_references() -> int:
    print("\n[4] script CFG_*.yaml references resolve...")
    cfg_dir = os.path.join(REPO_ROOT, "config")
    pattern = re.compile(r"CFG_[A-Z0-9_]*\s*=\s*['\"]([^'\"]+\.ya?ml)['\"]")
    issues = 0
    refs = set()
    src_dir = os.path.join(REPO_ROOT, "src")
    for dirpath, _, files in os.walk(src_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for cfg in pattern.findall(text):
                refs.add((cfg, os.path.relpath(path, REPO_ROOT)))

    for cfg, relpath in sorted(refs):
        if not os.path.exists(os.path.join(cfg_dir, cfg)):
            print(f"  FAIL: {cfg} referenced by {relpath} is missing")
            issues += 1
    if not refs:
        print("  warn: no CFG_*.yaml references found")
    elif issues == 0:
        print(f"  ok: {len(refs)} CFG references resolve")
    return issues


def test_cube_tracker_kwargs() -> int:
    """The 4 real push/pnp envs must expose the auto-launch kwargs that
    rl_training_validation's --cube-tracker CLI plumbs through to
    gym.make. Catches signature drift between the train scripts and
    the env files without needing hardware to construct the env."""
    print("\n[5] real push/pnp envs expose cube-tracker kwargs...")
    import importlib
    import inspect

    target_modules = [
        # RX200 real push + pnp
        ("rl_environments.rx200.real.task_envs.push.rx200_push_real", "RX200PushEnvReal"),
        ("rl_environments.rx200.real.task_envs.push.rx200_push_goal_real", "RX200PushGoalEnvReal"),
        ("rl_environments.rx200.real.task_envs.pnp.rx200_pnp_real", "RX200PnPEnvReal"),
        ("rl_environments.rx200.real.task_envs.pnp.rx200_pnp_goal_real", "RX200PnPGoalEnvReal"),
        # Ned2 real push + pnp
        ("rl_environments.ned2.real.task_envs.push.ned2_push_real", "NED2PushEnvReal"),
        ("rl_environments.ned2.real.task_envs.push.ned2_push_goal_real", "NED2PushGoalEnvReal"),
        ("rl_environments.ned2.real.task_envs.pnp.ned2_pnp_real", "NED2PnPEnvReal"),
        ("rl_environments.ned2.real.task_envs.pnp.ned2_pnp_goal_real", "NED2PnPGoalEnvReal"),
    ]
    required_kwargs = {
        "auto_launch_cube_tracker",
        "cube_tracker_camera",
        "cube_tracker_target_frame",
    }
    issues = 0
    for mod_path, cls_name in target_modules:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            print(f"  FAIL: import {mod_path}: {e}")
            issues += 1
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            # Class name guess may be wrong; just probe any class in the
            # module whose __init__ takes cube_pose_topic.
            for name in dir(mod):
                obj = getattr(mod, name)
                if inspect.isclass(obj) and "cube_pose_topic" in inspect.signature(obj.__init__).parameters:
                    cls = obj
                    break
        if cls is None:
            print(f"  FAIL: no task class found in {mod_path}")
            issues += 1
            continue
        params = set(inspect.signature(cls.__init__).parameters)
        missing = required_kwargs - params
        if missing:
            print(f"  FAIL: {cls.__name__} missing kwargs {sorted(missing)}")
            issues += 1
    if issues == 0:
        print(f"  ok: all {len(target_modules)} real envs expose {sorted(required_kwargs)}")
    return issues


def main() -> int:
    print("=" * 60)
    print("  Training-repo smoke test (no Gazebo, no hardware)")
    print("=" * 60)
    total = 0
    total += test_imports()
    total += test_env_safety_helpers()
    total += test_yaml_configs()
    total += test_script_config_references()
    total += test_cube_tracker_kwargs()
    print("\n" + "=" * 60)
    if total == 0:
        print("  ALL SMOKE TESTS PASSED")
        return 0
    print(f"  {total} issue(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

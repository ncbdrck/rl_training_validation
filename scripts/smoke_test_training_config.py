#!/usr/bin/env python3
"""
Smoke-test the training/validation repo's relationship with the
``rl_environments`` registry. Pure introspection — does NOT start
Gazebo, ROS, or hardware.

Verifies:
  1. ``rl_environments`` imports and at least the expected RX200 + NED2
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


# Ids that MUST be registered today. Keep this in lockstep with
# rl_environments/src/rl_environments/__init__.py.
EXPECTED_IDS = {
    # RX200 reach (kinect + zed2)
    "RX200ReacherSim-v0", "RX200ReacherGoalSim-v0",
    "RX200Zed2ReacherSim-v0", "RX200Zed2ReacherGoalSim-v0",
    # RX200 push (kinect + zed2)
    "RX200PushSim-v0", "RX200PushGoalSim-v0",
    "RX200Zed2PushSim-v0", "RX200Zed2PushGoalSim-v0",
    # RX200 PnP (kinect + zed2)
    "RX200PnPSim-v0", "RX200PnPGoalSim-v0",
    "RX200Zed2PnPSim-v0", "RX200Zed2PnPGoalSim-v0",
    # Ned2 reach (kinect)
    "NED2ReacherSim-v0", "NED2ReacherGoalSim-v0",
    # RX200 real reach
    "RX200ReacherReal-v0", "RX200ReacherGoalReal-v0",
}


def test_imports() -> int:
    print("\n[1] Imports + registration...")
    try:
        gym = _import_gym()
    except Exception as e:
        print(f"  FAIL: import error: {e}")
        return 1
    registered = set(gym.envs.registry.keys())
    missing = EXPECTED_IDS - registered
    if missing:
        print(f"  FAIL: missing from registry: {sorted(missing)}")
        return 1
    print(f"  ok: all {len(EXPECTED_IDS)} expected ids registered")
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
    ]:
        got = parse_env_id(eid)
        if got != want:
            print(f"  FAIL: parse_env_id({eid}) = {got}, expected {want}")
            issues += 1

    checks = [
        (not is_real("RX200ReacherSim-v0"),  "is_real Sim==False"),
        (is_real("RX200ReacherReal-v0"),     "is_real Real==True"),
        (is_goal_env("RX200ReacherGoalSim-v0"), "is_goal_env Goal==True"),
        (not is_goal_env("RX200ReacherSim-v0"), "is_goal_env non-Goal==False"),
        (is_registered("RX200PnPSim-v0"),    "is_registered PnP==True"),
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


def main() -> int:
    print("=" * 60)
    print("  Training-repo smoke test (no Gazebo, no hardware)")
    print("=" * 60)
    total = 0
    total += test_imports()
    total += test_env_safety_helpers()
    total += test_yaml_configs()
    total += test_script_config_references()
    print("\n" + "=" * 60)
    if total == 0:
        print("  ALL SMOKE TESTS PASSED")
        return 0
    print(f"  {total} issue(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

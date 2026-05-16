#!/usr/bin/env python3
"""
Environment availability + real-robot safety helpers for the training
and validation scripts in this repository.

This module is a thin shim on top of
``rl_environments.common.env_status`` and
``rl_environments.common.safety``. It exists so that training scripts
have a stable, repo-local API for:

  * filtering training configs to only the env IDs that are actually
    implemented behind ``UniROS-...`` in ``rl_environments``,
  * detecting "Real" env IDs and refusing to construct them unless
    the user has explicitly opted in to real-robot motion,
  * giving consistent, helpful error messages when a training script
    is pointed at a blocked or unimplemented env.

Nothing in this module touches Gazebo or any ROS topic.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Imports that may fail outside a sourced workspace
# ---------------------------------------------------------------------------

def _import_env_status():
    """Import the env-status table from rl_environments, or fail clearly."""
    try:
        from rl_environments.common import env_status  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "[rl_training_validation] Cannot import rl_environments.common.env_status: "
            f"{e}.\nMake sure rl_environments is installed (e.g. via catkin) and that "
            "your workspace's setup.bash is sourced."
        )
    return env_status


def _import_gym():
    try:
        import gymnasium as gym
        import rl_environments  # noqa: F401  triggers registration
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"[rl_training_validation] Cannot import gymnasium / rl_environments: {e}."
        )
    return gym


# ---------------------------------------------------------------------------
# Env-id helpers
# ---------------------------------------------------------------------------

_ROBOT_CAMEL = {"rx200": "RX200", "ned2": "Ned2", "ur5": "UR5", "ur5e": "UR5e"}
_TASK_CAMEL = {"reach": "Reach", "push": "Push", "pnp": "PnP", "slide": "Slide"}


def env_id_for(robot: str, task: str, mode: str, is_goal: bool) -> str:
    """Construct the canonical ``UniROS-...`` env id."""
    rc = _ROBOT_CAMEL[robot]
    tc = _TASK_CAMEL[task]
    gc = "Goal" if is_goal else ""
    mc = "Sim" if mode == "sim" else "Real"
    return f"UniROS-{rc}{tc}{gc}{mc}-v0"


def parse_env_id(env_id: str) -> Optional[Tuple[str, str, str, bool]]:
    """
    Parse a UniROS env id into ``(robot, mode, task, is_goal)`` or None.
    """
    if not env_id.startswith("UniROS-") or not env_id.endswith("-v0"):
        return None
    body = env_id[len("UniROS-"):-len("-v0")]
    # Match the LONGEST robot prefix first. "UR5e" must win over "UR5"
    # when the body is e.g. "UR5ePnPSim".
    sorted_robots = sorted(_ROBOT_CAMEL.items(), key=lambda kv: -len(kv[1]))
    for robot_lc, robot_cm in sorted_robots:
        if body.startswith(robot_cm):
            rest = body[len(robot_cm):]
            break
    else:
        return None
    if rest.endswith("Real"):
        mode = "real"
        rest = rest[:-len("Real")]
    elif rest.endswith("Sim"):
        mode = "sim"
        rest = rest[:-len("Sim")]
    else:
        return None
    is_goal = rest.endswith("Goal")
    if is_goal:
        rest = rest[:-len("Goal")]
    for task_lc, task_cm in _TASK_CAMEL.items():
        if rest == task_cm:
            return robot_lc, mode, task_lc, is_goal
    return None


def is_implemented(env_id: str) -> bool:
    parsed = parse_env_id(env_id)
    if parsed is None:
        return False
    robot, mode, task, is_goal = parsed
    env_status = _import_env_status()
    return env_status.is_implemented(robot, mode, task, is_goal)


def is_real(env_id: str) -> bool:
    parsed = parse_env_id(env_id)
    return parsed is not None and parsed[1] == "real"


def is_goal_env(env_id: str) -> bool:
    parsed = parse_env_id(env_id)
    return parsed is not None and parsed[3]


def list_implemented() -> List[str]:
    env_status = _import_env_status()
    return sorted(
        env_id_for(r, t, m, g) for (r, m, t, g) in env_status.implemented_ids()
    )


def list_unimplemented() -> List[str]:
    env_status = _import_env_status()
    return sorted(
        env_id_for(r, t, m, g) for (r, m, t, g) in env_status.unimplemented_ids()
    )


# ---------------------------------------------------------------------------
# Real-robot motion gate
# ---------------------------------------------------------------------------

ALLOW_REAL_ROBOT_FLAG_ENV = "ALLOW_REAL_ROBOT_MOTION"
ALLOW_REAL_ROBOT_FLAG_PARAM = "/allow_real_robot_motion"


def real_motion_consent_present() -> bool:
    """Read the consent flag without raising. Same logic as the env-side
    ``rl_environments.common.safety.real_robot_flag_set``."""
    try:
        from rl_environments.common.safety import real_robot_flag_set  # type: ignore
        return bool(real_robot_flag_set())
    except Exception:
        # Fall back to env-var check (no rospy available).
        return os.environ.get(ALLOW_REAL_ROBOT_FLAG_ENV, "").lower() in {
            "1", "true", "yes", "on"
        }


def add_real_motion_cli(parser: argparse.ArgumentParser) -> None:
    """Add a ``--allow-real-robot-motion`` flag to ``parser``."""
    parser.add_argument(
        "--allow-real-robot-motion",
        action="store_true",
        default=False,
        help=(
            "Required to construct any UniROS-...Real env in this script. "
            "Setting this flag exports ALLOW_REAL_ROBOT_MOTION=1 in the "
            "current process; you must ALSO `rosparam set "
            "/allow_real_robot_motion true` if the env's MoveIt path "
            "queries the parameter server."
        ),
    )


def enforce_real_motion_consent(env_id: str, allow_real_flag: bool) -> None:
    """
    Raise SystemExit unless we have BOTH:
      * the CLI ``--allow-real-robot-motion`` flag,
      * AND a real-motion consent (rosparam or env var) visible at the
        process level.

    Setting the CLI flag exports ``ALLOW_REAL_ROBOT_MOTION=1`` so that
    the env-side check inside ``rl_environments.common.safety`` will see
    consent without forcing the user to also export it manually.
    """
    if not is_real(env_id):
        return
    if not allow_real_flag:
        raise SystemExit(
            f"[rl_training_validation] {env_id} is a real-robot env. Refusing "
            "to construct without --allow-real-robot-motion. This is a safety "
            "measure to prevent accidental hardware motion."
        )
    # Propagate consent down to the env-side gate.
    os.environ[ALLOW_REAL_ROBOT_FLAG_ENV] = "1"
    if not real_motion_consent_present():
        warnings.warn(
            f"--allow-real-robot-motion set but {ALLOW_REAL_ROBOT_FLAG_ENV} "
            "could not be propagated. The env-side require_real_robot_flag() "
            f"may still raise. Set rosparam {ALLOW_REAL_ROBOT_FLAG_PARAM}=true "
            "to be safe."
        )


# ---------------------------------------------------------------------------
# Combined "is this env safe to construct now?" check
# ---------------------------------------------------------------------------

def check_env_constructable(env_id: str, allow_real_flag: bool = False) -> None:
    """
    Raise SystemExit if ``env_id`` is unimplemented, not a UniROS id, or a
    real env without explicit consent.
    """
    parsed = parse_env_id(env_id)
    if parsed is None:
        raise SystemExit(
            f"[rl_training_validation] '{env_id}' is not a UniROS-* env id "
            "(expected e.g. UniROS-RX200ReachSim-v0)."
        )
    if not is_implemented(env_id):
        raise SystemExit(
            f"[rl_training_validation] {env_id} is registered but blocked "
            "(routes to UnimplementedRLEnv). See "
            "rl_environments/common/env_status.py for the audited status."
        )
    enforce_real_motion_consent(env_id, allow_real_flag)


# ---------------------------------------------------------------------------
# Goal / HER plumbing helpers
# ---------------------------------------------------------------------------

def filter_to_implemented(env_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split a list of env ids into ``(implemented, blocked)``."""
    implemented, blocked = [], []
    for eid in env_ids:
        (implemented if is_implemented(eid) else blocked).append(eid)
    return implemented, blocked


def assert_goal_env(env_id: str) -> None:
    """Raise SystemExit if ``env_id`` is not a goal-conditioned env."""
    if not is_goal_env(env_id):
        raise SystemExit(
            f"[rl_training_validation] {env_id} is not a goal-conditioned env. "
            "HER replay buffers only support goal envs ({...Goal...} ids). "
            "Use a non-HER algorithm or switch to the Goal variant."
        )


def assert_non_goal_env(env_id: str) -> None:
    """Raise SystemExit if ``env_id`` is a goal-conditioned env."""
    if is_goal_env(env_id):
        raise SystemExit(
            f"[rl_training_validation] {env_id} is a goal-conditioned env. "
            "Use a HER-compatible algorithm (e.g. SAC_GOAL / TD3_GOAL) or "
            "switch to the non-Goal variant."
        )

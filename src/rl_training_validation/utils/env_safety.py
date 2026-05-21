#!/usr/bin/env python3
"""
Environment availability + real-robot safety helpers for the training
and validation scripts in this repository.

Pure registry-based: an env is "available" iff it's registered in the
Gymnasium registry (after ``import rl_environments``). No separate
implementation-status table — the registry IS the source of truth.

Goal-env detection is by id suffix (``...Goal*-v0``); real-env detection
is by id suffix (``...Real-v0``).

Nothing in this module touches Gazebo or any ROS topic.
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Gym import (deferred so non-ROS code paths can still import this module)
# ---------------------------------------------------------------------------

def _import_gym():
    try:
        import gymnasium as gym
        import rl_environments  # noqa: F401  triggers registration
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"[rl_training_validation] Cannot import gymnasium / rl_environments: {e}. "
            "Make sure rl_environments is installed (catkin or pip) and that your "
            "workspace's setup.bash is sourced."
        )
    return gym


# ---------------------------------------------------------------------------
# Env-id classification
# ---------------------------------------------------------------------------

# Our registry uses bare ids (no UniROS- prefix). Anything with one of these
# robot prefixes is considered "ours" for listing / cross-check purposes.
_ROBOT_PREFIXES = ("RX200", "NED2", "UR5e", "UR5")


def is_registered(env_id: str) -> bool:
    gym = _import_gym()
    return env_id in gym.envs.registry


# is_implemented kept as an alias so existing callers don't need to change.
is_implemented = is_registered


def is_real(env_id: str) -> bool:
    return env_id.endswith("Real-v0") or env_id.endswith("Real-v1")


def is_goal_env(env_id: str) -> bool:
    # Matches both ...GoalSim-v0 and ...GoalReal-v0.
    return "Goal" in env_id and (env_id.endswith("Sim-v0") or env_id.endswith("Real-v0"))


def list_implemented() -> List[str]:
    gym = _import_gym()
    return sorted(
        s for s in gym.envs.registry
        if any(s.startswith(p) for p in _ROBOT_PREFIXES)
    )


def list_unimplemented() -> List[str]:
    # No status table → nothing is "unimplemented but registered". Kept for
    # backward-compat with callers that expect both lists.
    return []


# ---------------------------------------------------------------------------
# Real-robot motion gate
# ---------------------------------------------------------------------------

ALLOW_REAL_ROBOT_FLAG_ENV = "ALLOW_REAL_ROBOT_MOTION"
ALLOW_REAL_ROBOT_FLAG_PARAM = "/allow_real_robot_motion"


def real_motion_consent_present() -> bool:
    """Check the env-var consent flag; rosparam check would require rospy."""
    return os.environ.get(ALLOW_REAL_ROBOT_FLAG_ENV, "").lower() in {
        "1", "true", "yes", "on",
    }


def add_real_motion_cli(parser: argparse.ArgumentParser) -> None:
    """Add a ``--allow-real-robot-motion`` flag to ``parser``."""
    parser.add_argument(
        "--allow-real-robot-motion",
        action="store_true",
        default=False,
        help=(
            "Required to construct any *Real-v0 env in this script. "
            "Setting this flag also exports ALLOW_REAL_ROBOT_MOTION=1 "
            "in the current process so downstream code can read consent "
            "from a single source; the env var is a propagation of the "
            "same gate, not an independent channel."
        ),
    )


def enforce_real_motion_consent(env_id: str, allow_real_flag: bool) -> None:
    """
    Single-channel consent gate for real-robot env construction.

    Raises ``SystemExit`` for any real env id unless
    ``--allow-real-robot-motion`` was passed. When the flag IS passed,
    also exports ``ALLOW_REAL_ROBOT_MOTION=1`` so downstream code can
    read consent from a single source.

    The env var is a propagation of the CLI flag, not a second
    independent channel — you don't need to set it manually, and
    unsetting it doesn't lock motion out once the flag is already
    passed. If you want a kill-switch that survives the CLI flag,
    enforce it inside the real RobotEnv itself (e.g. a rosparam check
    inside ``_set_init_pose`` / the publish path).
    """
    if not is_real(env_id):
        return
    if not allow_real_flag:
        raise SystemExit(
            f"[rl_training_validation] {env_id} is a real-robot env. Refusing "
            "to construct without --allow-real-robot-motion. This is a safety "
            "measure to prevent accidental hardware motion."
        )
    os.environ[ALLOW_REAL_ROBOT_FLAG_ENV] = "1"
    if not real_motion_consent_present():
        warnings.warn(
            f"--allow-real-robot-motion set but {ALLOW_REAL_ROBOT_FLAG_ENV} "
            "could not be propagated to os.environ for downstream readers."
        )


# ---------------------------------------------------------------------------
# Cube-tracker CLI plumbing for push / pnp real scripts
# ---------------------------------------------------------------------------

def add_cube_tracker_cli(parser: argparse.ArgumentParser) -> None:
    """Add --cube-tracker / --cube-tracker-camera / --cube-tracker-target-frame.

    Default behaviour is unchanged: vision pipeline is external. Pass
    ``--cube-tracker auto`` to have the env roslaunch
    ``rl_envs_cube_tracker/<camera>.launch`` automatically.
    """
    parser.add_argument(
        "--cube-tracker",
        choices=("none", "auto"),
        default="none",
        help=(
            "Vision pipeline source. 'none' (default): assume an external "
            "publisher on --cube-pose-topic. 'auto': env roslaunches "
            "rl_envs_cube_tracker on env construction."
        ),
    )
    parser.add_argument(
        "--cube-tracker-camera",
        choices=("kinect2", "zed2", "d405"),
        default="kinect2",
        help="Which camera launch file rl_envs_cube_tracker uses (only with --cube-tracker auto).",
    )
    parser.add_argument(
        "--cube-tracker-target-frame",
        default="",
        help=(
            "If non-empty, rl_envs_cube_tracker TF-transforms /cube_pose into "
            "this frame (e.g. rx200/base_link). Requires extrinsic calibration; "
            "see rl_envs_cube_tracker/config/extrinsics/README.md."
        ),
    )


def apply_cube_tracker_kwargs(env_kwargs: dict, args: argparse.Namespace) -> dict:
    """Merge cube-tracker CLI args into ``env_kwargs``. Returns it for chaining."""
    env_kwargs["auto_launch_cube_tracker"] = (args.cube_tracker == "auto")
    env_kwargs["cube_tracker_camera"] = args.cube_tracker_camera
    env_kwargs["cube_tracker_target_frame"] = args.cube_tracker_target_frame
    return env_kwargs


# ---------------------------------------------------------------------------
# Wrist-camera CLI plumbing (NED2 sim + real)
# ---------------------------------------------------------------------------

def add_wrist_camera_cli(parser: argparse.ArgumentParser) -> None:
    """Add ``--wrist-camera`` flag (default off).

    NED2 envs accept ``use_wrist_camera: bool`` as a kwarg. Default off
    so there's no extra Gazebo / vision-node load when the user doesn't
    need it.
    """
    parser.add_argument(
        "--wrist-camera",
        action="store_true",
        default=False,
        help=(
            "Enable the Niryo Ned2 wrist camera subscriber. Sim subscribes "
            "to /gazebo_camera/image_raw (raw); real subscribes to "
            "/niryo_robot_vision/compressed_video_stream (compressed). "
            "Decoded frame is exposed as self.cv_image_wrist on the env."
        ),
    )


def apply_wrist_camera_kwargs(env_kwargs: dict, args: argparse.Namespace) -> dict:
    """Merge ``--wrist-camera`` into ``env_kwargs``. Returns it for chaining."""
    env_kwargs["use_wrist_camera"] = bool(args.wrist_camera)
    return env_kwargs


# ---------------------------------------------------------------------------
# Combined "is this env safe to construct now?" check
# ---------------------------------------------------------------------------

def check_env_constructable(env_id: str, allow_real_flag: bool = False) -> None:
    """
    Raise SystemExit if ``env_id`` is unregistered or a real env without
    explicit consent.
    """
    if not is_registered(env_id):
        gym = _import_gym()
        sample = ", ".join(
            sorted(s for s in gym.envs.registry
                   if any(s.startswith(p) for p in _ROBOT_PREFIXES))[:6]
        )
        raise SystemExit(
            f"[rl_training_validation] '{env_id}' is not registered in the "
            f"Gymnasium registry. Available ids include: {sample} ... "
            "Run scripts/list_available_envs.py for the full list."
        )
    enforce_real_motion_consent(env_id, allow_real_flag)


# ---------------------------------------------------------------------------
# Goal / HER plumbing helpers
# ---------------------------------------------------------------------------

def filter_to_implemented(env_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split a list of env ids into ``(registered, missing)``."""
    registered, missing = [], []
    for eid in env_ids:
        (registered if is_registered(eid) else missing).append(eid)
    return registered, missing


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


# parse_env_id kept for backward-compat with scripts that expect a 4-tuple.
def parse_env_id(env_id: str):
    """Return ``(robot, mode, task, is_goal)`` or None.

    Parses bare ids like ``RX200PushGoalSim-v0`` / ``NED2ReacherReal-v0``.
    """
    if not env_id.endswith("-v0"):
        return None
    body = env_id[: -len("-v0")]
    for prefix in sorted(_ROBOT_PREFIXES, key=len, reverse=True):
        if body.startswith(prefix):
            robot_lc = {"RX200": "rx200", "NED2": "ned2",
                        "UR5e": "ur5e", "UR5": "ur5"}[prefix]
            rest = body[len(prefix):]
            break
    else:
        return None
    if rest.endswith("Real"):
        mode = "real"
        rest = rest[: -len("Real")]
    elif rest.endswith("Sim"):
        mode = "sim"
        rest = rest[: -len("Sim")]
    else:
        return None
    is_goal = rest.endswith("Goal")
    if is_goal:
        rest = rest[: -len("Goal")]
    # task: "Reacher" (RX200/NED2) or "Push"/"PnP"/...
    task_map = {"Reacher": "reach", "Push": "push", "PnP": "pnp"}
    # Also accept sensor-flavoured prefixes like "Zed2Reacher".
    for sensor_prefix in ("Zed2", ""):
        if rest.startswith(sensor_prefix):
            tail = rest[len(sensor_prefix):]
            if tail in task_map:
                return robot_lc, mode, task_map[tail], is_goal
    return None

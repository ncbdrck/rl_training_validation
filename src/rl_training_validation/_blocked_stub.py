#!/usr/bin/env python3
"""
Shared stub used by every train/validate script that points at an env id
which the audited ``rl_environments`` registry still has blocked.

Calling :func:`run_blocked_stub` from such a script prints a clear
message and exits ``1`` without constructing the env. This is preferable
to letting the script fail later with a confusing import error or — much
worse — wandering into an unsafe code path on real hardware.

The same stub is reused by the Ned2 / UR5 push / pnp / slide scripts and
all the Ned2 / UR5 / RX200 real push / pnp / slide scripts.
"""
from __future__ import annotations

import argparse
import sys


def run_blocked_stub(env_id: str, *, real: bool, reason: str = "") -> int:
    """Print why the env is blocked and exit ``1``."""
    parser = argparse.ArgumentParser(
        description=f"Stub for blocked env id '{env_id}'."
    )
    parser.add_argument("--allow-real-robot-motion", action="store_true",
                        help="Ignored — this env is registry-blocked.")
    parser.add_argument("--show-implemented", action="store_true",
                        help="Print the list of currently-implemented env ids.")
    args = parser.parse_args()

    try:
        from rl_training_validation.utils.env_safety import (
            list_implemented, parse_env_id, is_implemented,
        )
    except ImportError:
        print(f"[blocked-stub] cannot import env_safety; aborting safely.",
              file=sys.stderr)
        return 1

    # Sanity check: env_status may have been flipped to True in the
    # meantime. If so, the user should rewrite this script properly
    # rather than rely on the stub.
    if is_implemented(env_id):
        print(
            f"[blocked-stub] {env_id} is NOW implemented in env_status. "
            "This script needs to be rewritten to actually construct the "
            "env and run training/validation. See the existing RX200 reach "
            "train script as a template.",
            file=sys.stderr,
        )
        return 2

    print(f"\n[BLOCKED] {env_id} is registered but routes to "
          f"UnimplementedRLEnv (audited status).", file=sys.stderr)
    if reason:
        print(f"  reason: {reason}", file=sys.stderr)
    print("  Construction would raise NotImplementedError. Not starting.",
          file=sys.stderr)
    if real:
        print("  Note: this is a Real env. Even when unblocked, real motion "
              "requires --allow-real-robot-motion AND "
              "/allow_real_robot_motion=true.",
              file=sys.stderr)
    if args.show_implemented:
        print("\nCurrently implemented env ids:", file=sys.stderr)
        for eid in list_implemented():
            print(f"  - {eid}", file=sys.stderr)
    print(
        "\nSee rl_environments/common/env_status.py and the per-area audit "
        "docs in rl_environments/docs/ for what each blocked env needs "
        "before it can be unblocked.\n",
        file=sys.stderr,
    )
    return 1

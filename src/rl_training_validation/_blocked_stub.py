#!/usr/bin/env python3
"""
Shared stub used by every train/validate script that points at an env id
that isn't registered in ``rl_environments`` yet.

Calling :func:`run_blocked_stub` from such a script prints a clear
message and exits ``1`` without trying to ``gym.make`` the env. This is
preferable to letting the script fail later with a confusing import
error or — much worse — wandering into an unsafe code path on real
hardware.

Currently used by the Ned2 / UR5e real and sim push / pnp scripts and
the RX200 real push / pnp scripts (all envs that exist as scripts but
aren't registered yet).
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

    # If the env has been registered since this stub was written, the
    # user should replace this script with a real trainer (mirror an
    # existing one) instead of relying on the stub.
    if is_implemented(env_id):
        print(
            f"[blocked-stub] {env_id} IS NOW registered. "
            "Rewrite this script to construct the env properly. See the "
            "existing RX200 reach train script as a template.",
            file=sys.stderr,
        )
        return 2

    print(f"\n[BLOCKED] {env_id} is not registered in rl_environments.",
          file=sys.stderr)
    if reason:
        print(f"  reason: {reason}", file=sys.stderr)
    print("  Not constructing — would fail with EnvNotFound.", file=sys.stderr)
    if real:
        print("  Note: this is a Real env. Even when registered, real motion "
              "requires --allow-real-robot-motion AND "
              "/allow_real_robot_motion=true.",
              file=sys.stderr)
    if args.show_implemented:
        print("\nCurrently registered env ids:", file=sys.stderr)
        for eid in list_implemented():
            print(f"  - {eid}", file=sys.stderr)
    print(
        "\nRegister the env in rl_environments/src/rl_environments/__init__.py "
        "and add the corresponding task-env class. Then replace this stub "
        "with a real trainer script.\n",
        file=sys.stderr,
    )
    return 1

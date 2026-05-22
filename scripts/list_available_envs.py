#!/usr/bin/env python3
"""
List every RX200 / NED2 / VX300S / UR5e env id in the ``rl_environments``
Gymnasium registry.

Pure introspection — no Gazebo, no ROS, no hardware. Safe to run from
a stock Python interpreter as long as ``rl_environments`` is importable.

Usage::

    python3 scripts/list_available_envs.py          # human-readable table
    python3 scripts/list_available_envs.py --json   # JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="emit JSON instead of a human-readable table")
    args = p.parse_args()

    try:
        from rl_training_validation.utils.env_safety import (
            list_implemented, parse_env_id, is_goal_env, is_real,
        )
    except Exception as e:
        print(f"[ERROR] cannot import env_safety helpers: {e}", file=sys.stderr)
        return 1

    ids = list_implemented()

    if args.as_json:
        print(json.dumps({"count": len(ids), "ids": ids}, indent=2))
        return 0

    by_robot: dict = defaultdict(list)
    for eid in ids:
        parsed = parse_env_id(eid)
        robot = parsed[0] if parsed else "?"
        by_robot[robot].append(eid)

    print(f"\nrl_environments registry — {len(ids)} ids")
    print("=" * 60)
    for robot in ("rx200", "ned2", "vx300s", "ur5e", "ur5", "?"):
        entries = by_robot.get(robot, [])
        if not entries:
            continue
        print(f"\n[{robot}]")
        for eid in entries:
            tags = []
            if is_goal_env(eid):
                tags.append("goal")
            if is_real(eid):
                tags.append("real")
            tag_str = f"  ({', '.join(tags)})" if tags else ""
            print(f"  {eid}{tag_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

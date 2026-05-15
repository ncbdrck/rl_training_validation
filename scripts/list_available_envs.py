#!/usr/bin/env python3
"""
List which UniROS-... env ids in the rl_environments registry are
actually IMPLEMENTED today vs. still blocked behind the
``UnimplementedRLEnv`` placeholder.

Pure introspection — no Gazebo, no ROS, no hardware. Safe to run from
a stock Python interpreter as long as ``rl_environments`` is importable.

Usage::

    python3 scripts/list_available_envs.py            # default table
    python3 scripts/list_available_envs.py --json     # JSON output
    python3 scripts/list_available_envs.py --only-implemented
    python3 scripts/list_available_envs.py --only-blocked
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--only-implemented", action="store_true")
    group.add_argument("--only-blocked", action="store_true")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="emit JSON instead of a human-readable table")
    args = p.parse_args()

    try:
        from rl_training_validation.utils.env_safety import (
            list_implemented, list_unimplemented, parse_env_id,
        )
    except Exception as e:
        print(f"[ERROR] cannot import env_safety helpers: {e}", file=sys.stderr)
        return 1

    implemented = list_implemented()
    blocked = list_unimplemented()
    all_ids = sorted(set(implemented) | set(blocked))

    if args.as_json:
        out = {
            "implemented": implemented,
            "blocked": blocked,
            "counts": {
                "implemented": len(implemented),
                "blocked": len(blocked),
                "total": len(all_ids),
            },
        }
        print(json.dumps(out, indent=2))
        return 0

    show = set()
    if args.only_implemented:
        show.update(implemented)
    elif args.only_blocked:
        show.update(blocked)
    else:
        show.update(all_ids)

    # Group by robot for readability.
    by_robot: dict = defaultdict(list)
    for eid in sorted(show):
        parsed = parse_env_id(eid)
        if parsed is None:
            by_robot["?"].append((eid, "implemented" if eid in implemented else "blocked"))
            continue
        robot = parsed[0]
        by_robot[robot].append((eid, "implemented" if eid in implemented else "blocked"))

    print(f"\nUniROS env registry — {len(implemented)} / {len(all_ids)} implemented\n"
          f"{'=' * 60}")
    for robot in ("rx200", "ned2", "ur5"):
        entries = by_robot.get(robot, [])
        if not entries:
            continue
        print(f"\n[{robot}]")
        for eid, status in entries:
            tag = "✅" if status == "implemented" else "❌"
            print(f"  {tag}  {eid}")

    if not args.only_implemented and not args.only_blocked:
        print("\nLegend: ✅ implemented (entry class exists) — "
              "❌ blocked (constructs to NotImplementedError)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

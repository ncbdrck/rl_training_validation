#!/usr/bin/env python3
"""
Cross-check that every env id referenced by this training repo
(scripts + YAML configs) is registered in ``rl_environments``.

Pure introspection — no Gazebo, no ROS, no hardware.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Set, Tuple

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC)

from rl_training_validation.utils.env_safety import (  # noqa: E402
    is_registered, is_real, list_implemented, parse_env_id,
)


# Match string literals that look like our env ids (any of the known
# robot prefixes, possibly with a Zed2 sensor flavour, ending in -vN).
_ENV_ID_RE = re.compile(
    r"['\"]((?:RX200|NED2|VX300S)[A-Za-z0-9]+-v\d+)['\"]"
)


def _walk_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(exts):
                matches.append(os.path.join(dirpath, f))
    return matches


def _is_audit_script(path: str) -> bool:
    """Audit/smoke scripts may intentionally reference unregistered ids."""
    base = os.path.basename(path)
    return base.startswith(("check_", "smoke_test_", "list_available", "live_smoke"))


def main() -> int:
    print("=" * 60)
    print("  Training-repo env-availability audit")
    print("=" * 60)

    py_files = _walk_files(os.path.join(REPO_ROOT, "src"), (".py",))
    py_files += _walk_files(os.path.join(REPO_ROOT, "scripts"), (".py",))
    yaml_files = _walk_files(os.path.join(REPO_ROOT, "config"), (".yaml", ".yml"))

    issues = 0
    seen_envs: Set[str] = set()
    seen_envs_in_audits: Set[str] = set()

    for path in py_files + yaml_files:
        try:
            text = open(path, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        ids_here = _ENV_ID_RE.findall(text)
        if _is_audit_script(path):
            for m in ids_here:
                seen_envs_in_audits.add(m)
        else:
            for m in ids_here:
                seen_envs.add(m)

    print(f"\nFound {len(seen_envs)} unique env ids in non-audit files.")
    if seen_envs:
        unregistered: List[str] = []
        for eid in sorted(seen_envs):
            parsed = parse_env_id(eid)
            if parsed is None:
                print(f"  FAIL: {eid} — doesn't parse as a known env id")
                unregistered.append(eid)
                issues += 1
                continue
            ok = is_registered(eid)
            real = is_real(eid)
            tag = "ok  " if ok else "MISS"
            extra = " (real, needs --allow-real-robot-motion)" if real else ""
            print(f"  [{tag}] {eid}{extra}")
            if not ok:
                unregistered.append(eid)
        if unregistered:
            print(f"\nUnregistered env ids referenced from non-audit code:")
            for eid in unregistered:
                print(f"  - {eid}")
            print("  Either register the env in rl_environments or remove the "
                  "referencing script.")
            issues += len(unregistered)

    print(f"\nRegistered today ({len(list_implemented())} ids):")
    for eid in list_implemented():
        print(f"  - {eid}")

    print("\n" + "=" * 60)
    if issues == 0:
        print("  PASS: all env ids referenced by this repo are registered.")
        return 0
    print(f"  {issues} issue(s) above need attention.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

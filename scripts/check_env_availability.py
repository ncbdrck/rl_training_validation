#!/usr/bin/env python3
"""
Cross-check that:

  1. ``rl_environments`` is importable.
  2. Every env id this training repo's scripts and configs reference is
     actually IMPLEMENTED in the audited rl_environments status table.
  3. Every ``...Real`` env id is also gated behind
     ``--allow-real-robot-motion`` somewhere in the calling chain (we
     just confirm the env id is real; the gate itself is checked at
     construction time inside the safety module).

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
    is_implemented,
    is_real,
    list_implemented,
    parse_env_id,
)


# Match any string literal that looks like a UniROS env id.
_ENV_ID_RE = re.compile(r"['\"](UniROS-[A-Za-z0-9_]+-v\d+)['\"]")
# Legacy ids the previous training scripts used (pre-audit).
_LEGACY_ID_RE = re.compile(
    r"['\"]((?:RX200|Ned2|UR5)(?:Reacher|Push|PnP|Slide)[A-Za-z]*-v\d+)['\"]"
)


def _walk_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(exts):
                matches.append(os.path.join(dirpath, f))
    return matches


def _is_blocked_stub_file(text: str) -> bool:
    """True if this file's only job is to call run_blocked_stub for a blocked env."""
    return "from rl_training_validation._blocked_stub import run_blocked_stub" in text


def _is_audit_script(path: str) -> bool:
    """Audit/smoke scripts may intentionally reference blocked env ids."""
    base = os.path.basename(path)
    return base.startswith(("check_", "smoke_test_", "list_available"))


def main() -> int:
    print("=" * 60)
    print("  Training-repo env-availability audit")
    print("=" * 60)

    py_files = _walk_files(os.path.join(REPO_ROOT, "src"), (".py",))
    yaml_files = _walk_files(os.path.join(REPO_ROOT, "config"), (".yaml", ".yml"))
    py_files += _walk_files(os.path.join(REPO_ROOT, "scripts"), (".py",))

    issues = 0
    # env ids referenced in real (non-stub) scripts.
    seen_envs: Set[str] = set()
    # env ids that ONLY appear inside blocked-stub scripts (expected).
    seen_envs_in_stubs: Set[str] = set()
    seen_legacy: List[Tuple[str, str]] = []

    for path in py_files + yaml_files:
        try:
            text = open(path, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        ids_here = _ENV_ID_RE.findall(text)
        if _is_blocked_stub_file(text) or _is_audit_script(path):
            for m in ids_here:
                seen_envs_in_stubs.add(m)
        else:
            for m in ids_here:
                seen_envs.add(m)
        for m in _LEGACY_ID_RE.findall(text):
            seen_legacy.append((m, os.path.relpath(path, REPO_ROOT)))

    print(f"\nFound {len(seen_envs)} unique UniROS-... env ids in non-stub files.")
    if seen_envs:
        bad: List[str] = []
        for eid in sorted(seen_envs):
            parsed = parse_env_id(eid)
            if parsed is None:
                print(f"  ❌ {eid}  — not a recognised UniROS id pattern")
                bad.append(eid)
                issues += 1
                continue
            impl = is_implemented(eid)
            real = is_real(eid)
            tag = "✅" if impl else "❌"
            extra = " (real, needs --allow-real-robot-motion)" if real else ""
            print(f"  {tag}  {eid}{extra}")
            if not impl:
                bad.append(eid)
        if bad:
            print(f"\n[BLOCKED env ids referenced from non-stub code] {bad}")
            print("  Update the referencing config/script to use an implemented id "
                  "or rewrite it as a stub via _blocked_stub.run_blocked_stub.")
            issues += len(bad)

    if seen_envs_in_stubs:
        only_stub = seen_envs_in_stubs - seen_envs
        if only_stub:
            print(f"\n[blocked-stub-only references — OK] "
                  f"{len(only_stub)} env ids appear only in blocked-stub scripts:")
            for eid in sorted(only_stub):
                print(f"  ⏸  {eid}")

    print(f"\nImplemented today (for reference, {len(list_implemented())}):")
    for eid in list_implemented():
        print(f"  - {eid}")

    if seen_legacy:
        print(f"\n[Legacy id usage] {len(seen_legacy)} references to pre-audit env ids:")
        for eid, src in sorted(set(seen_legacy)):
            print(f"  ⚠️  {eid} in {src}")
        print("  These ids are NOT registered any more. Replace with their UniROS-... counterparts.")
        issues += len(set(seen_legacy))

    print("\n" + "=" * 60)
    if issues == 0:
        print("  ✅ All env ids referenced by this repo are implemented.")
        return 0
    print(f"  ⚠️  {issues} issue(s) above need attention.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

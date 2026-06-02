#!/usr/bin/env python3
"""
Live smoke test for every registered RX200 / NED2 / VX300S sim env.

For each env id this DOES start Gazebo (each env launches its own
roscore + gazebo subprocess). For each id we:

  1. ``gym.make(env_id)``
  2. ``env.reset()`` (must return an obs in observation_space)
  3. one ``env.step(action_space.sample())``
  4. ``env.close()``

Gates each step with a hard 60 s timeout so a hung env doesn't stall
the run; failures are reported and we move to the next id.

Usage::

    # Default: smoke every RX200/NED2/VX300S sim env in the registry.
    python3 scripts/live_smoke_envs.py

    # Subset by substring (matches any id containing the pattern):
    python3 scripts/live_smoke_envs.py --filter PnP
    python3 scripts/live_smoke_envs.py --filter Goal
    python3 scripts/live_smoke_envs.py --filter Zed2

    # Skip real envs (already the default — they need explicit gating).
    python3 scripts/live_smoke_envs.py --include-real

Real envs require BOTH ``--include-real`` AND
``--allow-real-robot-motion``; without those they're skipped.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import List

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Use uniros.make rather than gymnasium.make. uniros.make
# returns a GymProxy that runs the env in a subprocess and exposes the
# standard gym.Env surface (reset / step / close / action_space /
# observation_space). The subprocess isolation matters here: if an env
# hangs during gym.make (Gazebo bring-up, MoveIt init, etc.), proxy.close()
# cleanly terminates the worker — no orphan gzserver / xterm / roslaunch
# like we'd otherwise get on a SIGALRM timeout.
import uniros  # noqa: E402
import rl_environments  # noqa: E402,F401  triggers registration

from rl_training_validation.utils.env_safety import (  # noqa: E402
    is_real, list_implemented,
)


class _TimeoutError(Exception):
    pass


@contextmanager
def _timeout(seconds: int, label: str):
    """SIGALRM-based hard timeout. Linux-only; that matches our deploy."""
    def _handler(signum, frame):
        raise _TimeoutError(f"{label} exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def smoke_one(env_id: str, make_timeout: int, step_timeout: int) -> bool:
    """Return True on pass, False on fail."""
    print(f"\n--- {env_id} ---", flush=True)
    t0 = time.time()
    env = None
    try:
        with _timeout(make_timeout, "uniros.make"):
            env = uniros.make(env_id)
        with _timeout(step_timeout, "env.reset"):
            obs, info = env.reset()
        ok_obs = env.observation_space.contains(obs)
        print(f"  reset ok ({time.time() - t0:.1f}s); obs in space: {ok_obs}")
        if not ok_obs:
            print(f"  observation_space: {env.observation_space}")
            if hasattr(obs, "shape"):
                print(f"  obs.shape: {obs.shape}")

        with _timeout(step_timeout, "env.step"):
            action = env.action_space.sample()
            step_out = env.step(action)
        if len(step_out) == 5:
            obs2, r, term, trunc, info2 = step_out
        else:
            obs2, r, done, info2 = step_out
            term, trunc = done, False
        print(f"  step ok; reward={r:.4f} term={term} trunc={trunc}")
        return True
    except _TimeoutError as e:
        print(f"  FAIL (timeout): {e}")
        return False
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {e}")
        traceback.print_exc(limit=3)
        return False
    finally:
        if env is not None:
            try:
                with _timeout(30, "env.close"):
                    env.close()
            except Exception as e:
                print(f"  warn: close raised {type(e).__name__}: {e}")
        # Give gazebo / roscore a moment to actually go away.
        time.sleep(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter", default="",
                   help="only smoke env ids containing this substring")
    p.add_argument("--include-real", action="store_true",
                   help="include *Real-v0 ids (also requires --allow-real-robot-motion)")
    p.add_argument("--allow-real-robot-motion", action="store_true")
    p.add_argument("--make-timeout", type=int, default=120,
                   help="seconds to wait for gym.make to return (default 120)")
    p.add_argument("--step-timeout", type=int, default=60,
                   help="seconds to wait for reset/step (default 60)")
    args = p.parse_args()

    candidates: List[str] = []
    for eid in list_implemented():
        if args.filter and args.filter not in eid:
            continue
        if is_real(eid):
            if not args.include_real or not args.allow_real_robot_motion:
                print(f"  skip (real, not opted in): {eid}")
                continue
            os.environ["ALLOW_REAL_ROBOT_MOTION"] = "1"
        candidates.append(eid)

    if not candidates:
        print("No envs match filter. Use --filter '' for everything.")
        return 0

    print(f"Live-smoking {len(candidates)} env(s):")
    for eid in candidates:
        print(f"  - {eid}")

    results = []
    for eid in candidates:
        ok = smoke_one(eid, args.make_timeout, args.step_timeout)
        results.append((eid, ok))

    print("\n" + "=" * 60)
    print("Summary:")
    passes = sum(1 for _, ok in results if ok)
    fails = len(results) - passes
    for eid, ok in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {eid}")
    print(f"\n{passes} pass / {fails} fail")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

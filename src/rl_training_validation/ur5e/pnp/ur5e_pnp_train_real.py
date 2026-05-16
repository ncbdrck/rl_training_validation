#!/usr/bin/env python3
"""
Stub: UR5e Pick-and-Place (real) — train.

Registered as ``UniROS-UR5ePnPReal-v0`` but routes to
:class:`UnimplementedRLEnv` — UR5e real env class is not authored.
To unblock, implement the real UR5e env class and flip env_status.
"""
from __future__ import annotations

import sys

from rl_training_validation._blocked_stub import run_blocked_stub


if __name__ == "__main__":
    sys.exit(run_blocked_stub(
        "UniROS-UR5ePnPReal-v0",
        real=True,
        reason="UR5e real env class not yet implemented.",
    ))

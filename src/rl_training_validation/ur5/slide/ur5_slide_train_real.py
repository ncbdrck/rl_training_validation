#!/usr/bin/env python3
"""
Stub: UR5 Slide (real) — train.

This task is registered as ``UniROS-UR5SlideReal-v0`` but routes to
:class:`UnimplementedRLEnv` (the UR5 real env class is a 0-line stub).
Running this script prints a clear message and exits without
constructing any env. To unblock: implement the UR5 real env class and
flip env_status.
"""
from __future__ import annotations

import sys

from rl_training_validation._blocked_stub import run_blocked_stub


if __name__ == "__main__":
    sys.exit(run_blocked_stub(
        "UniROS-UR5SlideReal-v0",
        real=True,
        reason="UR5 real env class not yet implemented.",
    ))

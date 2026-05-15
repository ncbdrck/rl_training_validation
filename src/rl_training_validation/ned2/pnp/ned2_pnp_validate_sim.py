#!/usr/bin/env python3
"""
Stub: Ned2 Pick-and-Place (sim) — validate.

This task is registered as ``UniROS-Ned2PnPSim-v0`` but routes to
:class:`UnimplementedRLEnv`. Running this script prints a clear
message and exits without constructing any env. To unblock:
update ``rl_environments/common/env_status.py`` AFTER you have
actually implemented and tested the env class.
"""
from __future__ import annotations

import sys

from rl_training_validation._blocked_stub import run_blocked_stub


if __name__ == "__main__":
    sys.exit(run_blocked_stub(
        "UniROS-Ned2PnPSim-v0",
        real=False,
        reason="Ned2 sim cube/table + grasp interface not yet wired.",
    ))

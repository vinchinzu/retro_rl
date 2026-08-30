"""Shared private constants for Norfair controllers.

Pose sets and elevator geometry used by more than one Norfair hop module.
Keep helpers here only when they reduce real duplication — no thin wrappers.
"""

# Ticket k4.

from __future__ import annotations

# Standing / grounded poses (includes knockback 137/138 for settle checks).
_STANDING_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})
# Grounded poses only (exclude knockback 137/138) — door ledges, lip settles.
_LEDGE_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38})
# Business Center elevator platform height after Warehouse arrival.
_ELEVATOR_Y = 680

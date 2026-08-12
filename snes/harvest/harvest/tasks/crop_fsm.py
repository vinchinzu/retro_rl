"""CropWaterTask dual-FSM enums (rr-ds3 / rr-7f54).

``CropState`` is the outer step dispatch (detect/navigate/act/…).
``PlotPhase`` is the per-plot work arm (plant/hoe/water/refill/…).

Both are ``str, Enum`` so legacy string literals still compare equal, but
typed code should use the members and the frozenset groups below.
"""

from __future__ import annotations

from enum import Enum


class CropState(str, Enum):
    DETECT = "detect"
    NAVIGATE = "navigate"
    CENTER = "center"
    ACT = "act"
    VERIFY = "verify"
    TOOL_SWITCH = "tool_switch"
    FENCE_OPEN = "fence_open"
    DONE = "done"


class PlotPhase(str, Enum):
    PLANT = "plant"
    HOE = "hoe"
    WATER = "water"
    REFILL = "refill"
    STAGE_POND = "stage_pond"
    OPEN_POND = "open_pond"


# Membership sets for position / timeout policy (not thrash bands).
ON_APPROACH_PHASES = frozenset(
    {
        PlotPhase.PLANT,
        PlotPhase.WATER,
        PlotPhase.HOE,
        PlotPhase.STAGE_POND,
    }
)
# Soft-timeout owners for pond access (fence_open is CropState, not a phase).
POND_ACCESS_PHASES = frozenset(
    {
        PlotPhase.OPEN_POND,
        PlotPhase.STAGE_POND,
    }
)

WORK_MODE_FULL = "full"
WORK_MODE_ESTABLISH = "establish"
WORK_MODE_WATER = "water"

VALID_WORK_MODES = frozenset({WORK_MODE_FULL, WORK_MODE_ESTABLISH, WORK_MODE_WATER})

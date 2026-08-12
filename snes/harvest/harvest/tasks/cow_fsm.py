"""Cow chores phase enum and shared constants (rr-y80y).

``CowPhase`` is a ``str, Enum`` so legacy string comparisons and test
assignments like ``task._phase = "talk_nav"`` keep working.
"""

from __future__ import annotations

from enum import Enum

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import Tool


class CowPhase(str, Enum):
    TALK_NAV = "talk_nav"
    TALK_VERIFY = "talk_verify"
    BRUSH_SELECT = "brush_select"
    BRUSH_NAV = "brush_nav"
    BRUSH_VERIFY = "brush_verify"
    MILK_SELECT = "milk_select"
    MILK_NAV = "milk_nav"
    MILK_VERIFY = "milk_verify"
    MILK_SHIP_NAV = "milk_ship_nav"
    MILK_SHIP_VERIFY = "milk_ship_verify"
    FODDER_NAV = "fodder_nav"
    FODDER_VERIFY = "fodder_verify"
    FEED_PLACE_NAV = "feed_place_nav"
    FEED_VERIFY = "feed_verify"
    EXIT_PREP_NAV = "exit_prep_nav"
    DONE = "done"


CARE_PHASES = frozenset(
    {
        CowPhase.TALK_NAV,
        CowPhase.TALK_VERIFY,
        CowPhase.BRUSH_SELECT,
        CowPhase.BRUSH_NAV,
        CowPhase.BRUSH_VERIFY,
        CowPhase.MILK_SELECT,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

MILK_CARE_PHASES = frozenset(
    {
        CowPhase.MILK_SELECT,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

TOOL_CARE_PHASES = frozenset(
    {
        CowPhase.BRUSH_NAV,
        CowPhase.BRUSH_VERIFY,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

ADDR_TOOL_SELECTED = field_spec("tool_selected").address
ADDR_TOOL_BACKPACK = field_spec("tool_backpack").address
ADDR_PLAYER_ACTION = field_spec("player_action").address
BRUSH_TOOL_ID = int(Tool.BRUSH)
MILKER_TOOL_ID = int(Tool.MILKER)
MAX_BRUSH_ATTEMPTS = 3
MAX_TALK_ATTEMPTS = 3
MAX_MILK_ATTEMPTS = 3
MAX_COW_SLOT_CARE_FRAMES = 480
# Keep milk attempts well under the external 3600-frame stall watchdog.
# Pixel-lane nav can reset tile stasis while making zero net progress.
MAX_COW_SLOT_MILK_FRAMES = 720
MAX_NAV_FALLBACK_FRAMES = 12
MAX_COW_NAV_FAILURES = 45
MAX_CARE_DEFERRALS = 1
MAX_MILK_DEFERRALS = 2
PIXEL_NAV_STALL_FRAMES = 120
MAX_PIXEL_NAV_STALLS = 2
MAX_EXIT_PREP_FRAMES = 480

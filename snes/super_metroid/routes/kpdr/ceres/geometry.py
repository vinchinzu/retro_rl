"""Ceres geometry constants and room-chain tables.

Named elev / magnet bands and pose sets used by reactive arm-pump navigation.
Do not re-encode these thresholds inline in controllers.
"""

from __future__ import annotations

from super_metroid.takeoff import DEFAULT_PUMP_PERIOD, PlatformHop, TakeoffWindow
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)

# Classic arm-pump period — owned by takeoff, aliased for Ceres callers.
_CERES_ARM_PUMP_PERIOD = DEFAULT_PUMP_PERIOD
# Elevator geometry (smaller y = higher on screen).
# Falling→elev mid-transition can still show y≈139; gs=8 remaps to bottom ~651.
_CERES_ELEV_SHIP_Y = 80  # grounded ship pad band (product leave ~x145 y75 pose 2/10)
_CERES_ELEV_SHIP_X = 145  # product pad center before gs=32 Ceres-success
_CERES_ELEV_TOP_Y = 171  # s10 land / right-wall KB band
_CERES_ELEV_TOP_X = 211  # product right-wall contact (pose 137)
_CERES_ELEV_LEDGE_Y = 571  # mid shaft ledge after bottom LEFT+A
_CERES_ELEV_LEDGE_POSE = 2
_CERES_ELEV_BOTTOM_Y = 640  # bottom floor band after door remap


# 571/475/363 seats from the pin. Jump windows are incoming speed, not
# frame counts — subpixel + momentum decide height/distance.
# Shared type: ``takeoff.PlatformHop`` (every room, not a Ceres-only hop).
CERES_ELEV_HOPS: tuple[PlatformHop, ...] = (
    PlatformHop(571, 40, 130, TakeoffWindow((70, 110), "RIGHT", min_momentum=1)),
    PlatformHop(475, 90, 180, TakeoffWindow((118, 158), "RIGHT", min_momentum=1)),
    PlatformHop(363, 150, 220, TakeoffWindow((165, 205), "LEFT", min_momentum=1)),
)
# Magnet escape: leave door height ~y139; outbound mid ~y395.
_CERES_MAGNET_EXIT_Y = 200  # y at/below this → high enough for left exit

# Dead Scientist Room 0xE021: raised door alcoves (y≈139) over a pit (y≈187).
# Left alcove x≲80 — walk down, never jump (A bonks the ceiling / stalls).
# Right stairs takeoff is the pit floor, not the door lip.
_CERES_SCI_DOOR_Y = 139
_CERES_SCI_FLOOR_Y = 187
_CERES_SCI_ENTRY_LEDGE_X = 90
CERES_SCIENTIST_FLOOR_HOP = PlatformHop(
    _CERES_SCI_FLOOR_Y,
    280,
    430,
    TakeoffWindow((350, 410), "RIGHT", min_momentum=1),
)

# Outbound room chain (rightward).
_CERES_OUTBOUND_CHAIN = (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
    ROOM_CERES_FLAT,
    ROOM_CERES_RIDLEY,
)
# Escape reverse chain (leftward) before elevator shaft.
_CERES_ESCAPE_CHAIN = (
    ROOM_CERES_RIDLEY,
    ROOM_CERES_FLAT,
    ROOM_CERES_SCIENTIST,
    ROOM_CERES_MAGNET,
    ROOM_CERES_FALLING,
    ROOM_CERES_ELEVATOR,
)

__all__ = [
    "_CERES_ARM_PUMP_PERIOD",
    "_CERES_ELEV_SHIP_Y",
    "_CERES_ELEV_SHIP_X",
    "_CERES_ELEV_TOP_Y",
    "_CERES_ELEV_TOP_X",
    "_CERES_ELEV_LEDGE_Y",
    "_CERES_ELEV_LEDGE_POSE",
    "_CERES_ELEV_BOTTOM_Y",
    "CERES_ELEV_HOPS",
    "_CERES_MAGNET_EXIT_Y",
    "_CERES_SCI_DOOR_Y",
    "_CERES_SCI_FLOOR_Y",
    "_CERES_SCI_ENTRY_LEDGE_X",
    "CERES_SCIENTIST_FLOOR_HOP",
    "_CERES_OUTBOUND_CHAIN",
    "_CERES_ESCAPE_CHAIN",
]

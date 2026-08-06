"""Ceres geometry constants and room-chain tables.

Named elev / magnet bands and pose sets used by reactive arm-pump navigation.
Do not re-encode these thresholds inline in controllers.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)

# Classic arm-pump: dir+B with L↔R angle spam (``runway_dash`` period-2).
_CERES_ARM_PUMP_PERIOD = 2
# Knockback poses (shared with routes/skills/knockback).
_CERES_KB_POSES = frozenset({137, 138})
# Wall-latch pose (WJ ready).
_CERES_WALL_LATCH = 132
# Elevator geometry (smaller y = higher on screen).
# Falling→elev mid-transition can still show y≈139; gs=8 remaps to bottom ~651.
_CERES_ELEV_SHIP_Y = 80  # grounded ship pad band (product leave ~x145 y75 pose 2/10)
_CERES_ELEV_SHIP_X = 145  # product pad center before gs=32 Ceres-success
_CERES_ELEV_TOP_Y = 171  # s10 land / right-wall KB band
_CERES_ELEV_TOP_X = 211  # product right-wall contact (pose 137)
_CERES_ELEV_LEDGE_Y = 571  # mid shaft ledge after bottom LEFT+A
_CERES_ELEV_LEDGE_POSE = 2
_CERES_ELEV_BOTTOM_Y = 640  # bottom floor band after door remap
# Magnet escape: leave door height ~y139; outbound mid ~y395.
_CERES_MAGNET_EXIT_Y = 200  # y at/below this → high enough for left exit

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
    "_CERES_KB_POSES",
    "_CERES_WALL_LATCH",
    "_CERES_ELEV_SHIP_Y",
    "_CERES_ELEV_SHIP_X",
    "_CERES_ELEV_TOP_Y",
    "_CERES_ELEV_TOP_X",
    "_CERES_ELEV_LEDGE_Y",
    "_CERES_ELEV_LEDGE_POSE",
    "_CERES_ELEV_BOTTOM_Y",
    "_CERES_MAGNET_EXIT_Y",
    "_CERES_OUTBOUND_CHAIN",
    "_CERES_ESCAPE_CHAIN",
]

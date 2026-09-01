"""Ceres geometry constants and room-chain tables.

Named elev / magnet bands and pose sets used by reactive arm-pump navigation.
Do not re-encode these thresholds inline in controllers.
"""

from __future__ import annotations

from pathlib import Path

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
# Outbound first room (wiki Ceres 1 / Sniq 100% lsnes): pad y=72, short
# RIGHT hop, air-turn pose 25→26 at ~x142 y79, land y=75 (pose 229), spinning
# moonfall. Weave idles past 171/267; floor door ~(238, 651) pose 17.
_CERES_FIRST_PAD_Y = 72
_CERES_FIRST_TURN_X = 142
_CERES_FIRST_TURN_Y = 76
_CERES_FIRST_FLOOR_Y = 640
_CERES_FIRST_DOOR_X = 230
_CERES_ELEV_SHIP_Y = 80  # grounded ship pad band (product leave ~x145 y75 pose 2/10)
_CERES_ELEV_SHIP_X = 145  # product pad center before gs=32 Ceres-success
_CERES_ELEV_TOP_Y = 171  # s10 land / right-wall KB band
_CERES_ELEV_TOP_X = 211  # product right-wall contact (pose 137)
_CERES_ELEV_LEDGE_Y = 571  # mid-shaft ledge after a bottom LEFT+A recovery
_CERES_ELEV_BOTTOM_Y = 640  # bottom floor band after door remap
# Falling west door (TAS lsnes sniq_100): hop the y≈139 ledge and turn to
# pose 25 at x≲28. The reactive route takes a debris knockback that the TAS
# does not, so it releases A, rebuilds mx=2, and jumps farther inside.
# A floor transition remaps at y=651; a live jump must still be rising.
_CERES_FALLING_DOOR_LEDGE_Y = 139
_CERES_FALLING_DOOR_TURN_X = 28
_CERES_FALLING_DOOR_JUMP_X = 23
# Reverse Falling: run off y=139 onto y=187, hop x≈347 onto y=171, run
# LEFT, turn RIGHT at x≤314 leftover LEFT mx, 1f p25, B-only fall (no X),
# LEFT+B+A on p83/p80. Do not hold A into the y≈110 ceiling. RIGHT on the
# hit frame is p84 knock RIGHT. Door is pose 25 at (26,120) mx=2.
_CERES_FALLING_REV_FLOOR_Y = 187
_CERES_FALLING_REV_SHELF_Y = 171
_CERES_FALLING_REV_TILE_X = 300
# 1f p38 + 1f p25 + fall (no X) before 294 contact. 312+X is p47 overshoot.
_CERES_FALLING_REV_TURN_X = 314
CERES_FALLING_REV_FLOOR_HOP = PlatformHop(
    _CERES_FALLING_REV_FLOOR_Y,
    330,
    380,
    TakeoffWindow((342, 352), "LEFT", min_momentum=1),
)
# Outbound Falling Tile (wiki Ceres 2 / Sniq 100% lsnes): run off the y=139
# entry, short-hop the y=187 floor at x≈145–162 with LEFT+RIGHT+A so magnet
# feet plant y=171, L-pump the shelf, then jump at x≈326–337 into the right door.
_CERES_FALLING_OUT_ENTRY_Y = 139
_CERES_FALLING_OUT_FLOOR_Y = 187
_CERES_FALLING_OUT_PLAT_Y = 171
_CERES_FALLING_OUT_DOOR_X = 470
CERES_FALLING_FLOOR_HOP = PlatformHop(
    _CERES_FALLING_OUT_FLOOR_Y,
    140,
    180,
    TakeoffWindow((145, 168), "RIGHT", min_momentum=1),
)
CERES_FALLING_EXIT_HOP = PlatformHop(
    _CERES_FALLING_OUT_PLAT_Y,
    300,
    360,
    TakeoffWindow((320, 345), "RIGHT", min_momentum=1),
)
# Magnet escape: leave door height ~y139; outbound mid ~y395.
CERES_DATA_DIR = Path(__file__).resolve().parent / "data"
# Stair-top magnet-stop plants x≈85 y=347 mx=0. Jump window is HIGH_HOP (70, 78).
_CERES_MAGNET_STOP_X = 92
# East steam burns on this pin; wait before the door hop.
_CERES_MAGNET_DOOR_STEAM_FRAMES = 6
# Outbound Magnet Stairs (wiki Ceres 3 / Sniq 100% lsnes): from the west
# door (39, 139) run RIGHT, jump the y=139 ledge at x≈126–140, air-turn
# LEFT onto y=219 (~x192, not magnet-feet 219), run LEFT, jump the slope
# around y=255 x≈131, land y=347, RIGHT to the east door ~(236, 395).
# https://wiki.supermetroid.run/KPDR_Room_Strategies#Ceres_3
# Reverse (Sniq 100% lsnes magnet_escape): no east-door jump. Run 395
# stairs onto 347, jump x≈68–82 LEFT then air-turn RIGHT onto 267. Run
# RIGHT, jump x≈112–128, steam d-boost to 219, jump to 139. Do not jump
# x≤50 (west magnet-stop) or x≥85 from 347 (267 underside).
_CERES_MAGNET_TOP_Y = 139
_CERES_MAGNET_MID_Y = 219
_CERES_MAGNET_SHELF_Y = 267
_CERES_MAGNET_SLOPE_Y = 255
_CERES_MAGNET_BOT_Y = 347
_CERES_MAGNET_DOOR_Y = 395
_CERES_MAGNET_OUT_DOOR_X = 230
CERES_MAGNET_TOP_HOP = PlatformHop(
    _CERES_MAGNET_TOP_Y,
    40,
    180,
    TakeoffWindow((126, 145), "RIGHT", min_momentum=1),
)
CERES_MAGNET_MID_HOP = PlatformHop(
    _CERES_MAGNET_SLOPE_Y,
    110,
    145,
    TakeoffWindow((120, 138), "LEFT", min_momentum=1),
)
CERES_MAGNET_HIGH_HOP = PlatformHop(
    _CERES_MAGNET_BOT_Y,
    40,
    _CERES_MAGNET_STOP_X,
    TakeoffWindow((70, 78), "LEFT", min_momentum=1),
)
CERES_MAGNET_STEAM_HOP = PlatformHop(
    _CERES_MAGNET_SHELF_Y,
    60,
    140,
    TakeoffWindow((112, 128), "RIGHT", min_momentum=1),
)
CERES_MAGNET_MID_ESCAPE_HOP = PlatformHop(
    _CERES_MAGNET_MID_Y,
    170,
    210,
    TakeoffWindow((180, 200), "RIGHT", min_momentum=0),
)

# Dead Scientist Room 0xE021: raised door alcoves (y≈139) over a pit (y≈187).
# Sniq 100% lsnes never jumps: RIGHT+B+L/R off the left lip, run y=187, run
# the right stairs, east door ~(492,139). A on the alcove bonks the ceiling.
# Old floor takeoff x 350–410 caught the east ledge at _CERES_SCI_EXIT_LEDGE_X.
_CERES_SCI_DOOR_Y = 139
_CERES_SCI_FLOOR_Y = 187
_CERES_SCI_ENTRY_LEDGE_X = 90
_CERES_SCI_EXIT_LEDGE_X = 467

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
    "_CERES_FIRST_PAD_Y",
    "_CERES_FIRST_TURN_X",
    "_CERES_FIRST_TURN_Y",
    "_CERES_FIRST_FLOOR_Y",
    "_CERES_FIRST_DOOR_X",
    "_CERES_ELEV_SHIP_Y",
    "_CERES_ELEV_SHIP_X",
    "_CERES_ELEV_TOP_Y",
    "_CERES_ELEV_TOP_X",
    "_CERES_ELEV_LEDGE_Y",
    "_CERES_ELEV_BOTTOM_Y",
    "_CERES_FALLING_DOOR_LEDGE_Y",
    "_CERES_FALLING_DOOR_TURN_X",
    "_CERES_FALLING_DOOR_JUMP_X",
    "_CERES_FALLING_REV_FLOOR_Y",
    "_CERES_FALLING_REV_SHELF_Y",
    "_CERES_FALLING_REV_TILE_X",
    "_CERES_FALLING_REV_TURN_X",
    "CERES_FALLING_REV_FLOOR_HOP",
    "_CERES_FALLING_OUT_ENTRY_Y",
    "_CERES_FALLING_OUT_FLOOR_Y",
    "_CERES_FALLING_OUT_PLAT_Y",
    "_CERES_FALLING_OUT_DOOR_X",
    "CERES_FALLING_FLOOR_HOP",
    "CERES_FALLING_EXIT_HOP",
    "_CERES_MAGNET_TOP_Y",
    "_CERES_MAGNET_MID_Y",
    "_CERES_MAGNET_SHELF_Y",
    "_CERES_MAGNET_SLOPE_Y",
    "_CERES_MAGNET_BOT_Y",
    "_CERES_MAGNET_DOOR_Y",
    "_CERES_MAGNET_OUT_DOOR_X",
    "CERES_DATA_DIR",
    "CERES_MAGNET_TOP_HOP",
    "CERES_MAGNET_MID_HOP",
    "CERES_MAGNET_HIGH_HOP",
    "CERES_MAGNET_STEAM_HOP",
    "CERES_MAGNET_MID_ESCAPE_HOP",
    "_CERES_MAGNET_STOP_X",
    "_CERES_MAGNET_DOOR_STEAM_FRAMES",
    "_CERES_SCI_DOOR_Y",
    "_CERES_SCI_FLOOR_Y",
    "_CERES_SCI_ENTRY_LEDGE_X",
    "_CERES_SCI_EXIT_LEDGE_X",
    "_CERES_OUTBOUND_CHAIN",
    "_CERES_ESCAPE_CHAIN",
]

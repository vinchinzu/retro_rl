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
# moonfall. Weave idles past 171/267. Extra L at p17 x=205 is not TAS 8782
# B+RIGHT at (206, p17); skip once. Last-floor L on p17 carries leftover
# pose 17 at (39, 139). That pose-17 is held-aim; TAS leftover 17 + B+RIGHT
# is pose 15 at x=44, ours is still pose 9.
_CERES_FIRST_PAD_Y = 72
_CERES_FIRST_TURN_X = 142
_CERES_FIRST_TURN_Y = 76
_CERES_FIRST_FLOOR_Y = 640
_CERES_FIRST_DOOR_X = 230
# First inverted pulse (p17 + tape L). Do not L-every-p17 after 214.
_CERES_FIRST_INVERT_L_X = 205
_CERES_FIRST_INVERT_L_X_END = 214
# gs=9 through last gs=11 (Sniq 100% lsnes f8789–8949). B+RIGHT+R is the
# last fade frame; an L-pulse on the lip triggers 1px early (237 not 238).
_CERES_FIRST_DOOR_FADE = 161
_CERES_ELEV_SHIP_Y = 80  # grounded ship pad band (product leave ~x145 y75 pose 2/10)
_CERES_ELEV_SHIP_X = 145  # product pad center before gs=32 Ceres-success
_CERES_ELEV_TOP_Y = 171  # s10 land / right-wall KB band
_CERES_ELEV_TOP_X = 211  # shaft right wall; the entry wall jump kicks off it
_CERES_ELEV_LEDGE_Y = 571  # mid-shaft ledge; not a product recovery seat
_CERES_ELEV_BOTTOM_Y = 640  # bottom floor band after a missed door jump
# Shaft climb from the Falling-door entry (measured off ceres_first_control
# in scratch/ceres_elev_wj). The entry arrives four air frames into a spin
# jump, so the entry rise alone tops out at y=608 and the y=475 ledge is only
# reachable by riding the `_CERES_ELEV_TOP_X` right wall and kicking off it. Sniq's own
# elev_wj tape replays to the right places from this entry but never latches:
# it releases A for a single frame and stable-retro needs two.
#
# Above 475 the rungs are ordinary ground spin jumps between ledges, not wall
# jumps — a full ground spin jump rises 111px and the gaps are 112/96/96.
# Each launch x is the middle of its measured band; outside the band the jump
# clips a ledge lip and drops back down the shaft.
_CERES_ELEV_ENTRY_RISE_FRAMES = 16  # RIGHT+A up the wall before the kick
_CERES_ELEV_WJ_RELEASE_FRAMES = 2  # LEFT without A; 1f reads as a jump cut
_CERES_ELEV_WJ_KICK_FRAMES = 8
_CERES_ELEV_WJ_RIDE_FRAMES = 34  # pose 132 / movement type 20 carries to 474
_CERES_ELEV_475_LAUNCH_X = 137  # band 130-144, RIGHT onto 363
_CERES_ELEV_363_LAUNCH_X = 191  # band 185-211, LEFT onto 267
_CERES_ELEV_267_LAUNCH_X = 144  # band 132-156, LEFT onto 171
_CERES_ELEV_171_LAUNCH_X = 48  # band 41-55, RIGHT onto the ship pad (gs 32)
# Falling west door (TAS lsnes sniq_100). x=45 is the Ceres door enemy
# $E23F (`CERES_DOOR_ID`), shut for ~16f after the ledge: walking into it is
# pose-138 / movement-type-21, which zeroes momentum and parks the jump on
# the y=108 ceiling. Crouch it out east of x=45 instead, then run LEFT and
# jump at x<=33 so the leave is the 4th air frame — pose 26 rising vy=4 at
# y≈121, elev (216, 633) against TAS (216, 632). min_momentum is 1, not 2:
# ground momentum caps at 2.75 and halves once on the second air frame, so
# momentum_x reads 1 from there on however long the runway is. Floor remap
# y=651 is still a missed WJ.
_CERES_FALLING_DOOR_LEDGE_Y = 139
_CERES_FALLING_DOOR_JUMP_X = 33
# $E23F shutter wait, same shape as _CERES_MAGNET_DOOR_STEAM_FRAMES.
_CERES_FALLING_DOOR_SHUTTER_FRAMES = 10
CERES_FALLING_DOOR_HOP = PlatformHop(
    _CERES_FALLING_DOOR_LEDGE_Y,
    16,
    70,
    TakeoffWindow((16, _CERES_FALLING_DOOR_JUMP_X), "LEFT", min_momentum=1),
)
# Reverse Falling: run off y=139 onto y=187, hop x≈347 onto y=171, run
# LEFT, turn RIGHT at x≤314 leftover LEFT mx, 1f p25, B-only fall (no X),
# LEFT+B+A on p83/p80. Do not hold A into the y≈110 ceiling. RIGHT on the
# hit frame is p84 knock RIGHT. Door leave is pose 26 rising at (19,121).
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
    "_CERES_FIRST_INVERT_L_X",
    "_CERES_FIRST_INVERT_L_X_END",
    "_CERES_FIRST_DOOR_FADE",
    "_CERES_ELEV_SHIP_Y",
    "_CERES_ELEV_SHIP_X",
    "_CERES_ELEV_TOP_Y",
    "_CERES_ELEV_TOP_X",
    "_CERES_ELEV_LEDGE_Y",
    "_CERES_ELEV_BOTTOM_Y",
    "_CERES_ELEV_ENTRY_RISE_FRAMES",
    "_CERES_ELEV_WJ_RELEASE_FRAMES",
    "_CERES_ELEV_WJ_KICK_FRAMES",
    "_CERES_ELEV_WJ_RIDE_FRAMES",
    "_CERES_ELEV_475_LAUNCH_X",
    "_CERES_ELEV_363_LAUNCH_X",
    "_CERES_ELEV_267_LAUNCH_X",
    "_CERES_ELEV_171_LAUNCH_X",
    "_CERES_FALLING_DOOR_LEDGE_Y",
    "_CERES_FALLING_DOOR_JUMP_X",
    "_CERES_FALLING_DOOR_SHUTTER_FRAMES",
    "CERES_FALLING_DOOR_HOP",
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

"""Early Spazer Beam — **mainline K2.2** (Below climb → collect → return → West).

Spazer Room ``0xA447`` is on the continuous spine. Power-on reaches Below
Spazer ``0xA408`` with **Charge** from K1 (``play_big_pink_to_ghz``).

Mainline fuse:
:func:`~super_metroid.routes.kpdr.red_stack.play_below_spazer_to_west` →
:func:`play_spazer_detour` → West Tunnel ``0xCF54`` → Glass → East → Warehouse.

**Always detour when Spazer is missing** (floor entry included). There is no
Charge-only West skip on the product path.

Package layout
--------------
* ``geometry`` — named bands + pure predicates
* ``scripts`` — guide-shaped RLE tables
* ``helpers`` — ``play_script`` / lag break
* ``climb`` — floor→mid→solid top
* ``approach`` — solid top → Super door → Spazer Room
* ``collect`` — pedestal + return handoff
* ``drop`` — top→mid/floor → West
* ``detour`` — product fuse

West floor runner lives in
:mod:`super_metroid.routes.kpdr.below_spazer_west` (no import cycle with
red_stack).
"""

from __future__ import annotations

from super_metroid.routes.kpdr.rooms import ROOM_SPAZER
from super_metroid.routes.kpdr.spazer.approach import (
    approach_super_door_from_top,
    play_below_spazer_to_spazer,
)
from super_metroid.routes.kpdr.spazer.climb import (
    play_below_spazer_climb,
    play_below_spazer_floor_to_mid,
    play_below_spazer_mid_to_top,
)
from super_metroid.routes.kpdr.spazer.collect import (
    play_spazer_collect,
    play_spazer_return_to_below,
)
from super_metroid.routes.kpdr.spazer.detour import play_spazer_detour
from super_metroid.routes.kpdr.spazer.drop import (
    play_spazer_top_to_mid,
    play_spazer_top_to_west,
)
from super_metroid.routes.kpdr.spazer.geometry import (
    SOLID_TOP_X_MIN,
    SOLID_TOP_Y,
    SPAZER_BEAM_MASK,
    WJ_LEFT,
    WJ_PAIR,
    WJ_RIGHT,
    mid_band,
    on_mid_or_floor,
    on_solid_top,
    on_super_door_approach,
    solid_ish_top,
    standing_mid_seat,
)
from super_metroid.routes.kpdr.spazer.scripts import (
    FLOOR_MID_RLE,
    TOP_DOOR_APPROACH_RLE,
    TOP_MID_RLE,
)

# Compat aliases used by probes / older residual notes.
_FLOOR_MID_RLE = FLOOR_MID_RLE
_TOP_MID_RLE = TOP_MID_RLE
_TOP_DOOR_APPROACH_RLE = TOP_DOOR_APPROACH_RLE
_SOLID_TOP_Y = SOLID_TOP_Y
_SOLID_TOP_X_MIN = SOLID_TOP_X_MIN
_SPAZER_WJ_LEFT = WJ_LEFT
_SPAZER_WJ_RIGHT = WJ_RIGHT
_SPAZER_WJ_PAIR = WJ_PAIR
_on_solid_top = on_solid_top
_standing_mid_seat = standing_mid_seat
_mid_band = mid_band
_on_mid_or_floor = on_mid_or_floor
_on_super_door_approach = on_super_door_approach
_solid_ish_top = solid_ish_top
_approach_super_door_from_top = approach_super_door_from_top

__all__ = [
    "ROOM_SPAZER",
    "SPAZER_BEAM_MASK",
    "play_below_spazer_climb",
    "play_below_spazer_floor_to_mid",
    "play_below_spazer_mid_to_top",
    "play_below_spazer_to_spazer",
    "play_spazer_collect",
    "play_spazer_detour",
    "play_spazer_return_to_below",
    "play_spazer_top_to_mid",
    "play_spazer_top_to_west",
]

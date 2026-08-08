"""Ice / Moat / West Ocean human-tape guide stubs (post-Wave K5–K6)."""

from __future__ import annotations

from retro_harness.path_overlay import RoomGuide
from super_metroid.routes.kpdr.guides.common import (
    ROOM_KIHUNTER,
    ROOM_MOAT,
    ROOM_WEST_OCEAN,
    ROOM_WS_ENTRANCE,
    _C,
    _pts,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUSINESS,
    ROOM_ICE,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
)

# Business Center (0xA7DE): mid shaft Super green LEFT → Ice Gate.
# Pins from speed_to_wave_ice_moat_human.json + pure dual ~(61,922) Super plant.
GUIDE_BUSINESS_TO_ICE = RoomGuide(
    room_id=ROOM_BUSINESS,
    name="Business → Ice Gate",
    color=_C["ice"],
    points=_pts(
        (128, 680, "elev-settle"),
        (100, 920, "door-band"),
        (61, 922, "green-super"),  # Super plant left → Ice Gate
    ),
)

# Pure dual entry settles right lip ~(1752, 630–651); Acid exit is further left.
GUIDE_ICE_GATE = RoomGuide(
    room_id=ROOM_ICE_GATE,
    name="Ice Beam Gate Room",
    color=_C["ice"],
    points=_pts(
        (1752, 651, "business-entry"),
        (900, 651, "mid-right"),
        (40, 651, "acid-left"),  # → Acid Room 0xA75D (tape path)
    ),
)

GUIDE_ICE_TUTORIAL = RoomGuide(
    room_id=ROOM_ICE_TUTORIAL,
    name="Ice Tutorial",
    color=_C["ice"],
    points=_pts(
        (200, 140, "entry"),
        (40, 140, "left-door"),
    ),
)

GUIDE_ICE_SNAKE = RoomGuide(
    room_id=ROOM_ICE_SNAKE,
    name="Ice Snake Room",
    color=_C["ice"],
    points=_pts(
        (40, 200, "entry"),
        (200, 140, "climb"),
        (400, 140, "right-door"),
    ),
)

GUIDE_ICE_ROOM = RoomGuide(
    room_id=ROOM_ICE,
    name="Ice Beam collect",
    color=_C["ice"],
    points=_pts(
        (40, 140, "entry"),
        (120, 180, "chozo"),  # approximate Ice PLM
        (40, 140, "exit"),
    ),
)

# Crateria Kihunter → Moat (K6 stretch; pure spark already green from pin).
GUIDE_KIHUNTER_MOAT = RoomGuide(
    room_id=ROOM_KIHUNTER,
    name="Crateria Kihunter pre-Moat",
    color=_C["moat"],
    points=_pts(
        (100, 180, "clear"),
        (200, 180, "runway"),
        (400, 178, "charge"),
        (503, 178, "trench"),
        (555, 140, "hop-spark"),
    ),
)

GUIDE_MOAT = RoomGuide(
    room_id=ROOM_MOAT,
    name="The Moat",
    color=_C["moat"],
    points=_pts(
        (40, 120, "entry-spark"),
        (250, 120, "corridor"),
        (475, 120, "jam"),
        (700, 140, "blue-door"),  # West Ocean
    ),
)

# West Ocean (0x93FE): post-Moat spark lower-left entry → lower green Super
# door to Wrecked Ship. Waypoints are coarse (8×6 room); refine after first
# human tape. Pin: scratch/post_moat_west_ocean_spark.state ~(49,1163).
GUIDE_WEST_OCEAN = RoomGuide(
    room_id=ROOM_WEST_OCEAN,
    name="West Ocean",
    color=_C["west_ocean"],
    points=_pts(
        (49, 1163, "moat-entry"),  # pure spark handoff
        (300, 1179, "lower-left"),
        (700, 1179, "lower-mid"),
        (1200, 1179, "lower-right-run"),
        (1700, 1163, "pre-green"),
        (2010, 1163, "green-super-ws"),  # lower bottom-right → 0xCA08
    ),
)

# Wrecked Ship Entrance (0xCA08): free-play after green door; coarse center.
GUIDE_WS_ENTRANCE = RoomGuide(
    room_id=ROOM_WS_ENTRANCE,
    name="Wrecked Ship Entrance",
    color=_C["ws"],
    points=_pts(
        (40, 180, "west-door"),
        (200, 180, "mid"),
        (360, 180, "ship-shaft"),
    ),
)

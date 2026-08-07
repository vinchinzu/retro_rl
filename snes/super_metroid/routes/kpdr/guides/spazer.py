"""Early Spazer human guides: Below Spazer climb, collect, top-drop return."""

from __future__ import annotations

from retro_harness.path_overlay import RoomGuide
from super_metroid.routes.kpdr.guides.common import _C, _pts
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_SPAZER

# Below Spazer (0xA408) — early Spazer wall-jump detour (not West Tunnel bottom).
# Room is 2×2 screens (512×512). Preferred source (Charge on spine):
# post_below_spazer_with_charge_continuous / pre_spazer_door_with_charge.
# Legacy no-Charge: post_below_spazer_for_spazer_pure (power beam X only).
#
# Climb: left shaft (sm-json 1→4 canWallJump). Floor→mid spin to y≲300 (human
# peak ~284); mid→top double WJ probe-green from y≈260 → top ledge ~y125.
# Keep x≥40 — left door is Bat (door_ptr 0x9102); hugging it door-traps.
# Bomb top gap (4→3), Super green door block [31,7] → Spazer Room.
# TRAP: bottom-right blue door [31,23] is West Tunnel (any% skip path).
GUIDE_BELOW_SPAZER_EARLY = RoomGuide(
    room_id=ROOM_BELOW_SPAZER,
    name="Below Spazer early WJ → green Super",
    color=_C["below_spazer"],
    points=_pts(
        (49, 395, "entry"),  # continuous-like source / left door settle
        (60, 395, "off-door"),  # nudge RIGHT first — avoid Bat door suck
        (42, 300, "spin-mid"),  # floor→mid spin peak band (human ~284)
        (55, 260, "wj-zone"),  # double-WJ works from here (probe min_y~125)
        (50, 200, "wj-2"),
        (55, 150, "wj-3"),
        (59, 126, "top-left"),  # node 4 ledge (probe land)
        (200, 120, "bomb-gap"),  # morph bombs toward top right
        (320, 120, "mid-plat"),
        (420, 120, "ledge"),
        (480, 120, "green-door"),  # Super → Spazer (block ~31,7)
    ),
)

# Spazer Room (0xA447) — 1 screen; Chozo item block [11,9] → ~(176,144).
# Collect then return LEFT through blue door block [0,7] back to Below Spazer.
GUIDE_SPAZER_ROOM = RoomGuide(
    room_id=ROOM_SPAZER,
    name="Spazer collect + return",
    color=_C["spazer"],
    points=_pts(
        (40, 121, "entry"),  # just inside left door after Super
        (120, 140, "approach"),
        (176, 144, "chozo"),  # Spazer pedestal
        (120, 140, "return"),
        (40, 121, "exit-left"),  # back to Below Spazer top right
    ),
)

# Return handoff → clean mid/floor drop (SM-SPAZER-TOP-MID human source).
# Pure pin: post_spazer_return_pure ~(380,155) beams 0x1004, clear of Super.
# Do **not** RIGHT (open Super re-enters Spazer). Clean drop only — never the
# excluded enemy-fall thrash (human f9403–11066).
GUIDE_BELOW_SPAZER_TOP_DROP = RoomGuide(
    room_id=ROOM_BELOW_SPAZER,
    name="Below Spazer top→mid clean drop",
    color=_C["below_spazer"],
    points=_pts(
        (380, 155, "handoff"),  # pure return pin, x≲400
        (320, 155, "left-safe"),  # further clear of Super door
        (280, 200, "drop-1"),
        (240, 235, "mid-seat"),  # mid platform band y≥220
        (180, 280, "drop-2"),
        (120, 360, "floor-band"),  # toward floor / West path
        (80, 395, "floor"),  # stock floor before West
    ),
)

# Climb polyline remains the room default in GUIDE_BY_ROOM for early-spazer;
# top-drop uses its own route preset (same room_id, different ROUTE_PRESETS).

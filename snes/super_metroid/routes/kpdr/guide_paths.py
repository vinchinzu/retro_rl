"""Human-recording guide polylines for KPDR practice segments.

Waypoints are approximate KPDR route pins (room world pixels). Cathedral and
Rising Tide are aligned to the 2026-08-03 human demo
(``tasks/cathedral_to_bat_human.json``). Bubble marks **traps** (Save / wrong
doors) and the MapRando cavity climb — not a freeze-Ice path (Ice is post-Speed).

Also: post-Torizo **Parlor Alcatraz** left wall-jump climb (Flyway door shaft),
sm-json-data “Alcatraz Escape” family — **not** the product Terminator platform
hop / bomb-tunnel path.

Post-supers **Big Pink Charge** (main shaft → Chozo collect → ordinary return →
optional GHZ green door) is the K1 detour inside ``play_big_pink_to_ghz``.

Early Spazer (K2.2 optional): continuous-like Below Spazer ``0xA408`` left
entry → left-shaft wall-jump → top green Super door → Spazer Room collect →
return. Overlay + one-pager: ``docs/tasks/EARLY_SPAZER_HUMAN.md``.
"""

from __future__ import annotations

from retro_harness.path_overlay import GuidePoint, RoomGuide
from super_metroid.routes.skills.policies.bubble_to_bat import LOWER_SHELVES
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUBBLE,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_GHZ,
    ROOM_RISING_TIDE,
    ROOM_SPAZER,
)

# Bubble Mountain Save Room (wrong-door trap at left mid).
ROOM_BUBBLE_SAVE = 0xB0DD
# Parlor and Alcatraz (post-Bomb Torizo / Flyway return).
ROOM_PARLOR = 0x92FD

_C = {
    "entrance": (120, 200, 255),
    "cathedral": (80, 255, 120),
    "rising": (255, 200, 80),
    "bubble": (200, 120, 255),
    "trap": (255, 60, 60),
    "bat": (255, 100, 100),
    "parlor": (80, 220, 255),
    "charge": (255, 180, 60),
    "ghz": (120, 255, 160),
    "spazer": (255, 140, 220),
    "below_spazer": (220, 90, 140),
}


def _pts(*pairs: tuple[int, int, str]) -> tuple[GuidePoint, ...]:
    return tuple(GuidePoint(x, y, label) for x, y, label in pairs)


# Cathedral Entrance (0xA7B3): bomb left floor → cross → mid climb → Super door.
GUIDE_CATHEDRAL_ENTRANCE = RoomGuide(
    room_id=ROOM_CATHEDRAL_ENTRANCE,
    name="Cathedral Entrance",
    color=_C["entrance"],
    points=_pts(
        (39, 139, "entry"),
        (82, 139, "morph"),
        (90, 360, "bomb-drop"),
        (320, 420, "floor"),
        (620, 400, "climb-base"),
        (620, 300, "mid"),
        (680, 150, "door"),
    ),
)

# Cathedral (0xA788): human demo — upper ridge then drop to lower-right Super.
# Exit pin human f≈1081: (748, 395). Not the upper y≈120 lip.
GUIDE_CATHEDRAL = RoomGuide(
    room_id=ROOM_CATHEDRAL,
    name="Cathedral",
    color=_C["cathedral"],
    points=_pts(
        (50, 139, "entry"),
        (180, 144, "ridge-1"),
        (300, 140, "ridge-2"),
        (400, 110, "crest"),
        (550, 100, "high"),
        (650, 180, "drop"),
        (710, 280, "lower"),
        (720, 365, "super-lip"),
        (748, 395, "super-door"),
    ),
)

# Rising Tide (0xAFA3): 5×1 screens — right blue door is near x≈1260, not ~1000.
# Human exit f≈2908: (1262, 139).
GUIDE_RISING_TIDE = RoomGuide(
    room_id=ROOM_RISING_TIDE,
    name="Rising Tide",
    color=_C["rising"],
    points=_pts(
        (39, 139, "entry"),
        (200, 120, "plat-1"),
        (450, 100, "mid"),
        (750, 100, "plat-2"),
        (1000, 130, "far"),
        (1180, 130, "door-run"),
        (1262, 139, "blue-door"),
    ),
)

# Bubble Mountain (0xACB3): no Ice on first visit — save-door runway → spin
# glide → walljump (maprando left climb / human bubble_jump_try). Traps:
# Save left mid 0xB0DD, left high Missiles, right mid Single Chamber.
# Goal: top-right Super → Bat 0xB07A. Human peak ~(240,160); lip-clear open.
_bubble_lower = tuple(
    GuidePoint(x, y, f"shelf-{i + 1}") for i, (x, y) in enumerate(LOWER_SHELVES)
)
GUIDE_BUBBLE = RoomGuide(
    room_id=ROOM_BUBBLE,
    name="Bubble Mountain",
    color=_C["bubble"],
    points=(
        GuidePoint(48, 637, "entry"),
        *_bubble_lower,
        GuidePoint(80, 430, "lip"),
        # Runway OUTSIDE Save — do not enter 0xB0DD (trap).
        GuidePoint(55, 395, "save-runway"),
        GuidePoint(120, 350, "spin-glide"),
        GuidePoint(240, 280, "wall-contact"),
        GuidePoint(240, 160, "wj-peak"),
        GuidePoint(340, 220, "near-top"),
        GuidePoint(420, 130, "door-ledge"),
        GuidePoint(496, 112, "super-door"),
    ),
)

# Drawn red if player enters Save — not on the Speed spine.
GUIDE_BUBBLE_SAVE = RoomGuide(
    room_id=ROOM_BUBBLE_SAVE,
    name="Bubble Save (TRAP — leave RIGHT)",
    color=_C["trap"],
    points=_pts(
        (40, 140, "entered-trap"),
        (220, 140, "exit-right"),
    ),
)

GUIDE_BAT_CAVE = RoomGuide(
    room_id=ROOM_BAT_CAVE,
    name="Bat Cave",
    color=_C["bat"],
    points=_pts(
        (40, 140, "entry"),
        (200, 140, "center"),
    ),
)

# Parlor and Alcatraz (0x92FD) — post-BT Flyway door → **left shaft WJ climb**.
# sm-json-data node 5 “Alcatraz Door” (= Flyway return) + notable “Alcatraz Escape”:
# wall-jump up the shaft left of the door, midair morph out (no Terminator bomb
# tunnel). Pins from continuous post-BT settle + spore debug waypoint names.
#
# NOT the product parlor_chimney open-loop that wanders right-side platforms
# toward Terminator — that is a different chimney.
GUIDE_PARLOR_ALCATRAZ = RoomGuide(
    room_id=ROOM_PARLOR,
    name="Parlor Alcatraz LEFT WJ (Flyway door)",
    color=_C["parlor"],
    points=_pts(
        # Human demos parlor_left_human{,2}.json — see PARLOR_ALCATRAZ_HUMAN.md
        (968, 651, "flyway-door"),
        (895, 539, "mid-plat"),  # human2 first land
        (850, 459, "mid-ledge"),  # setup ledge
        (805, 355, "left-wall"),  # left face before spin-up
        (830, 310, "spin-up"),  # RIGHT+A p131 toward right wall
        (858, 256, "high-contact"),  # human min_y / p132 class
        (830, 210, "shaft-lip"),  # goal class (not human-cleared yet)
        (900, 200, "morph-out"),
        (980, 180, "central-high"),
    ),
)

# Big Pink (0x9D19) — post-supers main shaft → Charge Chozo → ordinary return.
# Pins from human tasks/charge_human.json (2026-08-04): missiles + R-angle open.
GUIDE_BIG_PINK_CHARGE = RoomGuide(
    room_id=ROOM_BIG_PINK,
    name="Big Pink Charge collect+return",
    color=_C["charge"],
    points=_pts(
        (746, 1465, "main-shaft"),  # start / human f0
        (620, 1600, "drop"),  # standing fall
        (565, 1659, "missiles"),  # human +5 missiles f255
        (683, 1689, "hole-shot"),  # crouch-shot floor
        (747, 1755, "mass-right"),
        (681, 1902, "charge-fall"),
        (613, 1915, "chozo"),  # R-angle open + collect
        (715, 1860, "return-1"),
        (747, 1704, "return-2"),
        (685, 1587, "tunnel-lip"),  # human min_y after climb
        (715, 1625, "lip-shot"),
        (936, 1680, "green-door"),  # Super → GHZ
    ),
)

# Green Hill Zone entry after Super door from Big Pink lower-right.
GUIDE_GHZ_ENTRY = RoomGuide(
    room_id=ROOM_GHZ,
    name="GHZ (post-Charge exit)",
    color=_C["ghz"],
    points=_pts(
        (40, 140, "entry"),
        (200, 140, "center"),
    ),
)

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

# Return land on top-right of Below Spazer after Spazer collect (optional pin).
# Same room as climb; second guide is not registered in GUIDE_BY_ROOM — climb
# polyline remains the room default for early-spazer recording.

GUIDE_BY_ROOM: dict[int, RoomGuide] = {
    g.room_id: g
    for g in (
        GUIDE_CATHEDRAL_ENTRANCE,
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
        GUIDE_PARLOR_ALCATRAZ,
        GUIDE_BIG_PINK_CHARGE,
        GUIDE_GHZ_ENTRY,
        GUIDE_BELOW_SPAZER_EARLY,
        GUIDE_SPAZER_ROOM,
    )
}

ROUTE_PRESETS: dict[str, tuple[RoomGuide, ...]] = {
    "cathedral-to-bat": (
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
    ),
    "cathedral-to-bubble": (
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
    ),
    "entrance-to-bat": (
        GUIDE_CATHEDRAL_ENTRANCE,
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
    ),
    "bubble-to-bat": (GUIDE_BUBBLE, GUIDE_BUBBLE_SAVE, GUIDE_BAT_CAVE),
    "cathedral-only": (GUIDE_CATHEDRAL,),
    "rising-only": (GUIDE_RISING_TIDE,),
    "bubble-only": (GUIDE_BUBBLE, GUIDE_BUBBLE_SAVE),
    # Post-Torizo: Flyway door → Alcatraz left wall-jump shaft (human demo).
    "parlor-left": (GUIDE_PARLOR_ALCATRAZ,),
    "parlor-alcatraz": (GUIDE_PARLOR_ALCATRAZ,),
    # Post-supers K1: Charge detour (collect + ordinary return); optional GHZ.
    "charge-collect-return": (GUIDE_BIG_PINK_CHARGE,),
    "big-pink-to-ghz": (GUIDE_BIG_PINK_CHARGE, GUIDE_GHZ_ENTRY),
    # Early Spazer (100% / K2.2): wall-jump climb + collect + return.
    "early-spazer": (GUIDE_BELOW_SPAZER_EARLY, GUIDE_SPAZER_ROOM),
    "spazer-collect-return": (GUIDE_BELOW_SPAZER_EARLY, GUIDE_SPAZER_ROOM),
    "below-spazer-only": (GUIDE_BELOW_SPAZER_EARLY,),
    "spazer-only": (GUIDE_SPAZER_ROOM,),
}


def guide_for_room(room_id: int) -> RoomGuide | None:
    return GUIDE_BY_ROOM.get(int(room_id))

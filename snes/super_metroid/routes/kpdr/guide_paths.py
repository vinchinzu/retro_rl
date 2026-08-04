"""Human-recording guide polylines for the Cathedral → Bubble → Bat spine.

Waypoints are approximate KPDR route pins (room world pixels). Cathedral and
Rising Tide are aligned to the 2026-08-03 human demo
(``tasks/cathedral_to_bat_human.json``). Bubble marks **traps** (Save / wrong
doors) and the MapRando cavity climb — not a freeze-Ice path (Ice is post-Speed).

Also: post-Torizo **Parlor Alcatraz** left wall-jump climb (Flyway door shaft),
sm-json-data “Alcatraz Escape” family — **not** the product Terminator platform
hop / bomb-tunnel path.

See ``docs/tasks/HUMAN_CATHEDRAL_TO_BAT_VALIDATE.md``.
"""

from __future__ import annotations

from retro_harness.path_overlay import GuidePoint, RoomGuide
from super_metroid.routes.skills.policies.bubble_to_bat import LOWER_SHELVES
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_RISING_TIDE,
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
}


def guide_for_room(room_id: int) -> RoomGuide | None:
    return GUIDE_BY_ROOM.get(int(room_id))

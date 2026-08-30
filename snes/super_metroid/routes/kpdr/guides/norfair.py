"""Norfair human guides: Cathedral → Bubble climb + post-Speed return pins."""

# Ticket k4.

from __future__ import annotations

from retro_harness.path_overlay import GuidePoint, RoomGuide
from super_metroid.routes.skills.policies.bubble_to_bat import LOWER_SHELVES
from super_metroid.routes.kpdr.guides.common import ROOM_BUBBLE_SAVE, _C, _pts
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_RISING_TIDE,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
)

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

# ---------------------------------------------------------------------------
# Post-Speed return pins (Speed → Hall → Bat → Bubble handoff)
# ---------------------------------------------------------------------------

# Speed Booster Room (0xAD1B): post-collect chozo shelf → left blue door.
GUIDE_SPEED_ROOM_EXIT = RoomGuide(
    room_id=ROOM_SPEED,
    name="Speed Room → Hall",
    color=_C["speed"],
    points=_pts(
        (169, 123, "post-collect"),  # post_speed_collected pin
        (120, 130, "shelf"),
        (60, 139, "approach"),
        (39, 139, "left-door"),
    ),
)

# Speed Booster Hall (0xACF0): right lip → LEFT+B dash → left blue → Bat.
GUIDE_SPEED_HALL_RETURN = RoomGuide(
    room_id=ROOM_SPEED_HALL,
    name="Speed Hall → Bat",
    color=_C["speed"],
    points=_pts(
        (480, 139, "right-entry"),  # human ~f390 right side
        (300, 139, "mid-dash"),
        (120, 139, "left-run"),
        (40, 139, "left-door"),
    ),
)

# Bat Cave (0xB07A) **return** (post-Speed): top-right shelf → bomb cavity →
# floor lava gaps → bottom-left blue → Bubble. Not the outbound climb guide.
GUIDE_BAT_CAVE_RETURN = RoomGuide(
    room_id=ROOM_BAT_CAVE,
    name="Bat Cave return → Bubble",
    color=_C["bat"],
    points=_pts(
        (200, 140, "top-entry"),
        (168, 140, "shelf-bomb"),  # morph bomb band ~x165–175
        (151, 250, "cavity-hole"),  # DOWN+X hole band
        (120, 395, "floor"),
        (45, 395, "left-door"),
    ),
)

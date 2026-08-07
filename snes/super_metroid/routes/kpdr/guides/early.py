"""Early KPDR human guides: Parlor Alcatraz + Big Pink Charge detour."""

from __future__ import annotations

from retro_harness.path_overlay import RoomGuide
from super_metroid.routes.kpdr.guides.common import ROOM_PARLOR, _C, _pts
from super_metroid.routes.kpdr.rooms import ROOM_BIG_PINK, ROOM_GHZ

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

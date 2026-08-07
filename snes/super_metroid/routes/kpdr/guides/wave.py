"""Post-Speed Wave branch human guides: Bubble → Single → Double → Wave."""

from __future__ import annotations

from retro_harness.path_overlay import RoomGuide
from super_metroid.routes.kpdr.guides.common import _C, _pts
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
)

# Bubble Mountain (0xACB3) **post-Speed Wave branch** (not first-visit climb):
# bottom-left entry from Bat → top settle optional → drop shaft ~x385 →
# floor sill → middle-right blue → Single Chamber.
GUIDE_BUBBLE_TO_SINGLE = RoomGuide(
    room_id=ROOM_BUBBLE,
    name="Bubble → Single (Wave branch)",
    color=_C["wave"],
    points=_pts(
        (19, 395, "bat-return"),  # human f2131
        (80, 300, "climb-opt"),
        (472, 115, "top-right"),  # pure return pin band
        (385, 200, "drop-shaft"),
        (381, 360, "floor"),
        (492, 395, "single-door"),  # middle-right blue
    ),
)

# Single Chamber (0xAD5E): top-left entry → mid → floor missiles → red door.
GUIDE_SINGLE_CHAMBER = RoomGuide(
    room_id=ROOM_SINGLE_CHAMBER,
    name="Single Chamber → Double",
    color=_C["wave"],
    points=_pts(
        (39, 139, "top-entry"),
        (130, 139, "top-walk"),
        (80, 267, "mid"),
        (124, 395, "missile-seat"),
        (220, 395, "red-door"),
    ),
)

# Double Chamber (0xADAD): take04 reference (tasks/dc_missile_v1/dc_missile_v1_take04).
# Phases: P1 hop → P2 gate → P3 missile free → P4 runway → P5 Super → Wave.
# Spazer freeze at missile PLM ~x494: RIGHT+B free ~406f to x≥510 (take04).
# Path data: routes/kpdr/data/dc_missile_wave_take04_paths.json
GUIDE_DOUBLE_CHAMBER = RoomGuide(
    room_id=ROOM_DOUBLE_CHAMBER,
    name="Double Chamber main (take04)",
    color=_C["wave"],
    points=_pts(
        (61, 139, "P1-entry"),
        (148, 129, "P1-hop1"),
        (214, 122, "P1-hop2"),
        (280, 139, "P1-hop3"),
        (335, 146, "P1-pre-seat"),
        (379, 139, "P2-gate-seat"),
        (411, 109, "P2-gate-peak"),
        (480, 139, "P2-past-gate"),
        (494, 139, "P3-missile"),
        (510, 139, "P3-free"),
        (437, 139, "P4-runway"),
        (600, 139, "P4-edge"),
        (647, 60, "P5-peak"),
        (903, 248, "P5-door-WJ"),
        (929, 139, "P5-sill"),
        (1004, 139, "P5-wave-door"),
    ),
)

# Fallback when P1 (or open) dumps to floor — climb back to gate seat, rejoin P2.
# Same room_id as main; guided_human draws both polylines.
GUIDE_DOUBLE_CHAMBER_RECOVER = RoomGuide(
    room_id=ROOM_DOUBLE_CHAMBER,
    name="Double Chamber floor recover (fallback)",
    color=_C["trap"],
    points=_pts(
        (351, 302, "fall-pin"),
        (326, 403, "floor"),
        (338, 360, "floor-climb"),
        (380, 202, "mid-rejoin"),
        (379, 139, "reseat-P2"),
    ),
)

# Wave Beam Room (0xADDE): left entry → chozo PLM (beam bit 0x0001).
GUIDE_WAVE_ROOM = RoomGuide(
    room_id=ROOM_WAVE,
    name="Wave Beam collect",
    color=_C["wave"],
    points=_pts(
        (40, 139, "entry"),
        (120, 140, "approach"),
        (176, 144, "chozo"),  # approximate pedestal
        (40, 139, "exit-left"),  # return for Ice path
    ),
)

"""Post-Bomb-Torizo Morph+Bombs path: Parlor Flyway door → Gauntlet Entrance.

Vanilla / Map Rando Hard: ``h_useMorphBombs`` + ``canLongIBJ`` (or consecutive
wall-jumps) from Landing Site floor to the ledge under Obstacle A, then
``h_carefullyDestroyBombWalls`` into the top-left Gauntlet door.

Pixel bands are RAM windows from sm-json-data node tiles + live pin dumps.
Not a continuous tip — side-quest probe only.
"""

from __future__ import annotations

from super_metroid.leave_specs import LeaveSpec
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import MORPH_POSES, POSE_WALL_LATCH
from super_metroid.routes.kpdr.room_ids import (
    ROOM_GAUNTLET_ENTRANCE,
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
)

# --- Parlor (0x92FD): Flyway door (node 5) → top-right landing door (node 4)

# Product post-BT alignment before Alcatraz; same seat, climb the RIGHT shaft.
FLYWAY_X = (940, 980)
FLYWAY_Y = (640, 670)
# Right shaft of parlor (map col 3). Left of this is the central void.
PARLOR_SHAFT_X = (800, 1040)
PARLOR_SHAFT_MID_X = 920
PARLOR_TOP_Y = 180
PARLOR_DOOR_X = 1180
PARLOR_CLIMB_BUDGET = 2400

# --- Landing Site (0x91F8): bottom-left parlor door → Gauntlet ledge → door

LANDING_FLOOR_Y = (1100, 1220)
LANDING_CAVE_EXIT_X = 430
# Open ship floor, right of the jumpable cave wall (x≈495).
SHIP_FLOOR_MIN_X = 650
SHIP_FLOOR_X = (650, 920)
# Open-air long IBJ toward the ship — NOT the V-gap (x 380–500) and NOT
# hugging the cliff (x 516–640). Live: x 760–850, minY 643.
IBJ_X = (840, 900)
IBJ_CENTER_X = 870
# Node 7 / ship-facing cliff lip (live (613, 801) stand). Below Obstacle A.
LEDGE_X = (480, 640)
LEDGE_Y = (640, 820)
LIP_X = (600, 630)
LIP_Y = (780, 820)
# Obstacle A: purple bomb wall at the Gauntlet cave mouth (col 1/2, row 2).
BOMB_WALL_X = 500
GAUNTLET_DOOR_X = 48
GAUNTLET_DOOR_Y = (560, 760)

# Rest first-bomb lift starts ~frame 55; wait 52 after X then 18/30 is the
# live-stable climb at x≈870 (minY 639 / 20 cycles). 20/32 falls.
IBJ_FIRST_WAIT = 52
IBJ_WAIT1 = 18
IBJ_WAIT2 = 30
# Door height is y≈640. Hand off just above it; the drift/cliff cadence owns
# the remaining height and must not send Samus to the top of the room.
IBJ_STOP_Y = 700
IBJ_DOOR_Y = 680
IBJ_MAX_CYCLES = 90
BOMB_WALL_MAX_FRAMES = 650
BOMB_WALL_PULSE_PERIOD = 8
BOMB_WALL_EXIT_X = 450
CAVE_HOP_MAX = 8

# --- Gauntlet Entrance (0x92B3): enter from the right door (node 2)

GAUNTLET_ENTRY_X = (1080, 1280)
GAUNTLET_ENTRY_Y = (60, 220)

STAND_POSES = frozenset({1, 2, 9, 10, 12, 27, 28, 137, 138})
GROUND_POSES = frozenset({1, 2, 9, 10})

PARLOR_TO_LANDING = LeaveSpec(
    hop="parlor_to_landing",
    room=ROOM_LANDING_SITE,
    x=(20, 200),
    y=LANDING_FLOOR_Y,
    pose_class="any",
)

LANDING_TO_GAUNTLET = LeaveSpec(
    hop="landing_to_gauntlet",
    room=ROOM_GAUNTLET_ENTRANCE,
    x=GAUNTLET_ENTRY_X,
    y=GAUNTLET_ENTRY_Y,
    pose_class="any",
)

PARLOR_TO_GAUNTLET = LANDING_TO_GAUNTLET


def at_flyway_door(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_PARLOR
        and FLYWAY_X[0] <= int(state.samus_x) <= FLYWAY_X[1]
        and FLYWAY_Y[0] <= int(state.samus_y) <= FLYWAY_Y[1]
    )


def in_parlor_shaft(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_PARLOR
        and PARLOR_SHAFT_X[0] <= int(state.samus_x) <= PARLOR_SHAFT_X[1]
    )


def at_parlor_top(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_PARLOR and int(state.samus_y) <= PARLOR_TOP_Y


def at_landing_floor(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_LANDING_SITE
        and LANDING_FLOOR_Y[0] <= int(state.samus_y) <= LANDING_FLOOR_Y[1]
    )


def at_ship_floor(state: SuperMetroidState) -> bool:
    return (
        at_landing_floor(state)
        and SHIP_FLOOR_X[0] <= int(state.samus_x) <= SHIP_FLOOR_X[1]
    )


def at_gauntlet_ledge(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_LANDING_SITE
        and LEDGE_X[0] <= int(state.samus_x) <= LEDGE_X[1]
        and LEDGE_Y[0] <= int(state.samus_y) <= LEDGE_Y[1]
    )


def at_cliff_lip(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_LANDING_SITE
        and LIP_X[0] <= int(state.samus_x) <= LIP_X[1]
        and LIP_Y[0] <= int(state.samus_y) <= LIP_Y[1]
    )


def at_gauntlet_entry(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_GAUNTLET_ENTRANCE
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def is_wall_latch(pose: int) -> bool:
    return int(pose) == POSE_WALL_LATCH


def is_grounded(state: SuperMetroidState) -> bool:
    pose = int(state.pose)
    return pose in GROUND_POSES and int(state.velocity_y) == 0


def is_morph_pose(pose: int) -> bool:
    return int(pose) in MORPH_POSES


__all__ = [
    "BOMB_WALL_EXIT_X",
    "BOMB_WALL_MAX_FRAMES",
    "BOMB_WALL_PULSE_PERIOD",
    "BOMB_WALL_X",
    "CAVE_HOP_MAX",
    "FLYWAY_X",
    "FLYWAY_Y",
    "GAUNTLET_DOOR_X",
    "GAUNTLET_DOOR_Y",
    "GAUNTLET_ENTRY_X",
    "GAUNTLET_ENTRY_Y",
    "GROUND_POSES",
    "IBJ_CENTER_X",
    "IBJ_DOOR_Y",
    "IBJ_FIRST_WAIT",
    "IBJ_MAX_CYCLES",
    "IBJ_STOP_Y",
    "IBJ_WAIT1",
    "IBJ_WAIT2",
    "IBJ_X",
    "LANDING_CAVE_EXIT_X",
    "LANDING_FLOOR_Y",
    "LANDING_TO_GAUNTLET",
    "LEDGE_X",
    "LEDGE_Y",
    "LIP_X",
    "LIP_Y",
    "PARLOR_CLIMB_BUDGET",
    "PARLOR_DOOR_X",
    "PARLOR_SHAFT_MID_X",
    "PARLOR_SHAFT_X",
    "PARLOR_TO_GAUNTLET",
    "PARLOR_TO_LANDING",
    "PARLOR_TOP_Y",
    "SHIP_FLOOR_MIN_X",
    "SHIP_FLOOR_X",
    "STAND_POSES",
    "at_cliff_lip",
    "at_flyway_door",
    "at_gauntlet_entry",
    "at_gauntlet_ledge",
    "at_landing_floor",
    "at_parlor_top",
    "at_ship_floor",
    "in_parlor_shaft",
    "is_grounded",
    "is_morph_pose",
    "is_wall_latch",
]

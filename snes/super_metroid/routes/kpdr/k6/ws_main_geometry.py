"""Powered Main Shaft geometry: bands, hops, region / phase classifier.

Single source of truth for x/y/pose contracts used by climb, overlay, and
play. Controllers import predicates — do not re-encode magic thresholds.

Take02 two-hop: short A at ~1166 facing LEFT (fails), land, walk LEFT to
1156, committed A, RIGHT+A at y~1920. Observable fire-slope land is
grate_seat region; usable outgoing pin is leave_specs.WS_MAIN_GRATE_SEAT.
Stairs leftover (1111, 1899) is PIT, not the Wave-hole shelf.
"""

from __future__ import annotations

import enum

from super_metroid.leave_specs import (
    WS_MAIN_ATTIC_SEAT,
    WS_MAIN_GRATE_SEAT,
    WS_MAIN_MID_CLIMB,
    WS_MAIN_WEST_SUPER,
)
from super_metroid.ram import FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.room_ids import (
    ROOM_WS_ATTIC,
    ROOM_WS_MAIN,
    ROOM_WS_SAVE,
    ROOM_WS_WEST_SUPER,
)
from super_metroid.takeoff import PlatformHop, TakeoffWindow

WS_MAIN_PHASES: tuple[str, ...] = (
    "pit_shot",
    "grate_seat",
    "west_super",
    "mid_climb",
    "attic_seat",
    "attic_door",
)


class ShaftRegion(enum.Enum):
    PIT = "pit"
    GRATE_SEAT = "grate_seat"
    SHELF = "shelf"
    SAVE_ALCOVE = "save_alcove"
    SAVE_COLUMN = "save_column"
    SHAFT = "shaft"
    ATTIC_SEAT = "attic_seat"
    ATTIC = "attic"


# ---------------------------------------------------------------------------
# Named bands
# ---------------------------------------------------------------------------

WS_MAIN_SAVE_X = 1240
WS_MAIN_STAIR_Y = 1920
WS_MAIN_FLOOR_Y = 1960
WS_MAIN_PIT_Y = 1850
WS_MAIN_SHAFT_CENTER = 1152
WS_MAIN_ATTIC_DOOR_X = (WS_MAIN_ATTIC_SEAT.x[0] + WS_MAIN_ATTIC_SEAT.x[1]) // 2
TUNNEL_CLEAR_X = 1088

# Observable fire-slope land after the pit two-hop. Pocket x=1177 is out.
# Usable outgoing pin is leave_specs.WS_MAIN_GRATE_SEAT ~(1223,1860), not
# this band: (1189,1883) p2 and take04 (1195,1883) land here.
GRATE_LAND_X = (1188, 1232)
GRATE_LAND_Y = (1852, 1888)
FIRST_JUMP_LAND_X = GRATE_LAND_X
FIRST_JUMP_LAND_Y = GRATE_LAND_Y
FIRST_JUMP_LAND_TARGET_X = 1223
GRATE_SEAT_X = FIRST_JUMP_LAND_X
GRATE_SEAT_Y = FIRST_JUMP_LAND_Y

# Floor TakeoffWindows — take02 recipe, not a facing-RIGHT retune.
SHORT_HOP = TakeoffWindow((1163, 1171), "LEFT", min_momentum=0)
COMMITTED_HOP = TakeoffWindow((1138, 1162), "RIGHT", min_momentum=0)
SHORT_HOP_X = SHORT_HOP.x_range
FIRST_JUMP_TAKEOFF_X = COMMITTED_HOP.x_range
FIRST_JUMP_TAKEOFF_TARGET_X = 1156
PIT_EXIT_RIGHT_X = 1104

# Right lip / save-ledge shoot seat. Alcove / save x≳1232.
LIP_SHOT_X = (1164, 1227)
LIP_SHOT_Y = (1852, 1896)
LIP_FIRE_X = (1188, 1227)
POCKET_RELEASE_CHARGE = 8
MORPH_DROP_X = (1176, 1216)
MORPH_DROP_Y = (1765, 1810)
MORPH_DROP_BOMB_FRAMES = 12

# Planted metal shelf ~(1082, 1878). Must not contain stairs (1111, 1899).
LEFT_PLATFORM_X = (1064, 1100)
LEFT_PLATFORM_Y = (1860, 1888)
LEFT_PLATFORM_TARGET_X = 1082

SAVE_LEDGE_X = (1208, 1232)
SAVE_LEDGE_Y = (1836, 1876)
SAVE_COLUMN_WJ_X = (1212, 1232)
SAVE_COLUMN_WJ_Y = (1701, 1888)
SAVE_COLUMN_LATCH_X = 1216

WEST_SUPER_Y = WS_MAIN_WEST_SUPER.y
MID_CLIMB_Y = WS_MAIN_MID_CLIMB.y
SHAFT_X = WS_MAIN_WEST_SUPER.x

THREE_SHOT_X_MIN = 1168
THREE_SHOT_X_MAX = 1210
THREE_SHOT_FRAMES = 240

GROUNDED_POSES = frozenset({1, 2, 3, 4, 9, 10})
AIM_POSES = frozenset({5, 6, 7, 8})
CROUCH_POSES = frozenset({39, 40})
TURN_POSES = frozenset({37, 38})
AIR_POSES = frozenset({19, 20, 21, 25, 26, 47, 48, 75, 76, 77, 78, 81, 82, 83, 84})
HURT_POSES = frozenset({41, 129, 130})
DROP_MORPH_POSES = frozenset({56, 57})
WJ_POSES = frozenset({19, 20, 132})
TURNING_MOVEMENT = 14

SHAFT_HOPS: tuple[PlatformHop, ...] = (
    # Powered take02 lands near x=1108, repositions to x=1062, then launches
    # RIGHT to the y=1543 ledge (frames 742-828).
    PlatformHop(1675, 1048, 1220, TakeoffWindow((1054, 1070), "RIGHT", min_momentum=0)),
    PlatformHop(1468, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(1288, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(1163, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(857, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(680, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(200, 1100, 1180, TakeoffWindow((1110, 1160), "LEFT", min_momentum=0)),
)


def ws_main_phase_index(name: str) -> int:
    """Index in ``WS_MAIN_PHASES``. Accepts hyphen or underscore."""
    key = str(name).strip().lower().replace("-", "_")
    try:
        return WS_MAIN_PHASES.index(key)
    except ValueError:
        raise ValueError(
            f"unknown Main Shaft phase {name!r}; use one of {WS_MAIN_PHASES}"
        ) from None


def _in_room_main(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_WS_MAIN


def _in_main(state: SuperMetroidState) -> bool:
    return _in_room_main(state) and int(state.game_state) == 8


def ws_main_attic_settled(state: SuperMetroidState) -> bool:
    """Ordinary Attic handoff: room ``0xCA52`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_ATTIC
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_main_attic_door_seat(state: SuperMetroidState) -> bool:
    """Standing / planted under the blue ceiling door to Attic."""
    pose = int(state.pose)
    return (
        _in_room_main(state)
        and abs(int(state.samus_x) - WS_MAIN_ATTIC_DOOR_X) <= 24
        and int(state.samus_y) <= 160
        and pose in (1, 2, 9, 10)
        and abs(int(state.velocity_y)) <= 1
    )


def at_ws_main_first_jump_land(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on observable fire-slope land. Not the hatch-lip pocket."""
    x, y = int(samus_x), int(samus_y)
    return (
        FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
        and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
        and int(pose) in GROUNDED_POSES | CROUCH_POSES
        and abs(int(velocity_y)) <= 1
    )


def at_ws_main_grate_seat(state: SuperMetroidState) -> bool:
    """Observable fire-slope land. Usable pin is WS_MAIN_GRATE_SEAT glance."""
    return _in_main(state) and at_ws_main_first_jump_land(
        int(state.samus_x),
        int(state.samus_y),
        int(state.pose),
        int(state.velocity_y),
    )


def at_ws_main_usable_grate_seat(state: SuperMetroidState) -> bool:
    """take02 fire-slope handoff. Observable land is at_ws_main_grate_seat."""
    if not _in_main(state):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        WS_MAIN_GRATE_SEAT.x[0] <= x <= WS_MAIN_GRATE_SEAT.x[1]
        and WS_MAIN_GRATE_SEAT.y[0] <= y <= WS_MAIN_GRATE_SEAT.y[1]
        and abs(int(state.velocity_y)) <= 1
    )


def at_ws_main_pit(state: SuperMetroidState) -> bool:
    """Hatch floor, pocket, cubby, stairs leftover. Not the fire slope."""
    if not _in_room_main(state) or at_ws_main_grate_seat(state):
        return False
    return int(state.samus_y) >= WS_MAIN_PIT_Y


def at_ws_main_left_platform(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Planted on the metal shelf ~(1082, 1878). Stairs leftover is out."""
    x, y = int(samus_x), int(samus_y)
    pose_i = int(pose)
    return (
        LEFT_PLATFORM_X[0] <= x <= LEFT_PLATFORM_X[1]
        and LEFT_PLATFORM_Y[0] <= y <= LEFT_PLATFORM_Y[1]
        and pose_i not in AIR_POSES
        and pose_i not in HURT_POSES
        and not is_morph(pose_i)
        and abs(int(velocity_y)) <= 1
    )


def at_ws_main_lip_shot_seat(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on the right lip / save-ledge shoot seat. Not the save alcove."""
    x, y = int(samus_x), int(samus_y)
    return (
        LIP_SHOT_X[0] <= x <= LIP_SHOT_X[1]
        and LIP_SHOT_Y[0] <= y <= LIP_SHOT_Y[1]
        and int(pose) in GROUNDED_POSES | AIM_POSES | CROUCH_POSES | TURN_POSES
        and abs(int(velocity_y)) <= 1
        and not is_morph(int(pose))
    )


def at_ws_main_morph_drop(
    samus_x: int, samus_y: int, pose: int = 0, velocity_y: int = 0
) -> bool:
    """Morph-drop hole after the Wave-block spawn. Not the lip, not the alcove."""
    del pose, velocity_y
    x, y = int(samus_x), int(samus_y)
    return MORPH_DROP_X[0] <= x <= MORPH_DROP_X[1] and MORPH_DROP_Y[0] <= y <= MORPH_DROP_Y[1]


def at_ws_main_west_super_band(state: SuperMetroidState) -> bool:
    """First shaft hop y~1675. Must stay in Main — save/west-super doors are out."""
    if int(state.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SAVE, ROOM_WS_ATTIC):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        _in_main(state)
        and SHAFT_X[0] <= x <= SHAFT_X[1]
        and WEST_SUPER_Y[0] <= y <= WEST_SUPER_Y[1]
    )


def at_ws_main_mid_climb(state: SuperMetroidState) -> bool:
    """Mid-shaft hop y~680 (past sponge / save height)."""
    if int(state.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SAVE, ROOM_WS_ATTIC):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        _in_main(state)
        and SHAFT_X[0] <= x <= SHAFT_X[1]
        and MID_CLIMB_Y[0] <= y <= MID_CLIMB_Y[1]
    )


def at_ws_main_save_alcove(state: SuperMetroidState) -> bool:
    """Planted on the save-door alcove ~(1235, 1851). Jump LEFT into the shaft."""
    pose = int(state.pose)
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        _in_room_main(state)
        and WS_MAIN_GRATE_SEAT.x[1] < x < WS_MAIN_SAVE_X
        and SAVE_LEDGE_Y[0] <= y <= SAVE_LEDGE_Y[1]
        and pose in GROUNDED_POSES | TURN_POSES
        and abs(int(state.velocity_y)) <= 1
        and not is_morph(pose)
    )


def at_ws_main_save_column_wj(state: SuperMetroidState) -> bool:
    """Air against the save-column LEFT face. Not the save door, not the lip."""
    pose = int(state.pose)
    x, y = int(state.samus_x), int(state.samus_y)
    airborne = (
        (pose not in GROUNDED_POSES and pose not in CROUCH_POSES)
        or abs(int(state.velocity_y)) > 1
    )
    return (
        _in_room_main(state)
        and SAVE_COLUMN_WJ_X[0] <= x < SAVE_COLUMN_WJ_X[1]
        and SAVE_COLUMN_WJ_Y[0] <= y <= SAVE_COLUMN_WJ_Y[1]
        and (airborne or pose in WJ_POSES)
        and not is_morph(pose)
        and pose not in CROUCH_POSES
        and int(state.facing) == FACING_RIGHT
    )


def classify_region_xy(
    samus_x: int,
    samus_y: int,
    pose: int,
    velocity_y: int = 0,
    *,
    lip_hit: bool = False,
) -> ShaftRegion:
    """In-room xy dispatch. Stairs leftover is PIT, not the Wave-hole shelf."""
    if (
        abs(int(samus_x) - WS_MAIN_ATTIC_DOOR_X) <= 24
        and int(samus_y) <= 160
        and int(pose) in (1, 2, 9, 10)
        and abs(int(velocity_y)) <= 1
    ):
        return ShaftRegion.ATTIC_SEAT
    if at_ws_main_first_jump_land(samus_x, samus_y, pose, velocity_y):
        return ShaftRegion.GRATE_SEAT
    if lip_hit and at_ws_main_left_platform(samus_x, samus_y, pose, velocity_y):
        return ShaftRegion.SHELF
    if int(samus_y) >= WS_MAIN_PIT_Y:
        return ShaftRegion.PIT
    return ShaftRegion.SHAFT


def classify_region(
    state: SuperMetroidState, *, lip_hit: bool = False
) -> ShaftRegion:
    """Dispatch region. No y>=1760 steal — stairs leftover is PIT."""
    if ws_main_attic_settled(state) or int(state.room_id) == ROOM_WS_ATTIC:
        return ShaftRegion.ATTIC
    if at_ws_main_attic_door_seat(state):
        return ShaftRegion.ATTIC_SEAT
    if at_ws_main_save_alcove(state):
        return ShaftRegion.SAVE_ALCOVE
    if at_ws_main_save_column_wj(state):
        return ShaftRegion.SAVE_COLUMN
    return classify_region_xy(
        int(state.samus_x),
        int(state.samus_y),
        int(state.pose),
        int(state.velocity_y),
        lip_hit=lip_hit,
    )


def classify_ws_main_phase(state: SuperMetroidState) -> str:
    """Highest phase this still satisfies. Pin / pocket / stairs is ``pit_shot``."""
    if ws_main_attic_settled(state) or int(state.room_id) == ROOM_WS_ATTIC:
        return "attic_door"
    if at_ws_main_attic_door_seat(state) or (
        _in_main(state) and int(state.samus_y) <= 160
    ):
        return "attic_seat"
    x, y = int(state.samus_x), int(state.samus_y)
    in_shaft = _in_main(state) and SHAFT_X[0] <= x <= SHAFT_X[1]
    if at_ws_main_mid_climb(state) or (in_shaft and y <= MID_CLIMB_Y[1]):
        return "mid_climb"
    if at_ws_main_west_super_band(state) or (
        in_shaft and y <= WEST_SUPER_Y[1] and y < GRATE_SEAT_Y[0]
    ):
        return "west_super"
    if at_ws_main_grate_seat(state):
        return "grate_seat"
    return "pit_shot"


__all__ = [
    "AIM_POSES",
    "AIR_POSES",
    "COMMITTED_HOP",
    "CROUCH_POSES",
    "DROP_MORPH_POSES",
    "FIRST_JUMP_LAND_TARGET_X",
    "FIRST_JUMP_LAND_X",
    "FIRST_JUMP_LAND_Y",
    "FIRST_JUMP_TAKEOFF_TARGET_X",
    "FIRST_JUMP_TAKEOFF_X",
    "GRATE_LAND_X",
    "GRATE_LAND_Y",
    "GRATE_SEAT_X",
    "GRATE_SEAT_Y",
    "GROUNDED_POSES",
    "HURT_POSES",
    "LEFT_PLATFORM_TARGET_X",
    "LEFT_PLATFORM_X",
    "LEFT_PLATFORM_Y",
    "LIP_FIRE_X",
    "LIP_SHOT_X",
    "LIP_SHOT_Y",
    "MID_CLIMB_Y",
    "MORPH_DROP_BOMB_FRAMES",
    "MORPH_DROP_X",
    "MORPH_DROP_Y",
    "PIT_EXIT_RIGHT_X",
    "POCKET_RELEASE_CHARGE",
    "SAVE_COLUMN_LATCH_X",
    "SAVE_COLUMN_WJ_X",
    "SAVE_COLUMN_WJ_Y",
    "SAVE_LEDGE_X",
    "SAVE_LEDGE_Y",
    "SHAFT_HOPS",
    "SHAFT_X",
    "SHORT_HOP",
    "SHORT_HOP_X",
    "THREE_SHOT_FRAMES",
    "THREE_SHOT_X_MAX",
    "THREE_SHOT_X_MIN",
    "TUNNEL_CLEAR_X",
    "TURNING_MOVEMENT",
    "TURN_POSES",
    "WEST_SUPER_Y",
    "WJ_POSES",
    "WS_MAIN_ATTIC_DOOR_X",
    "WS_MAIN_FLOOR_Y",
    "WS_MAIN_PHASES",
    "WS_MAIN_PIT_Y",
    "WS_MAIN_SAVE_X",
    "WS_MAIN_SHAFT_CENTER",
    "WS_MAIN_STAIR_Y",
    "ShaftRegion",
    "at_ws_main_attic_door_seat",
    "at_ws_main_first_jump_land",
    "at_ws_main_grate_seat",
    "at_ws_main_usable_grate_seat",
    "at_ws_main_left_platform",
    "at_ws_main_lip_shot_seat",
    "at_ws_main_mid_climb",
    "at_ws_main_morph_drop",
    "at_ws_main_pit",
    "at_ws_main_save_alcove",
    "at_ws_main_save_column_wj",
    "at_ws_main_west_super_band",
    "classify_region",
    "classify_region_xy",
    "classify_ws_main_phase",
    "ws_main_attic_settled",
    "ws_main_phase_index",
]

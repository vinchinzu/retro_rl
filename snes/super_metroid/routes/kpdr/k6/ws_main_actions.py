"""RAM-driven Main Shaft climb actions (rr-kw8t hop 2).

Pit two-hop takeoff lives in ``ws_main_pit``. Hop ``side`` is D-pad
``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.k6.ws_ceiling_door import ceiling_door_action
from super_metroid.routes.kpdr.k6.ws_main_grate import (
    LIP_FIRE_X,
    LIP_SHOT_X,
    LIP_SHOT_Y,
    at_ws_main_lip_shot_seat,
    at_ws_main_morph_drop,
    grate_lip_action,
    grate_morph_action,
)
from super_metroid.routes.kpdr.k6.ws_main_pit import (
    FIRST_JUMP_AIR_Y,
    FIRST_JUMP_LAND_TARGET_X,
    FIRST_JUMP_LAND_X,
    FIRST_JUMP_LAND_Y,
    FIRST_JUMP_TAKEOFF_TARGET_X,
    FIRST_JUMP_TAKEOFF_X,
    PIT_EXIT_RIGHT_X,
    SHORT_HOP_X,
    THREE_SHOT_FRAMES,
    THREE_SHOT_X_MAX,
    THREE_SHOT_X_MIN,
    WS_MAIN_FLOOR_Y,
    WS_MAIN_PIT_Y,
    WS_MAIN_STAIR_Y,
    at_ws_main_first_jump_land,
    at_ws_main_pit,
    pit_exit_action,
    three_shot_action,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.takeoff import PlatformHop, TakeoffWindow, spin_jump, walk_toward_x

WS_MAIN_SAVE_X = 1240
WS_MAIN_HATCH_X_MIN = 1135
WS_MAIN_HATCH_X_MAX = 1165
WS_MAIN_ATTIC_DOOR_X = 1135
WS_MAIN_SHAFT_CENTER = 1152
TUNNEL_CLEAR_X = 1088
# Tunnel unmorph stands on the metal shelf ~(1082, 1878) p10, next to the
# remaining Wave blocks. Shoot a standing hole, gun-jump RIGHT. Do not spin.
LEFT_PLATFORM_X = (1064, 1112)
LEFT_PLATFORM_Y = (1860, 1904)
LEFT_PLATFORM_TARGET_X = 1082
TUNNEL_EXIT_X_MAX = 1112
# Human WJ seat on the save-column wall. Stay left of the save door x=1240.
SAVE_LEDGE_X = (1208, 1232)
SAVE_LEDGE_Y = (1836, 1876)
_AIR_POSES = frozenset({19, 20, 21, 25, 26, 47, 48, 75, 76, 77, 78, 81, 82, 83, 84})
_TURNING = 14
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_CROUCH = frozenset({39, 40})
_HURT = frozenset({41, 129, 130})

SHAFT_HOPS: tuple[PlatformHop, ...] = (
    PlatformHop(1675, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(1468, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(1288, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(1163, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(857, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(680, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(200, 1100, 1180, TakeoffWindow((1110, 1160), "LEFT", min_momentum=0)),
)


def plant_then_spin(facing: int, turning: bool, pose: int) -> tuple[str, ...]:
    """Uncrouch, face RIGHT, then spin-jump."""
    if int(pose) in _CROUCH:
        return ("UP",)
    if int(facing) != FACING_RIGHT or turning:
        return ("RIGHT",)
    return spin_jump("RIGHT")


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
        int(state.room_id) == ROOM_WS_MAIN
        and abs(int(state.samus_x) - WS_MAIN_ATTIC_DOOR_X) <= 24
        and int(state.samus_y) <= 160
        and pose in (1, 2, 9, 10)
        and abs(int(state.velocity_y)) <= 1
    )


def at_ws_main_save_ledge(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Planted on the save-column ledge ~(1219, 1864). Human WJ seat."""
    x, y = int(samus_x), int(samus_y)
    pose_i = int(pose)
    return (
        SAVE_LEDGE_X[0] <= x <= SAVE_LEDGE_X[1]
        and SAVE_LEDGE_Y[0] <= y <= SAVE_LEDGE_Y[1]
        and pose_i not in _AIR_POSES
        and pose_i not in _HURT
        and not is_morph(pose_i)
        and abs(int(velocity_y)) <= 1
    )


def at_ws_main_left_platform(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Planted on the metal shelf ~(1082, 1878). Pose 37 turning still counts."""
    x, y = int(samus_x), int(samus_y)
    pose_i = int(pose)
    return (
        LEFT_PLATFORM_X[0] <= x <= LEFT_PLATFORM_X[1]
        and LEFT_PLATFORM_Y[0] <= y <= LEFT_PLATFORM_Y[1]
        and pose_i not in _AIR_POSES
        and pose_i not in _HURT
        and not is_morph(pose_i)
        and abs(int(velocity_y)) <= 1
    )


def west_super_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    frame: int = 0,
    velocity_y: int = 0,
    movement_type: int = 0,
) -> tuple[str, ...]:
    """Leftover takeoff: shelf gun-jump, morph-tunnel leftover, air steer.

    Shelf hole (X-cycle) lives in ``climb_until``. Save-column WJ and alcove
    leftover live in the climb overlay. Peak (1085, 1843) p78 is over the
    gap — steer LEFT. Wave UP is reserved for the hole, not the stairs.
    """
    del frame
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    turning = int(movement_type) == _TURNING
    if x >= WS_MAIN_SAVE_X - 8:
        return ("LEFT", "B")
    airborne = (
        pose_i in _AIR_POSES or pose_i in _HURT or abs(int(velocity_y)) > 1
    )
    if at_ws_main_left_platform(x, y, pose_i, velocity_y):
        if pose_i in _CROUCH:
            return ("UP",)
        if facing_i != FACING_RIGHT:
            return ("RIGHT",)
        if turning:
            return ()
        return ("A",)
    if airborne:
        if x < LEFT_PLATFORM_X[0]:
            return ("RIGHT", "A")
        if (
            x <= LEFT_PLATFORM_TARGET_X + 2
            and LEFT_PLATFORM_Y[0] <= y <= LEFT_PLATFORM_Y[1]
        ):
            return ()
        if x <= TUNNEL_EXIT_X_MAX:
            if y > LEFT_PLATFORM_Y[1]:
                return ("LEFT", "A")
            if facing_i == FACING_RIGHT:
                return ("RIGHT", "A")
            return ("LEFT",)
        if facing_i == FACING_LEFT:
            return ("LEFT", "A") if x > WS_MAIN_SHAFT_CENTER else ("A",)
        if x < FIRST_JUMP_LAND_TARGET_X - 4:
            return ("RIGHT", "A")
        return ("A",)
    if at_ws_main_save_ledge(x, y, pose_i, velocity_y):
        return plant_then_spin(facing_i, turning, pose_i)
    if x <= TUNNEL_EXIT_X_MAX:
        if pose_i in _CROUCH:
            return ("UP",)
        if facing_i != FACING_LEFT or turning:
            return ("LEFT",)
        return ("LEFT", "A")
    if facing_i != FACING_RIGHT or turning:
        return ("RIGHT",)
    return ("RIGHT", "A")


def climb_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int = FACING_RIGHT,
    velocity_y: int = 0,
    movement_type: int = 0,
    frame: int = 0,
    lip_hit: bool = False,
    charge: int = 0,
) -> tuple[str, ...]:
    """Stay in the shaft and spin-hop up. Lip takeoff is shoot-up until PLM hit."""
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    turning = int(movement_type) == _TURNING
    if x >= WS_MAIN_SAVE_X - 16:
        return ("LEFT", "B")
    if x < 1040:
        return ("RIGHT", "B")
    if y >= WS_MAIN_STAIR_Y:
        return pit_exit_action(x, y, pose, facing, movement_type, velocity_y)
    # Grate band. Fire-slope shoots until 0xD080 spawns, then LEFT+A
    # through the hole (take02). Pocket x<1195 keeps first-jump. Morph
    # later at ~(1189, 1785) — never DOWN on the lip.
    if y >= 1760:
        on_fire = x >= LIP_FIRE_X[0]
        if at_ws_main_lip_shot_seat(x, y, pose_i, velocity_y) and (
            lip_hit or on_fire
        ):
            return grate_lip_action(
                pose_i, bool(lip_hit), facing_i, x, int(charge)
            )
        if at_ws_main_morph_drop(x, y, pose_i, velocity_y):
            morph = grate_morph_action(pose_i, bool(lip_hit))
            if morph is not None:
                return morph
        if lip_hit:
            return grate_lip_action(
                pose_i, True, facing_i, x, int(charge)
            )
        if (
            not lip_hit
            and LIP_SHOT_X[0] <= x < FIRST_JUMP_LAND_TARGET_X
            and y >= FIRST_JUMP_AIR_Y
            and not at_ws_main_first_jump_land(x, y, pose_i, velocity_y)
        ):
            return pit_exit_action(x, y, pose, facing, movement_type, velocity_y)
        return west_super_action(
            x, y, pose_i, facing_i, frame, velocity_y, movement_type
        )
    if pose_i in _AIR_POSES:
        if x > WS_MAIN_SHAFT_CENTER + 24:
            return ("LEFT", "A")
        if x < WS_MAIN_SHAFT_CENTER - 24:
            return ("RIGHT", "A")
        return spin_jump("LEFT") if facing_i == FACING_LEFT else spin_jump("RIGHT")
    hop = next((h for h in SHAFT_HOPS if abs(y - h.y) <= 24), None)
    side = hop.side if hop is not None else ("LEFT" if x > WS_MAIN_SHAFT_CENTER else "RIGHT")
    want = FACING_LEFT if side == "LEFT" else FACING_RIGHT
    if facing_i != want or turning:
        return (side,)
    in_window = hop is not None and hop.takeoff.x_range[0] <= x <= hop.takeoff.x_range[1]
    if hop is not None and not in_window:
        return walk_toward_x(x, (hop.takeoff.x_range[0] + hop.takeoff.x_range[1]) // 2, slack=8)
    return spin_jump(side)


def attic_door_action(
    samus_x: int, samus_y: int, pose: int, frame: int
) -> tuple[str, ...]:
    """Open the blue ceiling door from the seat, then jump through. Never L."""
    names = ceiling_door_action(
        samus_x,
        samus_y,
        pose,
        frame,
        seat_x=WS_MAIN_ATTIC_DOOR_X,
        lip_y=160,
        shaft_y=50,
        slack=12,
        hold_charge=True,
    )
    if names is not None:
        return names
    return climb_action(int(samus_x), int(samus_y), pose)


def grate_clear_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    frame: int,
    charge: int = 0,
    velocity_y: int = 0,
    movement_type: int = 0,
    lip_hit: bool = False,
) -> tuple[str, ...] | None:
    """From a grate seat, shoot UP until a Wave-block PLM spawns, then jump.

    Right lip / save-ledge ~(1223,1860): ``shoot_up`` until 0xD080-family
    spawn. Morph only after that spawn, at ~(1189,1785) — not on the lip.
    Wave UP from the left shelf y~1845 still breaks the floor you stand
    on — that is not this seat. None outside the band so the climb loop
    can fall through.
    """
    y = int(samus_y)
    if y < 1760 or y >= WS_MAIN_STAIR_Y:
        return None
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    pose_i = int(pose)
    facing_i = int(facing)
    on_fire = x >= LIP_FIRE_X[0]
    if at_ws_main_lip_shot_seat(x, y, pose_i, velocity_y) and (
        lip_hit or on_fire
    ):
        return grate_lip_action(
            pose_i, bool(lip_hit), facing_i, x, int(charge)
        )
    if at_ws_main_morph_drop(x, y, pose_i, velocity_y):
        morph = grate_morph_action(pose_i, bool(lip_hit))
        if morph is not None:
            return morph
    if lip_hit:
        # Dual leftover (1202, 1854) p77: hole is open; do not RIGHT-A
        # at the save-column. Take02 keeps LEFT+A to ~(1189, 1785).
        return grate_lip_action(pose_i, True, facing_i, x, int(charge))
    if (
        not lip_hit
        and LIP_SHOT_X[0] <= x < FIRST_JUMP_LAND_TARGET_X
        and y >= FIRST_JUMP_AIR_Y
        and not at_ws_main_first_jump_land(x, y, pose_i, velocity_y)
    ):
        return pit_exit_action(x, y, pose, facing, movement_type, velocity_y)
    airborne = (
        pose_i in _AIR_POSES or pose_i in _HURT or abs(int(velocity_y)) > 1
    )
    # Fire-slope landing air: let climb_action / pit_exit steer.
    if airborne and x >= FIRST_JUMP_LAND_X[0] and facing_i == FACING_RIGHT:
        return None
    return west_super_action(
        x, y, pose_i, facing_i, frame, velocity_y, movement_type
    )


__all__ = [
    "FIRST_JUMP_AIR_Y",
    "FIRST_JUMP_LAND_TARGET_X",
    "FIRST_JUMP_LAND_X",
    "FIRST_JUMP_LAND_Y",
    "FIRST_JUMP_TAKEOFF_TARGET_X",
    "FIRST_JUMP_TAKEOFF_X",
    "LEFT_PLATFORM_TARGET_X",
    "LEFT_PLATFORM_X",
    "LEFT_PLATFORM_Y",
    "LIP_FIRE_X",
    "LIP_SHOT_X",
    "LIP_SHOT_Y",
    "PIT_EXIT_RIGHT_X",
    "SAVE_LEDGE_X",
    "SAVE_LEDGE_Y",
    "SHAFT_HOPS",
    "SHORT_HOP_X",
    "THREE_SHOT_FRAMES",
    "THREE_SHOT_X_MAX",
    "THREE_SHOT_X_MIN",
    "TUNNEL_CLEAR_X",
    "TUNNEL_EXIT_X_MAX",
    "WS_MAIN_ATTIC_DOOR_X",
    "WS_MAIN_FLOOR_Y",
    "WS_MAIN_PIT_Y",
    "WS_MAIN_SAVE_X",
    "WS_MAIN_SHAFT_CENTER",
    "WS_MAIN_STAIR_Y",
    "at_ws_main_attic_door_seat",
    "at_ws_main_first_jump_land",
    "at_ws_main_left_platform",
    "at_ws_main_lip_shot_seat",
    "at_ws_main_morph_drop",
    "at_ws_main_pit",
    "at_ws_main_save_ledge",
    "attic_door_action",
    "climb_action",
    "grate_clear_action",
    "grate_lip_action",
    "grate_morph_action",
    "pit_exit_action",
    "plant_then_spin",
    "three_shot_action",
    "west_super_action",
    "ws_main_attic_settled",
]

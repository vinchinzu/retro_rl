"""RAM-driven Main Shaft climb actions (rr-kw8t hop 2).

Hop ``side`` is D-pad ``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.takeoff import PlatformHop, TakeoffWindow, spin_jump, walk_toward_x

WS_MAIN_SAVE_X = 1240
WS_MAIN_HATCH_X_MIN = 1135
WS_MAIN_HATCH_X_MAX = 1165
WS_MAIN_PIT_Y = 1850
WS_MAIN_STAIR_Y = 1920
WS_MAIN_FLOOR_Y = 1960
WS_MAIN_ATTIC_DOOR_X = 1135
WS_MAIN_SHAFT_CENTER = 1152
THREE_SHOT_X_MIN = 1168
THREE_SHOT_X_MAX = 1210
THREE_SHOT_FRAMES = 240
TUNNEL_CLEAR_X = 1088
PIT_EXIT_RIGHT_X = 1104
# Hatch column has no ceiling. Pin x=1173 is under the right lip (bonk y~1940).
# Human tape: A from (1149,1979) p75 → land (1184,1883) p9. Floor HiJump peaks
# ~1868; left (1075,1845) is above that. Gun-jump A, not spin, not X.
FIRST_JUMP_TAKEOFF_X = (1138, 1162)
FIRST_JUMP_LAND_X = (1170, 1210)
FIRST_JUMP_LAND_Y = (1868, 1896)
FIRST_JUMP_LAND_TARGET_X = 1184
_AIR_POSES = frozenset({19, 20, 21, 25, 26, 47, 48, 75, 76, 77, 78, 81, 82, 83, 84})
_TURNING = 14
_DOOR_SHOOT_FRAMES = 240
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_CROUCH = frozenset({39, 40})

SHAFT_HOPS: tuple[PlatformHop, ...] = (
    PlatformHop(1675, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(1468, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(1288, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(1163, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(857, 1080, 1220, TakeoffWindow((1100, 1180), "RIGHT", min_momentum=0)),
    PlatformHop(680, 1080, 1220, TakeoffWindow((1100, 1180), "LEFT", min_momentum=0)),
    PlatformHop(200, 1100, 1180, TakeoffWindow((1110, 1160), "LEFT", min_momentum=0)),
)


def ws_main_attic_settled(state: SuperMetroidState) -> bool:
    """Ordinary Attic handoff: room ``0xCA52`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_ATTIC
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_main_pit(state: SuperMetroidState) -> bool:
    """True on the hatch-floor pit under the Wave 3-shot blocks."""
    return int(state.room_id) == ROOM_WS_MAIN and int(state.samus_y) >= WS_MAIN_PIT_Y


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


def at_ws_main_first_jump_land(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on the right hatch-lip ~(1184, 1883). First stable seat."""
    x, y = int(samus_x), int(samus_y)
    return (
        FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
        and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
        and int(pose) in _GROUNDED
        and abs(int(velocity_y)) <= 1
    )


def pit_exit_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    movement_type: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...]:
    """First jump: hatch-column gun-jump onto the right lip ~(1184, 1883).

    Walk LEFT off the pin into the hatch column, face RIGHT, hold A. Drift
    RIGHT at peak. Cubby: release A and walk RIGHT — do not spin into the
    ceiling. Never DOWN (hatch). Never X (bonk-jump). Never shoulder L.
    """
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    facing_i = int(facing)
    turning = int(movement_type) == _TURNING
    airborne = int(pose) in _AIR_POSES or abs(int(velocity_y)) > 1
    if at_ws_main_first_jump_land(x, y, pose, velocity_y):
        return ()
    in_land_xy = (
        FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
        and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
    )
    if in_land_xy and airborne:
        if x < FIRST_JUMP_LAND_TARGET_X - 4:
            return ("RIGHT",)
        if x > FIRST_JUMP_LAND_TARGET_X + 8:
            return ("LEFT",)
        return ()
    if x <= PIT_EXIT_RIGHT_X:
        if airborne:
            return ("RIGHT",)
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    if airborne:
        # Under the right lip: release A, land, walk back to the hatch column.
        if x >= FIRST_JUMP_TAKEOFF_X[1] and y >= WS_MAIN_STAIR_Y:
            return ("LEFT",)
        if y > FIRST_JUMP_LAND_Y[1]:
            # Rise with A only — human added RIGHT at y~1880, not off the floor.
            if x < FIRST_JUMP_TAKEOFF_X[0]:
                return ("RIGHT", "A") if facing_i == FACING_RIGHT else ("A",)
            return ("A",)
        if x < FIRST_JUMP_LAND_X[0]:
            return ("RIGHT", "A") if facing_i == FACING_RIGHT else ("A",)
        if x > FIRST_JUMP_LAND_X[1]:
            return ("LEFT",)
        return ()
    if x > FIRST_JUMP_TAKEOFF_X[1]:
        return ("LEFT",)
    if x < FIRST_JUMP_TAKEOFF_X[0]:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    target = (FIRST_JUMP_TAKEOFF_X[0] + FIRST_JUMP_TAKEOFF_X[1]) // 2
    if abs(x - target) > 6:
        return walk_toward_x(x, target, slack=6)
    if facing_i != FACING_RIGHT or turning:
        return ("RIGHT",)
    return ("A",)


def three_shot_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    frame: int,
    charge: int = 0,
    movement_type: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...]:
    """Pit floor is the first jump. Charge-jump from x=1173 bonks at y~1940.

    Above the pit, keep the Wave charge cycle so remaining grate blocks can
    still open. Never DOWN. Never shoulder L.
    """
    if int(pose) in (137, 138):
        return ()
    if is_morph(int(pose)):
        return ("LEFT",) if int(samus_y) < WS_MAIN_FLOOR_Y else ("UP",)
    x = int(samus_x)
    y = int(samus_y)
    if y >= WS_MAIN_PIT_Y:
        return pit_exit_action(x, y, pose, facing, movement_type, velocity_y)
    turning = int(movement_type) == _TURNING
    if turning or int(facing) != FACING_LEFT:
        return ("LEFT",)
    if x > THREE_SHOT_X_MAX:
        return ("LEFT", "B")
    if x < THREE_SHOT_X_MIN:
        return walk_toward_x(x, THREE_SHOT_X_MIN, slack=6)
    cycle = 80
    phase = int(frame) % cycle
    charged = int(charge) >= CHARGE_FULL or phase >= 62
    if charged:
        if phase < 70:
            return ("A",)
        return ("LEFT", "A")
    return ("X", "A")


def climb_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int = FACING_RIGHT,
    velocity_y: int = 0,
    movement_type: int = 0,
) -> tuple[str, ...]:
    """Stay in the shaft and spin-hop up. DOWN is morph on the lip only; never hatch."""
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
    # Grate band. Hatch-column gun-jump faces RIGHT onto the lip; west_super
    # takeoff faces LEFT through the hole. Do not mix the two airs.
    if y >= 1760:
        airborne = pose_i in _AIR_POSES or abs(int(velocity_y)) > 1
        if airborne:
            if x < 1080:
                return ("RIGHT", "A")
            if facing_i == FACING_LEFT:
                return ("LEFT", "A") if x > WS_MAIN_SHAFT_CENTER else ("A",)
            if x < FIRST_JUMP_LAND_TARGET_X - 4:
                return ("RIGHT", "A")
            return ("A",)
        if at_ws_main_first_jump_land(x, y, pose_i, velocity_y) or (
            FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
            and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
            and pose_i in _CROUCH
        ):
            return ("DOWN",)
        if x < FIRST_JUMP_LAND_X[0]:
            if facing_i != FACING_RIGHT or turning:
                return ("RIGHT",)
            return spin_jump("RIGHT")
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("UP", "A")
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
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    slack = 12
    phase = int(frame) % 80
    if y < 50:
        if x > WS_MAIN_ATTIC_DOOR_X + 4:
            return ("LEFT", "A")
        if x < WS_MAIN_ATTIC_DOOR_X - 4:
            return ("RIGHT", "A")
        return ("A",)
    if y < 160:
        if x > WS_MAIN_ATTIC_DOOR_X + slack:
            return ("LEFT", "UP", "A")
        if x < WS_MAIN_ATTIC_DOOR_X - slack:
            return ("RIGHT", "UP", "A")
        if phase < 60:
            return ("UP", "X")
        if int(frame) < _DOOR_SHOOT_FRAMES or phase < 68:
            return ("UP",)
        return ("UP", "A")
    return climb_action(x, y, pose)


def grate_clear_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    frame: int,
    charge: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...] | None:
    """From a grate seat, enter the opened shaft. Do not Wave UP.

    Right lip ~(1177,1883): remaining Wave blocks are a morph tunnel (AFS
    3-shot). DOWN to morph, then roll LEFT. Jumping LEFT falls to the pit
    (ceiling y~1843). Left seat ~(1075,1845): spin-jump RIGHT. Wave UP from
    y~1845 breaks the floor you stand on. None outside the band so the
    climb loop can fall through.
    """
    del frame, charge
    y = int(samus_y)
    if y < 1760 or y >= WS_MAIN_STAIR_Y:
        return None
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    pose_i = int(pose)
    facing_i = int(facing)
    airborne = pose_i in _AIR_POSES or abs(int(velocity_y)) > 1
    if airborne:
        if x < 1080:
            return ("RIGHT", "A")
        if facing_i == FACING_LEFT:
            return ("LEFT", "A") if x > WS_MAIN_SHAFT_CENTER else ("A",)
        return None
    if at_ws_main_first_jump_land(x, y, pose_i, velocity_y) or (
        FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
        and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
        and pose_i in _CROUCH
    ):
        return ("DOWN",)
    if x < FIRST_JUMP_LAND_X[0]:
        if facing_i != FACING_RIGHT:
            return ("RIGHT",)
        return spin_jump("RIGHT")
    return None


__all__ = [
    "FIRST_JUMP_LAND_TARGET_X",
    "FIRST_JUMP_LAND_X",
    "FIRST_JUMP_LAND_Y",
    "FIRST_JUMP_TAKEOFF_X",
    "PIT_EXIT_RIGHT_X",
    "SHAFT_HOPS",
    "THREE_SHOT_FRAMES",
    "THREE_SHOT_X_MAX",
    "THREE_SHOT_X_MIN",
    "TUNNEL_CLEAR_X",
    "WS_MAIN_ATTIC_DOOR_X",
    "WS_MAIN_FLOOR_Y",
    "WS_MAIN_PIT_Y",
    "WS_MAIN_SAVE_X",
    "WS_MAIN_SHAFT_CENTER",
    "at_ws_main_attic_door_seat",
    "at_ws_main_first_jump_land",
    "at_ws_main_pit",
    "attic_door_action",
    "climb_action",
    "grate_clear_action",
    "pit_exit_action",
    "three_shot_action",
    "ws_main_attic_settled",
]

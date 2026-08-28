"""RAM-driven Main Shaft climb actions (rr-kw8t hop 2).

One ``climb_action`` switches on ``ShaftRegion``. Pit is take02 two-hop.
Hop ``side`` is D-pad ``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.k6.ws_ceiling_door import ceiling_door_action
from super_metroid.routes.kpdr.k6.ws_main_geometry import (
    AIR_POSES,
    CROUCH_POSES,
    DROP_MORPH_POSES,
    FIRST_JUMP_LAND_TARGET_X,
    FIRST_JUMP_LAND_X,
    FIRST_JUMP_TAKEOFF_TARGET_X,
    FIRST_JUMP_TAKEOFF_X,
    GROUNDED_POSES,
    LIP_FIRE_X,
    LIP_SHOT_Y,
    PIT_EXIT_RIGHT_X,
    POCKET_RELEASE_CHARGE,
    SHAFT_HOPS,
    SHORT_HOP_X,
    THREE_SHOT_X_MAX,
    THREE_SHOT_X_MIN,
    TURNING_MOVEMENT,
    WS_MAIN_ATTIC_DOOR_X,
    WS_MAIN_FLOOR_Y,
    WS_MAIN_PIT_Y,
    WS_MAIN_SAVE_X,
    WS_MAIN_SHAFT_CENTER,
    WS_MAIN_STAIR_Y,
    ShaftRegion,
    at_ws_main_first_jump_land,
    at_ws_main_morph_drop,
    classify_region_xy,
)
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.takeoff import spin_jump, walk_toward_x


def plant_then_spin(facing: int, turning: bool, pose: int) -> tuple[str, ...]:
    """Uncrouch, face RIGHT, then spin-jump."""
    if int(pose) in CROUCH_POSES:
        return ("UP",)
    if int(facing) != FACING_RIGHT or turning:
        return ("RIGHT",)
    return spin_jump("RIGHT")


def grate_lip_action(
    pose: int,
    lip_hit: bool,
    facing: int = FACING_LEFT,
    samus_x: int = 1223,
    charge: int = 0,
) -> tuple[str, ...]:
    """Shoot the Wave blocks until a 0xD080-family PLM spawns, then jump LEFT.

    Fire slope x>=1188: take02/04 UP+X. After spawn: LEFT+A, never DOWN.
    """
    if int(pose) in CROUCH_POSES:
        return ("UP",)
    x = int(samus_x)
    if not lip_hit:
        if x < LIP_FIRE_X[0]:
            if int(facing) != FACING_LEFT:
                return ("LEFT",)
            if int(charge) >= POCKET_RELEASE_CHARGE:
                return ()
            return ("X",)
        if x > LIP_FIRE_X[1]:
            return ("LEFT",)
        if int(charge) >= CHARGE_FULL:
            return ("UP",)
        return shoot_up_action()
    if is_morph(int(pose)):
        return ("LEFT",)
    if int(facing) != FACING_LEFT:
        return ("LEFT",)
    return ("LEFT", "A")


def grate_morph_action(pose: int, lip_hit: bool) -> tuple[str, ...] | None:
    """DOWN-morph after a 0xD080-family spawn. None before spawn or in air."""
    if not lip_hit:
        return None
    p = int(pose)
    if p in CROUCH_POSES:
        return ("DOWN",)
    if is_morph(p) or p in DROP_MORPH_POSES:
        return ("X",) if is_morph(p) else ("DOWN",)
    if p in GROUNDED_POSES:
        return ("DOWN",)
    return None


def pit_exit_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    movement_type: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...]:
    """Take02 two-hop onto the fire slope, then walk to ~(1223,1860).

    Floor: short A at ~1166 facing LEFT (fails), land, walk to 1156,
    committed A. Committed hop holds A then RIGHT+A at y~1920. Pocket
    ~(1177,1883) gun-jumps RIGHT+A. Never DOWN / X / L.
    """
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    facing_i = int(facing)
    turning = int(movement_type) == TURNING_MOVEMENT
    airborne = int(pose) in AIR_POSES or abs(int(velocity_y)) > 1
    if at_ws_main_first_jump_land(x, y, pose, velocity_y):
        if x < FIRST_JUMP_LAND_TARGET_X - 4:
            return ("RIGHT",)
        return ()
    if x <= PIT_EXIT_RIGHT_X:
        if airborne:
            return ("RIGHT",)
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    if airborne:
        if x >= FIRST_JUMP_TAKEOFF_X[1] and y >= WS_MAIN_STAIR_Y:
            return ("LEFT",)
        if x > FIRST_JUMP_LAND_X[1]:
            return ("LEFT",)
        if y >= WS_MAIN_STAIR_Y:
            if x < FIRST_JUMP_TAKEOFF_X[0]:
                return ("RIGHT", "A") if facing_i == FACING_RIGHT else ("A",)
            return ("A",)
        if x < FIRST_JUMP_LAND_TARGET_X:
            return ("RIGHT", "A")
        return ()
    if LIP_SHOT_Y[0] <= y <= LIP_SHOT_Y[1] and x < FIRST_JUMP_LAND_X[0]:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("A",)
    if SHORT_HOP_X[0] <= x <= SHORT_HOP_X[1] and y >= WS_MAIN_FLOOR_Y:
        if facing_i != FACING_LEFT or turning:
            return ("LEFT",)
        return ("A",)
    if x > FIRST_JUMP_TAKEOFF_X[1]:
        return ("LEFT",)
    if x < FIRST_JUMP_TAKEOFF_X[0]:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    if abs(x - FIRST_JUMP_TAKEOFF_TARGET_X) > 6:
        return walk_toward_x(x, FIRST_JUMP_TAKEOFF_TARGET_X, slack=6)
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
    """Pit floor is the two-hop takeoff. Charge cycle is only above the floor."""
    if int(pose) in (137, 138):
        return ()
    if is_morph(int(pose)):
        return ("LEFT",) if int(samus_y) < WS_MAIN_FLOOR_Y else ("UP",)
    x = int(samus_x)
    y = int(samus_y)
    if y >= WS_MAIN_PIT_Y:
        return pit_exit_action(x, y, pose, facing, movement_type, velocity_y)
    turning = int(movement_type) == TURNING_MOVEMENT
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


def _shelf_action(
    pose: int, facing: int, turning: bool
) -> tuple[str, ...]:
    if int(pose) in CROUCH_POSES:
        return ("UP",)
    if int(facing) != FACING_RIGHT:
        return ("RIGHT",)
    if turning:
        return ()
    return ("A",)


def _shaft_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    turning: bool,
    velocity_y: int,
    lip_hit: bool,
) -> tuple[str, ...]:
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    if at_ws_main_morph_drop(x, y, pose_i, velocity_y):
        morph = grate_morph_action(pose_i, bool(lip_hit))
        if morph is not None:
            return morph
    if pose_i in AIR_POSES:
        if x > WS_MAIN_SHAFT_CENTER + 24:
            return ("LEFT", "A")
        if x < WS_MAIN_SHAFT_CENTER - 24:
            return ("RIGHT", "A")
        return spin_jump("LEFT") if facing_i == FACING_LEFT else spin_jump("RIGHT")
    hop = next((h for h in SHAFT_HOPS if abs(y - h.y) <= 24), None)
    side = hop.side if hop is not None else (
        "LEFT" if x > WS_MAIN_SHAFT_CENTER else "RIGHT"
    )
    want = FACING_LEFT if side == "LEFT" else FACING_RIGHT
    if facing_i != want or turning:
        return (side,)
    in_window = (
        hop is not None and hop.takeoff.x_range[0] <= x <= hop.takeoff.x_range[1]
    )
    if hop is not None and not in_window:
        mid = (hop.takeoff.x_range[0] + hop.takeoff.x_range[1]) // 2
        return walk_toward_x(x, mid, slack=8)
    return spin_jump(side)


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
    *,
    region: ShaftRegion | None = None,
) -> tuple[str, ...]:
    """Stay in the shaft. Dispatch is ``ShaftRegion`` — no y>=1760 steal."""
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    turning = int(movement_type) == TURNING_MOVEMENT
    if x >= WS_MAIN_SAVE_X - 16:
        return ("LEFT", "B")
    if x < 1040:
        return ("RIGHT", "B")
    if region is None:
        region = classify_region_xy(
            x, y, pose_i, velocity_y, lip_hit=bool(lip_hit)
        )
    if region is ShaftRegion.PIT:
        return pit_exit_action(x, y, pose_i, facing_i, movement_type, velocity_y)
    if region is ShaftRegion.GRATE_SEAT:
        return grate_lip_action(pose_i, bool(lip_hit), facing_i, x, int(charge))
    if region is ShaftRegion.SHELF:
        return _shelf_action(pose_i, facing_i, turning)
    if region is ShaftRegion.ATTIC_SEAT:
        return attic_door_action(x, y, pose_i, int(frame))
    return _shaft_action(x, y, pose_i, facing_i, turning, velocity_y, bool(lip_hit))


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
    return climb_action(
        int(samus_x), int(samus_y), pose, region=ShaftRegion.SHAFT
    )


__all__ = [
    "attic_door_action",
    "climb_action",
    "grate_lip_action",
    "grate_morph_action",
    "pit_exit_action",
    "plant_then_spin",
    "three_shot_action",
]

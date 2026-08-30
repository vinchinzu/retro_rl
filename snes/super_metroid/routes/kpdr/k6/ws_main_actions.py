"""RAM-driven Main Shaft climb actions (rr-kw8t hop 2).

One ``climb_action`` switches on ``ShaftRegion``. Pit is take02 two-hop.
Hop ``side`` is D-pad ``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.k6.ws_ceiling_door import ceiling_door_action
from super_metroid.routes.kpdr.k6.ws_main_departure import (
    SLOPE_LEFT_A,
    SLOPE_LEFT_A_Y,
    TAKE02_LIP_FIRE,
)
from super_metroid.routes.kpdr.k6.ws_main_geometry import (
    AIR_POSES,
    CROUCH_POSES,
    DROP_MORPH_POSES,
    FIRST_JUMP_LAND_TARGET_X,
    FIRST_JUMP_LAND_X,
    FIRST_JUMP_TAKEOFF_TARGET_X,
    FIRST_JUMP_TAKEOFF_X,
    GROUNDED_POSES,
    HURT_POSES,
    LIP_FIRE_X,
    LIP_SHOT_Y,
    PIT_EXIT_RIGHT_X,
    POCKET_RELEASE_CHARGE,
    SAVE_LEDGE_Y,
    SHAFT_HOPS,
    SHORT_HOP_X,
    SLOPE_1130_AIR_Y,
    SLOPE_1130_Y,
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
    at_ws_main_slope_1130,
    at_ws_main_stairs_1543,
    classify_region_xy,
)
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.takeoff import hop_for_y, spin_jump, walk_toward_x


def at_take02_departure(samus_x: int, samus_y: int, velocity_y: int = 0) -> bool:
    """Take02 fire-to-takeoff slope, including moving-aim pose 15/16."""
    return (
        TAKE02_LIP_FIRE[0] - 2 <= int(samus_x) <= SLOPE_LEFT_A.x_range[1] + 1
        and SLOPE_LEFT_A_Y[0] - 1 <= int(samus_y) <= TAKE02_LIP_FIRE[1] + 2
        and abs(int(velocity_y)) <= 1
    )


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
    samus_y: int = 1860,
    velocity_y: int = 0,
    charge: int = 0,
) -> tuple[str, ...]:
    """Shoot the Wave blocks until a 0xD080-family PLM spawns, then jump LEFT.

    Observable land walks RIGHT to take02 ~(1223,1860) before UP+X.
    After spawn: LEFT+A directly from the grounded takeoff window, never
    a bare LEFT turn first. Take04 alcove is not this seat.
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
        walk = walk_toward_x(x, FIRST_JUMP_LAND_TARGET_X, slack=0)
        if walk:
            return walk
        if x > LIP_FIRE_X[1]:
            return ("LEFT",)
        if int(facing) != FACING_RIGHT:
            return ("RIGHT",)
        if int(charge) >= CHARGE_FULL:
            return ("UP",)
        return shoot_up_action()
    if is_morph(int(pose)):
        return ("LEFT",)
    if not at_take02_departure(samus_x, samus_y, velocity_y):
        return ()
    _, hi = SLOPE_LEFT_A.x_range
    if x < hi:
        if x == hi - 1 and int(pose) in (15, 16):
            return ("UP",)
        return ("UP", "RIGHT")
    if x > hi:
        return ("UP", "LEFT")
    grounded_takeoff = (
        SLOPE_LEFT_A_Y[0] <= int(samus_y) <= SLOPE_LEFT_A_Y[1]
        and int(pose) in GROUNDED_POSES
        and abs(int(velocity_y)) <= 1
    )
    if not grounded_takeoff:
        return ("UP",) if abs(int(velocity_y)) <= 1 else ()
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


_SHAFT_HOP_Y_SLACK = 20


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


def _hop_below(y: int):
    below = [h for h in SHAFT_HOPS if h.y > int(y)]
    return min(below, key=lambda h: h.y) if below else None


def _approach_or_jump(
    x: int, facing: int, turning: bool, hop
) -> tuple[str, ...]:
    lo, hi = hop.takeoff.x_range
    mid = (lo + hi) // 2
    want = FACING_LEFT if hop.side == "LEFT" else FACING_RIGHT
    in_window = lo <= int(x) <= hi
    if in_window and int(facing) == want and not turning:
        if hop.y == 1675:
            return (hop.side, "A")
        return spin_jump(hop.side)
    if not in_window or abs(int(x) - mid) > 6:
        return walk_toward_x(x, mid, slack=6)
    return (hop.side,)


def _stairs_1543_hop():
    for hop in SHAFT_HOPS:
        if hop.y == 1543:
            return hop
    return None


def _stairs_1543_action(
    x: int,
    y: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    vy: int,
    hop,
) -> tuple[str, ...]:
    """Tape-derived 1675→1543: land, dash to x~1252–1259, launch LEFT."""
    del y
    lo, hi = hop.takeoff.x_range
    if pose_i in HURT_POSES or pose_i == 42:
        return ("LEFT", "A")
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if airborne:
        if x >= lo - 16:
            if facing_i != FACING_LEFT or turning:
                return ("LEFT",)
            return spin_jump("LEFT")
        if int(vy) >= 0:
            if facing_i != FACING_RIGHT or turning:
                return ("RIGHT",)
            return ("RIGHT",)
        return ("RIGHT", "A")
    if x < lo:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    if x > hi:
        return ("LEFT",)
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    return spin_jump("LEFT")


def _slope_1130_hop():
    for hop in SHAFT_HOPS:
        if hop.y == SLOPE_1130_Y:
            return hop
    return None


def _owns_slope_1130(x: int, y: int, pose_i: int, vy: int, hop, slope) -> bool:
    """Grounded 1130 slope + wall-launch air onto 1019. 1245 LEFT+A is out."""
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if at_ws_main_slope_1130(x, y, pose_i, vy):
        return True
    if slope is not None and hop is slope and not airborne:
        return True
    if not airborne:
        return False
    air_lo, air_hi = SLOPE_1130_AIR_Y
    if not (air_lo <= int(y) <= air_hi):
        return False
    # Incoming 1245→1130: x≳1160, or x>1080 while still near 1130.
    if int(x) >= 1160:
        return False
    if int(y) >= 1100 and int(x) > 1080:
        return False
    return True


def _slope_1130_action(
    x: int,
    y: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    vy: int,
    hop,
) -> tuple[str, ...]:
    """Tape-derived 1130→1019: B+LEFT to the wall, p138, LEFT+A, RIGHT+A."""
    lo, hi = hop.takeoff.x_range
    if pose_i in HURT_POSES or pose_i == 42:
        return ("LEFT", "A")
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if airborne:
        # take02: p76 LEFT+A, p78 A, p48 RIGHT+A at y=1072 vy=5. Bare RIGHT
        # on the turn frame kills A and the bounce (leftover vy=0 at 1072).
        if int(y) > 1072:
            if pose_i == 78 and not turning:
                return ("A",)
            return ("LEFT", "A")
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT", "A")
        # take02 holds A through ~(1062, 980). y==1028 used to drop A.
        peaked = int(y) <= 1020 and int(vy) <= 1 and pose_i in (47, 81, 82)
        if peaked:
            return ("B", "RIGHT")
        if int(y) <= 1060:
            return ("B", "RIGHT", "A")
        return ("RIGHT", "A")
    if x > hi:
        if facing_i != FACING_LEFT or turning:
            return ("LEFT",)
        return ("LEFT", "B")
    if x < lo:
        return ("RIGHT",)
    # In the wall window: keep LEFT+B so take 02 can plant p138, then jump.
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    return ("LEFT", "B")


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
    vy = int(velocity_y)
    if at_ws_main_morph_drop(x, y, pose_i, velocity_y):
        if lip_hit and not is_morph(pose_i):
            plant = walk_toward_x(x, 1189, slack=1)
            if plant:
                return plant
        morph = grate_morph_action(pose_i, bool(lip_hit))
        if morph is not None:
            return morph
    hop = hop_for_y(y, SHAFT_HOPS, slack=_SHAFT_HOP_Y_SLACK)
    if hop is None and SHAFT_HOPS and y > SHAFT_HOPS[0].y:
        hop = SHAFT_HOPS[0]
    airborne = pose_i in AIR_POSES or abs(vy) > 1
    stairs = _stairs_1543_hop()
    if stairs is not None and (
        hop is stairs or at_ws_main_stairs_1543(x, y, pose_i, vy)
    ):
        return _stairs_1543_action(x, y, pose_i, facing_i, turning, vy, stairs)
    slope = _slope_1130_hop()
    if slope is not None and _owns_slope_1130(x, y, pose_i, vy, hop, slope):
        return _slope_1130_action(x, y, pose_i, facing_i, turning, vy, slope)
    if hop is not None:
        lo, hi = hop.takeoff.x_range
        mid = (lo + hi) // 2
        in_window = lo <= x <= hi
        over_seat = hop.x_lo - 8 <= x <= hop.x_hi + 8
        if airborne and hop.y == SLOPE_1130_Y:
            return spin_jump("LEFT")
        if airborne:
            below_floor = y > hop.y + 2
            coming_down = over_seat and vy >= 0 and 8 <= (hop.y - y) <= 32
            if below_floor:
                walk = walk_toward_x(x, mid, slack=8)
                return (*walk, "A") if walk else ("A",)
            if coming_down:
                return walk_toward_x(x, mid, slack=8)
            if hop.y == 1675:
                return ("RIGHT", "A")
            return spin_jump(hop.side)
        if y > hop.y + 8:
            want = FACING_LEFT if hop.side == "LEFT" else FACING_RIGHT
            if facing_i != want or turning:
                return (hop.side,)
            if hop.y == 1675:
                return ("RIGHT", "A")
            return spin_jump(hop.side)
        return _approach_or_jump(x, facing_i, turning, hop)
    src = _hop_below(y)
    if airborne and src is not None:
        origin = SHAFT_HOPS[0]
        if stairs is not None and src is stairs:
            return ("RIGHT", "A")
        if src is origin:
            return ("RIGHT", "A")
        if src is not origin:
            over_src = src.x_lo - 16 <= x <= src.x_hi + 16
            if vy >= 0 and over_src and src.y - 64 <= y <= src.y + 8:
                lo, hi = src.takeoff.x_range
                return walk_toward_x(x, (lo + hi) // 2, slack=8)
        return spin_jump(src.side)
    if pose_i in AIR_POSES:
        if x > WS_MAIN_SHAFT_CENTER + 24:
            return ("LEFT", "A")
        if x < WS_MAIN_SHAFT_CENTER - 24:
            return ("RIGHT", "A")
        return spin_jump("LEFT") if facing_i == FACING_LEFT else spin_jump("RIGHT")
    side = "LEFT" if x > WS_MAIN_SHAFT_CENTER else "RIGHT"
    want = FACING_LEFT if side == "LEFT" else FACING_RIGHT
    if facing_i != want or turning:
        return (side,)
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
    take02_active: bool = False,
) -> tuple[str, ...]:
    """Stay in the shaft. Dispatch is ``ShaftRegion`` — no y>=1760 steal."""
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    turning = int(movement_type) == TURNING_MOVEMENT
    if region is None:
        region = classify_region_xy(
            x, y, pose_i, velocity_y, lip_hit=bool(lip_hit)
        )
    if (
        x >= WS_MAIN_SAVE_X - 16
        and y >= SAVE_LEDGE_Y[0]
        and region is not ShaftRegion.GRATE_SEAT
        and not take02_active
    ):
        return ("LEFT", "B")
    if x < 1040:
        return ("RIGHT", "B")
    if region is ShaftRegion.PIT:
        if take02_active:
            return _shaft_action(
                x, y, pose_i, facing_i, turning, velocity_y, bool(lip_hit)
            )
        return pit_exit_action(x, y, pose_i, facing_i, movement_type, velocity_y)
    if region is ShaftRegion.GRATE_SEAT:
        return grate_lip_action(
            pose_i,
            bool(lip_hit),
            facing_i,
            x,
            y,
            int(velocity_y),
            int(charge),
        )
    if region is ShaftRegion.SHELF:
        return _shelf_action(pose_i, facing_i, turning)
    if region is ShaftRegion.ATTIC_SEAT:
        return attic_door_action(x, y, pose_i, int(frame))
    return _shaft_action(
        x,
        y,
        pose_i,
        facing_i,
        turning,
        velocity_y,
        bool(lip_hit),
    )


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
    "at_take02_departure",
    "attic_door_action",
    "climb_action",
    "grate_lip_action",
    "grate_morph_action",
    "pit_exit_action",
    "plant_then_spin",
    "three_shot_action",
]

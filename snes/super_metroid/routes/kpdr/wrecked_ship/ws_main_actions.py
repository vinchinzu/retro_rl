"""RAM-driven Main Shaft climb actions (rr-kw8t hop 2).

One ``climb_action`` switches on ``ShaftRegion``. Pit is take02 two-hop.
Hop ``side`` is D-pad ``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.wrecked_ship.ws_ceiling_door import ceiling_door_action
from super_metroid.routes.kpdr.wrecked_ship.ws_main_departure import (
    SLOPE_LEFT_A,
    SLOPE_LEFT_A_Y,
    TAKE02_LIP_FIRE,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_geometry import (
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
    SLOPE_523_FIRE,
    SLOPE_523_GAP,
    SLOPE_523_SEAT_Y,
    SLOPE_523_TAKEOFF_X,
    SLOPE_651_AIR_Y,
    SLOPE_651_SEAT_Y,
    SLOPE_651_Y,
    UPPER_WALL_SHOT_X,
    WALL_SHOT_FIRE,
    WALL_SHOT_GAP,
    SLOPE_827_AIR_Y,
    SLOPE_827_Y,
    SLOPE_1019_AIR_Y,
    SLOPE_1019_Y,
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
    at_ws_main_slope_651,
    at_ws_main_slope_827,
    at_ws_main_slope_1019,
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


def _slope_1019_hop():
    for hop in SHAFT_HOPS:
        if hop.y == SLOPE_1019_Y:
            return hop
    return None


def _slope_827_hop():
    for hop in SHAFT_HOPS:
        if hop.y == SLOPE_827_Y:
            return hop
    return None


def _slope_651_hop():
    for hop in SHAFT_HOPS:
        if hop.y == SLOPE_651_Y:
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


def _owns_slope_1019(x: int, y: int, pose_i: int, vy: int, hop, slope) -> bool:
    """Grounded 1019 slope + right-wall air onto 827. 1019 spin-jump is out."""
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if at_ws_main_slope_1019(x, y, pose_i, vy):
        return True
    if slope is not None and hop is slope and not airborne:
        return True
    if not airborne:
        return False
    air_lo, air_hi = SLOPE_1019_AIR_Y
    if not (air_lo <= int(y) <= air_hi):
        return False
    return int(x) >= 1180


def _slope_1019_action(
    x: int,
    y: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    vy: int,
    hop,
) -> tuple[str, ...]:
    """Tape-derived 1019→827: B+RIGHT to the wall, p137, LEFT+A."""
    lo, hi = hop.takeoff.x_range
    if pose_i in HURT_POSES or pose_i == 42:
        return ("LEFT", "A")
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if airborne:
        # Rise on A at the wall until y=850, then LEFT to 827. Drop A once
        # off the wall so p165 plants; 827 slope owns the land (no A).
        peaked = int(y) <= 820
        off_wall = int(y) <= 830 and int(x) < 1240
        if peaked or off_wall:
            if facing_i != FACING_LEFT or turning:
                return ("LEFT",)
            return ("B", "LEFT")
        if int(y) > 850:
            return ("A",)
        if facing_i != FACING_LEFT or turning:
            return ("LEFT", "A")
        return ("B", "LEFT", "A")
    if x < lo:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("RIGHT", "B")
    if x > hi:
        return ("LEFT",)
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    return ("LEFT", "B")


def _owns_slope_827(x: int, y: int, pose_i: int, vy: int, hop, slope) -> bool:
    """Grounded 827 slope + left-wall air onto mid_climb. Jump-on-land is out."""
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if at_ws_main_slope_827(x, y, pose_i, vy):
        return True
    if slope is not None and hop is slope and not airborne:
        return True
    if not airborne:
        return False
    air_lo, air_hi = SLOPE_827_AIR_Y
    if not (air_lo <= int(y) <= air_hi):
        return False
    return int(x) <= 1180


def _slope_827_action(
    x: int,
    y: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    vy: int,
    hop,
) -> tuple[str, ...]:
    """Tape-derived 827→680: B+LEFT to the wall, p138, A, RIGHT+A."""
    lo, hi = hop.takeoff.x_range
    if pose_i in HURT_POSES or pose_i == 42:
        return ("RIGHT", "A")
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if airborne:
        # take02: p76 A, p78 A, p48 RIGHT+A. LEFT+A on p165 hops off 827.
        if int(y) > 740:
            if pose_i == 78 and not turning:
                return ("A",)
            return ("A",) if int(x) <= hi + 8 else ("LEFT", "A")
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT", "A")
        return ("B", "RIGHT", "A")
    if x > hi:
        if facing_i != FACING_LEFT or turning:
            return ("LEFT",)
        return ("LEFT", "B")
    if x < lo:
        return ("RIGHT",)
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    return ("LEFT", "B")


def wall_up_shot_action(shot_frame: int, charge: int = 0) -> tuple[str, ...]:
    """Take02 587-wall cadence: 5f UP+X, 6f UP. Full charge always releases."""
    if int(charge) >= CHARGE_FULL:
        return ("UP",)
    cycle = WALL_SHOT_FIRE + WALL_SHOT_GAP
    if int(shot_frame) % cycle < WALL_SHOT_FIRE:
        return shoot_up_action()
    return ("UP",)


def _owns_slope_651(x: int, y: int, pose_i: int, vy: int, hop, slope) -> bool:
    """Grounded 651 slope + right-wall air onto 523. 651 spin-jump is out."""
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    if at_ws_main_slope_651(x, y, pose_i, vy):
        return True
    ledge_lo, ledge_hi = SLOPE_523_SEAT_Y
    if ledge_lo <= int(y) <= ledge_hi:
        return True
    # 488 ceiling jam (p144 at y=499) is not an air pose.
    if 430 <= int(y) <= ledge_hi and int(x) <= 1100:
        return True
    if slope is not None and hop is slope and not airborne:
        return True
    if not airborne:
        return False
    air_lo, air_hi = SLOPE_651_AIR_Y
    if air_lo <= int(y) <= air_hi and int(x) >= 1180:
        return True
    # take02 skip (1131, 640) p83. Leftover (1061, 752) p77 is the 827 wall.
    seat_lo, seat_hi = SLOPE_651_SEAT_Y
    return seat_lo <= int(y) <= seat_hi and int(x) >= 1080


def _slope_523_action(
    x: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    shot_frame: int,
) -> tuple[str, ...]:
    """Take02 523 traverse: LEFT+X to ~1077. Take03 UP+A clears the 488 ceiling."""
    take_lo, take_hi = SLOPE_523_TAKEOFF_X
    if pose_i in HURT_POSES or pose_i == 42:
        return ("LEFT", "A")
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    if x > take_hi:
        cycle = SLOPE_523_FIRE + SLOPE_523_GAP
        if int(shot_frame) % cycle < SLOPE_523_FIRE:
            return ("LEFT", "X")
        return ("LEFT",)
    if x < take_lo:
        return ("RIGHT",)
    # 488 shot-blocks respawn. take03 UP+X then UP+A from 1077. Ice first.
    if pose_i in (3, 4, 85, 86):
        cycle = SLOPE_523_FIRE + SLOPE_523_GAP
        if int(shot_frame) % cycle < SLOPE_523_FIRE:
            return ("UP", "X")
        return ("UP", "A")
    return ("UP",)


def _slope_651_action(
    x: int,
    y: int,
    pose_i: int,
    facing_i: int,
    turning: bool,
    vy: int,
    hop,
    ceiling_open: bool = False,
    wall_shot_frame: int = 0,
    charge: int = 0,
) -> tuple[str, ...]:
    """Tape-derived 651→587: B+RIGHT, tap-UP opens 572, LEFT+A at 1231."""
    lo, hi = hop.takeoff.x_range
    if pose_i in HURT_POSES or pose_i == 42:
        return ("LEFT", "A")
    airborne = pose_i in AIR_POSES or abs(int(vy)) > 1
    ledge_lo, ledge_hi = SLOPE_523_SEAT_Y
    bounce_air = (
        430 <= int(y) < ledge_lo
        and int(x) <= 1100
        and (airborne or pose_i in (22, 86, 144))
    )
    if airborne or bounce_air:
        if ledge_lo - 80 <= int(y) <= ledge_hi and int(x) < 1180:
            if int(y) > 460:
                if pose_i in (22, 86, 4, 144):
                    return ("UP", "A")
                if pose_i == 78 and not turning:
                    return ("A",)
                return ("LEFT", "A") if int(x) <= 1088 else ("A",)
            if facing_i != FACING_RIGHT or turning:
                return ("RIGHT", "A")
            return ("RIGHT", "A")
        if int(x) < 1180:
            if facing_i != FACING_RIGHT or turning:
                return ("RIGHT",)
            return ("RIGHT", "B")
        # Overhang at x≳1228 y≲572. take02 holds LEFT+A until ~1220.
        if int(x) >= 1220:
            return ("LEFT", "A")
        peaked = int(y) <= 500
        off_wall = int(y) <= 520
        if peaked or off_wall:
            if facing_i != FACING_LEFT or turning:
                return ("LEFT",)
            return ("B", "LEFT")
        if int(y) > 540:
            return ("A",)
        if facing_i != FACING_LEFT or turning:
            return ("LEFT", "A")
        return ("B", "LEFT", "A")
    if ledge_lo <= int(y) <= ledge_hi:
        return _slope_523_action(x, pose_i, facing_i, turning, wall_shot_frame)
    if x < lo:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        # take02 drops B at ~1202 on the 587 ledge so momentum does not
        # carry past 1231 into the wall.
        if int(y) <= 590 and int(x) >= 1198:
            return ("RIGHT",)
        return ("RIGHT", "B")
    # take02 taps UP+X once x≳1228. Holding X never fires; jumping from
    # aim-up (p3/p4) is an 8px hop that bonks y=572.
    if not ceiling_open and int(y) <= 590 and int(x) >= UPPER_WALL_SHOT_X:
        return wall_up_shot_action(wall_shot_frame, charge)
    if x > hi:
        return ("LEFT",)
    if facing_i != FACING_LEFT or turning:
        return ("LEFT",)
    if pose_i not in (1, 2, 9, 10):
        return ("LEFT",)
    return ("LEFT", "A")


def _shaft_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    facing: int,
    turning: bool,
    velocity_y: int,
    lip_hit: bool,
    ceiling_open: bool = False,
    wall_shot_frame: int = 0,
    charge: int = 0,
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
    slope_1019 = _slope_1019_hop()
    if slope_1019 is not None and _owns_slope_1019(
        x, y, pose_i, vy, hop, slope_1019
    ):
        return _slope_1019_action(
            x, y, pose_i, facing_i, turning, vy, slope_1019
        )
    slope_827 = _slope_827_hop()
    if slope_827 is not None and _owns_slope_827(
        x, y, pose_i, vy, hop, slope_827
    ):
        return _slope_827_action(
            x, y, pose_i, facing_i, turning, vy, slope_827
        )
    slope_651 = _slope_651_hop()
    if slope_651 is not None and _owns_slope_651(
        x, y, pose_i, vy, hop, slope_651
    ):
        return _slope_651_action(
            x,
            y,
            pose_i,
            facing_i,
            turning,
            vy,
            slope_651,
            ceiling_open,
            wall_shot_frame,
            charge,
        )
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
    ceiling_open: bool = False,
    wall_shot_frame: int = 0,
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
        ceiling_open=bool(ceiling_open),
        wall_shot_frame=int(wall_shot_frame),
        charge=int(charge),
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
        hold_charge=False,
        fire_phase=8,
        wait_phase=18,
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
    "wall_up_shot_action",
]

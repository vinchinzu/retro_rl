"""Powered Main Shaft pit takeoff (rr-kw8t hop 2).

Take02 two-hop: short A at ~(1166,1979) that fails, land, walk LEFT to
1156, then committed A and RIGHT+A at y~1920. Lands the fire slope
~(1208,1875) p9 → (1223,1860) p3. One-hop air-steer peaked (1194,1836)
and fell back — do not retune that peak.

Hop ``side`` is D-pad ``LEFT``/``RIGHT``, never shoulder L.
"""

from __future__ import annotations

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import is_morph
from super_metroid.routes.kpdr.k6.ws_main_grate import LIP_SHOT_Y
from super_metroid.routes.kpdr.room_ids import ROOM_WS_MAIN
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.takeoff import walk_toward_x

WS_MAIN_PIT_Y = 1850
WS_MAIN_STAIR_Y = 1920
WS_MAIN_FLOOR_Y = 1960
THREE_SHOT_X_MIN = 1168
THREE_SHOT_X_MAX = 1210
THREE_SHOT_FRAMES = 240
PIT_EXIT_RIGHT_X = 1104
# Hatch column has no ceiling. Pin x=1173 is under the right lip (bonk y~1940).
# Take02: short A from (1166,1979), land, walk LEFT to 1156, committed A,
# RIGHT+A over the lip wall (1181 @ y=1883) at y~1920, land (1208,1875)
# p9, walk to fire (1223,1860) p3. Take04 lands (1195,1883). Pocket
# ~(1177,1883) cannot hit. Gun-jump A, not spin, not X. Do not walk RIGHT
# from the grounded pocket.
FIRST_JUMP_TAKEOFF_X = (1138, 1162)
FIRST_JUMP_TAKEOFF_TARGET_X = 1156
# take02 wasted short hop. Pin 1173 walks LEFT into this band; facing
# RIGHT after land walks LEFT to 1156 instead of hopping again.
SHORT_HOP_X = (1163, 1171)
FIRST_JUMP_LAND_X = (1188, 1232)
FIRST_JUMP_LAND_Y = (1852, 1888)
FIRST_JUMP_LAND_TARGET_X = 1223
# Peak over the lip wall is y~1836 (dual leftover min_y), above LIP_SHOT_Y.
# Morph-drop hole ends y=1810 — do not steal that with pit_exit.
FIRST_JUMP_AIR_Y = 1820
_AIR_POSES = frozenset({19, 20, 21, 25, 26, 47, 48, 75, 76, 77, 78, 81, 82, 83, 84})
_TURNING = 14
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_CROUCH = frozenset({39, 40})


def at_ws_main_pit(state: SuperMetroidState) -> bool:
    """True on the hatch-floor pit under the Wave 3-shot blocks."""
    return int(state.room_id) == ROOM_WS_MAIN and int(state.samus_y) >= WS_MAIN_PIT_Y


def at_ws_main_first_jump_land(
    samus_x: int, samus_y: int, pose: int, velocity_y: int = 0
) -> bool:
    """Grounded on the take02/04 fire slope ~(1195–1223, 1856–1883). Not the pocket."""
    x, y = int(samus_x), int(samus_y)
    return (
        FIRST_JUMP_LAND_X[0] <= x <= FIRST_JUMP_LAND_X[1]
        and FIRST_JUMP_LAND_Y[0] <= y <= FIRST_JUMP_LAND_Y[1]
        and int(pose) in _GROUNDED | _CROUCH
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
    """Take02 two-hop onto the fire slope, then walk to ~(1223,1860).

    Floor: short A at ~1166 facing LEFT (fails), land, walk to 1156,
    committed A. Air of the short hop is LEFT (already x>takeoff at
    y>=1920). Committed hop holds A then RIGHT+A at y~1920. Pocket
    ~(1177,1883) is a miss — do not seat there. Cubby: release A and
    walk RIGHT. Never DOWN / X / L.
    """
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    facing_i = int(facing)
    turning = int(movement_type) == _TURNING
    airborne = int(pose) in _AIR_POSES or abs(int(velocity_y)) > 1
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
        # Under the right lip: release A, walk back to the hatch column.
        if x >= FIRST_JUMP_TAKEOFF_X[1] and y >= WS_MAIN_STAIR_Y:
            return ("LEFT",)
        if x > FIRST_JUMP_LAND_X[1]:
            return ("LEFT",)
        if y >= WS_MAIN_STAIR_Y:
            # Rise with A only — take02 adds RIGHT at y~1920, not off the floor.
            if x < FIRST_JUMP_TAKEOFF_X[0]:
                return ("RIGHT", "A") if facing_i == FACING_RIGHT else ("A",)
            return ("A",)
        # At/above lip: RIGHT+A even if facing LEFT (dual peak p26 then LEFT+A
        # to the stairs). Ceiling-release at y<1840 dropped the peak back to
        # (1170,1843) — keep A. Coast at 1177 drops in-pocket.
        if x < FIRST_JUMP_LAND_TARGET_X:
            return ("RIGHT", "A")
        return ()
    # Grounded short of the fire slope (pocket 1177 / leftover 1181): gun-jump,
    # then RIGHT in air. Do not walk LEFT into the hatch.
    if LIP_SHOT_Y[0] <= y <= LIP_SHOT_Y[1] and x < FIRST_JUMP_LAND_X[0]:
        if facing_i != FACING_RIGHT or turning:
            return ("RIGHT",)
        return ("A",)
    # take02 two-hop: short A from ~1166 facing LEFT. Facing RIGHT here is
    # the land — walk to 1156. Do not 4th-dual first-jump air-steer.
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
    """Pit floor is the two-hop takeoff. Charge-jump from x=1173 bonks at y~1940.

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


__all__ = [
    "FIRST_JUMP_AIR_Y",
    "FIRST_JUMP_LAND_TARGET_X",
    "FIRST_JUMP_LAND_X",
    "FIRST_JUMP_LAND_Y",
    "FIRST_JUMP_TAKEOFF_TARGET_X",
    "FIRST_JUMP_TAKEOFF_X",
    "PIT_EXIT_RIGHT_X",
    "SHORT_HOP_X",
    "THREE_SHOT_FRAMES",
    "THREE_SHOT_X_MAX",
    "THREE_SHOT_X_MIN",
    "WS_MAIN_FLOOR_Y",
    "WS_MAIN_PIT_Y",
    "WS_MAIN_STAIR_Y",
    "at_ws_main_first_jump_land",
    "at_ws_main_pit",
    "pit_exit_action",
    "three_shot_action",
]

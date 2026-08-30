"""Powered Main Shaft Ice keepaway (Atomics).

Overlay shim: ``combat.enemies`` owns scan + Stance. Pit 3-shot stays
hop-owned via the y>=1960 overlay skip. Frozen Atomics are solid.

https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

from super_metroid.combat.enemies import (
    ATOMIC_ID,
    COVERN_ID,
    Enemy,
    Intent,
    choose,
)
from super_metroid.ram import FACING_LEFT

SHAFT_ICE = Intent(engage=frozenset({ATOMIC_ID, COVERN_ID}))
# Shelf takeoff ~(1082, 1878). Stairs Covern (1048, 1928) is out.
# Wave hole must open (X-cycle) before Ice; standing shot, no jump-shot A.
SHELF_COVERN_XY = (1129, 1818)
SHELF_HOLE_FRAMES = 56
SHELF_COVERN_ICE = Intent(engage=frozenset({COVERN_ID}))
OVERLAY_SKIP_FLOOR_Y = 1960
# 1675→1543 conversion. Takes 02–05 use no X; overlay must not steal landing.
SHAFT_ICE_SKIP_Y = (1480, 1720)
# 1130→1019 slope-run and bounce air. Tape uses no X. Planted 1083 wall
# seat may Ice the overlapping Atomic; leftover (1045, 1066) p48 is air.
SLOPE_1130_ICE_SKIP_Y = (960, 1148)
# 1019→827 slope-run and bounce air. Tape uses no X. Leftover (1187, 817)
# p48 overlapping Atomic is air, not a planted 827 ice seat.
SLOPE_1019_ICE_SKIP_Y = (800, 1036)
# 827 slope-run. Jump-on-land Ice would steal B+LEFT.
SLOPE_827_ICE_SKIP_Y = (756, 840)
# 651 slope-run and 587 wall bounce. Tape uses no X from the plant.
# 523 bounce is the exception: leftover Atomics at (1084, 514) eat LEFT+A.
SLOPE_651_ICE_SKIP_Y = (490, 660)
WALL_1083_ICE_X = 1056
WALL_1083_ICE_Y = (1076, 1088)
# Planted 523 takeoff. take02 Atomic at ~1164 is out; overlap at 1084 is in.
SLOPE_523_ICE_X = 1088
SLOPE_523_ICE_Y = (508, 540)
SLOPE_523_ICE_DX = 40
SLOPE_523_ICE_DY = 32
SHAFT_RANGE_DX = 180
SHAFT_RANGE_DY = 96
SHAFT_FIRE_RANGE_PX = 80
SHAFT_FROZEN_WAIT = 28
SHELF_COVERN_X = (1089, 1169)
SHELF_COVERN_Y = (1778, 1858)


def _in_range(samus_x: int, samus_y: int, enemy: Enemy) -> bool:
    return (
        abs(int(enemy.x) - int(samus_x)) <= SHAFT_RANGE_DX
        and abs(int(enemy.y) - int(samus_y)) <= SHAFT_RANGE_DY
    )


def _planted_wall_1083(samus_x: int, samus_y: int, velocity_y: int) -> bool:
    """Grounded 1083 wall seat. Bounce air and the 1130 dash are out."""
    y_lo, y_hi = WALL_1083_ICE_Y
    return (
        int(velocity_y) == 0
        and int(samus_x) <= WALL_1083_ICE_X
        and y_lo <= int(samus_y) <= y_hi
    )


def _planted_523_bounce(samus_x: int, samus_y: int, velocity_y: int) -> bool:
    """Grounded 523 left plant. LEFT+X traverse at x>1088 stays movement."""
    y_lo, y_hi = SLOPE_523_ICE_Y
    return (
        abs(int(velocity_y)) <= 1
        and int(samus_x) <= SLOPE_523_ICE_X
        and y_lo <= int(samus_y) <= y_hi
    )


def _in_523_bounce_lane(samus_x: int, samus_y: int, enemy: Enemy) -> bool:
    return (
        int(enemy.enemy_id) == ATOMIC_ID
        and int(enemy.hp) > 0
        and abs(int(enemy.x) - int(samus_x)) <= SLOPE_523_ICE_DX
        and abs(int(enemy.y) - int(samus_y)) <= SLOPE_523_ICE_DY
    )


def ice_keepaway_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    enemies: tuple[Enemy, ...],
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...] | None:
    """Charge-release Ice at a nearby live Atomic. Skip in the pit 3-shot."""
    if int(samus_y) >= OVERLAY_SKIP_FLOOR_Y:
        return None
    skip_lo, skip_hi = SHAFT_ICE_SKIP_Y
    if skip_lo <= int(samus_y) <= skip_hi and abs(int(velocity_y)) > 1:
        return None
    planted_wall = _planted_wall_1083(samus_x, samus_y, velocity_y)
    slope_lo, slope_hi = SLOPE_1130_ICE_SKIP_Y
    if slope_lo <= int(samus_y) <= slope_hi and not planted_wall:
        return None
    skip_1019_lo, skip_1019_hi = SLOPE_1019_ICE_SKIP_Y
    if skip_1019_lo <= int(samus_y) <= skip_1019_hi:
        return None
    skip_827_lo, skip_827_hi = SLOPE_827_ICE_SKIP_Y
    if skip_827_lo <= int(samus_y) <= skip_827_hi:
        return None
    skip_651_lo, skip_651_hi = SLOPE_651_ICE_SKIP_Y
    planted_523 = _planted_523_bounce(samus_x, samus_y, velocity_y)
    if skip_651_lo <= int(samus_y) <= skip_651_hi and not planted_523:
        return None
    if planted_523:
        nearby = tuple(
            e for e in enemies if _in_523_bounce_lane(samus_x, samus_y, e)
        )
        if not nearby:
            return None
        choice = choose(
            int(samus_x),
            int(samus_y),
            int(facing),
            nearby,
            SHAFT_ICE,
            movement_type=int(movement_type),
            charge=int(charge),
            velocity_y=int(velocity_y),
            fire_range_px=SHAFT_FIRE_RANGE_PX,
            frozen_wait_gap=None,
        )
        return choice.buttons
    # Overlap at (1045, 1083) with dx=0 faces RIGHT after the bounce
    # turn and never charges. Climb faces LEFT; Ice only from that seat.
    if planted_wall and (
        int(movement_type) == 14 or int(facing) != FACING_LEFT
    ):
        return None
    nearby = tuple(e for e in enemies if _in_range(samus_x, samus_y, e))
    choice = choose(
        int(samus_x),
        int(samus_y),
        int(facing),
        nearby,
        SHAFT_ICE,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=SHAFT_FIRE_RANGE_PX,
        frozen_wait_gap=None if planted_wall else SHAFT_FROZEN_WAIT,
    )
    if planted_wall and choice.target is not None and int(choice.target.freeze_timer) > 0:
        return None
    return choice.buttons


def shelf_covern_ice_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    enemies: tuple[Enemy, ...],
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...] | None:
    """Ice only the shelf Covern. Frozen or absent → None so the hop jumps.

    Jump-shot after the Wave hole is open: a standing 45° from the shelf
    passes under (1129, 1818).
    """
    x_lo, x_hi = SHELF_COVERN_X
    y_lo, y_hi = SHELF_COVERN_Y
    nearby = tuple(
        e
        for e in enemies
        if x_lo <= int(e.x) <= x_hi
        and y_lo <= int(e.y) <= y_hi
        and _in_range(samus_x, samus_y, e)
    )
    choice = choose(
        int(samus_x),
        int(samus_y),
        int(facing),
        nearby,
        SHELF_COVERN_ICE,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=SHAFT_FIRE_RANGE_PX,
    )
    if choice.target is not None and int(choice.target.freeze_timer) > 0:
        return None
    return choice.buttons


__all__ = [
    "ATOMIC_ID",
    "COVERN_ID",
    "SHAFT_FIRE_RANGE_PX",
    "SHAFT_FROZEN_WAIT",
    "OVERLAY_SKIP_FLOOR_Y",
    "SHAFT_ICE_SKIP_Y",
    "SLOPE_523_ICE_DX",
    "SLOPE_523_ICE_DY",
    "SLOPE_523_ICE_X",
    "SLOPE_523_ICE_Y",
    "SLOPE_651_ICE_SKIP_Y",
    "SLOPE_827_ICE_SKIP_Y",
    "SLOPE_1019_ICE_SKIP_Y",
    "SLOPE_1130_ICE_SKIP_Y",
    "WALL_1083_ICE_X",
    "WALL_1083_ICE_Y",
    "SHAFT_ICE",
    "SHELF_COVERN_ICE",
    "SHELF_COVERN_XY",
    "SHELF_HOLE_FRAMES",
    "ice_keepaway_action",
    "shelf_covern_ice_action",
]

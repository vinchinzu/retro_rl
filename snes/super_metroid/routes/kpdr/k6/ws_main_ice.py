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

SHAFT_ICE = Intent(engage=frozenset({ATOMIC_ID, COVERN_ID}))
# Shelf takeoff ~(1082, 1878). Stairs Covern (1048, 1928) is out.
# Wave hole must open (X-cycle) before Ice; standing shot, no jump-shot A.
SHELF_COVERN_XY = (1129, 1818)
SHELF_HOLE_FRAMES = 56
SHELF_COVERN_ICE = Intent(engage=frozenset({COVERN_ID}))
OVERLAY_SKIP_FLOOR_Y = 1960
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
    nearby = tuple(e for e in enemies if _in_range(samus_x, samus_y, e))
    return choose(
        int(samus_x),
        int(samus_y),
        int(facing),
        nearby,
        SHAFT_ICE,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=SHAFT_FIRE_RANGE_PX,
        frozen_wait_gap=SHAFT_FROZEN_WAIT,
    ).buttons


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
    "SHAFT_ICE",
    "SHELF_COVERN_ICE",
    "SHELF_COVERN_XY",
    "SHELF_HOLE_FRAMES",
    "ice_keepaway_action",
    "shelf_covern_ice_action",
]

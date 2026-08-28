"""Powered Basement Ice keepaway (Atomics) and Workrobot avoid.

Overlay shim: ``combat.enemies`` owns scan + Stance. Ice-until-dead stays
the policy. Frozen Atomics are solid (movement stall). Workrobot is solid,
no damage. Coverns Absorb (tank) unless a hop Engage-overrides.

Hop geometry (x-band, max range, takeoff) stays here — not on Intent.

https://wiki.supermetroid.run/Basement
"""

from __future__ import annotations

from super_metroid.combat.enemies import (
    ATOMIC_ID,
    COVERN_ID,
    WORKROBOT_ID,
    Enemy,
    Intent,
    choose,
)
from super_metroid.combat.enemies.atomic import approach_clamp_x

BASEMENT_ICE = Intent(
    engage=frozenset({ATOMIC_ID}),
    absorb=frozenset({COVERN_ID}),
    ignore=frozenset({WORKROBOT_ID}),
)
BASEMENT_ROBOT = Intent(
    avoid=frozenset({WORKROBOT_ID}),
    ignore=frozenset({ATOMIC_ID, COVERN_ID}),
)
BASEMENT_ICE_X = (400, 1100)
BASEMENT_ICE_RANGE = 400
BASEMENT_TAKEOFF_X = 720


def basement_overlay_targets(
    samus_x: int, samus_y: int, enemies: tuple[Enemy, ...]
) -> tuple[Enemy, ...]:
    """Live enemies in the Ice walk band. Map-side Atomics stay out."""
    sx, sy = int(samus_x), int(samus_y)
    lo, hi = BASEMENT_ICE_X
    out: list[Enemy] = []
    for enemy in enemies:
        if int(enemy.hp) <= 0:
            continue
        if not (int(lo) <= int(enemy.x) <= int(hi)):
            continue
        dx = abs(int(enemy.x) - sx)
        dy = abs(int(enemy.y) - sy)
        if (dx * dx + dy * dy) ** 0.5 > float(BASEMENT_ICE_RANGE):
            continue
        out.append(enemy)
    return tuple(out)


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
    """Shoot path Atomics until hp=0. None = none left to kill."""
    return choose(
        int(samus_x),
        int(samus_y),
        int(facing),
        basement_overlay_targets(samus_x, samus_y, enemies),
        BASEMENT_ICE,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        clamp_solids=True,
    ).buttons


def workrobot_avoid_action(
    samus_x: int,
    samus_y: int,
    enemies: tuple[Enemy, ...],
    *,
    takeoff_x_min: int = BASEMENT_TAKEOFF_X,
) -> tuple[str, ...] | None:
    """Do not walk into Workrobot ``0xE8FF``. None = path is clear."""
    return choose(
        int(samus_x),
        int(samus_y),
        0,
        enemies,
        BASEMENT_ROBOT,
        takeoff_x_min=int(takeoff_x_min),
    ).buttons


__all__ = [
    "ATOMIC_ID",
    "BASEMENT_ICE",
    "BASEMENT_ICE_RANGE",
    "BASEMENT_ICE_X",
    "BASEMENT_ROBOT",
    "BASEMENT_TAKEOFF_X",
    "WORKROBOT_ID",
    "approach_clamp_x",
    "basement_overlay_targets",
    "ice_keepaway_action",
    "workrobot_avoid_action",
]

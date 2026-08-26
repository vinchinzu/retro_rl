"""Powered Main Shaft Ice keepaway (Atomics).

Overlay shim: ``combat.enemies`` owns scan + Stance. Pit 3-shot stays
hop-owned via ``SHAFT_ICE.skip_below_y``. Frozen Atomics are solid.

https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

from super_metroid.combat.enemies import (
    ATOMIC_ID,
    COVERN_ID,
    Enemy,
    Intent,
    choose,
    list_enemies,
)
from super_metroid.routes.runtime import ControllerSession

ShaftEnemy = Enemy
SHAFT_ICE = Intent(
    engage=frozenset({ATOMIC_ID, COVERN_ID}),
    skip_below_y=1960,
    range_dx=180,
    range_dy=96,
    frozen_wait_gap=28,
    fire_range_px=80,
)


def list_shaft_enemies(session: ControllerSession) -> tuple[Enemy, ...]:
    """Compatibility shim over the full room-enemy scan."""
    return list_enemies(session)


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
    return choose(
        int(samus_x),
        int(samus_y),
        int(facing),
        enemies,
        SHAFT_ICE,
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
    ).buttons


__all__ = [
    "ATOMIC_ID",
    "COVERN_ID",
    "SHAFT_ICE",
    "ShaftEnemy",
    "ice_keepaway_action",
    "list_shaft_enemies",
]

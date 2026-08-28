"""Atomic (and Ice Covern) Engage: ice-until-dead, do not walk into frozen solid.

Charge-release lives in ``charge_shot``. This adapter picks a target,
waits on frozen overlap when the hop asks for it, and clamps approach
around live/frozen solids (Workrobot, frozen Atomic).
"""

from __future__ import annotations

from super_metroid.combat.enemies.scan import Enemy
from super_metroid.combat.enemies.species import is_solid, species_of
from super_metroid.routes.skills.charge_shot import (
    FIRE_RANGE_PX,
    in_shot_seat,
    position_then_charge_action,
)

_SOLID_Y_BAND = 50


def approach_clamp_x(
    samus_x: int,
    samus_y: int,
    target_x: int,
    enemies: tuple[Enemy, ...],
) -> tuple[int | None, int | None]:
    """Do not walk through a solid. ``(min, max)`` along the walk, or None."""
    sx, sy, tx = int(samus_x), int(samus_y), int(target_x)
    x_min: int | None = None
    x_max: int | None = None
    for enemy in enemies:
        if int(enemy.hp) <= 0:
            continue
        if not is_solid(enemy):
            continue
        if abs(int(enemy.y) - sy) >= _SOLID_Y_BAND:
            continue
        gap = species_of(enemy.enemy_id).solid_gap
        if tx < sx and int(enemy.x) < sx:
            floor = int(enemy.x) + gap
            x_min = floor if x_min is None else max(x_min, floor)
        if tx > sx and int(enemy.x) > sx:
            ceil = int(enemy.x) - gap
            x_max = ceil if x_max is None else min(x_max, ceil)
    return x_min, x_max


def in_engage_seat(
    samus_x: int,
    samus_y: int,
    enemy: Enemy,
    enemies: tuple[Enemy, ...],
    *,
    fire_range_px: int = FIRE_RANGE_PX,
) -> bool:
    """True when keepaway would charge-release instead of walk."""
    x_min, x_max = approach_clamp_x(
        int(samus_x), int(samus_y), int(enemy.x), enemies
    )
    return in_shot_seat(
        int(samus_x),
        int(samus_y),
        int(enemy.x),
        int(enemy.y),
        fire_range_px=fire_range_px,
        approach_x_min=x_min,
        approach_x_max=x_max,
    )


def ice_engage_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    target: Enemy,
    enemies: tuple[Enemy, ...],
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
    fire_range_px: int = FIRE_RANGE_PX,
    frozen_wait_gap: int | None = None,
    clamp: bool = True,
) -> tuple[str, ...]:
    """One-frame Ice at ``target``. Empty = wait on frozen overlap."""
    if (
        frozen_wait_gap is not None
        and int(target.freeze_timer) > 0
        and species_of(target.enemy_id).is_solid(int(target.freeze_timer))
        and abs(int(target.x) - int(samus_x)) < int(frozen_wait_gap)
    ):
        return ()
    x_min = x_max = None
    if clamp:
        x_min, x_max = approach_clamp_x(
            int(samus_x), int(samus_y), int(target.x), enemies
        )
    return position_then_charge_action(
        int(samus_x),
        int(samus_y),
        int(facing),
        int(target.x),
        int(target.y),
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=int(fire_range_px),
        approach_x_min=x_min,
        approach_x_max=x_max,
    )


__all__ = [
    "approach_clamp_x",
    "ice_engage_action",
    "in_engage_seat",
]

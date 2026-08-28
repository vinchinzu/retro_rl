"""Workrobot Avoid: solid, no damage, do not walk through.

Also names movement stalls (turning / knockback / solid overlap) so hops
and probes can log why a frame cannot walk. Live Atomic knockback poses
stay the hop's ``is_knockback`` guard; this only labels them.
"""

from __future__ import annotations

from super_metroid.combat.enemies.scan import Enemy
from super_metroid.combat.enemies.species import (
    WORKROBOT_ID,
    Contact,
    is_solid,
    species_of,
)
from super_metroid.routes.skills.charge_shot import MOVEMENT_TURNING

_ROBOT_Y_BAND = 50
_FROZEN_Y_BAND = 32
_FLEE_GAP = 16
_KNOCKBACK_POSES = frozenset({137, 138})
_TURN_POSES = frozenset({37, 38})


def _y_band(enemy: Enemy) -> int:
    spec = species_of(enemy.enemy_id)
    if spec.live_contact is Contact.SOLID:
        return _ROBOT_Y_BAND
    return _FROZEN_Y_BAND


def floor_solids(
    enemies: tuple[Enemy, ...], samus_y: int, *, enemy_id: int | None = None
) -> tuple[Enemy, ...]:
    """Solids in the Y band. Optional id filter (Workrobot-only Avoid)."""
    sy = int(samus_y)
    out: list[Enemy] = []
    for enemy in enemies:
        if int(enemy.hp) <= 0:
            continue
        if enemy_id is not None and int(enemy.enemy_id) != int(enemy_id):
            continue
        if not is_solid(enemy):
            continue
        if abs(int(enemy.y) - sy) >= _y_band(enemy):
            continue
        out.append(enemy)
    return tuple(out)


def stall_reason(
    samus_x: int,
    samus_y: int,
    movement_type: int,
    pose: int,
    enemies: tuple[Enemy, ...],
) -> str | None:
    """Why this frame cannot walk. None = free."""
    if int(movement_type) == MOVEMENT_TURNING or int(pose) in _TURN_POSES:
        return "turning"
    if int(pose) in _KNOCKBACK_POSES:
        return "knockback"
    sx, sy = int(samus_x), int(samus_y)
    for enemy in floor_solids(enemies, sy):
        gap = species_of(enemy.enemy_id).solid_gap
        if abs(int(enemy.x) - sx) >= gap:
            continue
        if int(enemy.enemy_id) == WORKROBOT_ID:
            return "workrobot"
        if int(enemy.freeze_timer) > 0:
            return f"frozen_{species_of(enemy.enemy_id).name.lower()}"
    return None


def avoid_action(
    samus_x: int,
    samus_y: int,
    target: Enemy,
    *,
    takeoff_x_min: int | None = None,
) -> tuple[str, ...] | None:
    """Do not walk into one solid Avoid target. None = path is clear.

    Empty tuple = idle. Under a hop-supplied takeoff lip, flee east.
    """
    if int(target.hp) <= 0 or not is_solid(target):
        return None
    if abs(int(target.y) - int(samus_y)) >= _y_band(target):
        return None
    sx = int(samus_x)
    gap = int(target.x) - sx
    if abs(gap) >= species_of(target.enemy_id).solid_gap:
        return None
    if takeoff_x_min is not None and sx < int(takeoff_x_min):
        if gap > _FLEE_GAP:
            return ("RIGHT", "B")
        if gap > 0:
            return ()
        return ("RIGHT", "B")
    return ()


__all__ = [
    "avoid_action",
    "floor_solids",
    "is_solid",
    "stall_reason",
]

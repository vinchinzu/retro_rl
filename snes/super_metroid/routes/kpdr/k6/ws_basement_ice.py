"""Powered Basement Ice keepaway (Atomics) and Workrobot avoid.

Ice-until-dead stays the policy. Shots go through
``charge_shot.position_then_charge_action`` — tap-X from x=879 does not
connect through the hatch pillar. Frozen Atomics are solid (movement stall);
do not walk into them. Workrobot is solid, no damage.

https://wiki.supermetroid.run/Basement
"""

from __future__ import annotations

from typing import NamedTuple

from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.charge_shot import (
    FIRE_RANGE_PX,
    MOVEMENT_TURNING,
    in_shot_seat,
    position_then_charge_action,
)

ATOMIC_ID = 0xE9FF
WORKROBOT_ID = 0xE8FF
_ENEMY_BASE = 0x0F78
_ENEMY_STRIDE = 0x40
_MAX_ENEMY_SLOTS = 8
_ICE_KEEPAWAY_PX = 400
_ATOMIC_PATH_X_MIN = 400
_ATOMIC_PATH_X_MAX = 1100
_ATOMIC_OVERLAP_PX = 24
_ROBOT_GAP_PX = 48
_ROBOT_Y_BAND = 50


class BasementEnemy(NamedTuple):
    """One enemy slot from ``$0F78 + i*0x40`` (id/x/y/hp/freeze at +0x26)."""

    slot: int
    enemy_id: int
    x: int
    y: int
    hp: int
    freeze_timer: int


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def list_basement_enemies(session: ControllerSession) -> tuple[BasementEnemy, ...]:
    """Scan slots 0–7. Empty when the session has no ``env.get_ram``."""
    env = getattr(session, "env", None)
    get_ram = getattr(env, "get_ram", None) if env is not None else None
    if get_ram is None:
        return ()
    ram = get_ram()
    out: list[BasementEnemy] = []
    for slot in range(_MAX_ENEMY_SLOTS):
        base = _ENEMY_BASE + slot * _ENEMY_STRIDE
        enemy_id = _u16(ram, base)
        if enemy_id == 0:
            continue
        hp = _u16(ram, base + 0x14)
        if hp <= 0:
            continue
        x = _u16(ram, base + 0x02)
        y = _u16(ram, base + 0x06)
        if x >= 0xFE00 or y >= 0xFE00:
            continue
        out.append(
            BasementEnemy(
                slot=slot,
                enemy_id=enemy_id,
                x=x,
                y=y,
                hp=hp,
                freeze_timer=_u16(ram, base + 0x26),
            )
        )
    return tuple(out)


def _path_atomics(enemies: tuple[BasementEnemy, ...]) -> tuple[BasementEnemy, ...]:
    return tuple(
        enemy
        for enemy in enemies
        if int(enemy.enemy_id) == ATOMIC_ID
        and int(enemy.hp) > 0
        and _ATOMIC_PATH_X_MIN <= int(enemy.x) <= _ATOMIC_PATH_X_MAX
    )


def _floor_robots(
    enemies: tuple[BasementEnemy, ...], samus_y: int
) -> tuple[BasementEnemy, ...]:
    return tuple(
        enemy
        for enemy in enemies
        if int(enemy.enemy_id) == WORKROBOT_ID
        and int(enemy.hp) > 0
        and abs(int(enemy.y) - int(samus_y)) < _ROBOT_Y_BAND
    )


def approach_clamp_x(
    samus_x: int,
    samus_y: int,
    target_x: int,
    enemies: tuple[BasementEnemy, ...],
) -> tuple[int | None, int | None]:
    """Do not walk through a Workrobot or a frozen Atomic. (min, max) or None."""
    sx, sy, tx = int(samus_x), int(samus_y), int(target_x)
    x_min: int | None = None
    x_max: int | None = None
    for enemy in _floor_robots(enemies, sy):
        gap = int(enemy.x) - sx
        if tx < sx and int(enemy.x) < sx:
            floor = int(enemy.x) + _ROBOT_GAP_PX
            x_min = floor if x_min is None else max(x_min, floor)
        if tx > sx and int(enemy.x) > sx:
            ceil = int(enemy.x) - _ROBOT_GAP_PX
            x_max = ceil if x_max is None else min(x_max, ceil)
        del gap
    for enemy in _path_atomics(enemies):
        if int(enemy.freeze_timer) <= 0:
            continue
        if abs(int(enemy.y) - sy) >= _ROBOT_Y_BAND:
            continue
        if tx < sx and int(enemy.x) < sx:
            floor = int(enemy.x) + _ATOMIC_OVERLAP_PX
            x_min = floor if x_min is None else max(x_min, floor)
        if tx > sx and int(enemy.x) > sx:
            ceil = int(enemy.x) - _ATOMIC_OVERLAP_PX
            x_max = ceil if x_max is None else min(x_max, ceil)
    return x_min, x_max


def movement_stall_reason(
    samus_x: int,
    samus_y: int,
    movement_type: int,
    pose: int,
    enemies: tuple[BasementEnemy, ...],
) -> str | None:
    """Why this frame cannot walk. None = free.

    turning (mov=14 / pose 37–38) is not stun — firing X during the turn
    is the pose-37 stall. Frozen Atomics and Workrobots are solid.
    Live Atomic contact is knockback (137/138), handled elsewhere.
    """
    if int(movement_type) == MOVEMENT_TURNING or int(pose) in (37, 38):
        return "turning"
    if int(pose) in (137, 138):
        return "knockback"
    sx, sy = int(samus_x), int(samus_y)
    for enemy in _floor_robots(enemies, sy):
        if abs(int(enemy.x) - sx) < _ROBOT_GAP_PX:
            return "workrobot"
    for enemy in _path_atomics(enemies):
        if int(enemy.freeze_timer) <= 0:
            continue
        if abs(int(enemy.x) - sx) < _ATOMIC_OVERLAP_PX and abs(int(enemy.y) - sy) < 32:
            return "frozen_atomic"
    return None


def ice_keepaway_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    enemies: tuple[BasementEnemy, ...],
    *,
    movement_type: int = 0,
    charge: int = 0,
    velocity_y: int = 0,
) -> tuple[str, ...] | None:
    """Shoot path Atomics until hp=0. None = none left to kill.

    Frozen is not dead. Position into a seat (east of the Workrobot), then
    charge-release with aim / jump-shot. Horizontal taps from x=879 miss
    the hatch-floor blob through the pillar.
    """
    atomics = _path_atomics(enemies)
    if not atomics:
        return None
    sx, sy = int(samus_x), int(samus_y)
    nearest = min(
        atomics, key=lambda enemy: (int(enemy.x) - sx) ** 2 + (int(enemy.y) - sy) ** 2
    )
    dist = ((int(nearest.x) - sx) ** 2 + (int(nearest.y) - sy) ** 2) ** 0.5
    if dist > _ICE_KEEPAWAY_PX:
        return None
    x_min, x_max = approach_clamp_x(sx, sy, int(nearest.x), enemies)
    return position_then_charge_action(
        sx,
        sy,
        int(facing),
        int(nearest.x),
        int(nearest.y),
        movement_type=int(movement_type),
        charge=int(charge),
        velocity_y=int(velocity_y),
        fire_range_px=FIRE_RANGE_PX,
        approach_x_min=x_min,
        approach_x_max=x_max,
    )


def workrobot_avoid_action(
    samus_x: int,
    samus_y: int,
    enemies: tuple[BasementEnemy, ...],
    *,
    takeoff_x_min: int = 720,
) -> tuple[str, ...] | None:
    """Do not walk into Workrobot ``0xE8FF``. None = path is clear.

    Empty tuple = idle (let the robot walk). Under-hatch is occupied —
    flee east to the takeoff band (``takeoff_x_min``, same 720 as the hop).
    """
    robots = _floor_robots(enemies, int(samus_y))
    if not robots:
        return None
    sx = int(samus_x)
    nearest = min(robots, key=lambda enemy: abs(int(enemy.x) - sx))
    gap = int(nearest.x) - sx
    if abs(gap) >= _ROBOT_GAP_PX:
        return None
    if sx < int(takeoff_x_min):
        if gap > 16:
            return ("RIGHT", "B")
        if gap > 0:
            return ()
        return ("RIGHT", "B")
    return ()


def atomic_in_shot_seat(
    samus_x: int,
    samus_y: int,
    enemy: BasementEnemy,
    enemies: tuple[BasementEnemy, ...],
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
        approach_x_min=x_min,
        approach_x_max=x_max,
    )


__all__ = [
    "ATOMIC_ID",
    "WORKROBOT_ID",
    "BasementEnemy",
    "approach_clamp_x",
    "atomic_in_shot_seat",
    "ice_keepaway_action",
    "list_basement_enemies",
    "movement_stall_reason",
    "workrobot_avoid_action",
]

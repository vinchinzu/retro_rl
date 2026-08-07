"""Sword hitbox / threat helpers for Zelda I combat policies.

Pure functions over Link + enemy positions. Used by the generic dungeon
controller so swings only fire when the blade can actually hit.
"""

from __future__ import annotations

from typing import Iterable

from zelda_i.ram import ZeldaObject, ZeldaSnapshot

# Conservative NES wooden-sword reach (engine ~16–24 px).
SWORD_REACH = 20
SWORD_HALF_WIDTH = 12
THREAT_RADIUS = 40
# Contact softlock guard: swing even if slightly off-axis when this close.
CONTACT_CHEBYSHEV = 12
CONTACT_MANHATTAN = 14

FACING_NORTH = 0x08
FACING_SOUTH = 0x04
FACING_EAST = 0x01
FACING_WEST = 0x02

_DIR_TO_FACING = {
    "UP": FACING_NORTH,
    "DOWN": FACING_SOUTH,
    "RIGHT": FACING_EAST,
    "LEFT": FACING_WEST,
}
_FACING_TO_DIR = {
    FACING_NORTH: "UP",
    FACING_SOUTH: "DOWN",
    FACING_EAST: "RIGHT",
    FACING_WEST: "LEFT",
}


def direction_to_facing(direction: str) -> int:
    """Map controller direction name to Link facing RAM value."""
    key = direction.upper()
    if key not in _DIR_TO_FACING:
        raise ValueError(f"unsupported direction: {direction}")
    return _DIR_TO_FACING[key]


def facing_to_direction(facing: int) -> str:
    """Map Link facing RAM value to controller direction name."""
    try:
        return _FACING_TO_DIR[int(facing)]
    except KeyError as exc:
        raise ValueError(f"unsupported facing: {facing:#x}") from exc


def manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(int(ax) - int(bx)) + abs(int(ay) - int(by))


def chebyshev(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def in_sword_hitbox(
    link_x: int,
    link_y: int,
    facing_or_direction: int | str,
    enemy_x: int,
    enemy_y: int,
    *,
    reach: int = SWORD_REACH,
    half_width: int = SWORD_HALF_WIDTH,
) -> bool:
    """True if enemy center is in the sword rectangle in front of Link.

    Facing UP: enemy y < link_y, |enemy_x-link_x| <= half_width, depth <= reach.
    Same pattern for the other three facings.
    """
    if isinstance(facing_or_direction, str):
        facing = direction_to_facing(facing_or_direction)
    else:
        facing = int(facing_or_direction)

    dx = int(enemy_x) - int(link_x)
    dy = int(enemy_y) - int(link_y)

    if facing == FACING_NORTH:
        # Toward smaller Y.
        return dy < 0 and abs(dx) <= half_width and -dy <= reach
    if facing == FACING_SOUTH:
        return dy > 0 and abs(dx) <= half_width and dy <= reach
    if facing == FACING_EAST:
        return dx > 0 and abs(dy) <= half_width and dx <= reach
    if facing == FACING_WEST:
        return dx < 0 and abs(dy) <= half_width and -dx <= reach
    return False


def nearest_enemy(
    link_x: int,
    link_y: int,
    enemies: Iterable[ZeldaObject],
) -> ZeldaObject | None:
    best: ZeldaObject | None = None
    best_d = 10**9
    for obj in enemies:
        d = manhattan(link_x, link_y, obj.x, obj.y)
        if d < best_d:
            best_d = d
            best = obj
    return best


def should_swing_at(
    link_x: int,
    link_y: int,
    direction: str,
    enemies: Iterable[ZeldaObject],
    *,
    swing_reach: int = SWORD_REACH,
    half_width: int = SWORD_HALF_WIDTH,
    threat_radius: int = THREAT_RADIUS,
    contact_chebyshev: int = CONTACT_CHEBYSHEV,
    contact_manhattan: int = CONTACT_MANHATTAN,
) -> bool:
    """Swing only if some enemy is in the sword hitbox for ``direction``,
    or extremely close so contact damage / softlocks are avoided.

    ``threat_radius`` is accepted for API symmetry with approach logic; it does
    not by itself authorize a swing.
    """
    del threat_radius  # approach threshold only; attack uses hitbox/contact
    enemies = tuple(enemies)
    if not enemies:
        return False

    for obj in enemies:
        if in_sword_hitbox(
            link_x,
            link_y,
            direction,
            obj.x,
            obj.y,
            reach=swing_reach,
            half_width=half_width,
        ):
            return True
        d_man = manhattan(link_x, link_y, obj.x, obj.y)
        d_cheb = chebyshev(link_x, link_y, obj.x, obj.y)
        if d_cheb <= contact_chebyshev or d_man <= contact_manhattan:
            return True
    return False


def overworld_threat_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Live-looking OW objects (enemies/projectiles in playfield bounds)."""
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and obj.type_id not in (0, 0xFF)
        and 40 < obj.y < 220
        and 8 < obj.x < 248
    )


__all__ = [
    "SWORD_REACH",
    "SWORD_HALF_WIDTH",
    "THREAT_RADIUS",
    "CONTACT_CHEBYSHEV",
    "CONTACT_MANHATTAN",
    "FACING_NORTH",
    "FACING_SOUTH",
    "FACING_EAST",
    "FACING_WEST",
    "direction_to_facing",
    "facing_to_direction",
    "manhattan",
    "chebyshev",
    "in_sword_hitbox",
    "nearest_enemy",
    "should_swing_at",
    "overworld_threat_objects",
]

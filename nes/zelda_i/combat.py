"""Sword hitbox / threat helpers for Zelda I combat policies.

Pure functions over Link + enemy positions. Used by the generic dungeon
controller so swings only fire when the blade can actually hit.
"""

from __future__ import annotations

from typing import Iterable

from zelda_i.dungeon.ids import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATE,
    CLOCK_DROP_OBJECT_TYPE,
    FAIRY_DROP_OBJECT_TYPE,
    FAIRY_DROP_STATE,
    FIVE_RUPEE_DROP_OBJECT_TYPE,
    FIVE_RUPEE_DROP_STATE,
    GHINI_FLYING_OBJECT_TYPE,
    HEART_DROP_OBJECT_TYPE,
    HEART_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
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
    hint: object | None = None,
) -> bool:
    """Swing only if some enemy is in the sword hitbox for ``direction``,
    or extremely close so contact damage / softlocks are avoided.

    ``threat_radius`` is accepted for API symmetry with approach logic; it does
    not by itself authorize a swing.

    Optional ``hint`` (see ``combat_behaviors.EngagementHint``) may veto:
    ``swing=False`` or ``retreat=True``. It cannot authorize a swing outside
    the hitbox / contact guard.
    """
    del threat_radius  # approach threshold only; attack uses hitbox/contact
    if hint is not None:
        if hasattr(hint, "swing") and not bool(getattr(hint, "swing")):
            return False
        if bool(getattr(hint, "retreat", False)):
            return False
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


# Live At4A: every floor drop is ObjType 0x60. Item code is ObjState
# (0x22 heart, 0x23 fairy, 0x18 rupee, 0x0F 5-rupee, 0x21 clock).
# Type 0x22 is ghini_flying, never a heart.
FLOOR_DROP_TYPES = frozenset(
    {
        RUPEE_DROP_OBJECT_TYPE,
        HEART_DROP_OBJECT_TYPE,
        FAIRY_DROP_OBJECT_TYPE,
        FIVE_RUPEE_DROP_OBJECT_TYPE,
        CLOCK_DROP_OBJECT_TYPE,
        BOMB_DROP_OBJECT_TYPE,
    }
)
HEART_OR_FAIRY_TYPES = frozenset({HEART_DROP_OBJECT_TYPE, FAIRY_DROP_OBJECT_TYPE})
HEART_OR_FAIRY_STATES = frozenset({HEART_DROP_STATE, FAIRY_DROP_STATE})
RUPEE_DROP_STATES = frozenset({RUPEE_DROP_STATE, FIVE_RUPEE_DROP_STATE})
BOMB_DROP_STATES = frozenset({BOMB_DROP_STATE})


def _in_drop_bounds(obj: ZeldaObject) -> bool:
    return obj.slot >= 1 and 8 < obj.x < 248 and 40 < obj.y < 220


def is_floor_drop(obj: ZeldaObject) -> bool:
    """True for an in-bounds floor drop sprite (never a living ghini).

    Live type_id is 0x60 (hp 0, flash 0x80). Heart vs rupee is ObjState
    (0x22 vs 0x18), not ObjType — 0x22 as type is ``ghini_flying``.
    """
    if not _in_drop_bounds(obj):
        return False
    if int(obj.type_id) == GHINI_FLYING_OBJECT_TYPE:
        return False
    return int(obj.type_id) in FLOOR_DROP_TYPES


def is_heart_or_fairy_drop(obj: ZeldaObject) -> bool:
    """Heart/fairy among 0x60 drops: ObjState item code 0x22 / 0x23."""
    return is_floor_drop(obj) and int(obj.state) in HEART_OR_FAIRY_STATES


def floor_drops(
    snap: ZeldaSnapshot,
    types: Iterable[int] | None = None,
    states: Iterable[int] | None = None,
) -> tuple[ZeldaObject, ...]:
    """In-bounds 0x60 floor drops, optionally filtered by ObjType and ObjState.

    Live item identity is ObjState (heart 0x22, fairy 0x23, rupee 0x18).
    Filtering by type alone returns every drop.
    """
    wanted_types = (
        FLOOR_DROP_TYPES if types is None else frozenset(int(t) for t in types)
    )
    wanted_states = (
        None if states is None else frozenset(int(s) for s in states)
    )
    return tuple(
        obj
        for obj in snap.objects
        if is_floor_drop(obj)
        and int(obj.type_id) in wanted_types
        and (wanted_states is None or int(obj.state) in wanted_states)
    )


def heart_or_fairy_drops(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return floor_drops(snap, states=HEART_OR_FAIRY_STATES)


def nearest_floor_drop(
    snap: ZeldaSnapshot | int,
    types: Iterable[int] | None = None,
    drops: Iterable[ZeldaObject] | None = None,
    *,
    states: Iterable[int] | None = None,
) -> ZeldaObject | None:
    """Nearest floor drop to Link. Also accepts ``(link_x, link_y, drops)``."""
    if drops is not None:
        return nearest_enemy(int(snap), int(types or 0), drops)
    if not isinstance(snap, ZeldaSnapshot):
        return None
    return nearest_enemy(
        snap.link_x, snap.link_y, floor_drops(snap, types, states=states)
    )


def nearest_heart_or_fairy(snap: ZeldaSnapshot) -> ZeldaObject | None:
    return nearest_floor_drop(snap, states=HEART_OR_FAIRY_STATES)


DOOR_EDGE = 16


def scoop_exits_room(
    link_x: int,
    link_y: int,
    drop: ZeldaObject,
    *,
    bounds: tuple[int, int, int, int],
    edge: int = DOOR_EDGE,
) -> bool:
    """True when walking to ``drop`` heads into a door-mouth wall edge.

    ``bounds`` is ``(xmin, xmax, ymin, ymax)``; the caller owns the room box.
    """
    xmin, xmax, ymin, ymax = bounds
    dx = int(drop.x) - int(link_x)
    dy = int(drop.y) - int(link_y)
    return (
        (int(drop.x) <= xmin + edge and dx < 0)
        or (int(drop.x) >= xmax - edge and dx > 0)
        or (int(drop.y) <= ymin + edge and dy < 0)
        or (int(drop.y) >= ymax - edge and dy > 0)
    )


def wants_heart_pickup(snap: ZeldaSnapshot) -> bool:
    """True when a container is empty (2/3 yes, 3/3 no). min_filled is the caller."""
    return not snap.health_is_full


def overworld_threat_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Live OW combatants: typed, in-bounds, hp>0, not floor drops.

    OW octoroks use HP; corpses (hp<=0) and type 0x60 drops (heart/rupee/fairy
    even with hp>0) are not threats. Type-only liveness (Keese) is a dungeon
    rule — kept local to avoid a combat→behaviors import cycle.
    """
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and obj.type_id not in (0, 0xFF)
        and int(obj.type_id) not in FLOOR_DROP_TYPES
        and int(obj.hp) > 0
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
    "FLOOR_DROP_TYPES",
    "HEART_OR_FAIRY_TYPES",
    "HEART_OR_FAIRY_STATES",
    "RUPEE_DROP_STATES",
    "BOMB_DROP_OBJECT_TYPE",
    "BOMB_DROP_STATE",
    "BOMB_DROP_STATES",
    "direction_to_facing",
    "facing_to_direction",
    "manhattan",
    "chebyshev",
    "in_sword_hitbox",
    "nearest_enemy",
    "should_swing_at",
    "overworld_threat_objects",
    "is_floor_drop",
    "is_heart_or_fairy_drop",
    "floor_drops",
    "heart_or_fairy_drops",
    "nearest_floor_drop",
    "nearest_heart_or_fairy",
    "scoop_exits_room",
    "DOOR_EDGE",
    "wants_heart_pickup",
]

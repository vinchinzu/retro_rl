"""Reusable Zelda I enemy engagement policies.

Pure helpers over Link + object slots. Room geometry stays on
``DungeonRoomSpec``; the generic dungeon controller can call these later.
Hitbox-gated sword in ``combat.should_swing_at`` remains the swing gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from zelda_i.dungeon import ids as _ids
from zelda_i.combat import should_swing_at
from zelda_i.dungeon.engine import AliveRule
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

# IDs already catalogued in dungeon_ids.
KEESE_TYPE = _ids.KEESE_OBJECT_TYPE
VIRE_SPLIT_KEESE_TYPE = _ids.VIRE_SPLIT_KEESE_TYPE
ROPE_TYPE = _ids.ROPE_OBJECT_TYPE
GORIYA_TYPE = _ids.GORIYA_OBJECT_TYPE
GORIYA_BLUE_TYPE = _ids.GORIYA_BLUE_OBJECT_TYPE
POLS_VOICE_TYPE = _ids.POLS_VOICE_OBJECT_TYPE
GIBDO_TYPE = _ids.GIBDO_OBJECT_TYPE
WALLMASTER_TYPE = _ids.WALLMASTER_OBJECT_TYPE
FIREBALL_TYPE = _ids.FIREBALL_OBJECT_TYPE
MANHANDLA_PROJECTILE_TYPE = _ids.MANHANDLA_PROJECTILE_TYPE

# Live-probe IDs not yet exported from dungeon_ids (L1 Stalfos; L5 Digdogger).
STALFOS_TYPE = 0x2A
DIGDOGGER_TYPE = 0x38  # large form, HP 240; whistle-immune to sword
DIGDOGGER_SHRUNK_TYPE = 0x18  # after recorder; sword-legal

_EMPTY_TYPES = frozenset({0, 0xFF})

# Aquamentus-style approach band (level1_finish); generalized to four facings.
PROJECTILE_AHEAD = 48
PROJECTILE_BEHIND = 8
PROJECTILE_HALF_WIDTH = 20

# Dormant Wallmasters park just outside the wall (L1 0x45: x=0).
WALLMASTER_X_LO = 16
WALLMASTER_X_HI = 240
WALLMASTER_Y_LO = 72
WALLMASTER_Y_HI = 208

ROPE_AXIS_BAND = 12

PROJECTILE_TYPES = frozenset({FIREBALL_TYPE, MANHANDLA_PROJECTILE_TYPE})

DIGDOGGER_POLICY = (
    "Whistle shrinks type 0x38 (HP 240) to 0x18 (HP 128); sword only after shrink."
)


class EnemyKind(Enum):
    STALFOS = "stalfos"
    KEESE = "keese"
    ROPE = "rope"
    GORIYA = "goriya"
    POLS_VOICE = "pols_voice"
    GIBDO = "gibdo"
    WALLMASTER = "wallmaster"
    DIGDOGGER = "digdogger"
    PROJECTILE = "projectile"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EngagementHint:
    """One-frame advice. ``should_swing_at`` may veto on ``swing`` / ``retreat``."""

    preferred_distance: int
    face: str
    swing: bool
    retreat: bool


@dataclass(frozen=True)
class KindPolicy:
    preferred_distance: int
    alive_rule: AliveRule
    type_only: bool = False
    whistle_then_sword: bool = False
    off_wall_only: bool = False
    projectile_aware: bool = False
    notes: str = ""


KIND_POLICY: dict[EnemyKind, KindPolicy] = {
    EnemyKind.STALFOS: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Chase-and-slash; default CombatTuning engage 48.",
    ),
    EnemyKind.KEESE: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE,
        type_only=True,
        notes="HP stays 0 while alive; never use TYPE_AND_HP alone.",
    ),
    EnemyKind.ROPE: KindPolicy(
        preferred_distance=64,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Charge on-axis; face the lane and slash as they enter the blade.",
    ),
    EnemyKind.GORIYA: KindPolicy(
        preferred_distance=72,
        alive_rule=AliveRule.TYPE_AND_HP,
        projectile_aware=True,
        notes="Boomerang / fireball slots: do not walk into the approach band.",
    ),
    EnemyKind.POLS_VOICE: KindPolicy(
        preferred_distance=72,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Sword works; keep mid-range (room specs use 72).",
    ),
    EnemyKind.GIBDO: KindPolicy(
        preferred_distance=56,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Tanky mummy; same chase-and-slash as Stalfos, slightly closer.",
    ),
    EnemyKind.WALLMASTER: KindPolicy(
        preferred_distance=80,
        alive_rule=AliveRule.TYPE_AND_HP,
        off_wall_only=True,
        notes="Engage only after leaving the wall; ignore x≈0 parked slots.",
    ),
    EnemyKind.DIGDOGGER: KindPolicy(
        preferred_distance=64,
        alive_rule=AliveRule.TYPE_AND_HP,
        whistle_then_sword=True,
        projectile_aware=True,
        notes=DIGDOGGER_POLICY,
    ),
    EnemyKind.PROJECTILE: KindPolicy(
        preferred_distance=40,
        alive_rule=AliveRule.TYPE,
        projectile_aware=True,
        notes="Not a sword target; step off the approach band.",
    ),
    EnemyKind.UNKNOWN: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Generic melee; hitbox still gates the swing.",
    ),
}

_TYPE_TO_KIND: dict[int, EnemyKind] = {
    STALFOS_TYPE: EnemyKind.STALFOS,
    KEESE_TYPE: EnemyKind.KEESE,
    VIRE_SPLIT_KEESE_TYPE: EnemyKind.KEESE,
    ROPE_TYPE: EnemyKind.ROPE,
    GORIYA_TYPE: EnemyKind.GORIYA,
    GORIYA_BLUE_TYPE: EnemyKind.GORIYA,
    POLS_VOICE_TYPE: EnemyKind.POLS_VOICE,
    GIBDO_TYPE: EnemyKind.GIBDO,
    WALLMASTER_TYPE: EnemyKind.WALLMASTER,
    DIGDOGGER_TYPE: EnemyKind.DIGDOGGER,
    DIGDOGGER_SHRUNK_TYPE: EnemyKind.DIGDOGGER,
    FIREBALL_TYPE: EnemyKind.PROJECTILE,
    MANHANDLA_PROJECTILE_TYPE: EnemyKind.PROJECTILE,
}


def kind_for_type(type_id: int) -> EnemyKind:
    return _TYPE_TO_KIND.get(int(type_id) & 0xFF, EnemyKind.UNKNOWN)


def policy_for(kind: EnemyKind | int) -> KindPolicy:
    return KIND_POLICY[_coerce_kind(kind)]


def default_alive_rule(kind: EnemyKind | int) -> AliveRule:
    return policy_for(kind).alive_rule


def _coerce_kind(kind: EnemyKind | int) -> EnemyKind:
    if isinstance(kind, EnemyKind):
        return kind
    return kind_for_type(int(kind))


def _link_xy(link: ZeldaSnapshot | tuple[int, int]) -> tuple[int, int]:
    if isinstance(link, tuple):
        return int(link[0]), int(link[1])
    return int(link.link_x), int(link.link_y)


def _rule_token(rule: object) -> str:
    token = getattr(rule, "value", rule)
    return str(token).lower()


def is_typed(obj: ZeldaObject) -> bool:
    return (int(obj.type_id) & 0xFF) not in _EMPTY_TYPES


def uses_type_only_liveness(obj_or_kind: ZeldaObject | EnemyKind | int) -> bool:
    if isinstance(obj_or_kind, EnemyKind):
        return KIND_POLICY[obj_or_kind].type_only
    if isinstance(obj_or_kind, int):
        return KIND_POLICY[kind_for_type(obj_or_kind)].type_only
    return KIND_POLICY[kind_for_type(obj_or_kind.type_id)].type_only


def liveness(obj: ZeldaObject, rule: AliveRule | str) -> bool:
    """True if ``obj`` is a living combatant under ``rule``.

    Keese (and Vire-split 0x1c) keep HP=0 while alive — type-only even when
    the room spec says TYPE_AND_HP. Empty type 0 / 0xFF is never live.
    """
    if not is_typed(obj):
        return False
    if uses_type_only_liveness(obj):
        return True
    if _rule_token(rule) == AliveRule.TYPE.value:
        return True
    return int(obj.hp) > 0


def live_among(
    objects: Iterable[ZeldaObject],
    rule: AliveRule | str,
) -> tuple[ZeldaObject, ...]:
    return tuple(obj for obj in objects if liveness(obj, rule))


def face_toward(
    link_x: int,
    link_y: int,
    enemy_x: int,
    enemy_y: int,
    *,
    dominant_axis: bool = False,
) -> str:
    dx = int(enemy_x) - int(link_x)
    dy = int(enemy_y) - int(link_y)
    if dominant_axis and abs(dy) > 10 and abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx >= 0 else "LEFT"
    return "DOWN" if dy >= 0 else "UP"


def rope_on_axis(
    link_x: int,
    link_y: int,
    enemy: ZeldaObject,
    *,
    band: int = ROPE_AXIS_BAND,
) -> bool:
    return (
        abs(int(enemy.x) - int(link_x)) <= band
        or abs(int(enemy.y) - int(link_y)) <= band
    )


def _face_rope(link_x: int, link_y: int, enemy: ZeldaObject) -> str:
    dx = int(enemy.x) - int(link_x)
    dy = int(enemy.y) - int(link_y)
    if abs(dy) <= ROPE_AXIS_BAND and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dx) <= ROPE_AXIS_BAND and dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return face_toward(link_x, link_y, enemy.x, enemy.y)


def is_off_wall(obj: ZeldaObject) -> bool:
    """True once a Wallmaster has left the wall-parked slot (not x≈0)."""
    x, y = int(obj.x), int(obj.y)
    return (
        WALLMASTER_X_LO < x < WALLMASTER_X_HI
        and WALLMASTER_Y_LO < y < WALLMASTER_Y_HI
    )


def is_projectile(obj: ZeldaObject) -> bool:
    return (int(obj.type_id) & 0xFF) in PROJECTILE_TYPES


def needs_whistle(obj: ZeldaObject) -> bool:
    """Digdogger large form: recorder first; sword is not legal yet."""
    return (int(obj.type_id) & 0xFF) == DIGDOGGER_TYPE


def is_shrunk(obj: ZeldaObject) -> bool:
    return (int(obj.type_id) & 0xFF) == DIGDOGGER_SHRUNK_TYPE


def sword_legal(kind: EnemyKind | int, enemy: ZeldaObject) -> bool:
    kind = _coerce_kind(kind)
    if kind is EnemyKind.PROJECTILE:
        return False
    if kind is EnemyKind.DIGDOGGER:
        return is_shrunk(enemy)
    if kind is EnemyKind.WALLMASTER:
        return is_off_wall(enemy)
    return True


def projectile_threats(
    link_x: int,
    link_y: int,
    objects: Iterable[ZeldaObject],
    *,
    direction: str = "RIGHT",
    ahead: int = PROJECTILE_AHEAD,
    behind: int = PROJECTILE_BEHIND,
    half_width: int = PROJECTILE_HALF_WIDTH,
) -> tuple[ZeldaObject, ...]:
    """Projectiles in the approach band along ``direction``.

    Aquamentus (facing east) used ``-8 <= dx <= 48`` and ``|dy| < 20``.
    """
    facing = direction.upper()
    hits: list[ZeldaObject] = []
    for obj in objects:
        if not is_projectile(obj):
            continue
        dx = int(obj.x) - int(link_x)
        dy = int(obj.y) - int(link_y)
        if facing == "RIGHT":
            in_band = -behind <= dx <= ahead and abs(dy) <= half_width
        elif facing == "LEFT":
            in_band = -ahead <= dx <= behind and abs(dy) <= half_width
        elif facing == "DOWN":
            in_band = -behind <= dy <= ahead and abs(dx) <= half_width
        elif facing == "UP":
            in_band = -ahead <= dy <= behind and abs(dx) <= half_width
        else:
            raise ValueError(f"unsupported direction: {direction}")
        if in_band:
            hits.append(obj)
    return tuple(hits)


def blocked_by_projectile(
    link_x: int,
    link_y: int,
    direction: str,
    objects: Iterable[ZeldaObject],
    *,
    ahead: int = PROJECTILE_AHEAD,
    behind: int = PROJECTILE_BEHIND,
    half_width: int = PROJECTILE_HALF_WIDTH,
) -> bool:
    """True if walking ``direction`` steps into a known projectile slot."""
    return bool(
        projectile_threats(
            link_x,
            link_y,
            objects,
            direction=direction,
            ahead=ahead,
            behind=behind,
            half_width=half_width,
        )
    )


def engagement_hint(
    kind: EnemyKind | int,
    link: ZeldaSnapshot | tuple[int, int],
    enemy: ZeldaObject,
    *,
    projectiles: Iterable[ZeldaObject] = (),
    facing: str | None = None,
) -> EngagementHint:
    """Preferred range / facing / swing / retreat for one enemy.

    ``swing`` is True only when the policy allows a sword *and* the blade
    hitbox (or contact guard) would hit. ``retreat`` means do not walk into
    a projectile band or a Wallmaster still on the wall.
    """
    kind = _coerce_kind(kind)
    policy = KIND_POLICY[kind]
    lx, ly = _link_xy(link)

    if kind is EnemyKind.ROPE:
        face = facing.upper() if facing else _face_rope(lx, ly, enemy)
    else:
        face = facing.upper() if facing else face_toward(
            lx,
            ly,
            enemy.x,
            enemy.y,
            dominant_axis=kind is EnemyKind.WALLMASTER,
        )

    shots = tuple(projectiles)
    retreat = False
    if policy.projectile_aware and blocked_by_projectile(lx, ly, face, shots):
        retreat = True
    if kind is EnemyKind.PROJECTILE:
        retreat = True
    if kind is EnemyKind.WALLMASTER and not is_off_wall(enemy):
        retreat = False  # park; do not chase into the wall

    allow = sword_legal(kind, enemy)
    swing = bool(allow and should_swing_at(lx, ly, face, (enemy,)))
    return EngagementHint(
        preferred_distance=policy.preferred_distance,
        face=face,
        swing=swing,
        retreat=retreat,
    )


__all__ = [
    "KEESE_TYPE",
    "VIRE_SPLIT_KEESE_TYPE",
    "ROPE_TYPE",
    "GORIYA_TYPE",
    "GORIYA_BLUE_TYPE",
    "POLS_VOICE_TYPE",
    "GIBDO_TYPE",
    "WALLMASTER_TYPE",
    "FIREBALL_TYPE",
    "MANHANDLA_PROJECTILE_TYPE",
    "STALFOS_TYPE",
    "DIGDOGGER_TYPE",
    "DIGDOGGER_SHRUNK_TYPE",
    "PROJECTILE_TYPES",
    "DIGDOGGER_POLICY",
    "EnemyKind",
    "EngagementHint",
    "KindPolicy",
    "KIND_POLICY",
    "kind_for_type",
    "policy_for",
    "default_alive_rule",
    "is_typed",
    "uses_type_only_liveness",
    "liveness",
    "live_among",
    "face_toward",
    "rope_on_axis",
    "is_off_wall",
    "is_projectile",
    "needs_whistle",
    "is_shrunk",
    "sword_legal",
    "projectile_threats",
    "blocked_by_projectile",
    "engagement_hint",
]

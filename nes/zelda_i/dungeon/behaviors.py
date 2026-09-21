"""Reusable Zelda I enemy engagement policies.

Pure helpers over Link + object slots. Room geometry stays on
``DungeonRoomSpec``; the generic dungeon controller can call these later.
Hitbox-gated sword in ``combat.should_swing_at`` remains the swing gate.

**This module owns policy: how we choose to fight a kind.** What the ROM says
a type *is* -- hitbox shape, HP, speed, damage -- lives in
``dungeon/species.py`` (sourced in ``scratch/enemy_constants_rom.md``). The two
touch in exactly two places, and neither value is stored twice:
``alive_rule == TYPE`` is true *because* the ROM HP is 0, and
``whistle_then_sword`` is true *because* type ``$38`` carries object attribute
``$20``. ``tests/test_species.py`` asserts both agreements rather than copying
the numbers across.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from zelda_i.dungeon import ids as _ids
from zelda_i.combat import should_swing_at
from zelda_i.dungeon.ids import AliveRule
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

# IDs already catalogued in dungeon_ids.
KEESE_TYPE = _ids.KEESE_OBJECT_TYPE
VIRE_SPLIT_KEESE_TYPE = _ids.VIRE_SPLIT_KEESE_TYPE
# $1D is the third Keese. `NoDropMonsterTypes` lists $1B/$1C/$1D together and
# the ROM gives $1D HP 0 (`species.species_of(0x1D).hp`), so leaving it out of
# `_TYPE_TO_KIND` made it EnemyKind.UNKNOWN -> AliveRule.TYPE_AND_HP -> dead on
# arrival, the exact failure the KEESE note exists to prevent.
KEESE_BLACK_TYPE = _ids.KEESE_BLACK_OBJECT_TYPE
GEL_TYPE = _ids.GEL_OBJECT_TYPE
# $14 is the Zol-split Gel residual: same ROM HP 0 as $15, same kind.
GEL_SPLIT_TYPE = _ids.GEL_SPLIT_OBJECT_TYPE
ROPE_TYPE = _ids.ROPE_OBJECT_TYPE
GORIYA_TYPE = _ids.GORIYA_OBJECT_TYPE
GORIYA_BLUE_TYPE = _ids.GORIYA_BLUE_OBJECT_TYPE
POLS_VOICE_TYPE = _ids.POLS_VOICE_OBJECT_TYPE
GIBDO_TYPE = _ids.GIBDO_OBJECT_TYPE
WALLMASTER_TYPE = _ids.WALLMASTER_OBJECT_TYPE
FIREBALL_TYPE = _ids.FIREBALL_OBJECT_TYPE
MANHANDLA_PROJECTILE_TYPE = _ids.MANHANDLA_PROJECTILE_TYPE
GORIYA_BOOMERANG_TYPE = _ids.GORIYA_BOOMERANG_OBJECT_TYPE
ROCK_PROJECTILE_TYPE = _ids.ROCK_PROJECTILE_TYPE
LYNEL_SWORD_SHOT_TYPE = _ids.LYNEL_SWORD_SHOT_TYPE
MOBLIN_ARROW_TYPE = _ids.MOBLIN_ARROW_OBJECT_TYPE
OCTOROK_TYPE = _ids.OCTOROK_OBJECT_TYPE
OCTOROK_FAST_TYPE = _ids.OCTOROK_FAST_OBJECT_TYPE
OCTOROK_BLUE_TYPE = _ids.OCTOROK_BLUE_OBJECT_TYPE
OCTOROK_BLUE_FAST_TYPE = _ids.OCTOROK_BLUE_FAST_OBJECT_TYPE
MOBLIN_TYPE = _ids.MOBLIN_OBJECT_TYPE
MOBLIN_BLUE_TYPE = _ids.MOBLIN_BLUE_OBJECT_TYPE
LYNEL_TYPE = _ids.LYNEL_OBJECT_TYPE
LYNEL_BLUE_TYPE = _ids.LYNEL_BLUE_OBJECT_TYPE
TEKTITE_TYPE = _ids.TEKTITE_OBJECT_TYPE
TEKTITE_BLUE_TYPE = _ids.TEKTITE_BLUE_OBJECT_TYPE
LEEVER_TYPE = _ids.LEEVER_OBJECT_TYPE
LEEVER_BLUE_TYPE = _ids.LEEVER_BLUE_OBJECT_TYPE
ZORA_TYPE = _ids.ZORA_OBJECT_TYPE
PEAHAT_TYPE = _ids.PEAHAT_OBJECT_TYPE
ARMOS_TYPE = _ids.ARMOS_OBJECT_TYPE
GHINI_TYPE = _ids.GHINI_OBJECT_TYPE
GHINI_FLYING_TYPE = _ids.GHINI_FLYING_OBJECT_TYPE

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

PROJECTILE_TYPES = _ids.PROJECTILE_TYPES

_SMALL_SHIELD_BLOCKS = frozenset(
    {ROCK_PROJECTILE_TYPE, MOBLIN_ARROW_TYPE, LYNEL_SWORD_SHOT_TYPE}
)
_MAGIC_SHIELD_BLOCKS = frozenset({FIREBALL_TYPE, MANHANDLA_PROJECTILE_TYPE})

DIGDOGGER_POLICY = (
    "Whistle shrinks type 0x38 (HP 240) to 0x18 (HP 128); sword only after shrink."
)

# --- Zora surfacing cycle (measured) ---------------------------------------
# ``scratch/probe_zora.py`` on the live pre-L1 walk, OW ``0x59``, four
# surfacings in ``scratch/zora1.json`` -- identical to the frame every time.
# ``ObjState`` (``$00AC``) counts the cycle; the Zora never moves inside one,
# and it picks a new tile for the next.
#
#   state  frames  what it is
#   0x00       2   surfacing begins
#   0x01      32   rising
#   0x02      15   surfaced, mouth open -- the tell
#   0x03      34   firing; the 0x55 slot appears 2f in and then *holds still*
#   0x04      16   submerging
#   0x05      96   submerged (nothing on screen to hit or dodge)
#
# Two numbers matter to a 1 px/frame walker. The shot is born 2 frames into
# ``0x03``, and it then sits on the Zora's mouth for ``ZORA_MUZZLE_DWELL``
# frames before it moves at all -- so ``ObjectTracker`` measures it at zero
# velocity and ``threat.assess`` calls it safe for the entire window in which
# it could still be walked away from. Counting from the ``0x01 -> 0x02`` edge
# there are 34 frames before anything travels, which is nearly three times
# ``threat.MIN_DODGE_SHOT``.
ZORA_CYCLE: dict[int, int] = {0x00: 2, 0x01: 32, 0x02: 15, 0x03: 34, 0x04: 16, 0x05: 96}
ZORA_STATE_SURFACING = 0x01
ZORA_STATE_AIMING = 0x02  # mouth open; the shot is 17 frames out
ZORA_STATE_FIRING = 0x03
ZORA_STATE_SUBMERGING = 0x04
ZORA_STATE_SUBMERGED = 0x05
# Frames into ``0x03`` before the ``0x55`` slot exists.
ZORA_SHOT_DELAY = 2
# Frames the shot holds at the muzzle before its first pixel of travel. This
# is the whole dodge window, and it is invisible to a velocity tracker.
ZORA_MUZZLE_DWELL = 17
# px/frame along the major axis once it launches (measured 1.65-1.76), against
# Link's 1.0. Aim is quantized at launch, not a clean bearing to Link: the
# four measured shots left at 180.0, 180.0, -171.1 and -124.2 degrees against
# bearings of 180.0, -172.7, -162.9 and -119.3. Close the angle, do not try to
# solve the line.
ZORA_SHOT_SPEED = 1.75
# A Zora is not a kill. ``UpdateZora`` submerges it with ``DestroyMonster``
# (no ``HandleMonsterDied``), so the slot vanishing banks no streak tick --
# see ``overworld.prey.SKIP_TYPES`` and ``scratch/drop_mechanics_rom.md``.
ZORA_POLICY = (
    "Surfaces on a 195f cycle; fires 2f into ObjState 0x03 and the shot holds "
    "17f at the muzzle. Never a sword target: it submerges on its own clock."
)


def zora_shot_eta(obj: ZeldaObject) -> int | None:
    """Frames until this Zora's shot starts travelling, or ``None``.

    ``None`` for anything that is not a surfaced Zora, and for one already
    submerging: there is nothing left to walk away from. The count is
    deliberately conservative inside a state -- it assumes the Zora just
    entered it -- because the phase offset is not in RAM and over-estimating
    the warning is the failure that gets Link shot.
    """
    if (int(obj.type_id) & 0xFF) != ZORA_TYPE:
        return None
    state = int(obj.state)
    if state >= ZORA_STATE_SUBMERGING:
        return None
    launch = ZORA_SHOT_DELAY + ZORA_MUZZLE_DWELL
    if state == ZORA_STATE_FIRING:
        return launch
    eta = launch
    for phase in range(state, ZORA_STATE_FIRING):
        eta += ZORA_CYCLE[phase]
    return eta


class EnemyKind(Enum):
    STALFOS = "stalfos"
    KEESE = "keese"
    GEL = "gel"
    ROPE = "rope"
    GORIYA = "goriya"
    POLS_VOICE = "pols_voice"
    GIBDO = "gibdo"
    WALLMASTER = "wallmaster"
    DIGDOGGER = "digdogger"
    OCTOROK = "octorok"
    MOBLIN = "moblin"
    LYNEL = "lynel"
    TEKTITE = "tektite"
    LEEVER = "leever"
    PEAHAT = "peahat"
    ZORA = "zora"
    GHINI = "ghini"
    ARMOS = "armos"
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
        notes=(
            "HP stays 0 while alive; never use TYPE_AND_HP alone. "
            "Three types: 0x1B, 0x1C (Vire split) and 0x1D (black)."
        ),
    ),
    EnemyKind.GEL: KindPolicy(
        preferred_distance=40,
        alive_rule=AliveRule.TYPE,
        notes="Slow blob. Open-floor rooms raise engage_distance; do not chase flyers the same way.",
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
    EnemyKind.OCTOROK: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        projectile_aware=True,
        notes="OW melee; rocks are type 0x53. HP>0 while alive.",
    ),
    EnemyKind.MOBLIN: KindPolicy(
        preferred_distance=56,
        alive_rule=AliveRule.TYPE_AND_HP,
        projectile_aware=True,
        notes="Spear throw 0x5B; same chase-and-slash as Goriya.",
    ),
    EnemyKind.LYNEL: KindPolicy(
        preferred_distance=72,
        alive_rule=AliveRule.TYPE_AND_HP,
        projectile_aware=True,
        notes="Sword beam 0x57; keep mid-range (Death Mountain).",
    ),
    EnemyKind.TEKTITE: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Hopping melee; hitbox still gates the swing.",
    ),
    EnemyKind.LEEVER: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Burrow/surface melee; HP>0 while alive.",
    ),
    EnemyKind.PEAHAT: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes=(
            "Invulnerable while flying: CheckMonsterCollisions only when "
            "Flyer_ObjFlyingState $444==5 (landed). ZeldaObject.state is "
            "ObjState $00AC, not flying state — cannot gate sword_legal."
        ),
    ),
    EnemyKind.ZORA: KindPolicy(
        preferred_distance=64,
        alive_rule=AliveRule.TYPE_AND_HP,
        projectile_aware=True,
        notes="Water spit is fireball 0x55; HP>0 while surfaced.",
    ),
    EnemyKind.GHINI: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes="Graveyard melee; 0x22 flying variant same HP rule.",
    ),
    EnemyKind.ARMOS: KindPolicy(
        preferred_distance=48,
        alive_rule=AliveRule.TYPE_AND_HP,
        notes=(
            "Type 0x1E is awake. Statue form is tile $66/$67, not an "
            "object slot — no RAM type/hp for dormant statues."
        ),
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
    KEESE_BLACK_TYPE: EnemyKind.KEESE,
    GEL_TYPE: EnemyKind.GEL,
    GEL_SPLIT_TYPE: EnemyKind.GEL,
    ROPE_TYPE: EnemyKind.ROPE,
    GORIYA_TYPE: EnemyKind.GORIYA,
    GORIYA_BLUE_TYPE: EnemyKind.GORIYA,
    POLS_VOICE_TYPE: EnemyKind.POLS_VOICE,
    GIBDO_TYPE: EnemyKind.GIBDO,
    WALLMASTER_TYPE: EnemyKind.WALLMASTER,
    DIGDOGGER_TYPE: EnemyKind.DIGDOGGER,
    DIGDOGGER_SHRUNK_TYPE: EnemyKind.DIGDOGGER,
    OCTOROK_TYPE: EnemyKind.OCTOROK,
    OCTOROK_FAST_TYPE: EnemyKind.OCTOROK,
    OCTOROK_BLUE_TYPE: EnemyKind.OCTOROK,
    OCTOROK_BLUE_FAST_TYPE: EnemyKind.OCTOROK,
    MOBLIN_TYPE: EnemyKind.MOBLIN,
    MOBLIN_BLUE_TYPE: EnemyKind.MOBLIN,
    LYNEL_TYPE: EnemyKind.LYNEL,
    LYNEL_BLUE_TYPE: EnemyKind.LYNEL,
    TEKTITE_TYPE: EnemyKind.TEKTITE,
    TEKTITE_BLUE_TYPE: EnemyKind.TEKTITE,
    LEEVER_TYPE: EnemyKind.LEEVER,
    LEEVER_BLUE_TYPE: EnemyKind.LEEVER,
    ZORA_TYPE: EnemyKind.ZORA,
    PEAHAT_TYPE: EnemyKind.PEAHAT,
    ARMOS_TYPE: EnemyKind.ARMOS,
    GHINI_TYPE: EnemyKind.GHINI,
    GHINI_FLYING_TYPE: EnemyKind.GHINI,
    FIREBALL_TYPE: EnemyKind.PROJECTILE,
    MANHANDLA_PROJECTILE_TYPE: EnemyKind.PROJECTILE,
    GORIYA_BOOMERANG_TYPE: EnemyKind.PROJECTILE,
    ROCK_PROJECTILE_TYPE: EnemyKind.PROJECTILE,
    LYNEL_SWORD_SHOT_TYPE: EnemyKind.PROJECTILE,
    MOBLIN_ARROW_TYPE: EnemyKind.PROJECTILE,
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


def shield_blocks(obj: ZeldaObject, *, magic_shield: bool = False) -> bool:
    """True if Link's shield stops ``obj`` while he faces it and does not swing.

    The small shield eats rocks, arrows and Lynel sword shots; fireballs
    (and Manhandla/Gleeok residuals) need the Magical Shield. A Goriya
    boomerang is not blockable at all — it stuns.
    """
    type_id = int(obj.type_id) & 0xFF
    if type_id in _SMALL_SHIELD_BLOCKS:
        return True
    return magic_shield and type_id in _MAGIC_SHIELD_BLOCKS


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


def fight_target(
    link_x: int,
    link_y: int,
    live: Iterable[ZeldaObject],
) -> ZeldaObject | None:
    """Nearest sword-legal combatant. Parked Wallmasters are not targets."""
    legal = tuple(obj for obj in live if sword_legal(obj.type_id, obj))
    if not legal:
        return None
    return min(
        legal,
        key=lambda obj: abs(int(obj.x) - int(link_x))
        + abs(int(obj.y) - int(link_y)),
    )


def may_close(
    *,
    distance: int,
    engage_distance: int,
    occupancy: bool,
    occupancy_dir: str | None,
) -> bool:
    """Leave patrol to close on a target.

    Open floor: ``engage_distance`` is the chase cap (Gel rooms raise it).
    Occupancy maze: follow a BFS path from any distance. No path + far
    stays on patrol waypoints (do not greedy through water). No path +
    inside the room cap still closes — occupancy may have miss-blocked
    the enemy's own pixel.
    """
    if occupancy and occupancy_dir is not None:
        return True
    return int(distance) < int(engage_distance)


__all__ = [
    "KEESE_TYPE",
    "KEESE_BLACK_TYPE",
    "VIRE_SPLIT_KEESE_TYPE",
    "GEL_SPLIT_TYPE",
    "ROPE_TYPE",
    "GORIYA_TYPE",
    "GORIYA_BLUE_TYPE",
    "POLS_VOICE_TYPE",
    "GIBDO_TYPE",
    "WALLMASTER_TYPE",
    "FIREBALL_TYPE",
    "shield_blocks",
    "MANHANDLA_PROJECTILE_TYPE",
    "GORIYA_BOOMERANG_TYPE",
    "ROCK_PROJECTILE_TYPE",
    "LYNEL_SWORD_SHOT_TYPE",
    "MOBLIN_ARROW_TYPE",
    "OCTOROK_TYPE",
    "OCTOROK_FAST_TYPE",
    "OCTOROK_BLUE_TYPE",
    "OCTOROK_BLUE_FAST_TYPE",
    "MOBLIN_TYPE",
    "MOBLIN_BLUE_TYPE",
    "LYNEL_TYPE",
    "LYNEL_BLUE_TYPE",
    "TEKTITE_TYPE",
    "TEKTITE_BLUE_TYPE",
    "LEEVER_TYPE",
    "LEEVER_BLUE_TYPE",
    "ZORA_TYPE",
    "PEAHAT_TYPE",
    "ARMOS_TYPE",
    "GHINI_TYPE",
    "GHINI_FLYING_TYPE",
    "STALFOS_TYPE",
    "GEL_TYPE",
    "DIGDOGGER_TYPE",
    "DIGDOGGER_SHRUNK_TYPE",
    "PROJECTILE_TYPES",
    "DIGDOGGER_POLICY",
    "ZORA_CYCLE",
    "ZORA_MUZZLE_DWELL",
    "ZORA_POLICY",
    "ZORA_SHOT_DELAY",
    "ZORA_SHOT_SPEED",
    "ZORA_STATE_AIMING",
    "ZORA_STATE_FIRING",
    "ZORA_STATE_SUBMERGED",
    "ZORA_STATE_SUBMERGING",
    "ZORA_STATE_SURFACING",
    "zora_shot_eta",
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
    "fight_target",
    "may_close",
]

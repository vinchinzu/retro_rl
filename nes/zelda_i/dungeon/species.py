"""ROM fact per object type: hitbox shape, HP, speed, damage.

`dungeon/threat.py` gives every object in the game `BODY_HALF = 8` and
`SHOT_HALF = 4`, and nothing in this tree knows a per-type speed -- speed is
*observed* by `dungeon.tracking`, never known. This module is the constants
half of that gap, read out of the disassembly and sourced line by line in
`scratch/enemy_constants_rom.md`.

It is **not** a port of enemy AI. `rollout.py` measured a savestate rollout
against the real ROM as bit-exact at ~6 us, so a re-implementation would be
less accurate and slower to build. Constants are what a rollout cannot hand
back cheaply, and what turns a wide search into a narrow one.

**Fact, not policy.** `dungeon/behaviors.py::KIND_POLICY` owns how we choose
to fight a kind (`preferred_distance`, `alive_rule`, `projectile_aware`,
`off_wall_only`, `whistle_then_sword`) and stays the seam the rest of the tree
imports. This module owns what the ROM says a type *is*. The two touch in two
places and neither is copied here:

* `KIND_POLICY.alive_rule == TYPE` (Keese, Gel) is true *because* the ROM HP is
  0 -- `tests/test_species.py` asserts the agreement.
* `KIND_POLICY.whistle_then_sword` (Digdogger) is true *because* type `$38`
  carries attribute bit `$20`, invincible to every weapon -- same test.

Nothing consumes this module yet. Wiring it into `threat` / `combat` is a
later card; adding a row here must not change any controller's behaviour.

Everything below is derived from three ROM byte arrays plus a small overlay of
per-type speeds. Sources (aldonunez/zelda1-disassembly @ `50a1c86`):

* `ObjectTypeToAttributes`  Z_07.asm#L5242
* `ObjectTypeToHpPairs`     Z_07.asm#L5256 (decode: Z_04.asm#L11002)
* `ObjTypeToDamagePoints`   Z_01.asm#L5574 (unpack: Z_01.asm#L5719)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from zelda_i.dungeon.ids import object_name

_SRC = (
    "https://github.com/aldonunez/zelda1-disassembly/blob/"
    "50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/"
)


def source(path: str, line: int) -> str:
    """Line-anchored URL into the pinned disassembly commit."""
    return f"{_SRC}{path}#L{line}"


# --- Hitbox: the ROM has no per-type extent -------------------------------
# `GetObjectMiddle` (Z_01.asm#L5550) puts every object's collision midpoint at
# (x + 8, y + 8), except that attribute $40 moves the *X* midpoint to x + 4.
# `DoObjectsCollide` (Z_01.asm#L6426) then compares |dmid| against one
# threshold on both axes, and for a monster touching Link that threshold is 9
# (Z_01.asm#L5647) -- for bodies and shots alike.
MID_OFFSET = 8
HALF_WIDTH_MID_OFFSET = 4
CONTACT_THRESHOLD = 9
ATTR_URL = source("Z_07.asm", 5242)
HP_URL = source("Z_07.asm", 5256)
DAMAGE_URL = source("Z_01.asm", 5574)
CONTACT_URL = source("Z_01.asm", 5647)
MID_URL = source("Z_01.asm", 5550)

# Conservative fallback for an id the ROM arrays do not cover. These mirror
# `threat.BODY_HALF` / `threat.SHOT_HALF` on purpose so an unlisted type keeps
# today's behaviour; `test_species.py` asserts they still agree. Not imported
# from `threat` -- this module stays pure data with no engine dependency.
DEFAULT_BODY_HALF = 8
DEFAULT_SHOT_HALF = 4

# --- Object attribute bits (each verified at the site that reads it) -------
ATTR_SELF_COLLIDE = 0x01  # Z_07.asm#L1969
ATTR_HALF_WIDTH_DRAW = 0x02  # Z_01.asm#L5120
ATTR_SELF_DRAW = 0x04  # Z_07.asm#L1974
ATTR_IGNORE_SPRITE_TABLE = 0x08  # Z_01.asm#L5126
ATTR_REVERSE_WHEN_BLOCKED = 0x10  # Z_07.asm#L2939 -- dead code, set on no type
ATTR_INVINCIBLE = 0x20  # Z_01.asm#L5480
ATTR_HALF_WIDTH_COLLISION = 0x40  # Z_01.asm#L5553
ATTR_REVERSE_AFTER_HIT = 0x80  # Z_01.asm#L6616

# --- Speed unit -----------------------------------------------------------
# `MoveObject` (Z_07.asm#L2768) applies the quarter speed four times a frame to
# an 8-bit position fraction, so px/frame = qspeed / 64. Every monster starts
# at $20 = 0.5 px/f (`InitMode_EnterRoom`, Z_05.asm#L1692); a type that never
# writes ObjQSpeedFrac runs at that.
QSPEED_PER_PIXEL = 64
DEFAULT_QSPEED = 0x20
DEFAULT_QSPEED_URL = source("Z_05.asm", 1692)
# Link is $60 = 1.5 px/f, not the 1.0 `threat.LINK_SPEED` assumes; $30 on
# overworld mountain-stair tiles $74/$75 (Z_05.asm#L7123).
LINK_QSPEED = 0x60
LINK_STAIRS_QSPEED = 0x30
LINK_QSPEED_URL = source("Z_05.asm", 7123)

# --- Weapon damage points (Z_01.asm#L6162, #L6108, #L6242, #L6039) --------
SWORD_DAMAGE = (0x10, 0x20, 0x40)  # wooden / white / magical
ARROW_DAMAGE = (0x20, 0x40)  # wooden / silver
BOMB_DAMAGE = 0x40
FIRE_DAMAGE = 0x10
ROD_DAMAGE = 0x20
MAGIC_SHOT_DAMAGE = 0x20
BOOMERANG_DAMAGE = 0x00  # never damages; stuns $10 (~$A0 frames)
SWORD_DAMAGE_URL = source("Z_01.asm", 6162)

# --- Damage-type bits, for ObjInvincibilityMask (Z_01.asm#L5921) ----------
DMG_SWORD = 0x01
DMG_BOOMERANG = 0x02
DMG_ARROW = 0x04
DMG_BOMB = 0x08
DMG_MAGIC = 0x10
DMG_FIRE = 0x20


class Contact(Enum):
    """What overlapping this body does to Link.

    Zelda has no solid monster -- Link walks into everything and is shoved
    (`BeginShove`, Z_01.asm#L5719). What differs per type is the *consequence*,
    and a bool cannot hold it:

    HARM      overlap costs hearts.
    HARMLESS  overlap runs `HarmLink` -> `Link_BeHarmed` with zero damage
              points, so it costs no hearts and still zeroes the kill streak
              (`$0627`/`$50`/`$51`) -- bubbles $2B-$2E, whirlwind. See
              `scratch/drop_mechanics_rom.md`.
    CAPTURE   overlap takes Link instead of hearts: Wallmaster $27 and
              Like-Like $17 increment `ObjCaptureTimer` (Z_01.asm#L5530).
    NONE      no collision test runs at all for this type in this state --
              a burrower under the sand (`no_contact_states`).
    """

    HARM = auto()
    HARMLESS = auto()
    CAPTURE = auto()
    NONE = auto()


_CAPTURE_TYPES = frozenset({0x17, 0x27})  # Z_01.asm#L5530
# `Burrower_AnimateDrawAndCheckCollisions` returns before any collision check
# when ObjState is 0 (Z_04.asm#L2664). `UpdateZora` runs the same updater
# (Z_04.asm#L1915), so a submerged Zora is as dormant as a buried leever --
# which is the ROM behind `combat.dormant_body`.
BURROWER_TYPES = frozenset({0x0F, 0x10, 0x11})
_BURROWER_TYPES = BURROWER_TYPES  # historical private name
BURROWER_DORMANT_STATE = 0
BURROWER_URL = source("Z_04.asm", 2664)


@dataclass(frozen=True)
class Speed:
    """One sourced q-speed, or a per-state tuple. `url` is not optional."""

    qspeed: int | tuple[int, ...]
    url: str
    per_state: bool = False
    note: str = ""

    def px_per_frame(self, state: int = 0) -> float:
        """px/frame at `state`; `state` is ignored unless `per_state`."""
        if self.per_state:
            assert isinstance(self.qspeed, tuple)
            index = int(state) % len(self.qspeed)
            value = self.qspeed[index]
        else:
            assert isinstance(self.qspeed, int)
            value = self.qspeed
        return value / QSPEED_PER_PIXEL


@dataclass(frozen=True)
class Species:
    """Static ROM facts for one `$03xx` object type.

    `hp` and `damage_points` are None where the ROM array does not reach the
    id -- blank, never a guessed zero. `speed` is None where no `STA
    ObjQSpeedFrac` of its own was found; such a type runs at `DEFAULT_QSPEED`.
    """

    type_id: int
    name: str
    attr: int
    hp: int | None
    damage_points: int | None
    contact: Contact
    speed: Speed | None = None
    invincible_to: int = 0
    no_contact_states: tuple[int, ...] = ()
    notes: str = ""

    @property
    def half_width(self) -> bool:
        """Attribute $40: the X collision midpoint is x+4, not x+8."""
        return bool(self.attr & ATTR_HALF_WIDTH_COLLISION)

    @property
    def mid_offset_x(self) -> int:
        return HALF_WIDTH_MID_OFFSET if self.half_width else MID_OFFSET

    @property
    def mid_offset_y(self) -> int:
        return MID_OFFSET

    @property
    def weapon_proof(self) -> bool:
        """Attribute $20: the whole weapon pass is skipped for this type."""
        return bool(self.attr & ATTR_INVINCIBLE)

    @property
    def contact_hearts(self) -> float:
        """Hearts one touch costs: low nibble whole, high nibble of 256."""
        if not self.damage_points:
            return 0.0
        return (self.damage_points & 0x0F) + (self.damage_points & 0xF0) / 256.0

    def qspeed(self, state: int = 0) -> int:
        """Sourced q-speed, or the room-entry default when none was found."""
        if self.speed is None:
            return DEFAULT_QSPEED
        if self.speed.per_state:
            assert isinstance(self.speed.qspeed, tuple)
            return self.speed.qspeed[int(state) % len(self.speed.qspeed)]
        assert isinstance(self.speed.qspeed, int)
        return self.speed.qspeed

    def px_per_frame(self, state: int = 0) -> float:
        return self.qspeed(state) / QSPEED_PER_PIXEL

    def weak_to(self, damage_type: int) -> bool:
        """False when the type is immune to that damage bit."""
        if self.weapon_proof:
            return False
        return not (self.invincible_to & int(damage_type))

    def hits_to_kill(self, damage_points: int) -> int | None:
        """ceil(HP / damage), or None when HP is not in the ROM array.

        Ignores immunity on purpose -- ask `weak_to` first.
        """
        if self.hp is None or int(damage_points) <= 0:
            return None
        if self.hp == 0:
            return 1
        return -(-self.hp // int(damage_points))

    def contact_in_state(self, state: int) -> Contact:
        """Contact class this frame; NONE while a burrower is under the sand."""
        if int(state) in self.no_contact_states:
            return Contact.NONE
        return self.contact


# --- ROM byte arrays, transcribed ---------------------------------------
# ObjectTypeToAttributes (Z_07.asm#L5242), ObjectTypeToHpPairs (#L5256),
# ObjTypeToDamagePoints (Z_01.asm#L5574). One row per byte, in ROM order.
_ATTRIBUTES = (
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
    0x05, 0x05, 0x05, 0x81, 0x81, 0x81, 0x81, 0x01,
    0x01, 0x81, 0x01, 0x01, 0x43, 0x43, 0x81, 0x81,
    0x81, 0x81, 0x01, 0x81, 0x81, 0x81, 0x01, 0x81,
    0x81, 0x81, 0x81, 0x81, 0x81, 0xC3, 0xC3, 0x89,
    0x89, 0x81, 0x81, 0x89, 0x89, 0x89, 0x89, 0x83,
    0x81, 0x89, 0x89, 0xC9, 0xC9, 0x81, 0x81, 0x81,
    0xA9, 0xA9, 0x41, 0x41, 0x89, 0x89, 0x81, 0x81,
    0x81, 0xC1, 0xC1, 0xC1, 0xC1, 0xC1, 0x81, 0x81,
    0x81, 0xA1, 0xA1, 0x81, 0x81, 0x81, 0x81, 0x81,
    0x81, 0x81, 0x81, 0xE3, 0xE3, 0xE3, 0xE3, 0xE3,
    0xE1, 0xE1, 0xE1, 0xE1, 0xE1, 0x81, 0x81,
)

_HP_PAIRS = (
    0x06, 0x43, 0x25, 0x31, 0x12, 0x24, 0x81, 0x14,
    0x22, 0x42, 0x00, 0xA9, 0x8F, 0x20, 0x00, 0x3F,
    0xF9, 0xFA, 0x46, 0x62, 0x11, 0x2F, 0xFF, 0xFF,
    0x7F, 0xF6, 0x2F, 0xFF, 0xFF, 0x22, 0x46, 0xF1,
    0xF2, 0xAA, 0xAA, 0xFB, 0xBF, 0xF0,
)

_DAMAGE_POINTS = (
    0x60, 0x02, 0x01, 0x80, 0x80, 0x01, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x01, 0x02, 0x80, 0x80, 0x01,
    0x80, 0x80, 0x01, 0x01, 0x80, 0x80, 0x02, 0x01,
    0x02, 0x00, 0x80, 0x80, 0x80, 0x80, 0x01, 0x80,
    0x80, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x80,
    0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x01, 0x01, 0x02, 0x02, 0x00, 0x00, 0x00,
    0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x04, 0x80,
    0x80, 0x80, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02,
    0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x80, 0x80, 0x80, 0x01, 0x02,
    0x02, 0x04, 0x04, 0x80, 0x01,
)

# --- Per-type speed overlay ------------------------------------------------
# Only types with an `STA ObjQSpeedFrac` of their own appear here. Everything
# else runs at DEFAULT_QSPEED and carries `speed=None` rather than a guess.
# Flyers ($1A, $1B-$1D, $21, $22, $46) do not use ObjQSpeedFrac at all --
# `MoveFlyer` (Z_04.asm#L11560) is a different unit, so they are left blank
# here and documented in `scratch/enemy_constants_rom.md` section 2.
_SPEEDS: dict[int, Speed] = {
    0x07: Speed(0x20, source("Z_04.asm", 2969), note="slow Octorok; 0 while shooting"),
    0x08: Speed(0x40, source("Z_04.asm", 2969), note="fast Octorok: $20 doubled"),
    0x09: Speed(0x20, source("Z_04.asm", 2969), note="slow blue Octorok"),
    0x0A: Speed(0x40, source("Z_04.asm", 2969), note="fast blue Octorok"),
    0x01: Speed(0x20, source("Z_04.asm", 1958), note="restored after shooting"),
    0x02: Speed(0x20, source("Z_04.asm", 1958), note="restored after shooting"),
    0x03: Speed(0x20, source("Z_04.asm", 1950), note="restored after shooting"),
    0x04: Speed(0x20, source("Z_04.asm", 1950), note="restored after shooting"),
    0x0B: Speed(0x20, source("Z_04.asm", 6449), note="red Darknut"),
    0x0C: Speed(0x28, source("Z_04.asm", 6449), note="blue Darknut"),
    0x0F: Speed(
        (0x08, 0x0A, 0x10, 0x20, 0x10, 0x0A),
        source("Z_04.asm", 2592),
        per_state=True,
        note="BlueLeeverStateQSpeeds, ObjState 0-5",
    ),
    0x10: Speed(
        (0x00, 0x00, 0x00, 0x20, 0x00, 0x00),
        source("Z_04.asm", 2735),
        per_state=True,
        note="RedLeeverStateQSpeeds: still in every state but 3",
    ),
    0x13: Speed(0x40, source("Z_04.asm", 1470), note="Zol, state 2"),
    0x15: Speed(0x40, source("Z_04.asm", 1470), note="Gel, state 2; $20 in state 0"),
    0x27: Speed(0x18, source("Z_04.asm", 4242), note="Wallmaster crawl"),
    0x28: Speed(0x20, source("Z_04.asm", 4586), note="slow on turn; $60 on the rush"),
    0x2A: Speed(0x20, source("Z_04.asm", 4670), note="cannot shoot in quest 1"),
    0x40: Speed(0x40, source("Z_04.asm", 1095), note="Bubble"),
    0x53: Speed(0x20, source("Z_04.asm", 2969), note="rock inherits the shooter's"),
    0x55: Speed(0x70, source("Z_04.asm", 981), note="FireballQSpeedsX major axis"),
    0x56: Speed(0x70, source("Z_04.asm", 981), note="FireballQSpeedsX major axis"),
    0x5B: Speed(0x80, source("Z_04.asm", 2098), note="Moblin arrow"),
    0x5C: Speed(0xA0, source("Z_04.asm", 602), note="Goriya boomerang on throw"),
}

# --- Per-type ObjInvincibilityMask ----------------------------------------
# Written at init or at the collision site; mask bit set = immune to that
# damage type. Types absent here take every weapon (mask 0).
_INVINCIBLE_TO: dict[int, tuple[int, str]] = {
    0x0B: (0xF6, source("Z_04.asm", 6449)),
    0x0C: (0xF6, source("Z_04.asm", 6449)),
    0x16: (0xFE, source("Z_04.asm", 6656)),
    0x23: (0xF6, source("Z_04.asm", 7590)),
    0x24: (0xF6, source("Z_04.asm", 7590)),
    0x32: (0xFF, source("Z_04.asm", 6055)),
    0x33: (0xFB, source("Z_04.asm", 7804)),
    0x34: (0xFB, source("Z_04.asm", 7804)),
    0x3C: (0xE2, source("Z_04.asm", 7738)),
    0x3D: (0xE2, source("Z_04.asm", 4842)),
    0x43: (0xFE, source("Z_04.asm", 7641)),
    0x44: (0xFE, source("Z_04.asm", 7641)),
    0x45: (0xFE, source("Z_04.asm", 7641)),
    0x47: (0xFE, source("Z_04.asm", 9526)),
    0x48: (0xFE, source("Z_04.asm", 9526)),
}

_NOTES: dict[int, str] = {
    0x10: "RedLeeverStateQSpeeds is 0 in every state but 3: combat.dormant_body.",
    0x0F: "Blue leever creeps at $08 even in state 0; still uncuttable there.",
    0x11: "Never a kill: UpdateZora submerges with DestroyMonster, no streak.",
    0x12: "Splits into two type $1C on ObjState >= 2 (Z_04.asm#L6915).",
    0x16: "Any arrow sets HP to 0 regardless of the $FE mask (Z_01.asm#L6295).",
    0x1A: "Cuttable only in Flyer_ObjFlyingState 5 (Z_04.asm#L4022).",
    0x1B: "ROM HP is 0 while alive -- KIND_POLICY alive_rule must stay TYPE.",
    0x1C: "ROM HP is 0 while alive -- KIND_POLICY alive_rule must stay TYPE.",
    0x1D: "ROM HP 0 and NoDropMonsterTypes groups it with $1B/$1C as Keese, "
    "but behaviors._TYPE_TO_KIND has no row for it.",
    0x1E: "Type $1E is the awake Armos; the statue is tile $66/$67, not a slot.",
    0x33: "Arrow damage lands only on eye part 3/4, eye state 3, arrow facing "
    "UP (Z_04.asm#L8466). HP 96 = 3 wooden arrows.",
    0x34: "Same eye gate as $33. HP 32 = 1 wooden arrow.",
    0x38: "Attribute $20: no weapon touches it until the recorder shrinks it "
    "to $18.",
    0x53: "Costs exactly half a heart ($80) -- the $FF -> $7F partial.",
}


def _hp_for(type_id: int) -> int | None:
    """ExtractHitPointValue: two types to a byte (Z_04.asm#L11002)."""
    index = type_id >> 1
    if index >= len(_HP_PAIRS):
        return None
    pair = _HP_PAIRS[index]
    return (pair & 0xF0) if type_id % 2 == 0 else ((pair & 0x0F) << 4)


def _contact_for(type_id: int, damage_points: int | None) -> Contact:
    if type_id in _CAPTURE_TYPES:
        return Contact.CAPTURE
    if damage_points is None:
        return Contact.HARM
    return Contact.HARM if damage_points else Contact.HARMLESS


def _build() -> dict[int, Species]:
    table: dict[int, Species] = {}
    for type_id in range(1, len(_ATTRIBUTES)):
        damage = (
            _DAMAGE_POINTS[type_id] if type_id < len(_DAMAGE_POINTS) else None
        )
        mask = _INVINCIBLE_TO.get(type_id, (0, ""))[0]
        table[type_id] = Species(
            type_id=type_id,
            name=object_name(type_id),
            attr=_ATTRIBUTES[type_id],
            hp=_hp_for(type_id),
            damage_points=damage,
            contact=_contact_for(type_id, damage),
            speed=_SPEEDS.get(type_id),
            invincible_to=mask,
            no_contact_states=(
                (BURROWER_DORMANT_STATE,) if type_id in _BURROWER_TYPES else ()
            ),
            notes=_NOTES.get(type_id, ""),
        )
    return table


_TABLE: dict[int, Species] = _build()

# An id the ROM arrays do not reach. Conservative on every axis: it can hurt
# you, it has no known HP, and its box is today's `threat.BODY_HALF` pad.
_UNKNOWN = Species(
    type_id=0,
    name="unknown",
    attr=0,
    hp=None,
    damage_points=None,
    contact=Contact.HARM,
    speed=None,
    invincible_to=0,
    notes="Not in ObjectTypeToAttributes; treated as a full-width harmful body.",
)


def species_of(type_id: int) -> Species:
    """Known row, or a conservative unknown. Never raises, never returns None."""
    value = int(type_id) & 0xFF
    found = _TABLE.get(value)
    if found is not None:
        return found
    return Species(
        type_id=value,
        name=object_name(value),
        attr=_UNKNOWN.attr,
        hp=_UNKNOWN.hp,
        damage_points=_UNKNOWN.damage_points,
        contact=_UNKNOWN.contact,
        speed=_UNKNOWN.speed,
        invincible_to=_UNKNOWN.invincible_to,
        no_contact_states=_UNKNOWN.no_contact_states,
        notes=_UNKNOWN.notes,
    )


__all__ = [
    "ARROW_DAMAGE",
    "ATTR_HALF_WIDTH_COLLISION",
    "ATTR_INVINCIBLE",
    "BOMB_DAMAGE",
    "BOOMERANG_DAMAGE",
    "BURROWER_DORMANT_STATE",
    "BURROWER_TYPES",
    "CONTACT_THRESHOLD",
    "DEFAULT_BODY_HALF",
    "DEFAULT_QSPEED",
    "DEFAULT_SHOT_HALF",
    "DMG_ARROW",
    "DMG_BOMB",
    "DMG_BOOMERANG",
    "DMG_FIRE",
    "DMG_MAGIC",
    "DMG_SWORD",
    "FIRE_DAMAGE",
    "HALF_WIDTH_MID_OFFSET",
    "LINK_QSPEED",
    "LINK_STAIRS_QSPEED",
    "MID_OFFSET",
    "QSPEED_PER_PIXEL",
    "ROD_DAMAGE",
    "SWORD_DAMAGE",
    "Contact",
    "Species",
    "Speed",
    "source",
    "species_of",
]

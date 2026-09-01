"""Species facts for room enemies.

RAM ``enemy_id`` (header pointer, e.g. ``0xE9FF``) is the key — not
sm-json-data numeric ids. Unknown ids are representable: lookup returns
a row whose default Stance is Ignore. First pass fills three rows;
later hops add rows when they actually need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from super_metroid.combat.enemies.scan import Enemy

ATOMIC_ID = 0xE9FF
WORKROBOT_ID = 0xE8FF
COVERN_ID = 0xEA3F
CERES_STEAM_ID = 0xE1FF
CERES_DOOR_ID = 0xE23F
# PJBoy $0F88 bit 2: steam hide instruction sets this (no Samus reaction).
STEAM_HIDDEN_BIT = 0x04


class Contact(Enum):
    """What overlap does. Frozen-solid is Contact, not a Stance."""

    KNOCKBACK = auto()
    SOLID = auto()
    PLATFORM = auto()
    NONE = auto()


class Stance(Enum):
    """This-frame overlay choice. Hops set Intent; Species supplies the default."""

    ENGAGE = auto()
    AVOID = auto()
    ABSORB = auto()
    IGNORE = auto()


@dataclass(frozen=True)
class Species:
    """Static facts for one RAM id. Hitboxes stay out until a hop probes them."""

    enemy_id: int
    name: str
    max_hp: int
    live_contact: Contact
    frozen_contact: Contact
    default_stance: Stance
    freezable: bool
    solid_gap: int = 24

    def is_solid(self, freeze_timer: int = 0) -> bool:
        """True when overlap would stall (live solid or frozen solid)."""
        if self.live_contact is Contact.SOLID:
            return True
        return self.frozen_contact is Contact.SOLID and int(freeze_timer) > 0


_UNKNOWN = Species(
    enemy_id=0,
    name="unknown",
    max_hp=0,
    live_contact=Contact.NONE,
    frozen_contact=Contact.NONE,
    default_stance=Stance.IGNORE,
    freezable=False,
)

_TABLE: dict[int, Species] = {
    ATOMIC_ID: Species(
        ATOMIC_ID,
        "Atomic",
        max_hp=250,
        live_contact=Contact.KNOCKBACK,
        frozen_contact=Contact.SOLID,
        default_stance=Stance.ENGAGE,
        freezable=True,
        solid_gap=24,
    ),
    WORKROBOT_ID: Species(
        WORKROBOT_ID,
        "Workrobot",
        max_hp=800,
        live_contact=Contact.SOLID,
        frozen_contact=Contact.SOLID,
        default_stance=Stance.AVOID,
        freezable=False,
        solid_gap=48,
    ),
    COVERN_ID: Species(
        COVERN_ID,
        "Covern",
        max_hp=300,
        live_contact=Contact.KNOCKBACK,
        frozen_contact=Contact.SOLID,
        default_stance=Stance.ABSORB,
        freezable=True,
        solid_gap=24,
    ),
    CERES_STEAM_ID: Species(
        CERES_STEAM_ID,
        "Ceres steam",
        max_hp=32767,
        live_contact=Contact.KNOCKBACK,
        frozen_contact=Contact.NONE,
        default_stance=Stance.ABSORB,
        freezable=False,
        solid_gap=16,
    ),
    CERES_DOOR_ID: Species(
        CERES_DOOR_ID,
        "Ceres door",
        max_hp=40,
        live_contact=Contact.SOLID,
        frozen_contact=Contact.SOLID,
        default_stance=Stance.AVOID,
        freezable=False,
        solid_gap=32,
    ),
}


def species_of(enemy_id: int) -> Species:
    """Known row, or Ignore. Never raises."""
    found = _TABLE.get(int(enemy_id))
    if found is None:
        return Species(
            enemy_id=int(enemy_id),
            name=_UNKNOWN.name,
            max_hp=_UNKNOWN.max_hp,
            live_contact=_UNKNOWN.live_contact,
            frozen_contact=_UNKNOWN.frozen_contact,
            default_stance=_UNKNOWN.default_stance,
            freezable=_UNKNOWN.freezable,
        )
    return found


def is_solid(enemy: Enemy) -> bool:
    """True when overlap would stall (live solid or frozen solid)."""
    return species_of(int(enemy.enemy_id)).is_solid(int(enemy.freeze_timer))


def steam_is_burning(enemy: Enemy) -> bool:
    """True when Ceres steam ($E1FF) is shown and can knockback.

    Hide instruction ``$A6:F11D`` sets ``$0F88`` bit 2 (intangible). Show
    ``$A6:F135`` clears it. x/y stay put; only the bit/spritemap cycle.
    """
    if int(enemy.enemy_id) != CERES_STEAM_ID:
        return False
    return (int(enemy.extra_props) & STEAM_HIDDEN_BIT) == 0


def enemy_overlaps(
    enemy: Enemy,
    x: int,
    y: int,
    *,
    samus_r: int = 8,
) -> bool:
    """Axis-aligned overlap using the slot radii (steam 8×8, door 8×32)."""
    xr = int(enemy.x_radius) if int(enemy.x_radius) else 8
    yr = int(enemy.y_radius) if int(enemy.y_radius) else 8
    return abs(int(enemy.x) - int(x)) <= xr + samus_r and abs(
        int(enemy.y) - int(y)
    ) <= yr + samus_r


__all__ = [
    "ATOMIC_ID",
    "COVERN_ID",
    "CERES_DOOR_ID",
    "CERES_STEAM_ID",
    "STEAM_HIDDEN_BIT",
    "WORKROBOT_ID",
    "Contact",
    "Species",
    "Stance",
    "enemy_overlaps",
    "is_solid",
    "species_of",
    "steam_is_burning",
]

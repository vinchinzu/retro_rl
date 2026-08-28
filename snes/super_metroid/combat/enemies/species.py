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


__all__ = [
    "ATOMIC_ID",
    "COVERN_ID",
    "WORKROBOT_ID",
    "Contact",
    "Species",
    "Stance",
    "is_solid",
    "species_of",
]

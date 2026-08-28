"""Level 6 play 0x09 south door after Rod cellar return.

Leftover (192,141) NE spawn; left 0x68 gone; remaining right block;
south mouth open on the leftover PNG. Occupancy to (120,189) then DOWN.
Dest is RAM after the transition — do not invent Gohma. Do not poke
bow/arrows/doors/keys. Halt north of leftover (stairs 0x71).
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    NORTH_HALT_Y,
    SOUTH09_MAX_FRAMES,
    SOUTH09_SPEC,
    SOUTH_BAND_Y,
    SOUTH_DOOR_TOL,
    SOUTH_DOOR_X,
    SOUTH_DOOR_Y,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "NORTH_HALT_Y",
    "SOUTH09_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SOUTH_DOOR_TOL",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South09Controller",
    "level6_south09_stages",
    "level6_south09_success",
    "make_south09_controller",
]

Level6South09Controller = Level6DoorHopController


def make_south09_controller() -> Level6DoorHopController:
    """Occupancy south of cleared 0x09. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(SOUTH09_SPEC)


def level6_south09_stages():
    """Play 0x09 leftover (192,141) → occupancy south door. Dest is RAM."""
    return door_hop_stages(SOUTH09_SPEC)


def level6_south09_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x09 with ADDR_ROD. Dest is RAM."""
    return door_hop_success(SOUTH09_SPEC, snap)

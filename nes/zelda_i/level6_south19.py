"""Level 6 play 0x19 south door after Rod return.

Leftover (120,77) north mouth. PNG south and east mouths open.
Walkthrough: south until Vires. West is already-cleared Gleeok 0x18.
Occupancy to (120,189) then DOWN. Never UP (back to 0x09). Dest is RAM.
Do not poke bow/arrows/doors/keys. Do not invent Gohma.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    SOUTH19_MAX_FRAMES,
    SOUTH19_SPEC,
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
    "SOUTH19_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SOUTH_DOOR_TOL",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South19Controller",
    "level6_south19_stages",
    "level6_south19_success",
    "make_south19_controller",
]

Level6South19Controller = Level6DoorHopController


def make_south19_controller() -> Level6DoorHopController:
    """Occupancy south of cleared 0x19. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(SOUTH19_SPEC)


def level6_south19_stages():
    """Play 0x19 leftover (120,77) → occupancy south door. Dest is RAM."""
    return door_hop_stages(SOUTH19_SPEC)


def level6_south19_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x19 with ADDR_ROD. Dest is RAM."""
    return door_hop_success(SOUTH19_SPEC, snap)

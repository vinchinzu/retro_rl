"""Level 6 play 0x19 west after inland29 enter-stop.

Leftover (120,205) south mouth. Forward 0x18→0x19 was occupancy y=141
RIGHT (PNG-black shutter, no key). Return LEFT at y=141. OccupancyWalker
first; halt at first miss (no path → stand). Do not KEY-UP 0x09. Skip Map.
Do not poke bow/arrows/doors/keys. Isolated BFS banned.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    WEST19_MAX_FRAMES,
    WEST19_SPEC,
    WEST_DOOR_X,
    WEST_DOOR_Y,
    WEST_SPAWN_XMIN,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "WEST19_MAX_FRAMES",
    "WEST_DOOR_X",
    "WEST_DOOR_Y",
    "WEST_SPAWN_XMIN",
    "Level6West19Controller",
    "level6_west19_stages",
    "level6_west19_success",
    "make_west19_controller",
]

Level6West19Controller = Level6DoorHopController


def make_west19_controller() -> Level6DoorHopController:
    """Occupancy west of 0x19. Do not poke bow/arrows/doors/keys. Skip Map."""
    return Level6DoorHopController(WEST19_SPEC)


def level6_west19_stages():
    """Play 0x19 leftover (120,205) → occupancy LEFT y=141 → play 0x18."""
    return door_hop_stages(WEST19_SPEC)


def level6_west19_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x18 with ADDR_ROD. Enter-stop; Gleeok already dead."""
    return door_hop_success(WEST19_SPEC, snap)

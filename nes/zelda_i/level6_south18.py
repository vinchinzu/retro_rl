"""Level 6 play 0x18 south after west19 enter-stop.

Leftover (208,141) east mouth. Gleeok gone; decorative north hole is not
an exit (stairs18 v1–v5 red; not mode 9). OccupancyWalker to (120,189)
then DOWN. South return clip LIVE-TBD — occupancy first, halt at first
miss (no path → stand). Do not KEY-UP 0x09. Do not CheckWarp the hole.
Do not poke bow/arrows/doors/keys. Isolated BFS banned.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    SOUTH18_MAX_FRAMES,
    SOUTH18_SPEC,
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
    "SOUTH18_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SOUTH_DOOR_TOL",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South18Controller",
    "level6_south18_stages",
    "level6_south18_success",
    "make_south18_controller",
]

Level6South18Controller = Level6DoorHopController


def make_south18_controller() -> Level6DoorHopController:
    """Occupancy south of 0x18. Do not poke bow/arrows/doors/keys. No CheckWarp."""
    return Level6DoorHopController(SOUTH18_SPEC)


def level6_south18_stages():
    """Play 0x18 leftover (208,141) → occupancy DOWN (120,189) → play 0x28."""
    return door_hop_stages(SOUTH18_SPEC)


def level6_south18_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x28 with ADDR_ROD. Enter-stop; enemies may be gone."""
    return door_hop_success(SOUTH18_SPEC, snap)

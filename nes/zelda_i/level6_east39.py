"""Level 6 play 0x39 east door after Vire clear.

Leftover (136,173) tile 118 boxed 4-cardinal (east39 v1). PNG east mouth
open; cur_opened_doors 9 = N+E. RIGHT+UP clip then occupancy y=141
RIGHT. Dest is RAM. Do not poke bow/arrows/doors/keys. Do not invent
Gohma.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    CLIP_Y,
    EAST39_MAX_FRAMES,
    EAST39_SPEC,
    EAST_DOOR_TOL,
    EAST_DOOR_X,
    EAST_DOOR_Y,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "CLIP_Y",
    "EAST39_MAX_FRAMES",
    "EAST_DOOR_TOL",
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "Level6East39Controller",
    "level6_east39_stages",
    "level6_east39_success",
    "make_east39_controller",
]

Level6East39Controller = Level6DoorHopController


def make_east39_controller() -> Level6DoorHopController:
    """Occupancy east of cleared 0x39. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(EAST39_SPEC)


def level6_east39_stages():
    """Play 0x39 leftover (136,173) → occupancy east door. Dest is RAM."""
    return door_hop_stages(EAST39_SPEC)


def level6_east39_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x39 with ADDR_ROD. Dest is RAM."""
    return door_hop_success(EAST39_SPEC, snap)

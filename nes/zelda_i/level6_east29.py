"""Level 6 play 0x29 east door after wizzrobe clear.

Leftover (55,133). PNG east mouth open. Y-align 141 then occupancy
RIGHT. Do not RIGHT at leftover y=133 into a shutter face. Dest is RAM.
Do not poke bow/arrows/doors/keys. Do not invent Gohma. Do not fight
Gohma (no arrows).
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    CLIP_Y,
    EAST29_MAX_FRAMES,
    EAST29_SPEC,
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
    "EAST29_MAX_FRAMES",
    "EAST_DOOR_TOL",
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "Level6East29Controller",
    "level6_east29_stages",
    "level6_east29_success",
    "make_east29_controller",
]

Level6East29Controller = Level6DoorHopController


def make_east29_controller() -> Level6DoorHopController:
    """Occupancy east of cleared 0x29. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(EAST29_SPEC)


def level6_east29_stages():
    """Play 0x29 leftover (55,133) → occupancy east door. Dest is RAM."""
    return door_hop_stages(EAST29_SPEC)


def level6_east29_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x29 with ADDR_ROD. Dest is RAM."""
    return door_hop_success(EAST29_SPEC, snap)

"""Level 6 play 0x29 south door after wizzrobe clear.

Leftover (55,133) tile 244 boxed 4-cardinal (east29 v1). RIGHT+DOWN
clip reaches y=141 (east29 v2). Then occupancy to (120,189) DOWN.
East is sealed (mask 12 = U+D). Dest is RAM. Do not poke
bow/arrows/doors/keys. Do not invent Gohma. Do not fight Gohma.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    CLIP_Y,
    SOUTH29_MAX_FRAMES,
    SOUTH29_SPEC,
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
    "CLIP_Y",
    "SOUTH29_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SOUTH_DOOR_TOL",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South29Controller",
    "level6_south29_stages",
    "level6_south29_success",
    "make_south29_controller",
]

Level6South29Controller = Level6DoorHopController


def make_south29_controller() -> Level6DoorHopController:
    """Occupancy south of cleared 0x29. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(SOUTH29_SPEC)


def level6_south29_stages():
    """Play 0x29 leftover (55,133) → clip then occupancy south. Dest is RAM."""
    return door_hop_stages(SOUTH29_SPEC)


def level6_south29_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x29 with ADDR_ROD. Dest is RAM."""
    return door_hop_success(SOUTH29_SPEC, snap)

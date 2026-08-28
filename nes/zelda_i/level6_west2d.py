"""Level 6 play 0x2D west door after south1d leftover.

Leftover (120,77) north mouth. ROM N/W=open, S/E=wall. Occupancy y=141
then LEFT to (32,141). Dest is exact play 0x2C. Keys stay 4 (west open).
Fail 0x1D backtrack and Gohma 0x1C. Do not batch KEY-UP / Gohma.
Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    WEST2D_MAX_FRAMES,
    WEST2D_SPEC,
    WEST_DOOR_X,
    WEST_DOOR_Y,
    WEST_SPAWN_XMIN,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.level6_south1d import level6_south1d_stages
from zelda_i.ram import ZeldaSnapshot
from zelda_i.screen_glance import WEST2D_LEAVE, GlanceLeftover, grade_controller

__all__ = [
    "WEST2D_MAX_FRAMES",
    "WEST_DOOR_X",
    "WEST_DOOR_Y",
    "WEST_SPAWN_XMIN",
    "Level6West2DController",
    "level6_west2d_glance",
    "level6_west2d_stages",
    "level6_west2d_success",
    "make_west2d_controller",
]

Level6West2DController = Level6DoorHopController


def make_west2d_controller() -> Level6DoorHopController:
    """Occupancy west of 0x2D. Do not poke bow/arrows/doors/keys."""
    return Level6DoorHopController(WEST2D_SPEC)


def level6_west2d_stages():
    """South 0x1D leftover (120,77) → occupancy west door → play 0x2C."""
    return (
        *level6_south1d_stages(),
        *door_hop_stages(WEST2D_SPEC),
    )


def level6_west2d_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x2C with ADDR_ROD. Keys stay 4. Do not accept 0x1C/0x1D."""
    return door_hop_success(WEST2D_SPEC, snap)


def level6_west2d_glance(controller) -> GlanceLeftover:
    """Live 0x2C leftover (224,141) after the 0x2D west door."""
    return grade_controller(controller, WEST2D_LEAVE)

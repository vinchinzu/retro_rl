"""Level 6 play 0x1D south door after cellar 0x08 B-side.

Leftover (96,157); ROM N/W/E=wall, S=open. Occupancy to (120,189) then DOWN.
Dest is exact play 0x2D. Do not batch 0x2D LEFT / 0x2C KEY-UP / Gohma.
Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from zelda_i.level6_cellar08 import level6_cellar08_stages
from zelda_i.level6_door_hop import (
    SOUTH1D_MAX_FRAMES,
    SOUTH1D_SPEC,
    SOUTH_BAND_Y,
    SOUTH_DOOR_TOL,
    SOUTH_DOOR_X,
    SOUTH_DOOR_Y,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.ram import ZeldaSnapshot
from zelda_i.screen_glance import SOUTH1D_LEAVE, GlanceLeftover, grade_controller

__all__ = [
    "SOUTH1D_MAX_FRAMES",
    "SOUTH_BAND_Y",
    "SOUTH_DOOR_TOL",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South1DController",
    "level6_south1d_glance",
    "level6_south1d_stages",
    "level6_south1d_success",
    "make_south1d_controller",
]

Level6South1DController = Level6DoorHopController


def make_south1d_controller() -> Level6DoorHopController:
    """Occupancy south of cellar B-side 0x1D. Do not poke bow/arrows/doors."""
    return Level6DoorHopController(SOUTH1D_SPEC)


def level6_south1d_stages():
    """Cellar 0x08 B-side leftover (96,157) → occupancy south door → play 0x2D."""
    return (
        *level6_cellar08_stages(),
        *door_hop_stages(SOUTH1D_SPEC),
    )


def level6_south1d_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x2D with ADDR_ROD. Keys stay 4. Do not accept 0x1C/0x2C."""
    return door_hop_success(SOUTH1D_SPEC, snap)


def level6_south1d_glance(controller) -> GlanceLeftover:
    """Live 0x2D leftover (120,77) after the 0x1D south door."""
    return grade_controller(controller, SOUTH1D_LEAVE)

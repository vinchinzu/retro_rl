"""Level 6 play 0x2C KEY-UP after west2d leftover.

Leftover (224,141) east mouth. ROM N=key, S=open, W=wall, E=open.
Occupancy x-align then KEY-UP (120,93). Dest is exact play 0x1C.
Keys 4→3. Fail 0x2D backtrack and south 0x3C. Enter-stop only —
do not fight Gohma. Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from zelda_i.level6_door_hop import (
    EAST_SPAWN_XMAX,
    NORTH2C_MAX_FRAMES,
    NORTH2C_SPEC,
    NORTH_DOOR_X,
    NORTH_DOOR_Y,
    Level6DoorHopController,
    door_hop_stages,
    door_hop_success,
)
from zelda_i.level6_west2d import level6_west2d_stages
from zelda_i.ram import ZeldaSnapshot
from zelda_i.screen_glance import NORTH2C_LEAVE, GlanceLeftover, grade_controller

__all__ = [
    "EAST_SPAWN_XMAX",
    "NORTH2C_MAX_FRAMES",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "Level6North2CController",
    "level6_north2c_glance",
    "level6_north2c_stages",
    "level6_north2c_success",
    "make_north2c_controller",
]

Level6North2CController = Level6DoorHopController


def make_north2c_controller() -> Level6DoorHopController:
    """Occupancy KEY-UP of 0x2C. Do not poke bow/arrows/doors/keys."""
    return Level6DoorHopController(NORTH2C_SPEC)


def level6_north2c_stages():
    """West 0x2D leftover (224,141) → occupancy KEY-UP → play 0x1C."""
    return (
        *level6_west2d_stages(),
        *door_hop_stages(NORTH2C_SPEC),
    )


def level6_north2c_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x1C with ADDR_ROD. Do not accept 0x2C/0x2D/0x3C."""
    return door_hop_success(NORTH2C_SPEC, snap)


def level6_north2c_glance(controller) -> GlanceLeftover:
    """Live 0x1C leftover after the 0x2C KEY-UP. Keys 4→3."""
    return grade_controller(controller, NORTH2C_LEAVE)

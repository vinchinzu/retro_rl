"""K4 Ice pure stack — outbound PLM + return prefix for K5.

Package from day 1 (do not extend Wave megafiles). Tape recon:
``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B / ``rr-dbu.11`` outbound;
return first hop under ``rr-dbu.8`` K5 Alpha PB pure stack.

Outbound path (entry order)::

    Business 0xA7DE
      → Ice Gate 0xA815          (rr-fg3)
        → Acid Room 0xA75D       (not Tutorial-first)
          → Ice Snake 0xA8B9     (prefer 2WJ)
            → Ice 0xA890 PLM

Return path (tape Phase B return; K5 predecessor)::

    Ice 0xA890
      → Ice Snake 0xA8B9         (ice_to_snake — this package)
        → Tutorial 0xA865        (snake_to_tutorial — this package)
          → Ice Gate 0xA815      (tutorial_to_gate — this package)
            → Business 0xA7DE
"""

from __future__ import annotations

from super_metroid.routes.kpdr.ice.acid_to_snake import play_ice_acid_to_snake
from super_metroid.routes.kpdr.ice.business_to_gate import play_business_to_ice_gate
from super_metroid.routes.kpdr.ice.gate_to_acid import play_ice_gate_to_acid
from super_metroid.routes.kpdr.ice.geometry import (
    ICE_BEAM_MASK,
    ICE_SUPER_DOOR_X,
    ICE_SUPER_LIP_X_MAX,
    ICE_SUPER_Y_MAX,
    ICE_SUPER_Y_MIN,
    on_ice_super_lip,
)
from super_metroid.routes.kpdr.ice.ice_to_snake import play_ice_to_snake
from super_metroid.routes.kpdr.ice.snake_to_ice import play_ice_snake_to_ice
from super_metroid.routes.kpdr.ice.snake_to_tutorial import play_ice_snake_to_tutorial
from super_metroid.routes.kpdr.ice.tutorial_to_gate import play_ice_tutorial_to_gate
from super_metroid.routes.kpdr.rooms import (
    ROOM_ICE,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
)

__all__ = [
    "ICE_BEAM_MASK",
    "ICE_SUPER_DOOR_X",
    "ICE_SUPER_LIP_X_MAX",
    "ICE_SUPER_Y_MAX",
    "ICE_SUPER_Y_MIN",
    "ROOM_ICE",
    "ROOM_ICE_ACID",
    "ROOM_ICE_GATE",
    "ROOM_ICE_SNAKE",
    "ROOM_ICE_TUTORIAL",
    "on_ice_super_lip",
    "play_business_to_ice_gate",
    "play_ice_acid_to_snake",
    "play_ice_gate_to_acid",
    "play_ice_snake_to_ice",
    "play_ice_snake_to_tutorial",
    "play_ice_tutorial_to_gate",
    "play_ice_to_snake",
]

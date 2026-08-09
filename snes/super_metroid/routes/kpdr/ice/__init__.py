"""K4 Ice pure stack — Business Super → Gate → Acid → Snake (2WJ) → Ice PLM.

Package from day 1 (do not extend Wave megafiles). Tape recon:
``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B / ``rr-dbu.11``.

Outbound path (entry order)::

    Business 0xA7DE
      → Ice Gate 0xA815          (this hop — rr-fg3)
        → Acid Room 0xA75D       (not Tutorial-first)
          → Ice Snake 0xA8B9     (prefer 2WJ)
            → Ice 0xA890 PLM
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
from super_metroid.routes.kpdr.ice.snake_to_ice import play_ice_snake_to_ice
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
]

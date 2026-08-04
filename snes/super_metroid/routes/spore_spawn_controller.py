"""Back-compat re-export — implementation lives in :mod:`super_metroid.routes.kpdr.spore_spawn`.

Historical import path ``super_metroid.routes.spore_spawn_controller`` remains
valid. Continuous report hashing uses ``kpdr/spore_spawn.py`` via
``SPORE_CONTROLLER_PATH``.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.spore_spawn import (  # noqa: F401
    SporeSpawnEvidence,
    play_main_shaft_to_spore_spawn,
    play_parlor_to_main_shaft,
    play_post_torizo_to_spore_spawn,
)

# Private symbols imported by unit tests (wall-jump chimney constants).
from super_metroid.routes.kpdr.spore_spawn import (  # noqa: F401
    _PARLOR_CHIMNEY_GAP,
    _PARLOR_CHIMNEY_WJ,
)

__all__ = [
    "SporeSpawnEvidence",
    "play_main_shaft_to_spore_spawn",
    "play_parlor_to_main_shaft",
    "play_post_torizo_to_spore_spawn",
]

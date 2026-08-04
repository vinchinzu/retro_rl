"""Deprecated path — SM platformer levels live under the game package.

Import ``super_metroid.platformer_levels`` (or this shim) to register levels.
"""

from super_metroid.platformer_levels import *  # noqa: F401,F403
from super_metroid.platformer_levels import (  # noqa: F401
    SM_ACTIONS,
    SM_RAM,
    _ROOMS,
    _sm_config,
    _sm_make,
)

"""Deprecated path — SMW platformer levels live under the game package.

Import ``SMW.platformer_levels`` (or this shim) to register levels.
"""

from SMW.platformer_levels import *  # noqa: F401,F403
from SMW.platformer_levels import (  # noqa: F401
    SMW_RAM,
    SMW_SPEED_ACTIONS,
    _LEVEL_GAME_MODE,
    _OVERWORLD_MODES,
    _smw_level,
)

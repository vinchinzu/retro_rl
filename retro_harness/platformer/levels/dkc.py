"""Deprecated path — DKC platformer levels live under the game package.

Import ``donkey_kong_country.platformer_levels`` (or this shim) to register levels.
"""

from donkey_kong_country.platformer_levels import *  # noqa: F401,F403
from donkey_kong_country.platformer_levels import (  # noqa: F401
    DKC_RAM,
    DKC_SPEED_ACTIONS,
    OLD_TO_SPEED,
    convert_old_to_speed,
)

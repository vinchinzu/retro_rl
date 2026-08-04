"""Deprecated path — SMB3 platformer levels live under the game package.

Import ``smb3.platformer_levels`` (or this shim) to register levels.
"""

from smb3.platformer_levels import *  # noqa: F401,F403
from smb3.platformer_levels import (  # noqa: F401
    SMB3_ACTIONS,
    SMB3_COMPUTED,
    SMB3_RAM,
)

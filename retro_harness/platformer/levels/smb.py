"""Deprecated path — SMB platformer levels live under the game package.

Import ``smb.platformer_levels`` (or this shim) to register levels.
"""

from smb.platformer_levels import *  # noqa: F401,F403
from smb.platformer_levels import (  # noqa: F401
    SMB_84_COMPUTED,
    SMB_ACTIONS,
    SMB_COMPUTED,
    SMB_RAM,
    _smb_84_segment,
    _smb_level,
)

"""Level definitions for supported games.

Import a module here to register its levels in the global registry.
Game-owned level packs live under ``snes/<game>/`` or ``nes/<game>/`` and
are imported by package name so RAM maps stay out of the shared harness tree.
"""

import donkey_kong_country.platformer_levels  # noqa: F401 — game-owned DKC LevelConfigs
import smb.platformer_levels  # noqa: F401 — game-owned SMB LevelConfigs
import smb3.platformer_levels  # noqa: F401 — game-owned SMB3 LevelConfigs
import SMW.platformer_levels  # noqa: F401 — game-owned SMW LevelConfigs
import super_metroid.platformer_levels  # noqa: F401 — game-owned SM LevelConfigs
import sm_rando.platformer_levels  # noqa: F401 — SM-rando entry-corpus consumer

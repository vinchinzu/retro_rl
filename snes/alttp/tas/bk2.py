"""BizHawk ``.bk2`` movie import for ALttP TAS seeds.

Parser lives in :mod:`retro_harness.bk2` (LogKey, then legacy reverse).
Same wrapper Super Metroid uses.
"""

from __future__ import annotations

from retro_harness.bk2 import (
    Bk2Movie,
    parse_bk2,
    parse_logkey_p1_to_env,
)

__all__ = [
    "Bk2Movie",
    "parse_bk2",
    "parse_logkey_p1_to_env",
]

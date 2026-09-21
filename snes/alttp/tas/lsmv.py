"""lsnes ``.lsmv`` movie import for ALttP TAS seeds.

Generic SNES parser from Super Metroid. Button field is env order
``BYsSudlrAXLR``. Published GEG #3898M uses ``ygamepad16`` (16-char P1);
the parser keeps the first 12 buttons.
"""

from __future__ import annotations

from super_metroid.tas.lsmv import LsmvMovie, parse_lsmv

__all__ = [
    "LsmvMovie",
    "parse_lsmv",
]

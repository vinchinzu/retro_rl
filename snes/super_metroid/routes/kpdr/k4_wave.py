"""K4 Wave branch — thin facade over :mod:`super_metroid.routes.kpdr.wave`.

Controllers live in the ``wave/`` package (geometry + per-hop modules).
This module re-exports the public API so registry, spine_hops, probes, and
tests that import ``k4_wave`` stay stable.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.wave import (
    ROOM_BUBBLE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
    WAVE_BEAM_MASK,
    play_bubble_to_single_chamber,
    play_double_chamber_to_wave,
    play_double_to_single_chamber,
    play_single_to_double_chamber,
    play_wave_to_double_chamber,
)

__all__ = [
    "WAVE_BEAM_MASK",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_double_chamber_to_wave",
    "play_wave_to_double_chamber",
    "play_double_to_single_chamber",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
]

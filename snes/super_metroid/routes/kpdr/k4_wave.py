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
    play_bubble_to_farm,
    play_bubble_to_single_chamber,
    play_double_chamber_to_wave,
    play_double_to_single_chamber,
    play_farm_to_speedway,
    play_frog_save_to_business,
    play_single_to_bubble,
    play_single_to_double_chamber,
    play_speedway_to_frog_save,
    play_wave_to_double_chamber,
)

__all__ = [
    "WAVE_BEAM_MASK",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_double_chamber_to_wave",
    "play_wave_to_double_chamber",
    "play_double_to_single_chamber",
    "play_single_to_bubble",
    "play_bubble_to_farm",
    "play_farm_to_speedway",
    "play_speedway_to_frog_save",
    "play_frog_save_to_business",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
]

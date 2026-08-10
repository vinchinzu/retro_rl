"""Unit scaffold for Wave→Business return stack (no emulator)."""

from __future__ import annotations

from super_metroid.routes.kpdr import get_segment
from super_metroid.routes.kpdr.wave import (
    play_double_to_single_chamber,
    play_wave_to_double_chamber,
)
from super_metroid.routes.kpdr.wave.geometry import (
    DTS_DOOR_X,
    DTS_FLOOR_Y_MIN,
    DTS_HOP_LAUNCH_X,
    DTS_SINGLE_SETTLE,
    WAVE_BEAM_MASK,
    WAVE_DOOR_X,
    WAVE_DOUBLE_SETTLE,
    WAVE_LEAVE_FRAMES,
    has_wave,
)
from types import SimpleNamespace


def test_wave_to_double_export_and_registry() -> None:
    assert get_segment("wave_to_double_chamber") is play_wave_to_double_chamber


def test_double_to_single_export_and_registry() -> None:
    assert get_segment("double_to_single_chamber") is play_double_to_single_chamber


def test_wave_return_geometry_constants() -> None:
    assert WAVE_BEAM_MASK == 0x0001
    assert WAVE_DOOR_X == 48
    assert WAVE_LEAVE_FRAMES >= 300
    assert WAVE_DOUBLE_SETTLE >= 200
    assert DTS_HOP_LAUNCH_X >= 900
    assert DTS_FLOOR_Y_MIN >= 400
    assert DTS_DOOR_X <= 50
    assert DTS_SINGLE_SETTLE >= 200


def test_has_wave_predicate() -> None:
    yes = SimpleNamespace(collected_beams=0x1005)
    no = SimpleNamespace(collected_beams=0x1004)
    assert has_wave(yes) is True  # type: ignore[arg-type]
    assert has_wave(no) is False  # type: ignore[arg-type]

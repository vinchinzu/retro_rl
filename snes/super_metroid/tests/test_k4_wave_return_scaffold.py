"""Unit scaffold for Wave→Business return stack (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from super_metroid.routes.kpdr import get_segment
from super_metroid.routes.kpdr.wave import (
    play_bubble_to_farm,
    play_double_to_single_chamber,
    play_farm_to_speedway,
    play_frog_save_to_business,
    play_single_to_bubble,
    play_speedway_to_frog_save,
    play_wave_to_double_chamber,
)
from super_metroid.routes.kpdr.wave.geometry import (
    BTF_DOOR_X,
    BTF_FARM_SETTLE,
    BTF_MID_Y,
    BTF_UPPER_Y,
    DTS_DOOR_X,
    DTS_FLOOR_Y_MIN,
    DTS_HOP_LAUNCH_X,
    DTS_SINGLE_SETTLE,
    FTB_BUSINESS_SETTLE,
    FTB_DOOR_X,
    FTS_DOOR_X,
    FTS_SPEEDWAY_SETTLE,
    SPEED_BOOSTER_MASK,
    STB_BUBBLE_SETTLE,
    STB_DEEP_Y_MIN,
    STB_DOOR_X,
    STB_TOP_Y_MAX,
    STF_DOOR_X,
    STF_FROG_SETTLE,
    WAVE_BEAM_MASK,
    WAVE_DOOR_X,
    WAVE_DOUBLE_SETTLE,
    WAVE_LEAVE_FRAMES,
    has_speed,
    has_wave,
)

_WAVE_RETURN_SEGMENTS = (
    ("wave_to_double_chamber", play_wave_to_double_chamber),
    ("double_to_single_chamber", play_double_to_single_chamber),
    ("single_to_bubble", play_single_to_bubble),
    ("bubble_to_farm", play_bubble_to_farm),
    ("farm_to_speedway", play_farm_to_speedway),
    ("speedway_to_frog_save", play_speedway_to_frog_save),
    ("frog_save_to_business", play_frog_save_to_business),
)


def test_wave_return_segments_registered() -> None:
    for segment_id, controller in _WAVE_RETURN_SEGMENTS:
        assert get_segment(segment_id) is controller


def test_wave_return_geometry_and_predicates() -> None:
    assert WAVE_BEAM_MASK == 0x0001
    assert WAVE_DOOR_X == 48
    assert WAVE_LEAVE_FRAMES >= 300
    assert WAVE_DOUBLE_SETTLE >= 200
    assert DTS_HOP_LAUNCH_X >= 900
    assert DTS_FLOOR_Y_MIN >= 400
    assert DTS_DOOR_X <= 50
    assert DTS_SINGLE_SETTLE >= 200
    assert STB_DEEP_Y_MIN >= 550
    assert STB_TOP_Y_MAX <= 180
    assert STB_DOOR_X <= 50
    assert STB_BUBBLE_SETTLE >= 200
    assert BTF_MID_Y[0] >= 360
    assert BTF_UPPER_Y[1] <= 200
    assert BTF_DOOR_X <= 50
    assert BTF_FARM_SETTLE >= 200
    assert SPEED_BOOSTER_MASK == 0x2000
    assert FTS_DOOR_X <= 50
    assert FTS_SPEEDWAY_SETTLE >= 200
    assert STF_DOOR_X <= 50
    assert STF_FROG_SETTLE >= 200
    assert FTB_DOOR_X <= 50
    assert FTB_BUSINESS_SETTLE >= 200

    assert has_wave(SimpleNamespace(collected_beams=0x1005)) is True  # type: ignore[arg-type]
    assert has_wave(SimpleNamespace(collected_beams=0x1004)) is False  # type: ignore[arg-type]
    # Product post-Speed items often 0x3105 (= Speed|Bombs|HJ|Morph|Varia).
    assert has_speed(SimpleNamespace(collected_items=0x3105)) is True  # type: ignore[arg-type]
    assert has_speed(SimpleNamespace(collected_items=0x1105)) is False  # type: ignore[arg-type]

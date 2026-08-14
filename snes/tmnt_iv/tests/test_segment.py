"""Offline checks for the shared TMNT IV stage-segment helper."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, StageSpec


def test_stage_specs_cover_1_through_9() -> None:
    assert set(STAGE_SPECS) == set(range(1, 10))
    for number, spec in STAGE_SPECS.items():
        assert isinstance(spec, StageSpec)
        assert spec.number == number
        assert spec.preferred_states
        assert spec.preferred_states[-1] == "NONE"


def test_stage1_keeps_on_sight_boss_snapshot() -> None:
    spec = STAGE_SPECS[1]
    assert spec.snapshot_boss_on_sight
    assert spec.save_hp_min == 40
    assert spec.require_lives
    assert spec.walk == "right"


def test_stage8_idles_and_heals() -> None:
    spec = STAGE_SPECS[8]
    assert spec.walk == "idle"
    assert spec.heal_low_hp_default
    assert spec.default_max_frames == 20000


def test_stage9_walks_right_and_heals() -> None:
    spec = STAGE_SPECS[9]
    assert spec.walk == "right"
    assert spec.heal_low_hp_default

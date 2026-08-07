"""Unit tests for Ceres classic L↔R arm-pump helpers (no emulator)."""

from __future__ import annotations

from super_metroid.routes.kpdr.early_spine import (
    _CERES_ARM_PUMP_PERIOD,
    _arm_pump_dash_spans,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.early_spine import MORPH_SPINE


def test_arm_pump_dash_spans_period_2_alternates_l_r() -> None:
    spans = _arm_pump_dash_spans("RIGHT", 10, "t", period=2)
    assert sum(s.frames for s in spans) == 10
    assert all("B" in s.names and "RIGHT" in s.names for s in spans)
    angs: list[str] = []
    for s in spans:
        ang = "L" if "L" in s.names else "R"
        angs.extend([ang] * s.frames)
    assert angs == ["L", "L", "R", "R", "L", "L", "R", "R", "L", "L"]


def test_arm_pump_period_default_is_runway_classic() -> None:
    assert _CERES_ARM_PUMP_PERIOD == 2


def test_morph_spine_ceres_hops_use_arm_pump_play() -> None:
    by_id = {h.hop_id: h for h in MORPH_SPINE}
    assert by_id["ridley_countdown"].play is play_ceres_outbound_to_ridley
    assert by_id["zebes_landing"].play is play_ceres_escape_to_landing

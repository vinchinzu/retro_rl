"""Leave-pin glance checks: room / mode / xy / TF / hearts. No emulator."""

from __future__ import annotations

from dataclasses import replace

import pytest

from zelda_i.route.chain import ControllerStageResult
from zelda_i.ram import CAVE_MODE, PLAY_MODE
from zelda_i.screen_glance import (
    BOW22_LEAVE,
    BOW_CELLAR_LEAVE,
    BOW_PICKUP_LEAVE,
    CELLAR08_LEAVE,
    CLEAR_3A,
    LeaveSpec,
    NORTH2C_LEAVE,
    SOUTH1D_LEAVE,
    STAIRS3A_DEST,
    WEST2D_LEAVE,
    grade_controller,
    grade_final,
    grade_report,
    grade_stage_report,
    parse_room,
)

_LEAVE_SPECS = (
    CLEAR_3A, CELLAR08_LEAVE, SOUTH1D_LEAVE, WEST2D_LEAVE, NORTH2C_LEAVE,
    BOW22_LEAVE, BOW_CELLAR_LEAVE, BOW_PICKUP_LEAVE, STAIRS3A_DEST,
)


def _ok(**overrides: object) -> dict:
    final: dict = {
        "room": "0x3A", "xy": [144, 141], "mode": PLAY_MODE, "triforce": 0x1F,
        "keys": 4, "bombs": 8, "rod": 1, "health": 0x66, "progression_writes": 0,
    }
    final.update(overrides)
    return final


def _leftover_for(spec: LeaveSpec, **overrides: object) -> dict:
    leftover: dict = {
        "x": (spec.x[0] + spec.x[1]) // 2,
        "y": (spec.y[0] + spec.y[1]) // 2,
        "screen": spec.room,
        "mode": spec.mode,
        "triforce": spec.triforce_bits,
    }
    if spec.keys is not None:
        leftover["keys"] = spec.keys
    if spec.bombs is not None:
        leftover["bombs"] = spec.bombs
    if spec.hearts_lo_eq_hi:
        leftover["health"] = 0x66
    leftover.update(overrides)
    return leftover


def test_parse_room_accepts_hex_and_int() -> None:
    assert parse_room(0x3A) == parse_room("0x3a") == parse_room("0x3A") == parse_room(58) == 0x3A


def test_wrong_room_is_a_glance_miss() -> None:
    assert any(m.startswith("room ") for m in grade_final(_ok(room="0x39"), CLEAR_3A))


def test_still_cave_mode_11_is_a_glance_miss() -> None:
    assert any("cave" in m for m in grade_final(_ok(mode=CAVE_MODE), CLEAR_3A))


def test_triforce_bit_missing_is_a_glance_miss() -> None:
    assert any("missing bits" in m for m in grade_final(_ok(triforce=0x0F), CLEAR_3A))
    assert grade_final(_ok(triforce=0x3F), CLEAR_3A) == []


def test_hearts_nibble_0xf_is_a_glance_miss() -> None:
    for health in (0x6F, 0x0F):
        assert any("low nibble 0xF" in m for m in grade_final(_ok(health=health), CLEAR_3A))
    loose = replace(CLEAR_3A, hearts_lo_eq_hi=False)
    assert any("low nibble 0xF" in m for m in grade_final(_ok(health=0x6F), loose))


def test_hearts_lo_eq_hi_passes() -> None:
    assert grade_final(_ok(health=0x66), CLEAR_3A) == []
    assert any("lo!=hi" in m for m in grade_final(_ok(health=0x65), CLEAR_3A))


def test_xy_outside_band_is_a_glance_miss() -> None:
    assert any(m.startswith("x=") for m in grade_final(_ok(xy=[16, 141]), CLEAR_3A))


def test_failed_run_is_a_glance_miss() -> None:
    report = {"success": False, "runs": [{"success": False, "final": _ok()}]}
    misses = grade_report(report, CLEAR_3A)
    assert "success is false" in misses
    assert "run 1 success is false" in misses


class _DummyHop:
    def __init__(self, leftover: dict, *, success: bool = False) -> None:
        self.leftover = leftover
        self.success = success
        self.failed = not success

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "leftover": dict(self.leftover),
        }


def test_wrong_room_miss_still_returns_leftover() -> None:
    graded = grade_controller(_DummyHop(_leftover_for(CLEAR_3A, screen=0x39)), CLEAR_3A)
    assert not graded.ok
    assert any(m.startswith("room ") for m in graded.misses)
    assert graded.leftover["screen"] == 0x39
    assert graded.leftover["xy"] == [144, 141]


def test_failed_east3a_stage_report_includes_nonclaim_leftover() -> None:
    leftover = {
        "x": 96, "y": 157, "mode": PLAY_MODE, "screen": 0x3A,
        "keys": 4, "bombs": 8, "triforce": 0x1F, "tile": 118,
    }
    stage = ControllerStageResult(
        name="level6_east_0x3a",
        controller=_DummyHop(leftover, success=False),
        max_frames=4000,
        frames=80,
        success=False,
    )
    payload = stage.report()
    assert payload["success"] is False
    assert payload["leftover"]["x"] == 96
    graded = grade_stage_report(payload, CELLAR08_LEAVE)
    assert not graded.ok
    assert any(m.startswith("room ") for m in graded.misses)
    assert graded.leftover["xy"] == [96, 157]


@pytest.mark.parametrize("spec", _LEAVE_SPECS, ids=lambda s: s.hop)
def test_leave_spec_midband_leftover_glances(spec: LeaveSpec) -> None:
    leftover = _leftover_for(spec)
    graded = grade_controller(_DummyHop(leftover), spec)
    assert graded.ok
    assert graded.misses == []
    if not spec.hearts_lo_eq_hi:
        assert "health" not in leftover

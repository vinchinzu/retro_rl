"""Leave-pin glance checks: room / mode / xy / TF / hearts. No emulator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from zelda_i.chain import ControllerStageResult
from zelda_i.ram import CAVE_MODE, PLAY_MODE
from zelda_i.screen_glance import (
    BOW22_LEAVE,
    CELLAR08_LEAVE,
    CLEAR_3A,
    FANFARE_MODE,
    NORTH2C_LEAVE,
    SOUTH1D_LEAVE,
    STAIRS3A_DEST,
    WEST2D_LEAVE,
    grade_controller,
    grade_final,
    grade_report,
    grade_stage_report,
    leftover_from_controller,
    parse_room,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _ok(**overrides: object) -> dict:
    final: dict = {
        "room": "0x3A",
        "xy": [144, 141],
        "mode": PLAY_MODE,
        "triforce": 0x1F,
        "keys": 4,
        "bombs": 8,
        "rod": 1,
        "health": 0x66,
        "progression_writes": 0,
    }
    final.update(overrides)
    return final


def test_l6_clear3a_fixture_glances_play() -> None:
    misses = grade_report(_load("screen_glance_l6_clear3a.json"), CLEAR_3A)
    assert misses == []


def test_parse_room_accepts_hex_and_int() -> None:
    assert parse_room(0x3A) == 0x3A
    assert parse_room("0x3a") == 0x3A
    assert parse_room("0x3A") == 0x3A
    assert parse_room(58) == 0x3A


def test_wrong_room_is_a_glance_miss() -> None:
    misses = grade_final(_ok(room="0x39"), CLEAR_3A)
    assert any(m.startswith("room ") for m in misses)


def test_still_cave_mode_11_is_a_glance_miss() -> None:
    misses = grade_final(_ok(mode=CAVE_MODE), CLEAR_3A)
    assert any("cave" in m for m in misses)


def test_fanfare_mode_18_is_a_glance_miss() -> None:
    misses = grade_final(_ok(mode=FANFARE_MODE), CLEAR_3A)
    assert any("fanfare" in m for m in misses)


def test_triforce_bit_missing_is_a_glance_miss() -> None:
    misses = grade_final(_ok(triforce=0x0F), CLEAR_3A)
    assert any("missing bits" in m for m in misses)
    assert grade_final(_ok(triforce=0x3F), CLEAR_3A) == []


def test_hearts_nibble_0xf_is_a_glance_miss() -> None:
    for health in (0x6F, 0x0F):
        misses = grade_final(_ok(health=health), CLEAR_3A)
        assert any("low nibble 0xF" in m for m in misses)
    loose = replace(CLEAR_3A, hearts_lo_eq_hi=False)
    misses = grade_final(_ok(health=0x6F), loose)
    assert any("low nibble 0xF" in m for m in misses)


def test_hearts_lo_eq_hi_passes() -> None:
    assert grade_final(_ok(health=0x66), CLEAR_3A) == []
    misses = grade_final(_ok(health=0x65), CLEAR_3A)
    assert any("lo!=hi" in m for m in misses)


def test_xy_outside_band_is_a_glance_miss() -> None:
    misses = grade_final(_ok(xy=[16, 141]), CLEAR_3A)
    assert any(m.startswith("x=") for m in misses)


def test_door_poke_progression_writes_is_a_glance_miss() -> None:
    misses = grade_final(_ok(progression_writes=1), CLEAR_3A)
    assert any("progression_writes" in m for m in misses)
    misses = grade_final(_ok(doors_poked=True), CLEAR_3A)
    assert "doors_poked" in misses


def test_keys_earned_count_miss_when_spec_set() -> None:
    misses = grade_final(_ok(keys=3), CLEAR_3A)
    assert any(m.startswith("keys=") for m in misses)
    misses = grade_final(_ok(keys=16), CLEAR_3A)
    assert any(m.startswith("keys=") for m in misses)


def test_failed_run_is_a_glance_miss() -> None:
    report = {
        "success": False,
        "runs": [{"success": False, "final": _ok()}],
    }
    misses = grade_report(report, CLEAR_3A)
    assert "success is false" in misses
    assert "run 1 success is false" in misses


def test_dual_frame_mismatch_still_glances_leave() -> None:
    """Glance is RAM identity. Frame mismatch is a dual-green concern, not a still."""
    report = {
        "success": True,
        "runs": [
            {"success": True, "final": _ok(), "frames": 1857},
            {"success": True, "final": _ok(), "frames": 1864},
        ],
    }
    assert grade_report(report, CLEAR_3A) == []


class _DummyHop:
    """Controller stand-in: leftover dict, no emulator."""

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


def _clear3a_leftover(**overrides: object) -> dict:
    leftover: dict = {
        "x": 144,
        "y": 141,
        "screen": 0x3A,
        "mode": PLAY_MODE,
        "triforce": 0x1F,
        "keys": 4,
        "bombs": 8,
        "health": 0x66,
    }
    leftover.update(overrides)
    return leftover


def test_leftover_from_controller_grades_clear_3a_green() -> None:
    leftover = leftover_from_controller(_DummyHop(_clear3a_leftover()))
    assert leftover["room"] == 0x3A
    assert leftover["xy"] == [144, 141]
    graded = grade_controller(_DummyHop(_clear3a_leftover()), CLEAR_3A)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [144, 141]


def test_wrong_room_miss_still_returns_leftover() -> None:
    dummy = _DummyHop(_clear3a_leftover(screen=0x39))
    graded = grade_controller(dummy, CLEAR_3A)
    assert not graded.ok
    assert any(m.startswith("room ") for m in graded.misses)
    assert graded.leftover
    assert graded.leftover["screen"] == 0x39
    assert graded.leftover["xy"] == [144, 141]


def test_failed_east3a_stage_report_includes_nonclaim_leftover() -> None:
    leftover = {
        "x": 96,
        "y": 157,
        "mode": PLAY_MODE,
        "screen": 0x3A,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
        "tile": 118,
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
    assert "leftover" in payload
    assert payload["leftover"]["x"] == 96
    assert payload["leftover"]["y"] == 157
    graded = grade_stage_report(payload, CELLAR08_LEAVE)
    assert not graded.ok
    assert any(m.startswith("room ") for m in graded.misses)
    assert graded.leftover["xy"] == [96, 157]


def test_cellar08_b_endpoint_0x1d_96_157_glances() -> None:
    leftover = {
        "x": 96,
        "y": 157,
        "mode": PLAY_MODE,
        "screen": 0x1D,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
        "tile": 118,
    }
    graded = grade_controller(_DummyHop(leftover), CELLAR08_LEAVE)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [96, 157]
    dest = {
        "x": 208,
        "y": 93,
        "mode": 9,
        "screen": 0x08,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
    }
    dest_graded = grade_controller(_DummyHop(dest), STAIRS3A_DEST)
    assert dest_graded.ok
    from zelda_i.level6_east3a import level6_east3a_glance

    miss = level6_east3a_glance(
        _DummyHop(
            {**leftover, "screen": 0x3A, "x": 96, "y": 141},
            success=False,
        )
    )
    assert not miss.ok
    assert any(m.startswith("x=") or m.startswith("y=") for m in miss.misses)
    assert miss.leftover["xy"] == [96, 141]


def test_south1d_predicted_0x2d_120_77_glances() -> None:
    leftover = {
        "x": 120,
        "y": 77,
        "mode": PLAY_MODE,
        "screen": 0x2D,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
    }
    graded = grade_controller(_DummyHop(leftover), SOUTH1D_LEAVE)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [120, 77]
    still = grade_controller(
        _DummyHop({**leftover, "screen": 0x1D, "x": 96, "y": 157}),
        SOUTH1D_LEAVE,
    )
    assert not still.ok
    from zelda_i.level6_south1d import level6_south1d_glance

    ok = level6_south1d_glance(_DummyHop(leftover))
    assert ok.ok


def test_west2d_live_0x2c_224_141_glances() -> None:
    leftover = {
        "x": 224,
        "y": 141,
        "mode": PLAY_MODE,
        "screen": 0x2C,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
    }
    graded = grade_controller(_DummyHop(leftover), WEST2D_LEAVE)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [224, 141]
    still = grade_controller(
        _DummyHop({**leftover, "screen": 0x2D, "x": 120, "y": 77}),
        WEST2D_LEAVE,
    )
    assert not still.ok
    from zelda_i.level6_west2d import level6_west2d_glance

    ok = level6_west2d_glance(_DummyHop(leftover))
    assert ok.ok
    spent = grade_controller(
        _DummyHop({**leftover, "keys": 3}),
        WEST2D_LEAVE,
    )
    assert not spent.ok


def test_north2c_live_0x1c_120_205_glances() -> None:
    leftover = {
        "x": 120,
        "y": 205,
        "mode": PLAY_MODE,
        "screen": 0x1C,
        "keys": 3,
        "bombs": 8,
        "triforce": 0x1F,
    }
    graded = grade_controller(_DummyHop(leftover), NORTH2C_LEAVE)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [120, 205]
    still = grade_controller(
        _DummyHop({**leftover, "screen": 0x2C, "x": 224, "y": 141, "keys": 4}),
        NORTH2C_LEAVE,
    )
    assert not still.ok
    from zelda_i.level6_north2c import level6_north2c_glance

    ok = level6_north2c_glance(_DummyHop(leftover))
    assert ok.ok
    unspent = grade_controller(
        _DummyHop({**leftover, "keys": 4}),
        NORTH2C_LEAVE,
    )
    assert not unspent.ok


def test_bow22_planned_0x22_east_mouth_glances() -> None:
    leftover = {
        "x": 224,
        "y": 141,
        "mode": PLAY_MODE,
        "screen": 0x22,
        "keys": 0,
        "triforce": 0,
    }
    graded = grade_controller(_DummyHop(leftover), BOW22_LEAVE)
    assert graded.ok
    assert graded.misses == []
    assert graded.leftover["xy"] == [224, 141]
    still = grade_controller(
        _DummyHop({**leftover, "screen": 0x23, "x": 32, "y": 141}),
        BOW22_LEAVE,
    )
    assert not still.ok
    from zelda_i.level1_bow import level1_bow_glance

    ok = level1_bow_glance(_DummyHop(leftover))
    assert ok.ok

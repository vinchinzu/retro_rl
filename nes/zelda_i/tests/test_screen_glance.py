"""Leave-pin glance checks: room / mode / xy / TF / hearts. No emulator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from zelda_i.ram import CAVE_MODE, PLAY_MODE
from zelda_i.screen_glance import (
    FANFARE_MODE,
    LeaveSpec,
    grade_final,
    grade_report,
    parse_room,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Published leftover: l6_clear3a_continuous_v1 play 0x3A (144,141).
CLEAR_3A = LeaveSpec(
    hop="level6-clear3a",
    room=0x3A,
    x=(128, 160),
    y=(133, 149),
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
)


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

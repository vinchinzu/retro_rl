"""ROM-free lock for take02 LEFT+A vs take04 walk-right. No climb_action."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.hop_glance import grade_final
from super_metroid.leave_specs import WS_MAIN_GRATE_SEAT
from super_metroid.routes.kpdr.wrecked_ship.ws_main_departure import (
    ALCOVE_LEFT_A_X,
    LIVING_POLICY,
    SLOPE_LEFT_A,
    SLOPE_LEFT_A_POLICY,
    SLOPE_LEFT_A_Y,
    TAKE02_LEFT_A_SEAT,
    TAKE02_LIP_FIRE,
    TAKE02_PEAK_Y,
    TAKE03_LEFT_A_SEAT,
    TAKE03_PEAK_Y,
    TAKE04_LEFT_A_SEAT,
    TAKE04_LIP_FIRE,
    TAKE04_PEAK_Y,
    TAKE05_LEFT_A_SEAT,
    TAKE05_PEAK_Y,
    WALK_RIGHT_ALCOVE_POLICY,
    at_alcove_left_a,
    at_slope_left_a,
    scan_grate_departure,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ws_main_grate_departure.json"
TAPES = Path(__file__).resolve().parents[1] / "tasks" / "ws_main_attic_v1"


def _final(xy: tuple[int, int], *, pose: int = 3) -> dict:
    return {
        "room": "0xCAF6",
        "xy": [xy[0], xy[1]],
        "pose": pose,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }


def _takes() -> dict[str, list[dict]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["kind"] == "ws_main_grate_departure_snippets"
    assert payload["living"] == "take02"
    return payload["takes"]


def test_living_window_is_take02_slope_left_a_not_airborne_1221() -> None:
    assert LIVING_POLICY == SLOPE_LEFT_A_POLICY
    assert SLOPE_LEFT_A.side == "LEFT"
    assert SLOPE_LEFT_A.min_momentum == 0
    assert SLOPE_LEFT_A.x_range == (1227, 1231)
    assert SLOPE_LEFT_A_Y == (1852, 1856)
    assert SLOPE_LEFT_A.x_range[1] - SLOPE_LEFT_A.x_range[0] <= 4
    assert SLOPE_LEFT_A_Y[1] - SLOPE_LEFT_A_Y[0] <= 4
    assert at_slope_left_a(*TAKE02_LEFT_A_SEAT[:2])
    assert at_slope_left_a(*TAKE03_LEFT_A_SEAT[:2])
    assert not at_slope_left_a(1221, 1807)
    assert not at_slope_left_a(1223, 1860)
    assert not at_slope_left_a(*TAKE04_LIP_FIRE[:2])
    assert not at_slope_left_a(*TAKE04_LEFT_A_SEAT[:2])
    assert not at_alcove_left_a(*TAKE02_LEFT_A_SEAT[:2])
    assert at_alcove_left_a(*TAKE04_LEFT_A_SEAT[:2])
    assert at_alcove_left_a(*TAKE05_LEFT_A_SEAT[:2])
    assert ALCOVE_LEFT_A_X[0] > WS_MAIN_GRATE_SEAT.x[1]


def test_glance_covers_slope_takeoff_rejects_take04() -> None:
    fire = TAKE02_LIP_FIRE
    seat = TAKE02_LEFT_A_SEAT
    assert grade_final(_final((fire[0], fire[1]), pose=fire[2]), WS_MAIN_GRATE_SEAT) == []
    assert grade_final(_final((seat[0], seat[1]), pose=seat[2]), WS_MAIN_GRATE_SEAT) == []
    assert grade_final(
        _final((TAKE03_LEFT_A_SEAT[0], TAKE03_LEFT_A_SEAT[1]), pose=1),
        WS_MAIN_GRATE_SEAT,
    ) == []
    take04_fire = grade_final(
        _final((TAKE04_LIP_FIRE[0], TAKE04_LIP_FIRE[1])), WS_MAIN_GRATE_SEAT
    )
    assert any(m.startswith("x=") for m in take04_fire)
    alcove = grade_final(
        _final((TAKE04_LEFT_A_SEAT[0], TAKE04_LEFT_A_SEAT[1]), pose=1),
        WS_MAIN_GRATE_SEAT,
    )
    assert any(m.startswith("x=") for m in alcove)


def test_fixture_scan_locks_take02_vs_take04_policies() -> None:
    takes = _takes()
    t02 = scan_grate_departure(takes["take02"])
    assert t02.policy == SLOPE_LEFT_A_POLICY
    assert t02.lip_fire is not None
    assert (t02.lip_fire.x, t02.lip_fire.y, t02.lip_fire.pose) == TAKE02_LIP_FIRE
    assert "X" in t02.lip_fire.buttons
    assert t02.grounded_takeoff is not None
    assert (t02.grounded_takeoff.x, t02.grounded_takeoff.y) == TAKE02_LEFT_A_SEAT[:2]
    assert t02.first_left_a is not None
    assert t02.first_left_a.x == 1231
    assert t02.first_left_a.y == 1852
    assert "LEFT" in t02.first_left_a.buttons and "A" in t02.first_left_a.buttons
    assert t02.peak_y == TAKE02_PEAK_Y
    assert t02.first_right_after_shot is not None
    assert t02.first_right_after_shot.x >= 1223
    assert t02.first_right_after_shot.y <= 1860

    t03 = scan_grate_departure(takes["take03"])
    assert t03.policy == SLOPE_LEFT_A_POLICY
    assert t03.grounded_takeoff is not None
    assert (t03.grounded_takeoff.x, t03.grounded_takeoff.y) == TAKE03_LEFT_A_SEAT[:2]
    assert t03.peak_y == TAKE03_PEAK_Y
    assert t03.first_right_after_shot is None

    t04 = scan_grate_departure(takes["take04"])
    assert t04.policy == WALK_RIGHT_ALCOVE_POLICY
    assert t04.lip_fire is not None
    assert (t04.lip_fire.x, t04.lip_fire.y, t04.lip_fire.pose) == TAKE04_LIP_FIRE
    assert t04.first_right_after_shot is not None
    assert t04.first_right_after_shot.x <= 1196
    assert t04.first_right_after_shot.y == 1883
    assert t04.grounded_takeoff is not None
    assert (t04.grounded_takeoff.x, t04.grounded_takeoff.y) == TAKE04_LEFT_A_SEAT[:2]
    assert t04.first_left_a is not None
    assert t04.first_left_a.x >= ALCOVE_LEFT_A_X[0]
    assert t04.peak_y == TAKE04_PEAK_Y

    t05 = scan_grate_departure(takes["take05"])
    assert t05.policy == WALK_RIGHT_ALCOVE_POLICY
    assert t05.grounded_takeoff is not None
    assert (t05.grounded_takeoff.x, t05.grounded_takeoff.y) == TAKE05_LEFT_A_SEAT[:2]
    assert t05.peak_y == TAKE05_PEAK_Y


def test_full_tapes_match_fixture_when_present() -> None:
    takes = _takes()
    for name in ("take02", "take03", "take04", "take05"):
        path = TAPES / f"ws_main_attic_v1_{name}.json"
        if not path.is_file():
            continue
        tape = json.loads(path.read_text(encoding="utf-8"))["trace"]
        from_tape = scan_grate_departure(tape)
        from_fix = scan_grate_departure(takes[name])
        assert from_tape.policy == from_fix.policy
        assert from_tape.lip_fire is not None and from_fix.lip_fire is not None
        assert (from_tape.lip_fire.x, from_tape.lip_fire.y, from_tape.lip_fire.pose) == (
            from_fix.lip_fire.x,
            from_fix.lip_fire.y,
            from_fix.lip_fire.pose,
        )
        assert from_tape.grounded_takeoff is not None
        assert from_fix.grounded_takeoff is not None
        assert (from_tape.grounded_takeoff.x, from_tape.grounded_takeoff.y) == (
            from_fix.grounded_takeoff.x,
            from_fix.grounded_takeoff.y,
        )
        assert from_tape.peak_y == from_fix.peak_y

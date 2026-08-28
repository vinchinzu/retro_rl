"""Leave-pin glance checks: room / gs / pose / xy / boss bit. No emulator."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.hop_glance import (
    LeaveMiss,
    final_from_state,
    grade_final,
    grade_report,
    pose_class,
    raise_leave_miss,
)
from super_metroid.leave_specs import (
    PHANTOON_LEAVE,
    RED_TO_HELLWAY,
    WS_BASEMENT_TO_MAIN,
    WS_BASEMENT_TO_PHANTOON,
    WS_ENTRANCE_TO_MAIN,
    WS_MAIN_TO_ATTIC,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_HELLWAY,
    ROOM_PHANTOON,
    ROOM_RED_TOWER,
    ROOM_WS_BASEMENT,
    ROOM_WS_MAIN,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_ws_entrance_to_main_fixture_glances_stand_in_main() -> None:
    misses = grade_report(_load("hop_glance_ws_entrance_to_main.json"), WS_ENTRANCE_TO_MAIN)
    assert misses == []


def test_phantoon_leave_fixture_glances_basement_stand_boss_dead() -> None:
    misses = grade_report(_load("hop_glance_phantoon_leave.json"), PHANTOON_LEAVE)
    assert misses == []


def test_wrong_room_is_a_glance_miss() -> None:
    final = {"room": hex(ROOM_PHANTOON), "xy": [1240, 139], "pose": 10, "gs": 8, "dt": 0, "health": 299, "boss": 1}
    misses = grade_final(final, PHANTOON_LEAVE)
    assert any(m.startswith("room ") for m in misses)


def test_still_morph_when_door_needs_stand() -> None:
    final = {"room": "0xCAF6", "xy": [1063, 907], "pose": 29, "gs": 8, "dt": 0, "health": 299}
    assert pose_class(29) == "morph"
    misses = grade_final(final, WS_ENTRANCE_TO_MAIN)
    assert any("not stand" in m for m in misses)


def test_boss_alive_is_a_glance_miss() -> None:
    final = {"room": "0xcc6f", "xy": [1240, 139], "pose": 10, "gs": 8, "dt": 0, "health": 299, "boss": 0}
    misses = grade_final(final, PHANTOON_LEAVE)
    assert "boss=0 != 1" in misses


def test_not_ordinary_gameplay_is_a_glance_miss() -> None:
    final = {"room": "0xCAF6", "xy": [1063, 907], "pose": 9, "gs": 11, "dt": 1, "health": 299}
    misses = grade_final(final, WS_ENTRANCE_TO_MAIN)
    assert "gs=11 != 8" in misses
    assert "dt=1 != 0" in misses


def test_xy_outside_door_band_is_a_glance_miss() -> None:
    final = {"room": "0xCAF6", "xy": [57, 139], "pose": 9, "gs": 8, "dt": 0, "health": 299}
    misses = grade_final(final, WS_ENTRANCE_TO_MAIN)
    assert any(m.startswith("x=") for m in misses)


def test_hellway_door_slot_fire_is_not_a_leave() -> None:
    """x=237 in Red Tower is the door-slot fire, not Hellway."""
    fire = {"room": hex(ROOM_RED_TOWER), "xy": [237, 139], "pose": 11, "gs": 8, "dt": 0, "health": 299}
    misses = grade_final(fire, RED_TO_HELLWAY)
    assert any(m.startswith("room ") for m in misses)
    sill = {"room": hex(ROOM_HELLWAY), "xy": [39, 139], "pose": 11, "gs": 8, "dt": 0, "health": 299}
    assert grade_final(sill, RED_TO_HELLWAY) == []


def test_dual_frame_mismatch_still_glances_leave() -> None:
    """Glance is RAM identity. Frame mismatch is a dual-green concern, not a still."""
    report = {
        "success": True,
        "runs": [
            {
                "success": True,
                "final": {
                    "room": "0xCAF6",
                    "xy": [1063, 907],
                    "pose": 9,
                    "gs": 8,
                    "dt": 0,
                    "health": 299,
                },
                "frames": 403,
            },
            {
                "success": True,
                "final": {
                    "room": "0xCAF6",
                    "xy": [1063, 907],
                    "pose": 9,
                    "gs": 8,
                    "dt": 0,
                    "health": 299,
                },
                "frames": 410,
            },
        ],
    }
    assert grade_report(report, WS_ENTRANCE_TO_MAIN) == []


def test_failed_run_is_a_glance_miss() -> None:
    report = {
        "success": False,
        "runs": [
            {
                "success": False,
                "final": {
                    "room": "0xcc6f",
                    "xy": [1240, 139],
                    "pose": 10,
                    "gs": 8,
                    "dt": 0,
                    "health": 299,
                    "boss": 1,
                },
            }
        ],
    }
    misses = grade_report(report, PHANTOON_LEAVE)
    assert "success is false" in misses
    assert "run 1 success is false" in misses


class _Duck:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_final_from_state_maps_samus_and_duck() -> None:
    state = _Duck(
        room_id=ROOM_WS_MAIN,
        samus_x=1063,
        samus_y=907,
        pose=9,
        game_state=8,
        door_transition=0,
        health=299,
    )
    final = final_from_state(state)
    assert final == {
        "room": "0xCAF6",
        "xy": [1063, 907],
        "pose": 9,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    duck = _Duck(room=0xCAF6, x=1063, y=907, pose=9, gs=8, dt=0, health=299)
    assert final_from_state(duck)["xy"] == [1063, 907]


def test_grade_final_keeps_final_on_miss() -> None:
    leftover = {"room": "0xCAF6", "xy": [57, 139], "pose": 9, "gs": 8, "dt": 0, "health": 299}
    assert leftover["xy"] == [57, 139]
    assert grade_final(leftover, WS_ENTRANCE_TO_MAIN)


def test_missing_health_is_a_glance_miss() -> None:
    final = {"room": "0xCAF6", "xy": [1063, 907], "pose": 9, "gs": 8, "dt": 0}
    misses = grade_final(final, WS_ENTRANCE_TO_MAIN)
    assert "missing health" in misses


def test_leave_miss_carries_leftover_xy_when_glance_misses() -> None:
    leftover = {
        "room": "0xCAF6",
        "xy": [1063, 907],
        "pose": 29,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    misses = grade_final(leftover, WS_ENTRANCE_TO_MAIN)
    assert any("not stand" in m for m in misses)
    err = LeaveMiss("ws_entrance_to_main", leftover, misses)
    assert err.leftover["xy"] == [1063, 907]
    assert err.leftover["pose"] == 29
    assert "leftover xy=[1063, 907]" in str(err)
    assert "pose 29" in str(err)


def test_leave_miss_populates_leftover_when_room_is_wrong() -> None:
    leftover = {
        "room": hex(ROOM_PHANTOON),
        "xy": [39, 128],
        "pose": 10,
        "gs": 8,
        "dt": 0,
        "health": 299,
        "boss": 1,
    }
    misses = grade_final(leftover, PHANTOON_LEAVE)
    err = LeaveMiss(
        "phantoon_loot_exit",
        leftover,
        misses,
        room_label="WS Basement (post-Phantoon)",
        to_room=ROOM_WS_BASEMENT,
    )
    assert "expected WS Basement (post-Phantoon) 0xCC6F, got 0xCD13" in str(err)
    assert err.leftover["xy"] == [39, 128]
    assert err.leftover["pose"] == 10
    assert err.leftover["gs"] == 8


def test_raise_leave_miss_grades_state_and_chains() -> None:
    state = _Duck(
        room_id=ROOM_WS_BASEMENT,
        samus_x=657,
        samus_y=91,
        pose=2,
        game_state=8,
        door_transition=0,
        health=299,
    )
    try:
        raise_leave_miss(
            state,
            "ws_main_to_attic",
            WS_MAIN_TO_ATTIC,
            room_label="Attic",
            to_room=0xCA52,
            exc=RuntimeError("wrong room"),
        )
    except LeaveMiss as err:
        assert err.leftover["xy"] == [657, 91]
        assert err.leftover["pose"] == 2
        assert "expected Attic 0xCA52, got 0xCC6F" in str(err)
        assert any("RuntimeError" in m for m in err.misses)
        assert err.__cause__.__class__ is RuntimeError
    else:
        raise AssertionError("raise_leave_miss must raise LeaveMiss")


def test_ws_basement_to_phantoon_air_or_stand_is_a_leave() -> None:
    """Dual-green exit of basement: spin or stand in 0xCD13, not morph."""
    assert WS_BASEMENT_TO_PHANTOON.room == ROOM_PHANTOON
    assert WS_BASEMENT_TO_PHANTOON.pose_class == "door"
    spin = {
        "room": hex(ROOM_PHANTOON),
        "xy": [39, 124],
        "pose": 81,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    assert grade_final(spin, WS_BASEMENT_TO_PHANTOON) == []
    stand = {**spin, "pose": 1}
    assert grade_final(stand, WS_BASEMENT_TO_PHANTOON) == []
    morph = {**spin, "pose": 29}
    assert pose_class(29) == "morph"
    misses = grade_final(morph, WS_BASEMENT_TO_PHANTOON)
    assert any("not door" in m for m in misses)


def test_ws_basement_to_main_spec_is_main_shaft_stand() -> None:
    assert WS_BASEMENT_TO_MAIN.room == ROOM_WS_MAIN
    assert WS_BASEMENT_TO_MAIN.pose_class == "stand"
    assert WS_BASEMENT_TO_MAIN.gs == 8
    assert WS_BASEMENT_TO_MAIN.dt == 0
    seated = {
        "room": "0xCAF6",
        "xy": [1144, 1900],
        "pose": 1,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    assert grade_final(seated, WS_BASEMENT_TO_MAIN) == []
    basement_hatch = {
        "room": hex(ROOM_WS_BASEMENT),
        "xy": [657, 163],
        "pose": 1,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    leftover = dict(basement_hatch)
    assert leftover["xy"] == [657, 163]
    assert any(m.startswith("room ") for m in grade_final(leftover, WS_BASEMENT_TO_MAIN))


def test_ws_main_to_attic_spec_is_attic_door() -> None:
    from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC

    assert WS_MAIN_TO_ATTIC.room == ROOM_WS_ATTIC
    assert WS_MAIN_TO_ATTIC.pose_class == "door"
    assert WS_MAIN_TO_ATTIC.gs == 8
    assert WS_MAIN_TO_ATTIC.dt == 0
    seated = {
        "room": "0xCA52",
        "xy": [1135, 184],
        "pose": 1,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    assert grade_final(seated, WS_MAIN_TO_ATTIC) == []
    air = {**seated, "pose": 21}
    assert grade_final(air, WS_MAIN_TO_ATTIC) == []
    transition = {
        "room": "0xCAF6",
        "xy": [1135, 31],
        "pose": 21,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    leftover = dict(transition)
    assert leftover["xy"] == [1135, 31]
    assert any(m.startswith("room ") for m in grade_final(leftover, WS_MAIN_TO_ATTIC))

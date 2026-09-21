"""Leave-pin glance checks: room / module / xy / sword / follower. No emulator."""

from __future__ import annotations

from types import SimpleNamespace

from alttp.ram import AlttpSnapshot, FOLLOWER_ZELDA, HYRULE_CASTLE_NW_ROOM
from alttp.screen_glance import (
    MODULE_INDOORS,
    MODULE_OVERWORLD,
    ROOM_01,
    ROOM_50,
    ROOM_72,
    LeaveSpec,
    grade_controller,
    grade_final,
    grade_leftover,
    grade_report,
    leftover_from_mapping,
    leftover_from_snapshot,
    parse_room,
)

_BOOT_KEYS = (
    "room",
    "module",
    "submodule",
    "x",
    "y",
    "sword",
    "follower",
    "keys",
    "indoors",
    "screen",
)


def _ok(**overrides: object) -> dict:
    final: dict = {
        "room": "0x50",
        "xy": [448, 2680],
        "module": MODULE_INDOORS,
        "submodule": 0,
        "sword": 1,
        "follower": 0,
        "keys": 0,
        "indoors": True,
        "screen": 0,
    }
    final.update(overrides)
    return final


def _snap(**overrides: object) -> AlttpSnapshot:
    kw: dict = {
        "game_mode": MODULE_INDOORS,
        "submodule": 0,
        "room_id": HYRULE_CASTLE_NW_ROOM,
        "indoors": True,
        "screen_id": 0,
        "link_x": 448,
        "link_y": 2680,
        "link_direction": 0,
        "link_action": 0,
        "camera_x": 0,
        "camera_y": 0,
        "dark_world": False,
        "sword_level": 1,
        "lamp_level": 1,
        "num_keys": 0,
        "follower": 0,
    }
    kw.update(overrides)
    return AlttpSnapshot(**kw)


class _DummyHop:
    def __init__(self, leftover: dict, *, success: bool = False) -> None:
        self.leftover = leftover
        self.success = success

    def report(self) -> dict:
        return {"success": self.success, "leftover": dict(self.leftover)}


def test_parse_room_accepts_hex_and_int() -> None:
    assert parse_room(0x50) == parse_room("0x50") == parse_room(80) == 0x50


def test_room_50_spec_is_spawn_east_door_band() -> None:
    assert ROOM_50.hop == "room_50"
    assert ROOM_50.room == 0x50
    assert ROOM_50.module == MODULE_INDOORS
    assert ROOM_50.submodule == 0
    assert ROOM_50.sword_min == 1
    assert ROOM_50.follower is None
    assert ROOM_50.indoors is True
    assert ROOM_50.keys is None
    # maps/room_50.json spawn (448, 2680) and east door (480, 2680) ±12.
    assert ROOM_50.x[0] <= 448 <= ROOM_50.x[1]
    assert ROOM_50.x[0] <= 480 <= ROOM_50.x[1]
    assert ROOM_50.y[0] <= 2680 <= ROOM_50.y[1]
    assert grade_final(_ok(xy=[448, 2680]), ROOM_50) == []
    assert grade_final(_ok(xy=[480, 2680]), ROOM_50) == []


def test_room_01_spec_is_f1_well_band() -> None:
    assert ROOM_01.hop == "room_01"
    assert ROOM_01.room == 0x01
    assert ROOM_01.module == MODULE_INDOORS
    assert ROOM_01.submodule == 0
    assert ROOM_01.sword_min == 1
    assert ROOM_01.follower is None
    assert ROOM_01.indoors is True
    assert ROOM_01.keys is None
    # corridor (760,120) ±4 union well (760,99) ±2 — not a looser default-4 well.
    assert ROOM_01.x == (756, 764)
    assert ROOM_01.y == (97, 124)
    assert grade_final(_ok(room="0x01", xy=[760, 99]), ROOM_01) == []
    assert grade_final(_ok(room="0x01", xy=[760, 120]), ROOM_01) == []
    assert any(m.startswith("y=") for m in grade_final(_ok(room="0x01", xy=[760, 95]), ROOM_01))
    assert any(m.startswith("x=") for m in grade_final(_ok(room="0x01", xy=[960, 120]), ROOM_01))
    assert any("submodule=" in m for m in grade_final(_ok(room="0x01", xy=[760, 99], submodule=14), ROOM_01))


def test_room_72_spec_is_b1_landing_band() -> None:
    assert ROOM_72.hop == "room_72"
    assert ROOM_72.room == 0x72
    assert ROOM_72.module == MODULE_INDOORS
    assert ROOM_72.submodule == 0
    assert ROOM_72.sword_min == 1
    assert ROOM_72.follower is None
    assert ROOM_72.indoors is True
    assert ROOM_72.keys is None
    # castle_b1_guard (1273,3665) / f1_stair_approach (1272,3656) ±12.
    assert ROOM_72.x == (1260, 1285)
    assert ROOM_72.y == (3644, 3677)
    assert grade_final(_ok(room="0x72", xy=[1273, 3665]), ROOM_72) == []
    assert grade_final(_ok(room="0x72", xy=[1272, 3656]), ROOM_72) == []
    # CastleB1Key is the same north wall, 48px east of the stair column.
    assert any(m.startswith("x=") for m in grade_final(_ok(room="0x72", xy=[1320, 3656]), ROOM_72))
    assert any("submodule=" in m for m in grade_final(_ok(room="0x72", xy=[1273, 3665], submodule=14), ROOM_72))


def test_wrong_room_is_a_glance_miss() -> None:
    assert any(m.startswith("room ") for m in grade_final(_ok(room="0x60"), ROOM_50))


def test_xy_outside_band_is_a_glance_miss() -> None:
    assert any(m.startswith("x=") for m in grade_final(_ok(xy=[16, 2680]), ROOM_50))
    assert any(m.startswith("y=") for m in grade_final(_ok(xy=[448, 16]), ROOM_50))


def test_module_and_submodule_must_be_control_ready() -> None:
    assert any("module=" in m for m in grade_final(_ok(module=MODULE_OVERWORLD), ROOM_50))
    assert any("submodule=" in m for m in grade_final(_ok(submodule=1), ROOM_50))


def test_sword_below_min_is_a_glance_miss() -> None:
    assert any(m.startswith("sword=") for m in grade_final(_ok(sword=0), ROOM_50))


def test_follower_none_does_not_require_zelda() -> None:
    assert grade_final(_ok(follower=0), ROOM_50) == []
    zelda = LeaveSpec(
        hop="escort",
        room=0x50,
        x=ROOM_50.x,
        y=ROOM_50.y,
        follower=FOLLOWER_ZELDA,
        indoors=True,
    )
    assert any("follower=" in m for m in grade_final(_ok(follower=0), zelda))
    assert grade_final(_ok(follower=FOLLOWER_ZELDA), zelda) == []


def test_optional_keys_and_indoors() -> None:
    assert grade_final(_ok(), ROOM_50) == []
    keyed = LeaveSpec(
        hop="keyed",
        room=0x50,
        x=ROOM_50.x,
        y=ROOM_50.y,
        keys=1,
        indoors=True,
    )
    assert any("keys=" in m for m in grade_final(_ok(keys=0), keyed))
    outdoor = dict(_ok(indoors=False))
    assert any("indoors=" in m for m in grade_final(outdoor, ROOM_50))


def test_failed_run_is_a_glance_miss() -> None:
    report = {"success": False, "runs": [{"success": False, "final": _ok()}]}
    misses = grade_report(report, ROOM_50)
    assert "success is false" in misses
    assert "run 1 success is false" in misses


def test_grade_report_top_level_final_and_diag_aliases() -> None:
    diag = {
        "game_mode": 0x07,
        "submodule": 0,
        "room_id": 0x50,
        "room_base_id": 0x50,
        "indoors": True,
        "screen_id": 0,
        "link_x": 448,
        "link_y": 2680,
        "sword_level": 1,
        "num_keys": 0,
        "follower": 0,
    }
    assert grade_report({"final": diag}, ROOM_50) == []
    dual = {
        "success": True,
        "runs": [
            {"success": True, "final": _ok()},
            {"success": True, "final": _ok(room="0x01")},
        ],
    }
    misses = grade_report(dual, ROOM_50)
    assert any("run 2:" in m and "room " in m for m in misses)


def test_wrong_room_miss_still_returns_leftover() -> None:
    leftover = leftover_from_snapshot(_snap(room_id=0x60, link_x=448, link_y=2680))
    graded = grade_controller(_DummyHop(leftover), ROOM_50)
    assert not graded.ok
    assert any(m.startswith("room ") for m in graded.misses)
    assert graded.leftover["room"] == 0x60
    assert graded.leftover["x"] == 448
    assert graded.leftover["y"] == 2680


def test_room_01_and_72_miss_still_returns_leftover() -> None:
    snap_01 = leftover_from_snapshot(_snap(room_id=0x50, link_x=760, link_y=99))
    graded_01 = grade_leftover(snap_01, ROOM_01)
    assert not graded_01.ok
    assert any(m.startswith("room ") for m in graded_01.misses)
    assert graded_01.leftover["room"] == 0x50
    assert graded_01.leftover["x"] == 760
    assert graded_01.leftover["y"] == 99
    mapped_72 = leftover_from_mapping(
        {
            "room": "0x01",
            "xy": [16, 3665],
            "module": MODULE_INDOORS,
            "submodule": 0,
            "sword": 1,
            "follower": 0,
            "keys": 0,
            "indoors": True,
            "screen": 0,
        }
    )
    graded_72 = grade_leftover(mapped_72, ROOM_72)
    assert not graded_72.ok
    assert any(m.startswith("room ") for m in graded_72.misses)
    assert graded_72.leftover["room"] == 0x01
    assert graded_72.leftover["x"] == 16
    assert graded_72.leftover["y"] == 3665
    assert leftover_from_snapshot(_snap(room_id=0x01, link_x=760, link_y=99))["room"] == 0x01
    assert leftover_from_snapshot(_snap(room_id=0x72, link_x=1273, link_y=3665))["room"] == 0x72
    assert grade_leftover(
        leftover_from_snapshot(_snap(room_id=0x01, link_x=760, link_y=99)), ROOM_01
    ).ok
    assert grade_leftover(
        leftover_from_snapshot(_snap(room_id=0x72, link_x=1273, link_y=3665)), ROOM_72
    ).ok


def test_leftover_from_snapshot_boot_keys() -> None:
    leftover = leftover_from_snapshot(_snap())
    for key in _BOOT_KEYS:
        assert key in leftover
    assert leftover["room"] == 0x50
    assert leftover["module"] == MODULE_INDOORS
    assert leftover["submodule"] == 0
    assert leftover["x"] == 448
    assert leftover["y"] == 2680
    assert leftover["sword"] == 1
    assert leftover["follower"] == 0
    assert leftover["keys"] == 0
    assert leftover["indoors"] is True
    assert leftover["screen"] == 0
    duck = leftover_from_snapshot(
        SimpleNamespace(
            link_x=480,
            link_y=2680,
            room_id=0x50,
            game_mode=0x07,
            submodule=0,
            sword_level=1,
            follower=0,
            num_keys=2,
            indoors=True,
            screen_id=0,
        )
    )
    assert duck["x"] == 480
    assert duck["keys"] == 2
    mapped = leftover_from_snapshot(
        {
            "game_mode": "0x07",
            "submodule": 0,
            "room_base_id": "0x50",
            "link_x": 448,
            "link_y": 2680,
            "sword_level": 1,
            "follower": 0,
            "num_keys": 0,
            "indoors": True,
            "screen_id": 0,
        }
    )
    for key in _BOOT_KEYS:
        assert key in mapped
    assert grade_leftover(mapped, ROOM_50).ok
    assert grade_leftover({}, ROOM_50).leftover == {}
    assert "missing leftover" in grade_leftover({}, ROOM_50).misses

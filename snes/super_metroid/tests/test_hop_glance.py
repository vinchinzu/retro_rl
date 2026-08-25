"""Leave-pin glance checks: room / gs / pose / xy / boss bit. No emulator."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.hop_glance import LeaveSpec, grade_final, grade_report, pose_class
from super_metroid.routes.kpdr.room_ids import (
    ROOM_HELLWAY,
    ROOM_PHANTOON,
    ROOM_RED_TOWER,
    ROOM_WS_BASEMENT,
    ROOM_WS_MAIN,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

WS_ENTRANCE_TO_MAIN = LeaveSpec(
    hop="ws_entrance_to_main",
    room=ROOM_WS_MAIN,
    x=(1000, 1100),
    y=(880, 940),
    pose_class="stand",
)

PHANTOON_LEAVE = LeaveSpec(
    hop="phantoon_loot_exit",
    room=ROOM_WS_BASEMENT,
    x=(1200, 1280),
    y=(120, 160),
    pose_class="stand",
    boss_bit=1,
)


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


HELLWAY_SILL = LeaveSpec(
    hop="red_to_hellway",
    room=ROOM_HELLWAY,
    x=(1, 80),
    y=(120, 160),
    pose_class="any",
)


def test_hellway_door_slot_fire_is_not_a_leave() -> None:
    """x=237 in Red Tower is the door-slot fire, not Hellway."""
    fire = {"room": hex(ROOM_RED_TOWER), "xy": [237, 139], "pose": 11, "gs": 8, "dt": 0, "health": 299}
    misses = grade_final(fire, HELLWAY_SILL)
    assert any(m.startswith("room ") for m in misses)
    sill = {"room": hex(ROOM_HELLWAY), "xy": [39, 139], "pose": 11, "gs": 8, "dt": 0, "health": 299}
    assert grade_final(sill, HELLWAY_SILL) == []


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

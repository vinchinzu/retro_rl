"""ROM-free glance locks for Main Shaft phase seats and hop-2 probe."""

from __future__ import annotations

from pathlib import Path

from super_metroid.hop_glance import grade_final
from super_metroid.leave_specs import (
    LEAVE_BY_HOP,
    WS_MAIN_ATTIC_SEAT,
    WS_MAIN_GRATE_SEAT,
    WS_MAIN_PHASE_SPECS,
    WS_MAIN_PIT_SHOT,
    WS_MAIN_TO_ATTIC,
    WS_MAIN_WEST_SUPER,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_WS_ATTIC,
    ROOM_WS_MAIN,
    ROOM_WS_WEST_SUPER,
)
from super_metroid.scripts.probe.ws_main_climb import phase_glance

PROBE = Path(__file__).resolve().parents[1] / "scripts" / "probe" / "ws_main_climb.py"


def _final(
    xy: tuple[int, int],
    *,
    room: str = "0xCAF6",
    pose: int = 3,
    gs: int = 8,
) -> dict:
    return {
        "room": room,
        "xy": [xy[0], xy[1]],
        "pose": pose,
        "gs": gs,
        "dt": 0,
        "health": 299,
    }


def test_grate_seat_glances_usable_handoff_not_observable_land() -> None:
    assert WS_MAIN_GRATE_SEAT.room == ROOM_WS_MAIN
    assert WS_MAIN_GRATE_SEAT.pose_class == "any"
    assert WS_MAIN_GRATE_SEAT.gs == 8
    assert WS_MAIN_GRATE_SEAT.x == (1216, 1232)
    assert WS_MAIN_GRATE_SEAT.y == (1852, 1868)
    assert grade_final(_final((1223, 1860)), WS_MAIN_GRATE_SEAT) == []
    assert grade_final(_final((1227, 1856)), WS_MAIN_GRATE_SEAT) == []
    assert grade_final(_final((1221, 1862), pose=4), WS_MAIN_GRATE_SEAT) == []
    land = grade_final(_final((1189, 1883), pose=2), WS_MAIN_GRATE_SEAT)
    assert any(m.startswith("x=") for m in land)
    assert any(m.startswith("y=") for m in land)
    take04 = grade_final(_final((1195, 1883)), WS_MAIN_GRATE_SEAT)
    assert any(m.startswith("x=") for m in take04)
    pocket = grade_final(_final((1177, 1883), pose=2), WS_MAIN_GRATE_SEAT)
    assert any(m.startswith("x=") for m in pocket)
    stairs = grade_final(_final((1111, 1899), pose=157), WS_MAIN_GRATE_SEAT)
    assert any(m.startswith("x=") for m in stairs)
    assert any(m.startswith("y=") for m in stairs)


def test_west_super_glances_shaft_not_side_room() -> None:
    seated = _final((1152, 1675), pose=10)
    assert grade_final(seated, WS_MAIN_WEST_SUPER) == []
    side = _final((1152, 1675), room=hex(ROOM_WS_WEST_SUPER), pose=10)
    misses = grade_final(side, WS_MAIN_WEST_SUPER)
    assert any(m.startswith("room ") for m in misses)


def test_ws_main_to_attic_still_dest_attic() -> None:
    assert WS_MAIN_TO_ATTIC.room == ROOM_WS_ATTIC
    assert WS_MAIN_TO_ATTIC.pose_class == "door"
    assert WS_MAIN_PHASE_SPECS["attic_door"] is WS_MAIN_TO_ATTIC
    seated = {
        "room": "0xCA52",
        "xy": [1135, 184],
        "pose": 1,
        "gs": 8,
        "dt": 0,
        "health": 299,
    }
    assert grade_final(seated, WS_MAIN_TO_ATTIC) == []
    leftover = _final((1177, 1883), pose=2)
    assert any(m.startswith("room ") for m in grade_final(leftover, WS_MAIN_TO_ATTIC))


def test_phase_specs_map_and_leave_by_hop_uses_spec_hop() -> None:
    assert set(WS_MAIN_PHASE_SPECS) == {
        "pit_shot",
        "grate_seat",
        "west_super",
        "mid_climb",
        "attic_seat",
        "attic_door",
    }
    assert WS_MAIN_PIT_SHOT.pose_class == "any"
    assert WS_MAIN_ATTIC_SEAT.pose_class == "stand"
    assert "grate_seat" not in LEAVE_BY_HOP
    assert LEAVE_BY_HOP["ws_main_grate_seat"] is WS_MAIN_GRATE_SEAT
    assert LEAVE_BY_HOP["ws_main_west_super"] is WS_MAIN_WEST_SUPER
    assert LEAVE_BY_HOP["ws_main_to_attic"] is WS_MAIN_TO_ATTIC


def test_phase_glance_pocket_is_red_fire_slope_is_ok() -> None:
    fire = _final((1223, 1860))
    ok, misses = phase_glance("grate_seat", fire, None)
    assert ok is True
    assert misses == []
    take04 = _final((1195, 1883))
    ok, misses = phase_glance("grate_seat", take04, None)
    assert ok is False
    assert misses
    land = _final((1189, 1883), pose=2)
    ok, misses = phase_glance("grate_seat", land, None)
    assert ok is False
    assert any(m.startswith("x=") for m in misses)
    pocket = _final((1177, 1883), pose=2)
    ok, misses = phase_glance("grate_seat", pocket, None)
    assert ok is False
    assert any(m.startswith("x=") for m in misses)
    stairs = _final((1111, 1899), pose=157)
    ok, misses = phase_glance("grate_seat", stairs, None)
    assert ok is False
    assert misses
    assert phase_glance(None, fire, None) == (False, [])
    assert phase_glance("grate_seat", fire, "boom")[0] is False


def test_probe_source_grades_phase_spec_and_writes_held_pin() -> None:
    src = PROBE.read_text(encoding="utf-8")
    assert "--stop-at" in src
    assert "WS_MAIN_PHASE_SPECS" in src
    assert "post_ws_main_grate_seat" in src
    assert "from super_metroid.routes.kpdr.wrecked_ship.ws_main_geometry import WS_MAIN_PHASES" in src
    assert "phase_glance" in src
    dual_lines = [
        line.strip()
        for line in src.splitlines()
        if "dual =" in line and "args.dual" in line
    ]
    assert dual_lines
    assert any(line == "dual = bool(args.dual) and not headed" for line in dual_lines)
    assert all("stop_at" not in line and "attic_door" not in line for line in dual_lines)
    leftover_block = src[src.index("if not ok") :]
    assert "ws_main_to_attic_leftover.state" in leftover_block
    assert "leftover.state" not in src[src.index("if hop_green") : src.index("if not ok")]
    pin_block = src[src.index("if phase_ok and phase_stop") : src.index("if not ok")]
    assert "post_ws_main_grate_seat" in src
    assert "phase_ok and phase_stop" in pin_block
    assert "TRACE_FRAMES" in src
    assert "WALL_STUCK_FRAMES" in src
    assert "planted-wall deadlock" in src
    assert "events" in src
    assert "atomics" in src

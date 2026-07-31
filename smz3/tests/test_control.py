"""Unit tests for shared control primitives (no emulator)."""

from __future__ import annotations

from smz3.control import Z3_MODULE_DEATH, is_z3_dead
from smz3.ram import ComboSnapshot
from smz3.segment import RoomVisit, SegmentResult, track_room
from smz3.world import ActiveWorld


def _snap(**overrides: object) -> ComboSnapshot:
    base = dict(
        frame=10,
        sm_game_state=8,
        sm_room_id=0x91F8,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=99,
        sm_max_health=99,
        sm_samus_x=100,
        sm_samus_y=100,
        sm_pose=1,
        z3_module=0x09,
        z3_submodule=0,
        z3_indoors=0,
        z3_room_id=0,
        z3_screen_id=0x35,
        z3_link_x=0,
        z3_link_y=0,
    )
    base.update(overrides)
    return ComboSnapshot(**base)  # type: ignore[arg-type]


def test_is_z3_dead() -> None:
    assert is_z3_dead(_snap(z3_module=Z3_MODULE_DEATH))
    assert not is_z3_dead(_snap(z3_module=0x09))


def test_track_room_transitions() -> None:
    visits: list[RoomVisit] = []
    track_room(visits, _snap(sm_room_id=0x91F8, frame=1), ActiveWorld.SUPER_METROID)
    track_room(visits, _snap(sm_room_id=0x91F8, frame=5), ActiveWorld.SUPER_METROID)
    track_room(visits, _snap(sm_room_id=0x92FD, frame=20), ActiveWorld.SUPER_METROID)
    assert len(visits) == 2
    assert visits[0].room_id == 0x91F8
    assert visits[0].leave_frame == 20
    assert visits[1].room_id == 0x92FD


def test_segment_result_dict() -> None:
    r = SegmentResult(ok=True, goal="x", frames=3, detail="ok")
    d = r.to_dict()
    assert d["ok"] is True
    assert d["goal"] == "x"
    assert d["frames"] == 3

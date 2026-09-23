"""Gathering pin runner. No emulator."""

from __future__ import annotations

from zelda_i.overworld.gather_run import (
    entry_health_byte,
    gather_glance,
    leave_payload,
    written_leave,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


class _Ctrl:
    def __init__(self, success: bool) -> None:
        self.success = success


def _snap() -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=0x7B,
        next_screen=0x7B,
        link_x=120,
        link_y=80,
        facing=8,
        sword=1,
        bombs=4,
        rupees=20,
        keys=0,
        health=0x22,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
        candle=1,
    )


def test_bfs_heart_byte_clamps_without_adding_a_container() -> None:
    """0x2F is the byte on the BFS pins: 15 hearts in 3 containers."""
    assert entry_health_byte(0x2F) == 0x22
    assert entry_health_byte(0x22) is None
    assert entry_health_byte(0x20) is None


def test_gather_glance_carries_the_leave_fields() -> None:
    glance = gather_glance(_snap())
    assert glance["screen_hex"] == "0x7B"
    assert glance["sword"] == 1
    assert glance["candle"] == 1
    assert glance["bombs"] == 4
    assert glance["rupees"] == 20
    assert glance["whole_hearts"] == 3
    assert glance["heart_containers"] == 3
    assert glance["mode"] == PLAY_MODE


def test_red_leave_can_still_save_a_pose() -> None:
    path = "/tmp/gather_leave_red.state"
    payload = leave_payload(_Ctrl(False), {"screen_hex": "0x7B"}, 42, "leave", path)
    assert payload["ok"] is False
    assert payload["saved"] == path


def test_success_leave_is_ok() -> None:
    payload = leave_payload(
        _Ctrl(True), {"screen_hex": "0x7B"}, 10, "leave", "/tmp/gather_leave_ok.state"
    )
    assert payload["ok"] is True
    assert payload["saved"] == "/tmp/gather_leave_ok.state"


def test_heart_leave_is_not_written_when_containers_stay() -> None:
    assert written_leave("GatherHeartL8Leave", False, "/tmp/x.state", save_red=False) is None
    assert written_leave("leave", False, "/tmp/x.state", save_red=True) == "/tmp/x.state"


def test_no_save_as_means_saved_is_none() -> None:
    payload = leave_payload(
        _Ctrl(True), {"screen_hex": "0x7B"}, 10, None, "/tmp/unused.state"
    )
    assert payload["ok"] is True
    assert payload["saved"] is None

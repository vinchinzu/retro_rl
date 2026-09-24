"""Gathering pin runner. No emulator."""

from __future__ import annotations

from zelda_i.overworld.gather_run import gather_glance
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot



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



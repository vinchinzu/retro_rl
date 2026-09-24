"""L6 0x19 -> 0x09: the walk north never parks on the south rows."""

from __future__ import annotations

from zelda_i.level6.room19 import NORTH_09_SOUTH_Y, make_room09_controller
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


def _snap(x: int, y: int) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=PLAY_MODE, level=6, screen=0x19, next_screen=0x19, link_x=x, link_y=y,
        facing=8, sword=2, bombs=8, rupees=0, keys=7, health=0x85, triforce=0x1F,
        compass=0, dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=1,
        room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0, objects=(),
    )


def test_knocked_onto_the_south_rows_walks_back_north() -> None:
    """Last-heart run 30: knocked to (75, 189), the south-row halt idled
    until the 4000-frame timeout. Idling keeps Link there; he must leave
    northward (never DOWN into the south key door)."""
    from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX

    for x in (75, 120, 132):
        ctl = make_room09_controller()
        act = ctl.step(_snap(x, NORTH_09_SOUTH_Y + 8))
        held = [d for d in ("UP", "DOWN", "LEFT", "RIGHT") if act.action[NES_BUTTON_NAME_TO_INDEX[d]]]
        assert held and "DOWN" not in held, (x, held)

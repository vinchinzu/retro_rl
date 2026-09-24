"""L4 0x60 stepladder cellar: the stage ends on the item, not on a pose."""

from __future__ import annotations

from zelda_i.level4.stepladder import StepladderPhase, make_stepladder_controller
from zelda_i.ram import ZeldaSnapshot


def _snap(x: int, y: int, *, ladder: int) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=9, level=4, screen=0x60, next_screen=0x60, link_x=x, link_y=y, facing=2,
        sword=2, bombs=15, rupees=82, keys=6, health=0x85, triforce=7, compass=0,
        dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0, room_obj_count=0,
        cur_opened_doors=0, open_doorway_mask=0, objects=(), ladder=ladder,
    )


def test_hunt_ends_when_the_ladder_byte_is_set_wherever_link_stands() -> None:
    """Last-heart run 29: the pickup landed ($ladder=1, Link halted $40 with
    the item up) while a hunt swing had moved him to (132, 141); the phase
    only counted frames on (136, 141), pressed RIGHT into the hold for 96
    frames and failed ``hunt_solid_132_141``."""
    ctrl = make_stepladder_controller(clear_first=False)
    ctrl.phase = StepladderPhase.HUNT
    ctrl.step(_snap(132, 141, ladder=1))
    assert ctrl.success

    ctrl = make_stepladder_controller(clear_first=False)
    ctrl.phase = StepladderPhase.HUNT
    ctrl.step(_snap(132, 141, ladder=0))
    assert not ctrl.success

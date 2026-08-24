"""Unit locks for Bat→Red dry-pipe traverse + water lip climb."""

from types import SimpleNamespace

from super_metroid.routes.kpdr.k5 import play_bat_to_red
from super_metroid.routes.kpdr.k5.bat_to_red import (
    _in_water,
    _is_crouch,
    _is_knockback,
    _is_morph_like,
    _on_left_door_seat,
    _traverse_buttons,
    _water_direction,
)
from super_metroid.routes.kpdr.k5.geometry import (
    BAT_TO_RED_DOOR_SEAT_X,
    BAT_TO_RED_HIGH_Y,
    BAT_TO_RED_JUMP_HOLD,
    BAT_TO_RED_RUNUP,
)
from super_metroid.routes.kpdr.rooms import ROOM_BAT, ROOM_RED_TOWER


def _state(**kwargs: object) -> SimpleNamespace:
    base = dict(room_id=ROOM_BAT, samus_x=472, samus_y=139, velocity_y=0, pose=12)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_bat_to_red_exports() -> None:
    assert ROOM_BAT == 0xA3DD
    assert ROOM_RED_TOWER == 0xA253
    assert callable(play_bat_to_red)


def test_bat_to_red_is_registered() -> None:
    from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS

    assert KPDR_SEGMENTS["bat_to_red"] is play_bat_to_red


def test_left_door_seat_is_high_left_not_morph_lip() -> None:
    assert BAT_TO_RED_DOOR_SEAT_X <= 80
    assert BAT_TO_RED_HIGH_Y <= 165
    assert _on_left_door_seat(_state(samus_x=40, samus_y=139, pose=12)) is True
    assert _on_left_door_seat(_state(samus_x=376, samus_y=189, pose=42)) is False
    assert _on_left_door_seat(_state(samus_x=411, samus_y=190, pose=42)) is False
    assert _on_left_door_seat(_state(samus_x=472, samus_y=139, pose=12)) is False


def test_sill_pose_12_is_crouch_not_morph() -> None:
    assert _is_crouch(12) is True
    assert _is_crouch(11) is True
    assert _is_morph_like(12) is False
    assert _is_morph_like(42) is True
    assert _is_morph_like(27) is False


def test_water_is_below_dry_pipe_tops() -> None:
    assert _in_water(_state(samus_x=293, samus_y=193, pose=26)) is True
    assert _in_water(_state(samus_x=376, samus_y=189, pose=1)) is True
    assert _in_water(_state(samus_x=472, samus_y=139, pose=12)) is False
    assert _water_direction(_state(samus_x=293, samus_y=193)) == "LEFT"
    assert _water_direction(_state(samus_x=24, samus_y=190)) == "RIGHT"


def test_high_path_runup_then_periodic_jump() -> None:
    high = _state(samus_x=300, samus_y=139, pose=2)
    assert _traverse_buttons(0, high) == ["LEFT", "B", "X"]
    jumping = _traverse_buttons(BAT_TO_RED_RUNUP, high)
    assert jumping[:3] == ["LEFT", "B", "X"]
    assert "A" in jumping
    running = _traverse_buttons(BAT_TO_RED_JUMP_HOLD, high)
    assert running == ["LEFT", "B", "X"]


def test_door_seat_knockback_is_not_morph() -> None:
    kb = _state(samus_x=37, samus_y=155, pose=138)
    assert _on_left_door_seat(kb) is True
    assert _is_knockback(kb) is True
    assert _is_morph_like(138) is False
    assert _is_crouch(138) is False

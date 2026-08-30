"""K5 geometry predicates (not hop-file identity locks)."""

from __future__ import annotations

from types import SimpleNamespace

from super_metroid.routes.kpdr.red_tower.bat_to_red import (
    _in_water,
    _is_crouch,
    _is_knockback,
    _is_morph_like,
    _on_left_door_seat,
    _traverse_buttons,
    _water_direction,
)
from super_metroid.routes.kpdr.red_tower.below_to_bat import _on_bat_right_sill
from super_metroid.routes.kpdr.red_tower.caterpillar_to_alpha_pb import _entry_shelf_dir
from super_metroid.routes.kpdr.red_tower.geometry import (
    BAT_SILL_X_MAX,
    BAT_SILL_X_MIN,
    BAT_SILL_Y_MAX,
    BAT_SILL_Y_MIN,
    BAT_TO_RED_DOOR_SEAT_X,
    BAT_TO_RED_HIGH_Y,
    BAT_TO_RED_JUMP_HOLD,
    BAT_TO_RED_RUNUP,
)
from super_metroid.routes.kpdr.red_tower.hellway_to_caterpillar import _in_right_door_band
from super_metroid.routes.kpdr.rooms import ROOM_BAT


def _state(**kwargs: object) -> SimpleNamespace:
    base = dict(room_id=ROOM_BAT, samus_x=472, samus_y=139, velocity_y=0, pose=12)
    base.update(kwargs)
    return SimpleNamespace(**base)


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


def test_bat_sill_window_covers_dual_green_pin() -> None:
    assert BAT_SILL_X_MIN <= 472 <= BAT_SILL_X_MAX
    assert BAT_SILL_Y_MIN <= 139 <= BAT_SILL_Y_MAX


def test_on_bat_right_sill_rejects_mid_platform_morph() -> None:
    ok = SimpleNamespace(samus_x=472, samus_y=139, velocity_y=0, pose=12)
    mid_morph = SimpleNamespace(samus_x=411, samus_y=190, velocity_y=0, pose=42)
    assert _on_bat_right_sill(ok) is True
    assert _on_bat_right_sill(mid_morph) is False


def test_entry_shelf_recenters_off_the_right_ledge() -> None:
    """Compose Cacatac knockback lands ~(155,1389); walk back, do not mash A."""
    assert _entry_shelf_dir(39) == "RIGHT"
    assert _entry_shelf_dir(77) == "RIGHT"
    assert _entry_shelf_dir(90) is None
    assert _entry_shelf_dir(100) is None
    assert _entry_shelf_dir(101) == "LEFT"
    assert _entry_shelf_dir(110) == "LEFT"
    assert _entry_shelf_dir(155) == "LEFT"


def test_hellway_right_door_band_rejects_door_slot_underflow() -> None:
    """x=65522 is the Red Tower slot wrap, not Caterpillar's door."""
    assert _in_right_door_band(700)
    assert not _in_right_door_band(39)
    assert not _in_right_door_band(237)
    assert not _in_right_door_band(690)
    assert not _in_right_door_band(65522)

"""Unit tests for KPDR Charge Plus Missiles Phantoon (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.phantoon import (
    ROOM_PHANTOON,
    SEAT_X,
    WEAPON_MISSILES,
    _go_to_seat,
)
from super_metroid.combat.phantoon_charge_missiles import (
    MISSILE_SPACING,
    MISSILES_PER_BARRAGE,
    ROUND_RECIPE,
    SUPER_KILL_HP,
    ChargeMissilesStrategy,
    fight_charge_missiles_action,
    go_to_seat,
    round_recipe,
    seated,
    should_fire_super,
)
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PHANTOON,
        "samus_x": SEAT_X,
        "samus_y": 187,
        "pose": 1,
        "facing": 8,
        "enemy0_x": 120,
        "enemy0_y": 108,
        "enemy0_hp": 2500,
        "enemy0_spritemap": 0xDEF1,
        "missiles": 20,
        "max_missiles": 20,
        "super_missiles": 5,
        "selected_item": WEAPON_MISSILES,
        "num_enemies": 4,
        "health": 299,
        "max_health": 299,
    }
    values.update(overrides)
    return replace(base, **values)


def test_recipe_is_two_two_charge() -> None:
    assert ROUND_RECIPE == ("missiles", "missiles", "charge")
    assert MISSILES_PER_BARRAGE == 2
    assert MISSILE_SPACING == 10
    assert round_recipe() == ("missiles", "missiles", "charge")
    assert round_recipe(allow_super=False, hp=400) == ("missiles", "missiles", "charge")
    assert round_recipe(allow_super=True, hp=400) == ("missiles", "missiles", "super")
    assert round_recipe(allow_super=True, hp=2500) == ("missiles", "missiles", "charge")


def test_never_super_when_hp_above_600() -> None:
    assert SUPER_KILL_HP == 600
    assert not should_fire_super(601, allow_super=True)
    assert not should_fire_super(2500, allow_super=True)
    assert not should_fire_super(600, allow_super=False)
    assert not should_fire_super(0, allow_super=True)
    assert should_fire_super(600, allow_super=True)
    assert should_fire_super(400, allow_super=True)
    armed = ChargeMissilesStrategy(allow_super=True)
    assert "X" not in fight_charge_missiles_action(
        _state(enemy0_hp=601), 0, armed, round_step="super"
    )
    assert "X" in fight_charge_missiles_action(
        _state(enemy0_hp=400), 0, armed, round_step="super"
    )


def test_zero_missiles_does_not_fire() -> None:
    state = _state(missiles=0)
    action = fight_charge_missiles_action(state, 0, round_step="missiles")
    assert "X" not in action
    assert fight_charge_missiles_action(_state(), 0, round_step="missiles") == ("X",)


def test_seat_helper_is_left_corner() -> None:
    assert go_to_seat is _go_to_seat
    assert seated(_state())
    assert not seated(_state(pose=29))
    assert not seated(_state(samus_x=90))
    assert "LEFT" in fight_charge_missiles_action(_state(samus_x=120, pose=1), 0)
    assert fight_charge_missiles_action(_state(enemy0_hp=0), 0) == ()

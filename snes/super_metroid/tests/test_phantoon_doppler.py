"""Unit tests for wiki missile-doppler Phantoon (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.phantoon import ROOM_PHANTOON, SEAT_X, WEAPON_MISSILES
from super_metroid.combat.phantoon_doppler import (
    DOPPLER_SPACING,
    GAP_FRAMES,
    MAX_DOPPLER_EXTRA,
    PAIR_SIZE,
    PAIR_WAIT_FRAMES,
    SUPER_KILL_HP,
    barrage_phase,
    fight_phantoon_doppler_action,
    missile_spacing_ok,
    should_fire_missile,
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
        "max_super_missiles": 5,
        "selected_item": WEAPON_MISSILES,
        "num_enemies": 4,
        "health": 299,
        "max_health": 299,
    }
    values.update(overrides)
    return replace(base, **values)


def test_doppler_spacing_is_10f() -> None:
    assert DOPPLER_SPACING == 10
    assert not missile_spacing_ok(9)
    assert missile_spacing_ok(10)
    assert missile_spacing_ok(11)
    assert "X" not in fight_phantoon_doppler_action(_state(), frames_since_fire=9)
    assert "X" in fight_phantoon_doppler_action(_state(), frames_since_fire=10)


def test_super_gated_on_hp_le_600() -> None:
    assert SUPER_KILL_HP == 600
    assert should_fire_super(600, 5)
    assert should_fire_super(1, 1)
    assert not should_fire_super(601, 5)
    assert not should_fire_super(0, 5)
    assert not should_fire_super(100, 0)
    assert "X" in fight_phantoon_doppler_action(_state(enemy0_hp=600, missiles=0, super_missiles=5))
    assert "X" not in fight_phantoon_doppler_action(_state(enemy0_hp=601, missiles=0, super_missiles=5))


def test_zero_ammo_does_not_fire() -> None:
    assert not should_fire_missile(0, hittable=True, frames_since_fire=99)
    action = fight_phantoon_doppler_action(_state(missiles=0, enemy0_hp=2500, super_missiles=0))
    assert "X" not in action
    assert "X" not in fight_phantoon_doppler_action(
        _state(missiles=0, super_missiles=0, enemy0_hp=400)
    )


def test_barrage_recipe_2_2_n() -> None:
    assert PAIR_SIZE == 2
    assert MAX_DOPPLER_EXTRA == 6
    assert barrage_phase(0, 0, 0, 0) == "pair1"
    assert barrage_phase(1, 0, 0, 0) == "pair1"
    assert barrage_phase(2, 0, 0, 0) == "wait1"
    assert barrage_phase(2, 0, 0, PAIR_WAIT_FRAMES - 1) == "wait1"
    assert barrage_phase(2, 0, 0, PAIR_WAIT_FRAMES) == "pair2"
    assert barrage_phase(2, 1, 0, PAIR_WAIT_FRAMES) == "pair2"
    assert barrage_phase(2, 2, 0, PAIR_WAIT_FRAMES, 0) == "gap"
    assert barrage_phase(2, 2, 0, PAIR_WAIT_FRAMES, GAP_FRAMES - 1) == "gap"
    assert barrage_phase(2, 2, 0, PAIR_WAIT_FRAMES, GAP_FRAMES) == "doppler"
    assert barrage_phase(2, 2, 5, PAIR_WAIT_FRAMES, GAP_FRAMES) == "doppler"
    assert barrage_phase(2, 2, 6, PAIR_WAIT_FRAMES, GAP_FRAMES) == "done"
    wait = fight_phantoon_doppler_action(_state(), pair1=2, wait_frames=0)
    assert "X" not in wait
    pair2 = fight_phantoon_doppler_action(
        _state(), pair1=2, wait_frames=PAIR_WAIT_FRAMES, pair2=0
    )
    assert "X" in pair2
    gap = fight_phantoon_doppler_action(
        _state(), pair1=2, pair2=2, wait_frames=PAIR_WAIT_FRAMES, gap_frames=0
    )
    assert "X" not in gap
    extra = fight_phantoon_doppler_action(
        _state(), pair1=2, pair2=2, wait_frames=PAIR_WAIT_FRAMES, gap_frames=GAP_FRAMES
    )
    assert "X" in extra
    done = fight_phantoon_doppler_action(
        _state(), pair1=2, pair2=2, extra=MAX_DOPPLER_EXTRA,
        wait_frames=PAIR_WAIT_FRAMES, gap_frames=GAP_FRAMES,
    )
    assert "X" not in done

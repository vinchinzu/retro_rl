"""Unit tests for Phantoon seat/window helpers and hop wiring (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.combat.features import phantoon_catalog
from super_metroid.combat.phantoon import (
    ADDR_WS_BOSS_BITS,
    PHANTOON_BOSS_BIT,
    ROOM_PHANTOON,
    SEAT_X,
    VULNERABLE_SPRITEMAPS,
    WEAPON_MISSILES,
    eye_ilist_open,
    eye_open,
    func_vulnerable,
    charge_window_ok,
    rain_charge_ok,
    rain_phase,
    rain_vulnerable,
    right_park,
    seated,
)
from super_metroid.combat.protocol import wrap_phantoon_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.routes.kpdr.wrecked_ship.phantoon_fight import (
    phantoon_boss_bit_set,
    play_phantoon_room_fight,
    require_phantoon_defeated,
)


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
        "enemy0_spritemap": 0xDEDD,
        "missiles": 20,
        "max_missiles": 20,
        "selected_item": WEAPON_MISSILES,
        "num_enemies": 4,
        "health": 299,
        "max_health": 299,
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    """Small deterministic session double for hop wiring."""

    def __init__(self, state, *, hp_after_step=None, set_boss_bit=False):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.hp_after_step = hp_after_step
        self.set_boss_bit = set_boss_bit
        self.env = None

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        updates = {"frame": self.frame}
        if self.hp_after_step is not None and len(self.actions) >= 8:
            updates["enemy0_hp"] = self.hp_after_step
        hp_now = updates.get("enemy0_hp", self.state.enemy0_hp)
        if self.set_boss_bit and hp_now == 0:
            bits = list(self.state.boss_bits)
            bits[3] |= 0x01
            updates["boss_bits"] = tuple(bits)
        self.state = replace(self.state, **updates)
        return self.state


def test_seated_is_standing_left_corner() -> None:
    assert seated(_state())
    assert not seated(_state(pose=29))
    assert not seated(_state(pose=81))
    assert not seated(_state(samus_x=90))
    assert not seated(_state(samus_y=120, pose=1))


def test_eye_open_uses_measured_body_spritemaps() -> None:
    assert 0xDEF1 in VULNERABLE_SPRITEMAPS
    assert eye_open(_state(enemy0_spritemap=0xDEF1))
    assert not eye_open(_state(enemy0_spritemap=0xDEDD))
    assert eye_ilist_open(0xCC53)
    assert eye_ilist_open(0xCCA1)
    assert not eye_ilist_open(0xCC7F)
    assert func_vulnerable(0xD60D)
    assert not func_vulnerable(0xD5E7)


def test_rain_vulnerable_is_d767_d788_not_figure8() -> None:
    assert rain_vulnerable(0xD788)
    assert rain_vulnerable(0xD767)
    assert not rain_vulnerable(0xD60D)
    assert not rain_vulnerable(0xD5E7)


def test_charge_window_ok_skips_rain_and_right_fig8() -> None:
    assert charge_window_ok(0xD60D, 120)
    assert charge_window_ok(0xD4A8, 120)
    assert not charge_window_ok(0xD60D, 203)
    assert not charge_window_ok(0xD788, 128)
    assert not charge_window_ok(0xD767, 128)
    assert not charge_window_ok(0xD82A, 203)
    assert not charge_window_ok(0xD788, 88)
    assert charge_window_ok(0xD767, 48, 96)
    assert not charge_window_ok(0xD767, 56, 113)
    assert not charge_window_ok(0xD788, 168)
    assert not charge_window_ok(0xD4A8, 53, 82)
    assert not charge_window_ok(0xD60D, 53, 82)
    assert not charge_window_ok(0xD4A8, 83, 64)


def test_rain_charge_ok_is_48_96_not_56_113() -> None:
    assert rain_charge_ok(48, 96)
    assert rain_charge_ok(56, 96)
    assert not rain_charge_ok(56, 113)
    assert not rain_charge_ok(48, 113)
    assert not rain_charge_ok(88, 96)
    assert not rain_charge_ok(128, 96)
    assert not rain_charge_ok(53, 82)


def test_rain_phase_is_cycle_not_fig8() -> None:
    assert rain_phase(0xD82A)
    assert rain_phase(0xD73F)
    assert rain_phase(0xD767)
    assert rain_phase(0xD788)
    assert rain_phase(0xD7D5)
    assert rain_phase(0xD7F7)
    assert not rain_phase(0xD60D)
    assert not rain_phase(0xD5E7)
    assert not rain_phase(0xD4A8)


def test_right_park() -> None:
    assert right_park(203)
    assert not right_park(120)


def test_phantoon_boss_bit_is_wrecked_ship_d82b() -> None:
    """Low WRAM parse never contains $D82B; fight peeks bank 7E like Kraid."""
    assert ADDR_WS_BOSS_BITS == 0xD82B
    assert PHANTOON_BOSS_BIT == 0x01


def test_wrapper_entry_room_and_catalog() -> None:
    strategy = wrap_phantoon_as_boss_strategy()
    assert strategy.entry.room_id == ROOM_PHANTOON
    assert strategy.catalog.name == "Phantoon"
    assert phantoon_catalog().max_hp == 2500


def test_play_phantoon_room_fight_wrong_room() -> None:
    session = _Session(_state(room_id=0xCA08))
    with pytest.raises(RuntimeError, match="phantoon_room_fight"):
        play_phantoon_room_fight(session)


def test_play_phantoon_room_fight_already_defeated() -> None:
    bits = (0, 0, 0, 0x01, 0, 0, 0, 0)
    session = _Session(_state(enemy0_hp=0, boss_bits=bits))
    assert phantoon_boss_bit_set(session)
    out = play_phantoon_room_fight(session)
    assert out.room_id == ROOM_PHANTOON
    assert out.enemy0_hp == 0
    assert session.actions == []


def test_play_phantoon_room_fight_rejects_timeout(monkeypatch) -> None:
    session = _Session(_state())

    class _Timeout:
        outcome = "timeout"

    monkeypatch.setattr(
        "super_metroid.combat.phantoon_doppler.play_phantoon_doppler_fight",
        lambda *args, **kwargs: _Timeout(),
    )
    with pytest.raises(RuntimeError, match="fight failed"):
        play_phantoon_room_fight(session)


def test_require_phantoon_defeated_after_hook() -> None:
    bits = (0, 0, 0, 0x01, 0, 0, 0, 0)
    ok = _Session(_state(enemy0_hp=0, boss_bits=bits))
    require_phantoon_defeated(ok, [], None)
    with pytest.raises(RuntimeError, match=r"\$D82B"):
        require_phantoon_defeated(_Session(_state()), [], None)

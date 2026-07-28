from __future__ import annotations

import numpy as np

from metroid.ram import (
    ADDR_ENGINE_MODE,
    ADDR_EQUIPMENT,
    ADDR_GAME_MODE,
    ADDR_HEALTH_HI,
    ADDR_HEALTH_LO,
    ADDR_MAP_X,
    ADDR_MAP_Y,
    ADDR_PAUSED,
    ADDR_SAMUS_X,
    ADDR_SAMUS_Y,
    EQUIP_MORPH,
    capabilities_from_snapshot,
    is_level1_ready,
    is_missiles_obtained,
    parse_game_state,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_ENGINE_MODE] = fields.get("engine", 0)
    ram[ADDR_GAME_MODE] = fields.get("game_mode", 3)
    ram[ADDR_PAUSED] = fields.get("paused", 0)
    ram[ADDR_MAP_X] = fields.get("map_x", 3)
    ram[ADDR_MAP_Y] = fields.get("map_y", 14)
    ram[ADDR_SAMUS_X] = fields.get("x", 128)
    ram[ADDR_SAMUS_Y] = fields.get("y", 176)
    ram[ADDR_HEALTH_LO] = fields.get("health_lo", 0)
    ram[ADDR_HEALTH_HI] = fields.get("health_hi", 3)
    return ram


def test_is_level1_ready_requires_play_mode() -> None:
    assert is_level1_ready(_ram()) is True
    assert is_level1_ready(_ram(game_mode=8)) is False
    assert is_level1_ready(_ram(engine=1)) is False
    assert is_level1_ready(_ram(paused=1)) is False


def test_capabilities_from_equipment() -> None:
    snap = read_snapshot(_ram())
    # Without env, equipment stays 0
    assert "morph_ball" not in capabilities_from_snapshot(snap)
    morph = snap.__class__(
        **{**snap.__dict__, "equipment": EQUIP_MORPH, "missile_capacity": 5}
    )
    caps = capabilities_from_snapshot(morph)
    assert "morph_ball" in caps
    assert "missiles" in caps


def test_parse_game_state_extras() -> None:
    state = parse_game_state(_ram(), frame=10)
    assert state.extras["map_x"] == 3
    assert state.extras["map_y"] == 14


def test_is_missiles_obtained_uses_capacity(monkeypatch) -> None:
    class _Env:
        pass

    env = _Env()
    monkeypatch.setattr(
        "metroid.ram.read_missile_capacity",
        lambda _env: 0,
    )
    assert is_missiles_obtained(env) is False
    monkeypatch.setattr(
        "metroid.ram.read_missile_capacity",
        lambda _env: 5,
    )
    assert is_missiles_obtained(env) is True

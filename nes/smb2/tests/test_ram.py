"""Offline RAM-gate tests for SMB2 1-1 start/control."""

from __future__ import annotations

import numpy as np

from smb2.ram import (
    ADDR_AREA,
    ADDR_CHARACTER,
    ADDR_HEARTS,
    ADDR_JUMP_PHYSICS,
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_SUBAREA,
    ADDR_WORLD,
    HEARTS_TWO,
    SPAWN_X,
    SPAWN_Y,
    is_level1_control,
    is_level1_start,
    parse_game_state,
    read_snapshot,
)
from retro_harness.ram_state import GameMode


def _spawn_ram(*, physics: int = 0, x: int = SPAWN_X, y: int = SPAWN_Y) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_PLAYER_X] = x
    ram[ADDR_PLAYER_Y] = y
    ram[ADDR_CHARACTER] = 2
    ram[ADDR_HEARTS] = HEARTS_TWO
    ram[ADDR_LIVES] = 3
    ram[ADDR_LEVEL] = 0
    ram[ADDR_WORLD] = 0
    ram[ADDR_AREA] = 0
    ram[ADDR_SUBAREA] = 0
    ram[ADDR_JUMP_PHYSICS] = physics
    return ram


def test_parse_game_state_boot_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
    assert state.extras["level1_control"] is False
    assert state.mode is GameMode.BOOT


def test_start_gate_opens_on_sky_spawn_without_physics() -> None:
    snap = read_snapshot(_spawn_ram(physics=0), frame=253)
    assert is_level1_start(snap) is True
    assert is_level1_control(snap) is False
    state = parse_game_state(_spawn_ram(physics=0), frame=253)
    assert state.mode is GameMode.CUTSCENE


def test_control_gate_requires_live_jump_physics_at_spawn() -> None:
    snap = read_snapshot(_spawn_ram(physics=7), frame=304)
    assert is_level1_start(snap) is True
    assert is_level1_control(snap) is True
    assert snap.abs_x == SPAWN_X
    state = parse_game_state(_spawn_ram(physics=7), frame=304)
    assert state.mode is GameMode.PLAYING
    assert state.extras["level1_control"] is True


def test_control_gate_rejects_later_room_and_title() -> None:
    later = _spawn_ram(physics=7)
    later[ADDR_SUBAREA] = 2
    assert is_level1_control(read_snapshot(later)) is False

    title = np.zeros(0x800, dtype=np.uint8)
    title[ADDR_LIVES] = 3
    title[ADDR_CHARACTER] = 2
    assert is_level1_start(read_snapshot(title)) is False
    assert is_level1_control(read_snapshot(title)) is False

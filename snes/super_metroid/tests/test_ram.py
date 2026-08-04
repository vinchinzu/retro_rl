from __future__ import annotations

import numpy as np

from super_metroid.ram import (
    ADDR_AREA_INDEX,
    ADDR_COLLECTED_ITEMS,
    ADDR_ENEMY0_HP,
    ADDR_ENEMY0_SPRITEMAP,
    ADDR_GAME_STATE,
    ADDR_HEALTH,
    ADDR_MAX_HEALTH,
    ADDR_MAX_MISSILES,
    ADDR_MISSILES,
    ADDR_ROOM_ID,
    ADDR_SAMUS_X,
    ADDR_SAMUS_Y,
    GameplayPhase,
    BOMBS_MASK,
    HI_JUMP_MASK,
    MORPH_BALL_MASK,
    VARIA_MASK,
    parse_state,
    phase_for_game_state,
)


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def test_parse_progress_state() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    for address, value in (
        (ADDR_GAME_STATE, 8),
        (ADDR_AREA_INDEX, 1),
        (ADDR_ROOM_ID, 0x9E9F),
        (ADDR_SAMUS_X, 1800),
        (ADDR_SAMUS_Y, 651),
        (ADDR_HEALTH, 99),
        (ADDR_MAX_HEALTH, 99),
        (ADDR_MISSILES, 3),
        (ADDR_MAX_MISSILES, 5),
        (ADDR_COLLECTED_ITEMS, MORPH_BALL_MASK),
        (ADDR_ENEMY0_HP, 800),
        (ADDR_ENEMY0_SPRITEMAP, 0xEEAF),
    ):
        _put_u16(ram, address, value)

    state = parse_state(ram, frame=42)

    assert state.phase is GameplayPhase.ORDINARY_GAMEPLAY
    assert state.room_id == 0x9E9F
    assert state.area_name == "Brinstar"
    assert state.morph_ball
    assert not state.bombs
    assert state.enemy0_hp == 800
    assert state.enemy0_spritemap == 0xEEAF
    assert state.progress_vector()[:5] == (0x9E9F, 1, 8, 0, MORPH_BALL_MASK)


def test_bombs_mask_is_exposed_separately_from_morph_ball() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    _put_u16(ram, ADDR_COLLECTED_ITEMS, MORPH_BALL_MASK | BOMBS_MASK)

    state = parse_state(ram)

    assert state.morph_ball
    assert state.bombs


def test_hi_jump_and_varia_masks() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    _put_u16(ram, ADDR_COLLECTED_ITEMS, HI_JUMP_MASK | VARIA_MASK)

    state = parse_state(ram)

    assert state.hi_jump
    assert state.varia
    assert not state.morph_ball


def test_source_defined_game_state_phases() -> None:
    assert phase_for_game_state(8) is GameplayPhase.ORDINARY_GAMEPLAY
    assert phase_for_game_state(11) is GameplayPhase.ROOM_TRANSITION
    assert phase_for_game_state(14) is GameplayPhase.PAUSE_OR_INVENTORY
    assert phase_for_game_state(19) is GameplayPhase.DEATH_OR_GAME_OVER
    assert phase_for_game_state(39) is GameplayPhase.ENDING_OR_CREDITS

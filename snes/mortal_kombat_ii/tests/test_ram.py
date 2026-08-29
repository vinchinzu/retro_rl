"""ROM-free tests for MK2 high-WRAM health and match-win gating."""

from __future__ import annotations

from pathlib import Path

from mortal_kombat_ii.paths import GAME_DIR
from mortal_kombat_ii.ram import (
    ADDR_P1_HEALTH,
    ADDR_P2_HEALTH,
    DECOY_NOT_HEALTH,
    GETRAM_OFFSET,
    MAX_HEALTH,
    WRAM_P1_HEALTH,
    WRAM_P2_HEALTH,
    is_match_lost,
    is_match_won,
    make_test_ram,
    parse_ram,
)


def test_getram_indices_are_wram_plus_offset() -> None:
    assert ADDR_P1_HEALTH == WRAM_P1_HEALTH + GETRAM_OFFSET == 0x4EFD
    assert ADDR_P2_HEALTH == WRAM_P2_HEALTH + GETRAM_OFFSET == 0x50AB
    assert DECOY_NOT_HEALTH == (0x020A, 0x020E)
    assert ADDR_P1_HEALTH not in DECOY_NOT_HEALTH
    assert ADDR_P2_HEALTH not in DECOY_NOT_HEALTH


def test_cheat_extractor_uses_same_health_indices() -> None:
    text = (GAME_DIR / "cheat_extractor.py").read_text(encoding="utf-8")
    assert "P1_HEALTH_GETRAM_ADDR = 0x4EFD" in text
    assert "P2_HEALTH_GETRAM_ADDR = 0x50AB" in text
    assert "0x7E2EFC" in text
    assert "0x7E30AA" in text


def test_parse_ram_ignores_low_wram_decoys() -> None:
    ram = make_test_ram(p1_health=80, p2_health=40, decoy_020a=161, decoy_020e=161)
    snap = parse_ram(ram)
    assert snap.p1_health == 80
    assert snap.p2_health == 40
    assert ram[DECOY_NOT_HEALTH[0]] == MAX_HEALTH
    assert ram[DECOY_NOT_HEALTH[1]] == MAX_HEALTH


def test_full_health_fight_ready_buffer() -> None:
    snap = parse_ram(make_test_ram())
    assert snap.p1_health == MAX_HEALTH
    assert snap.p2_health == MAX_HEALTH
    assert snap.ram_len > ADDR_P2_HEALTH


def test_match_win_requires_strict_majority() -> None:
    assert is_match_won(2, 0)
    assert is_match_won(2, 1)
    assert is_match_won(3, 2)
    assert not is_match_won(2, 2)
    assert not is_match_won(1, 0)
    assert not is_match_won(0, 2)
    assert is_match_lost(0, 2)
    assert is_match_lost(1, 2)
    assert not is_match_lost(2, 2)

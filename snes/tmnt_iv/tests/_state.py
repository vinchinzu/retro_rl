"""Shared GameState / RAM builders for TMNT IV tests."""

from __future__ import annotations

import numpy as np
from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.ram import (
    ADDR_LIVES,
    ADDR_MENU,
    ENEMY_BASES,
    PLAYER_BASE,
    MenuId,
    OFF_CHAR,
    OFF_HP,
    OFF_X,
    OFF_Y,
    write_u16le,
)

A = 8
B = 0
Y = 1


def playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    health: int = 80,
    lives: int = 2,
    camera_x: int = 0,
    frame: int = 1,
    stage: int = 0,
    boss_active: bool = False,
    extras: dict[str, object] | None = None,
) -> GameState:
    """Build a playing state with the fields tests vary."""
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        stage=stage,
        camera_x=camera_x,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=lives,
        enemies=enemies,
        screen_locked=bool(enemies),
        boss_active=boss_active,
        extras=extras or {},
    )


def enemy(
    x: int,
    y: int,
    health: int = 16,
    *,
    slot: int = 0,
    kind: int = 0,
    animation: int = 0x43,
) -> EnemyState:
    """Build a combat-slot enemy."""
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=health,
        active=True,
        animation=animation,
        kind=kind,
    )


def ram() -> np.ndarray:
    """Initialized TMNT IV WRAM for parser tests."""
    buf = np.zeros(0x20000, dtype=np.uint8)
    buf[ADDR_MENU] = MenuId.PLAYING
    buf[ADDR_LIVES] = 2
    buf[PLAYER_BASE + OFF_HP] = 80
    write_u16le(buf, PLAYER_BASE + OFF_X, 64)
    write_u16le(buf, PLAYER_BASE + OFF_Y, 160)
    return buf


def write_enemy(
    buf: np.ndarray,
    slot: int = 0,
    *,
    x: int,
    y: int,
    health: int,
    char_id: int = 0x60,
) -> None:
    """Write combat-relevant bytes for one enemy slot."""
    base = ENEMY_BASES[slot]
    write_u16le(buf, base + OFF_X, x)
    write_u16le(buf, base + OFF_Y, y)
    buf[base + OFF_HP] = health
    buf[base + OFF_CHAR] = char_id

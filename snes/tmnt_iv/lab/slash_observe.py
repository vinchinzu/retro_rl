"""Slash slot RAM helpers shared by KEEP trace and vuln map."""

from __future__ import annotations

from typing import Any

from tmnt_iv.ram import (
    ENEMY_BASES,
    OFF_CHAR,
    OFF_HP,
    OFF_X,
    OFF_Y,
    read_u8,
    read_u16le,
)
from tmnt_iv.tactics.slash import SLASH_CHAR


def side(player_x: int, slash_x: int) -> str:
    """Player vs Slash: left / right / overlap."""
    if player_x < slash_x:
        return "left"
    if player_x > slash_x:
        return "right"
    return "overlap"


def read_slash(ram: Any) -> tuple[int, int, int, int, int] | None:
    """Return (x, y, hp, status, char_id) for living Slash, else None."""
    for base in ENEMY_BASES:
        char_id = read_u8(ram, base + OFF_CHAR)
        if char_id != SLASH_CHAR:
            continue
        hp_raw = read_u8(ram, base + OFF_HP)
        if hp_raw == 0 or hp_raw > 0xC0:
            continue
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if not (0 < x < 512 and 0 < y < 256):
            continue
        status = read_u8(ram, base)  # EnemyState.animation = entity base+0
        return x, y, hp_raw, status, char_id
    return None

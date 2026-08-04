"""RAM fields for TMNT II (NES)."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import EnemyState, GameMode, GameState

# Confirmed via boot + combat probes (2026-07-27).
ADDR_LIVES = 0x004D
ADDR_HEALTH = 0x0568
ADDR_SCORE = 0x03F0  # low byte; HUD PTS (increments per kill)

# NES OAM mirror: y, tile, attr, x per sprite (64 sprites).
OAM_BASE = 0x0200
OAM_STRIDE = 4


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one WRAM byte."""
    return int(ram[address])


def player_screen_x(ram: np.ndarray) -> int:
    """Best-effort Leo screen X from OAM sprites in the play Y band."""
    xs: list[int] = []
    for i in range(40):
        base = OAM_BASE + i * OAM_STRIDE
        y = read_u8(ram, base)
        x = read_u8(ram, base + 3)
        if 100 <= y <= 175 and 8 <= x <= 248:
            xs.append(x)
    return max(xs) if xs else read_u8(ram, OAM_BASE + 3)


def player_screen_y(ram: np.ndarray) -> int:
    """Best-effort Leo screen Y from OAM (median of band sprites)."""
    ys: list[int] = []
    for i in range(40):
        base = OAM_BASE + i * OAM_STRIDE
        y = read_u8(ram, base)
        x = read_u8(ram, base + 3)
        if 100 <= y <= 175 and 8 <= x <= 248:
            ys.append(y)
    if not ys:
        return read_u8(ram, OAM_BASE)
    ys.sort()
    return ys[len(ys) // 2]


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Stage 1 combat health/lives are live."""
    health = int(ram[ADDR_HEALTH])
    lives = int(ram[ADDR_LIVES])
    if not (0 < health < 200 and lives >= 0):
        return False
    if obs_mean is not None and obs_mean <= 30.0:
        return False
    return True


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project confirmed fields into ``GameState``."""
    ready = is_level1_ready(ram)
    health = read_u8(ram, ADDR_HEALTH)
    lives = read_u8(ram, ADDR_LIVES)
    score = read_u8(ram, ADDR_SCORE)
    sx = player_screen_x(ram)
    sy = player_screen_y(ram)
    player_dead = health == 0 or (lives == 0 and health == 0)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "score": score,
        "player_sx": sx,
        "player_sy": sy,
        "health": health,
        "lives": lives,
    }
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if ready else GameMode.MENU,
        stage=1,
        room=0,
        player_x=sx,
        player_y=sy,
        health=health,
        lives=lives,
        enemies=(),
        player_dead=player_dead,
        extras=extras,
    )

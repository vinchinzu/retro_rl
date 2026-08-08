"""RAM fields for Mega Man 2 (NES).

Addresses from Data Crystal, verified against Level1 Air Man probes
(2026-08-08). System WRAM is 0x0000–0x07FF via fceumm ``get_ram()``.
"""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

# Camera / scroll
ADDR_CAMERA_STATE = 0x001B  # 0 idle, 1 nametable scroll, 2 freeze, 0x80 vert
ADDR_CAMERA_X = 0x001F
ADDR_CAMERA_X_SCREEN = 0x0020
ADDR_CAMERA_Y = 0x0022

# Stage select (menu); leftover value may persist into play
ADDR_STAGE_CURSOR = 0x002A  # 0 Wily, 1–8 clockwise from Bubble Man

# Collision / pose
ADDR_TILE_FEET = 0x0032  # 0 air, 1 ground, 2 ladder, 3 death, …
ADDR_TILE_CENTER = 0x0033
ADDR_TILE_OVERLAP = 0x0034
ADDR_SHOOT_POSE_TIMER = 0x0036
ADDR_IS_SHOOTING = 0x003D
ADDR_INVULN_TIMER = 0x004B

# Weapons / meta
ADDR_WEAPONS = 0x009A  # bitfield of unlocked weapons / stages beaten
ADDR_E_TANKS = 0x00A7
ADDR_LIVES = 0x00A8

# Object slots: Mega Man is index 0
ADDR_PLAYER_X = 0x0460  # screen-relative X
ADDR_PLAYER_Y = 0x04A0

# Health bars
ADDR_HEALTH = 0x06C0  # full bar often 28
ADDR_BOSS_HP = 0x06C1
ADDR_ENEMY_HP_BASE = 0x06C2  # through 0x06E1

# Fall / death heuristics (Air Man open sky)
FALL_Y_THRESHOLD = 200
FULL_HEALTH = 28


def read_u8(ram: np.ndarray | bytes, address: int) -> int:
    """Read one WRAM byte."""
    return int(ram[address])


def camera_progress_x(ram) -> int:
    """Horizontal scroll progress in pixels (screen*256 + fine X)."""
    return read_u8(ram, ADDR_CAMERA_X_SCREEN) * 256 + read_u8(ram, ADDR_CAMERA_X)


def player_screen_x(ram) -> int:
    """Mega Man on-screen X."""
    return read_u8(ram, ADDR_PLAYER_X)


def player_screen_y(ram) -> int:
    """Mega Man on-screen Y."""
    return read_u8(ram, ADDR_PLAYER_Y)


def is_fallen(ram) -> bool:
    """True when Mega Man has dropped off the playfield (pit death path)."""
    return player_screen_y(ram) >= FALL_Y_THRESHOLD


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once in-stage energy is live (not title/stage select)."""
    health = read_u8(ram, ADDR_HEALTH)
    lives = read_u8(ram, ADDR_LIVES)
    if not (0 < health <= FULL_HEALTH and 0 < lives < 10):
        return False
    if obs_mean is not None and obs_mean <= 50.0:
        return False
    return True


def is_playing(ram) -> bool:
    """Controllable stage play (ready health, not fallen death pose)."""
    return is_level1_ready(ram) and not is_fallen(ram) and read_u8(ram, ADDR_HEALTH) > 0


def parse_game_state(ram: np.ndarray, frame: int = 0, obs_mean: float | None = None) -> GameState:
    """Project confirmed fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    health = read_u8(ram, ADDR_HEALTH)
    lives = read_u8(ram, ADDR_LIVES)
    sx = player_screen_x(ram)
    sy = player_screen_y(ram)
    cam_x = read_u8(ram, ADDR_CAMERA_X)
    cam_scr = read_u8(ram, ADDR_CAMERA_X_SCREEN)
    progress = camera_progress_x(ram)
    fallen = is_fallen(ram)
    # Menu zeros must not count as death; only in-stage fall / HP collapse.
    player_dead = bool(ready and (health == 0 or fallen))
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "health": health,
        "lives": lives,
        "player_sx": sx,
        "player_sy": sy,
        "camera_x": cam_x,
        "camera_x_screen": cam_scr,
        "camera_y": read_u8(ram, ADDR_CAMERA_Y),
        "camera_state": read_u8(ram, ADDR_CAMERA_STATE),
        "progress_x": progress,
        "tile_feet": read_u8(ram, ADDR_TILE_FEET),
        "invuln": read_u8(ram, ADDR_INVULN_TIMER),
        "weapons": read_u8(ram, ADDR_WEAPONS),
        "boss_hp": read_u8(ram, ADDR_BOSS_HP),
        "e_tanks": read_u8(ram, ADDR_E_TANKS),
        "fallen": fallen,
        "is_shooting": read_u8(ram, ADDR_IS_SHOOTING),
    }
    if not ready:
        mode = GameMode.MENU
    elif player_dead:
        mode = GameMode.GAME_OVER
    else:
        mode = GameMode.PLAYING
    return GameState(
        frame=frame,
        mode=mode,
        stage=1,
        room=cam_scr,
        player_x=sx,
        player_y=sy,
        health=health,
        lives=lives,
        enemies=(),
        player_dead=player_dead,
        extras=extras,
    )

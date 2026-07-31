"""Shared SMB observation builder for neuroevolution / PPO-style policies.

Builds a fixed-size float vector from NES RAM:

  [0:169]     13×13 local tile grid (1=solid, -1=enemy, 0=empty)
  [169:185]   continuous player / camera / timer state (16 floats)
  [185:210]   5 enemy slots × 5 (rel_x, rel_y, type, active, rel_vx proxy)
  total       N_SMB_INPUTS = 210

The older 189-dim layout (stub in-air, no velocities) is available via
``read_smb_inputs_legacy`` for checkpoint compatibility.
"""

from __future__ import annotations

import numpy as np

from smb.ram import (
    ADDR_ENEMY_FLAG,
    ADDR_ENEMY_X,
    ADDR_ENEMY_X_PAGE,
    ADDR_ENEMY_Y,
    ADDR_PLAYER_FACING,
    ADDR_PLAYER_SCREEN_X,
    ADDR_PLAYER_STATUS,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_X_PAGE,
    is_in_air,
    player_x as abs_player_x,
    player_x_speed,
    player_y_speed,
    screen_left_x,
    timer_value,
)

# Grid
GRID_SIZE = 13
GRID_RADIUS = 6  # cells either side of center
N_GRID = GRID_SIZE * GRID_SIZE  # 169

# Continuous state vector after the grid
# y, x_speed, y_speed, power, in_air, bias, facing, timer, screen_rel_x,
# player_screen_x, x_page_frac, grounded, oper_alive, prev_action_stub×3
N_STATE = 16

# Enemies: 5 slots × (rel_x, rel_y, type, active, pad/vx)
N_ENEMY_SLOTS = 5
N_ENEMY_FEATURES = 5
N_ENEMY = N_ENEMY_SLOTS * N_ENEMY_FEATURES  # 25

N_SMB_INPUTS = N_GRID + N_STATE + N_ENEMY  # 210

# Legacy MarI/O-style layout used by existing neuro checkpoints
N_SMB_INPUTS_LEGACY = 189


def _read_tile(ram: np.ndarray, tile_page: int, tile_x: int, tile_y: int) -> float:
    """Sample SMB column-major level tile buffer at 0x0500.

    Layout: two pages of 13 rows × 16 columns. ``tile_x`` is pixel-ish offset
    (we divide by 16 for column). ``tile_y`` is row 0..12.
    """
    if tile_y < 0 or tile_y >= 13 or tile_page < 0:
        return 0.0
    col = (tile_x // 16) & 0x0F
    # page wraps every 2 screens in the 0x0500 buffer
    page_bit = tile_page % 2
    addr = 0x0500 + page_bit * 13 * 16 + tile_y * 16 + col
    if addr >= len(ram):
        return 0.0
    return 1.0 if int(ram[addr]) != 0 else 0.0


def _fill_tile_grid(ram: np.ndarray, inputs: np.ndarray, mario_x: int, mario_y: int) -> None:
    """Write 13×13 solid/empty grid centered on Mario into inputs[0:169]."""
    x_page = mario_x // 256
    x_offset = mario_x % 256
    row0 = mario_y // 16
    idx = 0
    for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
            # Center on Mario's tile column; dx is in tiles
            tile_x = x_offset + dx * 16
            tile_page = x_page
            if tile_x < 0:
                tile_x += 256
                tile_page -= 1
            elif tile_x >= 256:
                tile_x -= 256
                tile_page += 1
            tile_y = row0 + dy
            inputs[idx] = _read_tile(ram, tile_page, tile_x, tile_y)
            idx += 1


def _mark_enemies_on_grid(
    ram: np.ndarray,
    inputs: np.ndarray,
    mario_x: int,
    mario_y: int,
) -> None:
    """Overlay active enemies as -1 on the local grid when in range."""
    for slot in range(N_ENEMY_SLOTS):
        if int(ram[ADDR_ENEMY_FLAG + slot]) == 0:
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + slot]) * 256 + int(ram[ADDR_ENEMY_X + slot])
        enemy_y = int(ram[ADDR_ENEMY_Y + slot]) + 24
        ex = (enemy_x - mario_x) // 16 + GRID_RADIUS
        ey = (enemy_y - mario_y) // 16 + GRID_RADIUS
        if 0 <= ex < GRID_SIZE and 0 <= ey < GRID_SIZE:
            inputs[ey * GRID_SIZE + ex] = -1.0


def read_smb_inputs(
    ram: np.ndarray,
    *,
    prev_action: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Build the 210-dim observation vector from SMB RAM.

    Parameters
    ----------
    ram:
        Full NES RAM dump (at least 0x800 bytes).
    prev_action:
        Optional previous button vector; first 3 components (B, null, Select
        unused) are not stored — we pack RIGHT/B/A activity into state slots
        when provided.
    """
    inputs = np.zeros(N_SMB_INPUTS, dtype=np.float32)

    mario_x = abs_player_x(ram)
    mario_y = int(ram[ADDR_PLAYER_Y])
    if mario_y == 0:
        # fallback sprite Y used by some loaders
        mario_y = int(ram[0x03B8]) + 16

    _fill_tile_grid(ram, inputs, mario_x, mario_y)
    _mark_enemies_on_grid(ram, inputs, mario_x, mario_y)

    xs = player_x_speed(ram)
    ys = player_y_speed(ram)
    in_air = 1.0 if is_in_air(ram) else 0.0
    power = min(int(ram[ADDR_PLAYER_STATUS]) / 2.0, 1.0)
    facing = int(ram[ADDR_PLAYER_FACING])
    # facing: 1=right → +1, 2=left → -1, else 0
    facing_n = 1.0 if facing == 1 else (-1.0 if facing == 2 else 0.0)
    timer = timer_value(ram) / 400.0
    cam = screen_left_x(ram)
    screen_rel = (mario_x - cam) / 256.0
    player_sx = int(ram[ADDR_PLAYER_SCREEN_X]) / 256.0

    base = N_GRID
    inputs[base + 0] = mario_y / 240.0
    inputs[base + 1] = np.clip(xs / 40.0, -1.0, 1.0)  # run max ≈ 40
    inputs[base + 2] = np.clip(ys / 64.0, -1.0, 1.0)
    inputs[base + 3] = power
    inputs[base + 4] = in_air
    inputs[base + 5] = 1.0  # bias
    inputs[base + 6] = facing_n
    inputs[base + 7] = min(timer, 1.0)
    inputs[base + 8] = np.clip(screen_rel, -1.0, 2.0)
    inputs[base + 9] = np.clip(player_sx, 0.0, 1.0)
    inputs[base + 10] = (mario_x % 256) / 256.0
    inputs[base + 11] = 1.0 - in_air  # grounded
    inputs[base + 12] = 1.0 if int(ram[0x0770]) == 1 else 0.0  # playing
    # Previous action encoding (RIGHT, B/run, A/jump) when available
    if prev_action is not None and len(prev_action) >= 9:
        inputs[base + 13] = float(prev_action[7])  # RIGHT
        inputs[base + 14] = float(prev_action[0])  # B
        inputs[base + 15] = float(prev_action[8])  # A
    else:
        inputs[base + 13] = 0.0
        inputs[base + 14] = 0.0
        inputs[base + 15] = 0.0

    # Enemy feature slots
    ebase = N_GRID + N_STATE
    for slot in range(N_ENEMY_SLOTS):
        off = ebase + slot * N_ENEMY_FEATURES
        enemy_type = int(ram[ADDR_ENEMY_FLAG + slot])
        if enemy_type == 0:
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + slot]) * 256 + int(ram[ADDR_ENEMY_X + slot])
        enemy_y = int(ram[ADDR_ENEMY_Y + slot])
        inputs[off + 0] = (enemy_x - mario_x) / 256.0
        inputs[off + 1] = (enemy_y - mario_y) / 240.0
        inputs[off + 2] = enemy_type / 64.0
        inputs[off + 3] = 1.0  # active
        # Crude relative approach rate from horizontal separation only
        inputs[off + 4] = np.clip((mario_x - enemy_x) / 256.0, -1.0, 1.0)

    return inputs


def read_smb_inputs_legacy(ram: np.ndarray) -> np.ndarray:
    """189-dim observation matching the original ``platformer_common.neuro`` layout.

    Kept for loading old neuro checkpoints. In-air is now derived correctly
    (was hard-coded 0).
    """
    inputs = np.zeros(N_SMB_INPUTS_LEGACY, dtype=np.float32)
    x_page = int(ram[ADDR_X_PAGE])
    x_offset = int(ram[ADDR_PLAYER_X])
    mario_y = int(ram[0x03B8]) + 16
    mario_x = x_page * 256 + x_offset

    grid_idx = 0
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            tile_x = x_offset + dx
            tile_page = x_page
            if tile_x < 0:
                tile_x += 256
                tile_page -= 1
            elif tile_x >= 256:
                tile_x -= 256
                tile_page += 1
            tile_y = mario_y // 16 + dy
            # Legacy used tile_x as pixel offset then //16 in the address.
            if tile_y < 0 or tile_y >= 13 or tile_page < 0:
                inputs[grid_idx] = 0.0
            else:
                addr = 0x0500 + (tile_page % 2) * 13 * 16 + tile_y * 16 + (tile_x // 16)
                if addr < len(ram):
                    inputs[grid_idx] = 1.0 if int(ram[addr]) != 0 else 0.0
                else:
                    inputs[grid_idx] = 0.0
            grid_idx += 1

    for slot in range(5):
        enemy_type = int(ram[ADDR_ENEMY_FLAG + slot])
        if enemy_type == 0:
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + slot]) * 256 + int(ram[ADDR_ENEMY_X + slot])
        enemy_y = int(ram[ADDR_ENEMY_Y + slot]) + 24
        ex = (enemy_x - mario_x) // 16 + 6
        ey = (enemy_y - mario_y) // 16 + 6
        if 0 <= ex < 13 and 0 <= ey < 13:
            inputs[ey * 13 + ex] = -1.0

    inputs[169] = mario_y / 240.0
    inputs[170] = x_offset / 256.0
    player_status = int(ram[0x000E])
    inputs[171] = min(player_status / 2.0, 1.0)
    inputs[172] = 1.0 if is_in_air(ram) else 0.0
    inputs[173] = 1.0

    for slot in range(5):
        base = 174 + slot * 3
        enemy_type = int(ram[ADDR_ENEMY_FLAG + slot])
        if enemy_type == 0:
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + slot]) * 256 + int(ram[ADDR_ENEMY_X + slot])
        enemy_y = int(ram[ADDR_ENEMY_Y + slot])
        inputs[base] = (enemy_x - mario_x) / 256.0
        inputs[base + 1] = (enemy_y - mario_y) / 240.0
        inputs[base + 2] = enemy_type / 64.0

    return inputs


def observation_slices() -> dict[str, slice]:
    """Named slices into the 210-dim vector (for CNN head / debugging)."""
    return {
        "grid": slice(0, N_GRID),
        "state": slice(N_GRID, N_GRID + N_STATE),
        "enemies": slice(N_GRID + N_STATE, N_SMB_INPUTS),
    }


def grid_as_image(inputs: np.ndarray) -> np.ndarray:
    """Reshape the grid portion to (1, 13, 13) for a tiny CNN."""
    return np.asarray(inputs[:N_GRID], dtype=np.float32).reshape(1, GRID_SIZE, GRID_SIZE)

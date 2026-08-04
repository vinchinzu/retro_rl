"""Unit tests for SMB observation builder and velocity RAM (no emulator)."""

from __future__ import annotations

import numpy as np

from smb.obs import (
    N_GRID,
    N_SMB_INPUTS,
    N_SMB_INPUTS_LEGACY,
    N_STATE,
    grid_as_image,
    observation_slices,
    read_smb_inputs,
    read_smb_inputs_legacy,
)
from smb.ram import (
    ADDR_PLAYER_STATE,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_X_PAGE,
    ADDR_X_SPEED,
    ADDR_Y_SPEED,
    is_grounded,
    is_in_air,
    player_x_speed,
    player_y_speed,
    read_snapshot,
    s8,
    timer_value,
)


def _blank_ram() -> np.ndarray:
    return np.zeros(0x800, dtype=np.uint8)


def test_s8_and_speeds() -> None:
    assert s8(40) == 40
    assert s8(200) == 200 - 256
    ram = _blank_ram()
    ram[ADDR_X_SPEED] = 40
    ram[ADDR_Y_SPEED] = 200  # -56
    assert player_x_speed(ram) == 40
    assert player_y_speed(ram) == -56


def test_in_air_from_y_speed() -> None:
    ram = _blank_ram()
    ram[ADDR_PLAYER_STATE] = 0x08
    ram[ADDR_Y_SPEED] = 0
    assert is_in_air(ram) is False
    assert is_grounded(ram) is True
    ram[ADDR_Y_SPEED] = 10
    assert is_in_air(ram) is True
    assert is_grounded(ram) is False


def test_snapshot_includes_physics() -> None:
    ram = _blank_ram()
    ram[ADDR_X_PAGE] = 2
    ram[ADDR_PLAYER_X] = 100
    ram[ADDR_PLAYER_Y] = 120
    ram[ADDR_X_SPEED] = 32
    ram[ADDR_Y_SPEED] = 0
    ram[ADDR_PLAYER_STATE] = 0x08
    ram[0x07F8] = 3
    ram[0x07F9] = 5
    ram[0x07FA] = 0
    snap = read_snapshot(ram)
    assert snap.player_x == 2 * 256 + 100
    assert snap.x_speed == 32
    assert snap.timer == 350
    assert snap.in_air is False
    assert timer_value(ram) == 350


def test_obs_shape_and_bias() -> None:
    ram = _blank_ram()
    ram[ADDR_PLAYER_STATE] = 0x08
    ram[ADDR_X_SPEED] = 40
    ram[0x0770] = 1  # playing
    obs = read_smb_inputs(ram)
    assert obs.shape == (N_SMB_INPUTS,)
    assert obs.dtype == np.float32
    # bias slot
    assert obs[N_GRID + 5] == 1.0
    # x_speed normalized ~ 40/40 = 1
    assert abs(float(obs[N_GRID + 1]) - 1.0) < 1e-5
    # grounded when vy=0
    assert obs[N_GRID + 4] == 0.0  # in_air
    assert obs[N_GRID + 11] == 1.0  # grounded


def test_obs_prev_action() -> None:
    ram = _blank_ram()
    prev = [0] * 9
    prev[7] = 1  # RIGHT
    prev[0] = 1  # B
    obs = read_smb_inputs(ram, prev_action=prev)
    assert obs[N_GRID + 13] == 1.0
    assert obs[N_GRID + 14] == 1.0
    assert obs[N_GRID + 15] == 0.0


def test_legacy_shape() -> None:
    ram = _blank_ram()
    ram[ADDR_Y_SPEED] = 5
    leg = read_smb_inputs_legacy(ram)
    assert leg.shape == (N_SMB_INPUTS_LEGACY,)
    assert leg[172] == 1.0  # in-air derived, not stubbed


def test_grid_as_image_and_slices() -> None:
    ram = _blank_ram()
    obs = read_smb_inputs(ram)
    img = grid_as_image(obs)
    assert img.shape == (1, 13, 13)
    slices = observation_slices()
    assert slices["grid"].stop == N_GRID
    assert slices["state"].stop - slices["state"].start == N_STATE

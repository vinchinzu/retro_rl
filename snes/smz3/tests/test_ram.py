"""Combo RAM snapshot unit tests (no emulator)."""

from __future__ import annotations

import numpy as np

from smz3.ram import (
    SM_ADDR_DOOR_TRANSITION,
    SM_ADDR_GAME_STATE,
    SM_ADDR_HEALTH,
    SM_ADDR_ROOM_ID,
    SM_ADDR_SAMUS_X,
    SM_ADDR_SAMUS_Y,
    Z3_ADDR_MODULE,
    Z3_ADDR_SUBMODULE,
    read_snapshot,
)


def _poke(ram: np.ndarray, addr: int, value: int, width: int = 2) -> None:
    if width == 1:
        ram[addr] = value & 0xFF
    else:
        ram[addr] = value & 0xFF
        ram[addr + 1] = (value >> 8) & 0xFF


def test_landing_site_controllable_snapshot() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke(ram, SM_ADDR_GAME_STATE, 8)
    _poke(ram, SM_ADDR_ROOM_ID, 0x91F8)
    _poke(ram, SM_ADDR_DOOR_TRANSITION, 0)
    _poke(ram, SM_ADDR_HEALTH, 99)
    _poke(ram, SM_ADDR_SAMUS_X, 1152)
    _poke(ram, SM_ADDR_SAMUS_Y, 1088)
    # Z3 fields are garbage while SM owns WRAM
    ram[Z3_ADDR_MODULE] = 151

    snap = read_snapshot(ram, frame=900)
    assert snap.sm_controllable
    assert snap.sm_room_id == 0x91F8
    assert snap.sm_health == 99
    assert snap.sm_samus_x == 1152
    assert not snap.z3_controllable
    assert snap.to_dict()["sm_room_id_hex"] == "0x91F8"


def test_z3_overworld_controllable_fields() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    # SM side not a known engine state
    _poke(ram, SM_ADDR_GAME_STATE, 0xBEEF)
    ram[Z3_ADDR_MODULE] = 0x09
    ram[Z3_ADDR_SUBMODULE] = 0x00
    snap = read_snapshot(ram)
    assert snap.z3_controllable
    assert not snap.sm_controllable

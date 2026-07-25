"""Unit tests for ALTTP RAM snapshot helpers."""

from __future__ import annotations

import numpy as np

from alttp.ram import (
    DARK_WORLD_FLAG,
    HYRULE_CASTLE_SCREEN,
    INDOORS,
    MODULE,
    SCREEN_ID,
    SUBMODULE,
    read_snapshot,
)


def _ram(writes: dict[int, int]) -> np.ndarray:
    ram = np.zeros(0x2000, dtype=np.uint8)
    for addr, value in writes.items():
        ram[addr] = value & 0xFF
    return ram


def test_has_control_requires_gameplay_module_and_idle_submodule() -> None:
    snap = read_snapshot(_ram({MODULE: 0x09, SUBMODULE: 0x00}))
    assert snap.has_control is True
    snap = read_snapshot(_ram({MODULE: 0x09, SUBMODULE: 0x01}))
    assert snap.has_control is False


def test_title_and_file_select_modes() -> None:
    assert read_snapshot(_ram({MODULE: 0x01})).is_title_screen is True
    assert read_snapshot(_ram({MODULE: 0x02})).is_file_select is True


def test_on_castle_grounds() -> None:
    snap = read_snapshot(
        _ram(
            {
                MODULE: 0x09,
                SUBMODULE: 0x00,
                SCREEN_ID: HYRULE_CASTLE_SCREEN,
                INDOORS: 0,
                DARK_WORLD_FLAG: 0,
            }
        )
    )
    assert snap.on_castle_grounds is True

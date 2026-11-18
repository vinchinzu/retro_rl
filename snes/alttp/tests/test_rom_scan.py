"""Chest-table and cell-lock decoders on bytes. No ROM file, no emulator."""

from __future__ import annotations

import pytest

from alttp.dungeon.rom_scan import (
    _cell_lock_tiles,
    _chest_place,
    _chest_records,
    _snes_file_offset,
)

# Subtype-3 index 0x18: p2 low bits 1, p1 low bits 2, p0 low bits 0.
# p0 = 44 << 2, p1 = (14 << 2) | 2. That pair is tile (44, 14).
_LOCK_OBJECT = bytes((0xB0, 0x3A, 0xF9))


def test_snes_file_offset_maps_a_rom_window() -> None:
    assert _snes_file_offset(0x81E96E) == 0xE96E
    assert _snes_file_offset(0x1F8000) == 0xF8000
    with pytest.raises(SystemExit):
        _snes_file_offset(0x81696E)


def test_chest_records_group_a_room_and_the_big_bit() -> None:
    base = _snes_file_offset(0x81E96E)
    rom = bytearray(base + 168 * 3)
    rom[base : base + 3] = bytes((0x72, 0x00, 0x32))
    rom[base + 3 : base + 6] = bytes((0x72, 0x80, 0x01))
    found = _chest_records(bytes(rom))
    assert found[0x72] == [(base, False), (base + 3, True)]
    assert found[0][0] == (base + 6, False)
    assert len(found[0]) == 166


def test_chest_place_names_a_tile_or_a_cleared_slot() -> None:
    assert _chest_place(0) == "loc=$0000 cleared"
    assert _chest_place(0x093C) == "tile=(30,18) layer=0 loc=$093C"


def test_cell_lock_object_decodes_to_its_tile() -> None:
    data_ea = 0x1F8100
    ptr_off = _snes_file_offset(0x1F8000)
    data_off = _snes_file_offset(data_ea)
    rom = bytearray(data_off + 16)
    rom[ptr_off : ptr_off + 3] = bytes(
        (data_ea & 0xFF, (data_ea >> 8) & 0xFF, (data_ea >> 16) & 0xFF)
    )
    body = bytes((0x00, 0x00)) + _LOCK_OBJECT + bytes((0xFF, 0xFF)) * 3
    rom[data_off : data_off + len(body)] = body
    assert _cell_lock_tiles(bytes(rom), 0) == [(44, 14)]
    assert _cell_lock_tiles(bytes(rom), 1) == []
    assert _cell_lock_tiles(bytes(rom), 0x140) == []

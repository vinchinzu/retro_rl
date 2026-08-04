"""Unit tests for ALTTP RAM snapshot helpers."""

from __future__ import annotations

import numpy as np

from alttp.ram import (
    DARK_WORLD_FLAG,
    EQUIP_SWORD,
    HYRULE_CASTLE_SCREEN,
    INDOORS,
    MODULE,
    ROOM_ID,
    SCREEN_ID,
    SECRET_PASSAGE_ROOM,
    SUBMODULE,
    WRAM_IDX,
    castle_entry_accepted,
    read_snapshot,
    read_sword_level,
    secret_passage_accepted,
    uncle_sword_event_accepted,
    wram_index,
)


def _ram(writes: dict[int, int], *, size: int = 0x20000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    for addr, value in writes.items():
        if addr < len(ram):
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


def test_sword_and_passage_acceptance() -> None:
    assert wram_index(EQUIP_SWORD) == WRAM_IDX + EQUIP_SWORD
    sword_ram = _ram({wram_index(EQUIP_SWORD): 1})
    assert read_sword_level(sword_ram) == 1
    assert uncle_sword_event_accepted(read_snapshot(sword_ram)) is True

    passage = _ram(
        {
            MODULE: 0x07,
            SUBMODULE: 0x00,
            INDOORS: 1,
            DARK_WORLD_FLAG: 0,
            ROOM_ID: SECRET_PASSAGE_ROOM,
        }
    )
    snap = read_snapshot(passage)
    assert secret_passage_accepted(snap) is True
    assert castle_entry_accepted(snap) is True


def test_follower_keys_hold_up() -> None:
    from alttp.ram import (
        FOLLOWER,
        LINK_ACTION,
        NUM_KEYS,
        has_zelda_follower,
        zelda_rescued_accepted,
    )

    ram = _ram(
        {
            wram_index(FOLLOWER): 1,
            wram_index(NUM_KEYS): 2,
            LINK_ACTION: 21,
        }
    )
    snap = read_snapshot(ram)
    assert snap.has_zelda_follower is True
    assert snap.dungeon_key_count == 2
    assert snap.is_hold_up_item is True
    assert zelda_rescued_accepted(snap) is True

    class _Env:
        def get_ram(self) -> np.ndarray:
            return ram

    assert has_zelda_follower(_Env()) is True
    assert read_snapshot(_ram({wram_index(NUM_KEYS): 0xFF})).dungeon_key_count is None

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


def test_snapshot_extras_from_verified_offsets() -> None:
    from alttp.ram import (
        DUNGEON_FLOOR,
        DUNGEON_ID,
        LINK_ACTION,
        LINK_HP,
        LINK_IFRAMES,
        LINK_KNOCKBACK,
        LINK_LAYER,
        format_pin_probe,
        snapshot_to_diag,
    )

    ram = _ram(
        {
            DUNGEON_ID: 0x02,
            DUNGEON_ID + 1: 0x5A,  # high byte is not the id
            DUNGEON_FLOOR: 0xFF,
            LINK_LAYER: 1,
            LINK_ACTION: 2,
            LINK_IFRAMES: 57,
            LINK_KNOCKBACK: 18,
            LINK_HP: 0x5A,  # low mirror must not win
            wram_index(LINK_HP): 0x14,
            ROOM_ID: 0x72,
        }
    )
    snap = read_snapshot(ram)
    assert snap.dungeon_id == 0x02
    assert snap.floor == 0xFF
    assert snap.layer == 1
    assert snap.link_state == 2
    assert snap.link_action == 2
    assert snap.hp == 0x14
    assert snap.iframes == 57
    assert snap.knockback == 18
    diag = snapshot_to_diag(snap)
    assert diag["dungeon_id"] == 0x02
    assert diag["hp"] == 0x14
    line = format_pin_probe("CastleB1Guard", snap)
    assert "dungeon=0x02" in line
    assert "floor=0xFF" in line
    assert "layer=1" in line
    assert "state=2" in line
    assert "hp=0x14" in line
    assert "iframes=57" in line
    assert "knockback=18" in line


def test_room_flags_synthetic_word_and_pin_diff_report() -> None:
    from alttp.ram import (
        ROOM_FLAG_CHEST,
        ROOM_FLAG_DOOR,
        ROOM_FLAG_KEY,
        ROOM_FLAG_PIN_DIFFS,
        SAVE_DUNG_INFO,
        format_room_flag_diffs,
        room_flags,
    )

    room = 0x72
    addr = wram_index(SAVE_DUNG_INFO + 2 * room)
    house = 0x104
    house_addr = wram_index(SAVE_DUNG_INFO + 2 * house)
    # Poison the masked house id and the raw (non-indexed) save offset.
    ram = _ram(
        {
            addr: 0x0F,
            addr + 1: 0x84,
            house_addr: 0x12,
            wram_index(SAVE_DUNG_INFO + 2 * 0x04): 0xFF,
            SAVE_DUNG_INFO + 2 * room: 0x5A,
        }
    )
    word = room_flags(ram, room)
    assert word == 0x840F
    assert word & ROOM_FLAG_KEY
    assert word & ROOM_FLAG_DOOR
    assert not word & ROOM_FLAG_CHEST
    assert room_flags(ram, house) == 0x0012
    assert room_flags(ram, house) & ROOM_FLAG_CHEST
    assert room_flags(ram, 0x04) == 0x00FF
    named = {
        (row.before_pin, row.after_pin, row.room, row.bit, row.name)
        for row in ROOM_FLAG_PIN_DIFFS
    }
    assert (
        "FighterSword",
        "FighterSwordLamp",
        0x104,
        ROOM_FLAG_CHEST,
        "chest",
    ) in named
    assert (
        "CastleB3",
        "CastleB3Boomerang",
        0x71,
        ROOM_FLAG_CHEST,
        "chest",
    ) in named
    assert (
        "CastleB1Key",
        "CastleB1SecondKey",
        0x72,
        ROOM_FLAG_KEY,
        "key",
    ) in named
    assert (
        "CastleB1Key",
        "CastleB1SecondKey",
        0x72,
        ROOM_FLAG_DOOR,
        "door",
    ) in named
    report = format_room_flag_diffs()
    assert "bit=0x0010 chest" in report
    assert "bit=0x0400 key" in report
    assert "bit=0x8000 door" in report
    assert "boss" not in report


def test_ow_flags_event_byte_and_castle_hole_tile() -> None:
    from alttp.ram import (
        CASTLE_BUSH_HOLE_POS,
        CASTLE_BUSH_HOLE_SCREEN,
        MEMORIZED_TILE_ADDR,
        MEMORIZED_TILE_COUNT,
        MEMORIZED_TILE_VALUE,
        OW_FLAG_BOMB_OVERLAY,
        OW_FLAG_EVENT_OVERLAY,
        OW_TILE_HOLE,
        SAVE_OW_EVENT_INFO,
        memorized_tiles,
        ow_flags,
    )

    hole = SAVE_OW_EVENT_INFO + CASTLE_BUSH_HOLE_SCREEN
    event = SAVE_OW_EVENT_INFO + 0x07
    ram = _ram(
        {
            wram_index(hole): 0x00,
            hole: 0x20,
            wram_index(event): OW_FLAG_EVENT_OVERLAY | OW_FLAG_BOMB_OVERLAY,
            wram_index(MEMORIZED_TILE_COUNT): 2,
            wram_index(MEMORIZED_TILE_ADDR): 0x70,
            wram_index(MEMORIZED_TILE_ADDR) + 1: 0x05,
            wram_index(MEMORIZED_TILE_VALUE): 0xCC,
            wram_index(MEMORIZED_TILE_VALUE) + 1: 0x0D,
            wram_index(MEMORIZED_TILE_ADDR + 2): 0x18,
            wram_index(MEMORIZED_TILE_ADDR + 3): 0x08,
            wram_index(MEMORIZED_TILE_VALUE + 2): 0xA6,
            wram_index(MEMORIZED_TILE_VALUE + 3): 0x0D,
            MEMORIZED_TILE_ADDR: 0xFF,
        }
    )
    assert ow_flags(ram, CASTLE_BUSH_HOLE_SCREEN) == 0
    assert ow_flags(ram, 0x07) & OW_FLAG_EVENT_OVERLAY
    assert ow_flags(ram, 0x07) & OW_FLAG_BOMB_OVERLAY
    assert memorized_tiles(ram) == ((CASTLE_BUSH_HOLE_POS, OW_TILE_HOLE),)
    assert CASTLE_BUSH_HOLE_POS == 0x0570
    assert OW_TILE_HOLE == 0x0DCC
    try:
        ow_flags(ram, 0x80)
    except ValueError:
        pass
    else:
        raise AssertionError("screen 0x80 is past save_ow_event_info")

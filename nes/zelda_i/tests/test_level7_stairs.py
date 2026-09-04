"""L7 ROM stair/cellar table (no emulator)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zelda_i.level7.graph import (
    AQUAMENTUS,
    LEVEL7_ROOMS,
    NOSE_CELLAR,
    PRE_BOSS,
    TIP_OF_NOSE,
    TRIFORCE,
)
from zelda_i.level7.stairs import (
    AQUAMENTUS_ROM,
    CANDLE_CELLAR_ROM,
    CELLAR_LADDER_LEFT_X,
    CELLAR_LADDER_RIGHT_X,
    LEVEL7_CELLAR_ROOMS,
    LEVEL7_STAIR_LIST,
    LEVEL7_STAIR_LIST_INES,
    LEVEL7_STAIR_LIST_PRG,
    NOSE_CELLAR_ROM,
    PRE_BOSS_ROM,
    TIP_OF_NOSE_ROM,
    TRIFORCE_ROM,
    cellar_dest_for,
    cellar_for_play_room,
    other_play_endpoint,
    play_rooms_entering_cellar,
    rom_secret_name,
    rom_side,
    spawn_ladder_for_source,
)

_ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"


def test_stair_list_offsets_and_cellars() -> None:
    assert LEVEL7_STAIR_LIST_INES - LEVEL7_STAIR_LIST_PRG == 0x10
    assert LEVEL7_STAIR_LIST[:2] == (NOSE_CELLAR_ROM, CANDLE_CELLAR_ROM)
    assert all(slot == 0xFF for slot in LEVEL7_STAIR_LIST[2:])
    assert LEVEL7_CELLAR_ROOMS == (0x7B, 0x4A)


def test_nose_cellar_pairs_0d_to_29() -> None:
    """0x0D is AttrB of cellar 0x7B; far side is play 0x29. Not a dead belief."""
    assert cellar_for_play_room(0x0D) == (0x7B, "right")
    assert cellar_for_play_room(0x29) == (0x7B, "left")
    assert cellar_dest_for(0x7B, side="left") == PRE_BOSS_ROM
    assert cellar_dest_for(0x7B, side="right") == TIP_OF_NOSE_ROM
    assert other_play_endpoint(0x0D) == 0x29
    assert other_play_endpoint(0x29) == 0x0D
    assert play_rooms_entering_cellar(0x7B) == ((0x29, "left"), (0x0D, "right"))


def test_candle_cellar_is_treasure_both_sides_1a() -> None:
    assert cellar_for_play_room(0x1A) == (0x4A, "left")
    assert cellar_dest_for(0x4A, side="left") == 0x1A
    assert cellar_dest_for(0x4A, side="right") == 0x1A
    assert play_rooms_entering_cellar(0x4A) == ((0x1A, "left"),)


def test_initmode9_from_0d_spawns_right_ladder() -> None:
    assert spawn_ladder_for_source(0x7B, 0x0D) == "right"
    assert spawn_ladder_for_source(0x7B, 0x29) == "left"
    assert CELLAR_LADDER_LEFT_X == 0x30
    assert CELLAR_LADDER_RIGHT_X == 0xC0


def test_room_0d_rom_doors_are_west_bomb_and_block_stairs() -> None:
    assert rom_side(0x0D, "n") == 1
    assert rom_side(0x0D, "s") == 1
    assert rom_side(0x0D, "w") == 4
    assert rom_side(0x0D, "e") == 1
    assert rom_secret_name(0x0D) == "block_stairs"
    assert rom_side(0x0C, "e") == 4
    assert rom_side(0x29, "e") == 4
    assert rom_side(0x2A, "w") == 4
    assert rom_side(0x2A, "e") == 7
    assert rom_side(0x2B, "w") == 0


def test_levelinfo_boss_hallway_ids() -> None:
    assert PRE_BOSS_ROM == 0x29
    assert AQUAMENTUS_ROM == 0x2A
    assert TRIFORCE_ROM == 0x2B
    assert TIP_OF_NOSE_ROM == 0x0D
    assert NOSE_CELLAR_ROM == 0x7B


def test_graph_promotes_only_walked_rom_ids() -> None:
    """NOSE_CELLAR 0x7B is promoted: the 0x0D walk-on is live 2/2."""
    by_id = {room.source_id: room for room in LEVEL7_ROOMS}
    assert by_id[TIP_OF_NOSE].ram_id == 0x0D
    assert by_id[NOSE_CELLAR].ram_id == 0x7B
    assert by_id[NOSE_CELLAR].evidence == "fixture-live"
    assert by_id[NOSE_CELLAR].route_eligible is False
    assert by_id[PRE_BOSS].ram_id == 0x29
    assert by_id[AQUAMENTUS].ram_id == 0x2A
    assert by_id[TRIFORCE].ram_id == 0x2B
    assert by_id[PRE_BOSS].evidence == "fixture-live"
    assert by_id[PRE_BOSS].route_eligible is False


def test_cellar_module_locks_rom_attr_endpoints() -> None:
    from zelda_i.level7.cellar import (
        CELLAR_ROOM,
        DEST_ROOM,
        EAST_X,
        SOURCE_ROOM,
        WEST_X,
    )

    assert CELLAR_ROOM == NOSE_CELLAR_ROM == 0x7B
    assert DEST_ROOM == PRE_BOSS_ROM == 0x29
    assert SOURCE_ROOM == TIP_OF_NOSE_ROM == 0x0D
    assert WEST_X == CELLAR_LADDER_LEFT_X == 0x30
    assert EAST_X == CELLAR_LADDER_RIGHT_X == 0xC0
    assert spawn_ladder_for_source(CELLAR_ROOM, SOURCE_ROOM) == "right"
    assert cellar_dest_for(CELLAR_ROOM, side="left") == DEST_ROOM


@pytest.mark.skipif(not _ROM.is_file(), reason="local Zelda I ROM not present")
def test_rom_bytes_match_hardcoded_stair_list() -> None:
    data = _ROM.read_bytes()
    assert data[:4] == b"NES\x1a"
    prg = data[0x10:]
    blob = bytes(prg[LEVEL7_STAIR_LIST_PRG : LEVEL7_STAIR_LIST_PRG + 8])
    assert tuple(blob) == LEVEL7_STAIR_LIST
    assert prg[0x18A00 + 0x7B] == 0x29  # AttrA
    assert prg[0x18A00 + 128 + 0x7B] == 0x0D  # AttrB
    assert prg[0x18A00 + 0x4A] == 0x1A
    assert prg[0x18A00 + 128 + 0x4A] == 0x1A

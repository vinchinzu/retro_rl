"""Synthetic reads for the $F340..$F37B inventory block."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alttp.ram import (
    EQUIP_SWORD,
    INV_BLOCK_HI,
    INV_BLOCK_LO,
    INVENTORY_FIELDS,
    LINK_HP,
    LINK_ITEM_LAMP,
    LINK_MAX_HP,
    NUM_KEYS,
    WRAM_IDX,
    AlttpInventory,
    InventoryField,
    read_inventory,
    wram_index,
)

# snesrev/zelda3 variables.h save block. Size defaults to 1.
_DISASSEMBLY: dict[str, tuple[int, int]] = {
    "bow": (0xF340, 1),
    "boomerang": (0xF341, 1),
    "hookshot": (0xF342, 1),
    "bombs": (0xF343, 1),
    "powder": (0xF344, 1),
    "fire_rod": (0xF345, 1),
    "ice_rod": (0xF346, 1),
    "bombos": (0xF347, 1),
    "ether": (0xF348, 1),
    "quake": (0xF349, 1),
    "lamp": (0xF34A, 1),
    "hammer": (0xF34B, 1),
    "flute": (0xF34C, 1),
    "net": (0xF34D, 1),
    "book": (0xF34E, 1),
    "somaria": (0xF350, 1),
    "byrna": (0xF351, 1),
    "cape": (0xF352, 1),
    "mirror": (0xF353, 1),
    "gloves": (0xF354, 1),
    "boots": (0xF355, 1),
    "flippers": (0xF356, 1),
    "pearl": (0xF357, 1),
    "sword": (0xF359, 1),
    "shield": (0xF35A, 1),
    "armor": (0xF35B, 1),
    "bottle_1": (0xF35C, 1),
    "bottle_2": (0xF35D, 1),
    "bottle_3": (0xF35E, 1),
    "bottle_4": (0xF35F, 1),
    "rupees": (0xF362, 2),
    "compass": (0xF364, 2),
    "big_key": (0xF366, 2),
    "dungeon_map": (0xF368, 2),
    "heart_pieces": (0xF36B, 1),
    "max_hp": (0xF36C, 1),
    "hp": (0xF36D, 1),
    "magic": (0xF36E, 1),
    "keys": (0xF36F, 1),
    "pendants": (0xF374, 1),
    "arrows": (0xF377, 1),
    "ability": (0xF379, 1),
    "crystals": (0xF37A, 1),
}

# Live scan 2026-09-29. Every other field stayed 0 on those pins.
_VERIFIED = {
    "boomerang": "CastleB3Boomerang",
    "lamp": "FighterSwordLamp",
    "sword": "FighterSword",
    "shield": "FighterSword",
    "big_key": "CastleMainZeldaBoomerang",
    "max_hp": "CastleB1Key",
    "hp": "CastleB1Key",
    "keys": "CastleB1Key",
    "magic": "IcePalaceEntry",
}

_RAM_MAP = Path(__file__).resolve().parents[1] / "docs" / "ram_map.md"


def _ram(writes: dict[int, int], *, size: int = 0x20000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    for addr, value in writes.items():
        ram[int(addr)] = int(value) & 0xFF
    return ram


def test_fields_match_disassembly_and_span() -> None:
    assert [f.name for f in INVENTORY_FIELDS] == list(_DISASSEMBLY)
    for field in INVENTORY_FIELDS:
        offset, size = _DISASSEMBLY[field.name]
        assert isinstance(field, InventoryField)
        assert field.offset == offset
        assert field.size == size
        assert INV_BLOCK_LO <= field.offset <= INV_BLOCK_HI
        assert field.offset + field.size - 1 <= INV_BLOCK_HI
        assert field.verified_on == _VERIFIED.get(field.name, "unverified")
    assert [f.name for f in AlttpInventory.__dataclass_fields__.values()] == [
        f.name for f in INVENTORY_FIELDS
    ]
    assert LINK_ITEM_LAMP == 0xF34A
    assert EQUIP_SWORD == 0xF359
    assert LINK_MAX_HP == 0xF36C
    assert LINK_HP == 0xF36D
    assert NUM_KEYS == 0xF36F


def test_read_inventory_synthetic_ram() -> None:
    writes: dict[int, int] = {}
    expected: dict[str, int] = {}
    for index, field in enumerate(INVENTORY_FIELDS):
        # Poison the low mirror so a missing wram_index fails the read.
        writes[field.offset] = 0x5A
        if field.size == 1:
            value = (0x11 + index * 3) & 0xFF
            writes[wram_index(field.offset)] = value
        else:
            value = 0x2000 + index * 17
            assert wram_index(field.offset) + 1 == wram_index(field.offset + 1)
            writes[wram_index(field.offset)] = value & 0xFF
            writes[wram_index(field.offset) + 1] = (value >> 8) & 0xFF
        expected[field.name] = value
    inv = read_inventory(_ram(writes))
    for name, value in expected.items():
        assert getattr(inv, name) == value
    assert inv.sword != 0x5A
    assert inv.rupees > 0xFF


def test_keys_keep_blank_sentinel() -> None:
    ram = _ram({wram_index(NUM_KEYS): 0xFF, wram_index(EQUIP_SWORD): 1})
    inv = read_inventory(ram)
    assert inv.keys == 0xFF
    assert inv.sword == 1
    assert wram_index(EQUIP_SWORD) == WRAM_IDX + EQUIP_SWORD


def test_ram_map_lists_every_field() -> None:
    text = _RAM_MAP.read_text(encoding="utf-8")
    section = text.split("## Inventory", 1)[1].split("\n## ", 1)[0]
    found: dict[str, tuple[int, str]] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if cols[0] in {"Field", ""} or set(cols[0]) <= {"-", " "}:
            continue
        offset = int(cols[1].lstrip("$"), 16)
        found[cols[0]] = (offset, cols[3])
    assert found == {
        field.name: (field.offset, field.verified_on) for field in INVENTORY_FIELDS
    }

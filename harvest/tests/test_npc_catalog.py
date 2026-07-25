from __future__ import annotations

import unittest

import numpy as np

from harvest.core.npc_catalog import (
    ADDR_PLAYER_GOBJ_INDEX,
    GOBJ_STRUCT_BASE,
    GOBJ_STRUCT_STRIDE,
    dialogue_catalog,
    game_objects,
    heart_tier,
    parse_romance_assignment,
    relationship_status,
    romance_points_for_hearts,
    set_romance_hearts,
    status_flags,
)
from harvest.core.ram_catalog import field_spec


def _blank_ram(size: int = 0x20000) -> np.ndarray:
    return np.zeros(size, dtype=np.uint8)


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _write_gobj(ram: np.ndarray, slot: int, sprite: int, x: int, y: int, components: list[int]) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    _write_u16(ram, offset, 0x7777)
    _write_u16(ram, offset + 0x02, sprite)
    _write_u16(ram, offset + 0x08, x)
    _write_u16(ram, offset + 0x0A, y)
    ram[offset + 0x13] = len(components)
    for i, component in enumerate(components):
        ram[offset + 0x14 + i] = component


class FakeMutableState:
    def __init__(self) -> None:
        self.ram = bytearray(0x20000)

    def write_u16(self, address: int, value: int) -> None:
        self.ram[address] = value & 0xFF
        self.ram[address + 1] = (value >> 8) & 0xFF


class NpcCatalogTests(unittest.TestCase):
    def test_game_object_decoder_exports_player_and_candidate_npc_tiles(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, ADDR_PLAYER_GOBJ_INDEX, 0)
        _write_gobj(ram, 0, 0x0005, 16 * 8, 16 * 9, [0, 1])
        _write_gobj(ram, 1, 0x0230, 16 * 20, 16 * 21, [4, 5])

        objects = game_objects(ram)

        self.assertEqual(len(objects), 2)
        self.assertTrue(objects[0].is_player)
        self.assertEqual(objects[0].tile, (8, 9))
        self.assertTrue(objects[1].is_npc_candidate)
        self.assertEqual(objects[1].label, "candidate_npc_0230")
        self.assertEqual(objects[1].tile, (20, 21))

    def test_game_object_decoder_reads_low_wram_direct_in_live_sized_ram(self) -> None:
        ram = _blank_ram(0x24000)
        _write_u16(ram, ADDR_PLAYER_GOBJ_INDEX, 0)
        _write_gobj(ram, 0, 0x0005, 16 * 4, 16 * 5, [0, 1])

        objects = game_objects(ram)

        self.assertEqual(objects[0].tile, (4, 5))

    def test_coop_chicken_sprite_range_is_named_animal(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, ADDR_PLAYER_GOBJ_INDEX, 0)
        _write_gobj(ram, 0, 0x0005, 16 * 4, 16 * 5, [0, 1])
        _write_gobj(ram, 1, 0x0217, 16 * 12, 16 * 11, [22])

        objects = game_objects(ram)

        self.assertEqual(objects[1].label, "chicken")
        self.assertEqual(objects[1].kind, "animal")

    def test_barn_cow_sprite_range_is_named_animal(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, ADDR_PLAYER_GOBJ_INDEX, 0)
        _write_gobj(ram, 0, 0x0005, 16 * 4, 16 * 5, [0, 1])
        _write_gobj(ram, 1, 0x01A6, 16 * 11, 16 * 14, [0x31, 0x2F])

        objects = game_objects(ram)

        self.assertEqual(objects[1].label, "cow")
        self.assertEqual(objects[1].kind, "animal")

    def test_relationship_status_decodes_hearts_marriage_and_event_flags(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, field_spec("maria_hearts").address, 220)
        _write_u16(ram, field_spec("marriage_flags").address, 0x0001)
        _write_u16(ram, field_spec("romance_event_flags").address, 0x1000)

        status = relationship_status(ram)

        self.assertEqual(heart_tier(220), 3)
        self.assertEqual(status["maria"]["heart_tier"], 3)
        self.assertTrue(status["maria"]["married"])
        self.assertTrue(status["maria"]["heart_200_event_seen"])

    def test_romance_heart_setter_writes_requested_threshold(self) -> None:
        state = FakeMutableState()

        points = set_romance_hearts(state, "ann", 8)

        addr = field_spec("ann_hearts").address
        self.assertEqual(points, 599)
        self.assertEqual(state.ram[addr] | (state.ram[addr + 1] << 8), 599)
        self.assertEqual(romance_points_for_hearts(6), 399)

    def test_parse_romance_assignment_normalizes_npc(self) -> None:
        self.assertEqual(parse_romance_assignment(" Eve = 6 "), ("eve", 6))

    def test_status_flags_decode_named_bits_and_unknown_mask(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, field_spec("incubator_flags").address, 0xA000)

        flags = status_flags(ram)

        self.assertTrue(flags["incubator_flags"]["named_bits"]["egg_incubating"])
        self.assertEqual(flags["incubator_flags"]["unknown_mask"], "0x8000")

    def test_dialogue_catalog_extracts_maria_text_from_decomp(self) -> None:
        catalog = dialogue_catalog(npc="maria", compact=True)
        labels = {
            row["text_label"]
            for rows in catalog["groups"].values()
            for row in rows
        }

        self.assertGreater(catalog["record_count"], 0)
        self.assertIn("Text_Maria_Spring", labels)


if __name__ == "__main__":
    unittest.main()

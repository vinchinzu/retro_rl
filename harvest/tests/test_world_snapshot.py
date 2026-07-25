from __future__ import annotations

import unittest

import numpy as np

from harvest.core.npc_catalog import ADDR_PLAYER_GOBJ_INDEX, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH
from harvest.core.world_snapshot import WorldSnapshot, world_snapshot_dict


def _blank_ram(size: int = 0x20000) -> np.ndarray:
    return np.zeros(size, dtype=np.uint8)


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _write_u24(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF
    ram[addr + 2] = (value >> 16) & 0xFF


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int, base: int = ADDR_MAP) -> None:
    ram[base + ty * MAP_WIDTH + tx] = tile_id


def _write_gobj(ram: np.ndarray, slot: int, sprite: int, x: int, y: int) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    _write_u16(ram, offset, 0x7777)
    _write_u16(ram, offset + 0x02, sprite)
    _write_u16(ram, offset + 0x08, x)
    _write_u16(ram, offset + 0x0A, y)
    ram[offset + 0x13] = 1


class WorldSnapshotTests(unittest.TestCase):
    def test_snapshot_exports_player_resources_landmarks_and_crop_stage(self) -> None:
        ram = _blank_ram()
        _write_u16(ram, field_spec("player_x").address, 10 * 16 + 8)
        _write_u16(ram, field_spec("player_y").address, 12 * 16 + 8)
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("day").address] = 9
        ram[field_spec("hour").address] = 6
        ram[field_spec("stamina").address] = 88
        ram[field_spec("tool_selected").address] = 0x10
        _write_u16(ram, ADDR_PLAYER_GOBJ_INDEX, 0)
        _write_gobj(ram, 0, 0x0005, 10 * 16 + 8, 12 * 16 + 8)
        _write_gobj(ram, 1, 0x0230, 20 * 16, 20 * 16)
        _write_u16(ram, field_spec("maria_hearts").address, 220)
        _write_u16(ram, field_spec("romance_event_flags").address, 0x1000)
        _write_u24(ram, field_spec("money").address, 700)
        _set_tile(ram, 10, 12, 0xA1)
        _set_tile(ram, 20, 20, 0x54)
        _set_tile(ram, 21, 20, 0x55)
        _set_tile(ram, 20, 21, 0x58)

        snapshot = WorldSnapshot.from_ram(ram, bounds=(18, 18, 22, 22))
        data = snapshot.to_dict(compact=True)

        self.assertEqual(data["player"]["tile"], [10, 12])
        self.assertEqual(data["player"]["tilemap_name"], "farm")
        self.assertEqual(data["player"]["tool_name"], "watering_can")
        self.assertEqual(data["resources"]["money"], 7000)
        self.assertEqual(data["date"]["day"], 9)
        self.assertEqual(data["crop_plots"][0]["max_stage"], 29)
        self.assertEqual(data["crop_plots"][0]["dry_count"], 2)
        self.assertIn("shipping_bin", {landmark["name"] for landmark in data["map"]["landmarks"]})
        self.assertEqual(data["relationship_status"]["maria"]["heart_tier"], 3)
        self.assertTrue(data["relationship_status"]["maria"]["heart_200_event_seen"])
        self.assertEqual(data["entities"]["candidate_npcs"][0]["sprite_table_hex"], "0x0230")
        self.assertEqual(data["dialogue"]["registers"]["input_lock"], 0)
        self.assertNotIn("scalars", data)

    def test_snapshot_uses_live_wram_mirror_for_player_and_tiles(self) -> None:
        ram = _blank_ram(0x24000)
        live = LIVE_RAM_WRAM_OFFSET
        _write_u16(ram, live + field_spec("player_x").address, 5 * 16 + 8)
        _write_u16(ram, live + field_spec("player_y").address, 5 * 16 + 8)
        ram[field_spec("tilemap").address] = 0x00
        _set_tile(ram, 5, 5, 0x03, base=ADDR_MAP)
        _set_tile(ram, 5, 5, 0x04, base=live + ADDR_MAP)

        data = world_snapshot_dict(ram, bounds=(5, 5, 5, 5), include_grid=True, compact=True)

        self.assertEqual(data["player"]["tile"], [5, 5])
        self.assertEqual(data["player"]["tile_id"], 0x04)
        self.assertEqual(data["objects"][0]["label"], "stone")
        self.assertEqual(data["grid"], [[0x04]])


if __name__ == "__main__":
    unittest.main()

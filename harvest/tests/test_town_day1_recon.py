"""Unit tests for Spring D1 town recon helpers (no ROM)."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec, read_ram_value
from harvest.scripts.town_day1_recon import (
    D1_TOWN_BITS,
    GATE_PIXEL,
    TARGET_MASK,
    TOWN_TILEMAP,
    TownSnapshot,
    decode_mask_bits,
    is_town_gate_entry,
    read_town_snapshot,
)


def _write_u8(ram: np.ndarray, address: int, value: int) -> None:
    ram[LIVE_RAM_WRAM_OFFSET + address] = value & 0xFF


def _write_u16(ram: np.ndarray, address: int, value: int) -> None:
    base = LIVE_RAM_WRAM_OFFSET + address
    ram[base] = value & 0xFF
    ram[base + 1] = (value >> 8) & 0xFF


class TownDay1ReconTests(unittest.TestCase):
    def test_mask_field_in_catalog(self) -> None:
        spec = field_spec("d1_town_event_mask")
        self.assertEqual(spec.address, 0x11F74)
        self.assertEqual(spec.kind, "u8")
        self.assertEqual(field_spec("town_day1_event_mask").key, "d1_town_event_mask")

    def test_decode_mask_bits(self) -> None:
        self.assertEqual(decode_mask_bits(0), [])
        self.assertEqual(decode_mask_bits(0x01), ["Ann(0x01)"])
        self.assertEqual(
            decode_mask_bits(0x03),
            ["Ann(0x01)", "Eve(0x02)"],
        )
        self.assertEqual(len(decode_mask_bits(TARGET_MASK)), len(D1_TOWN_BITS))
        self.assertEqual(TARGET_MASK, 0x3F)

    def test_read_town_snapshot_from_live_ram(self) -> None:
        ram = np.zeros(LIVE_RAM_WRAM_OFFSET + 0x20000, dtype=np.uint8)
        # tilemap uses live_offset=False
        ram[field_spec("tilemap").address] = TOWN_TILEMAP
        _write_u16(ram, field_spec("player_x").address, GATE_PIXEL[0])
        _write_u16(ram, field_spec("player_y").address, GATE_PIXEL[1])
        _write_u8(ram, field_spec("day").address, 1)
        _write_u8(ram, field_spec("season").address, 0)
        _write_u8(ram, field_spec("hour").address, 7)
        _write_u8(ram, field_spec("minute").address, 0)
        # input_lock is a live-index field (no WRAM mirror offset).
        ram[field_spec("input_lock").address] = 1
        _write_u8(ram, field_spec("d1_town_event_mask").address, 0x03)

        snap = read_town_snapshot(ram, frame=12)
        self.assertEqual(snap.frame, 12)
        self.assertEqual(snap.tilemap, TOWN_TILEMAP)
        self.assertEqual((snap.x, snap.y), GATE_PIXEL)
        self.assertEqual(snap.mask, 0x03)
        self.assertEqual(snap.bits_set, ["Ann(0x01)", "Eve(0x02)"])
        self.assertIn("Nina(0x04)", snap.bits_missing)
        self.assertTrue(is_town_gate_entry(snap))
        self.assertEqual(read_ram_value(ram, "d1_town_event_mask"), 0x03)

    def test_gate_entry_rejects_wrong_map_or_day(self) -> None:
        good = TownSnapshot(
            frame=0,
            tilemap=TOWN_TILEMAP,
            x=GATE_PIXEL[0],
            y=GATE_PIXEL[1],
            day=1,
            season=0,
            hour=7,
            minute=0,
            mask=0,
            input_lock=1,
        )
        self.assertTrue(is_town_gate_entry(good))
        self.assertFalse(is_town_gate_entry(TownSnapshot(**{**good.__dict__, "tilemap": 0x02})))
        self.assertFalse(is_town_gate_entry(TownSnapshot(**{**good.__dict__, "day": 2})))
        self.assertFalse(is_town_gate_entry(TownSnapshot(**{**good.__dict__, "x": 100})))


if __name__ == "__main__":
    unittest.main()

"""Event-script decoder on a synthetic LoROM image (no ROM needed)."""

from __future__ import annotations

import unittest

from harvest.scripts import event_dis as ed


def _rom() -> bytearray:
    return bytearray(0x200000)


def _put(rom: bytearray, addr: int, data: bytes) -> None:
    off = ed.rom_offset(addr)
    rom[off : off + len(data)] = data


class DecodeTest(unittest.TestCase):
    def test_lorom_offset_matches_documented_script_bytes(self) -> None:
        self.assertEqual(ed.rom_offset(0xB38000), 0x33 * 0x8000)
        self.assertEqual(ed.rom_offset(0xB5805A), 0x35 * 0x8000 + 0x5A)

    def test_follows_jumps_and_stops_at_a_jump_back(self) -> None:
        rom = _rom()
        _put(rom, 0xB3ED5D, bytes.fromhex("12 5a ed"))
        _put(rom, 0xB3ED5A, bytes.fromhex("1f 66 ed"))
        _put(rom, 0xB3ED66, bytes.fromhex("1c 1b 01 00 12 5a ed"))
        seen = ed.decode(bytes(rom), 0xB3ED5D)
        self.assertEqual(sorted(seen), [0xB3ED5A, 0xB3ED5D, 0xB3ED66, 0xB3ED6A])
        self.assertIn("text 0x011b", seen[0xB3ED66][2])

    def test_talk_and_gift_branches_are_both_followed(self) -> None:
        rom = _rom()
        _put(rom, 0xB3D22C, bytes.fromhex("2d d4 d2 5e d2 10"))
        _put(rom, 0xB3D2D4, bytes.fromhex("10"))
        _put(rom, 0xB3D25E, bytes.fromhex("10"))
        seen = ed.decode(bytes(rom), 0xB3D22C)
        self.assertEqual(sorted(seen), [0xB3D22C, 0xB3D231, 0xB3D25E, 0xB3D2D4])

    def test_signed_heart_add_and_pointer(self) -> None:
        rom = _rom()
        _put(rom, 0xB3D2A4, bytes.fromhex("41 27 1f 7f f8 ff 10"))
        seen = ed.decode(bytes(rom), 0xB3D2A4)
        self.assertIn("[7F1F27] += -8", seen[0xB3D2A4][2])

    def test_unknown_opcode_stops_the_walk_and_says_so(self) -> None:
        rom = _rom()
        _put(rom, 0xB3F106, bytes.fromhex("4f 50 7a"))
        seen = ed.decode(bytes(rom), 0xB3F106)
        self.assertIsNone(seen[0xB3F108][1])
        self.assertIn("no known length", seen[0xB3F108][2])

    def test_event_table_ends_at_the_lowest_script(self) -> None:
        rom = _rom()
        _put(rom, 0xB3E9DA, (0xE9DA + 4).to_bytes(2, "little") + (0xE9DA + 6).to_bytes(2, "little"))
        _put(rom, 0xB3E9DE, bytes.fromhex("10"))
        starts = ed.event_starts(bytes(rom), 0xB3E9DA)
        self.assertEqual(starts, [0xB3E9DE, 0xB3E9E0])

    def test_heart_sites_report_the_events_that_reach_them(self) -> None:
        rom = _rom()
        _put(rom, ed.GROUP_TABLE, (0xB3A000).to_bytes(3, "little"))
        _put(rom, 0xB3A000, (0xA004).to_bytes(2, "little"))
        _put(rom, 0xB3A004, bytes.fromhex("41 27 1f 7f 1e 00 10"))
        sites = ed.heart_sites(bytes(rom), "eve")
        self.assertEqual(sites, [(0xB3A004, 30, ["G00.E00"])])


if __name__ == "__main__":
    unittest.main()

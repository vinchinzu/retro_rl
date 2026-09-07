"""Decode which overworld cave actually holds the White Sword.

`route/item_gate_hops.py` places it on screen 0x0A from a walkthrough letter
grid, and 0x0A has now been shown to have NO walking entrance: its west
neighbour 0x09 is a sealed pocket, its east neighbour 0x0B has no west exit,
and its south neighbour 0x1A has no north exit at any column (live BFS over
40 northern screens, probe_ow_nw_bfs.py). Before hunting further, check the
premise against ROM.

Self-validating like dump_l9_rom_rooms.py: the cave-item table is *found* by
requiring the wooden sword cave (ROM cave id 16, screen 0x77, live since M3)
to contain item 0x01 = Wood Sword, using the same item-id encoding that
decode already validated underworld-side (0x09 = Silver Arrow in L9 cellar
0x4F).

    uv run python nes/zelda_i/scratch/dump_ow_cave_items.py
"""
from __future__ import annotations

from pathlib import Path

ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
INES = 0x10

OW_SCREEN_TABLE = 0x18490  # file offset; (byte >> 2) & 0x3F = cave id
FIRST_CAVE_ID = 16
N_CAVES = 40

ITEM_NAMES = {
    0x00: "Bomb", 0x01: "Wood Sword", 0x02: "White Sword", 0x03: "Magic Sword",
    0x04: "Food", 0x05: "Recorder", 0x06: "Blue Candle", 0x07: "Red Candle",
    0x08: "Wood Arrow", 0x09: "Silver Arrow", 0x0A: "Bow", 0x0B: "Magic Key",
    0x0C: "Raft", 0x0D: "Ladder", 0x0E: "Power Bracelet", 0x0F: "Letter",
    0x10: "Blue Ring", 0x11: "Red Ring", 0x12: "Magic Rod", 0x13: "Book",
    0x14: "Blue Potion", 0x15: "Red Potion", 0x16: "Boomerang",
    0x17: "Magic Boomerang", 0x18: "Heart Container", 0x19: "Triforce Piece",
    0x1A: "Magic Shield", 0x1B: "Key", 0x1C: "Heart", 0x1D: "Rupee",
    0x1E: "5 Rupees", 0x1F: "Clock",
}


def cave_ids(rom: bytes) -> dict[int, list[int]]:
    """screen -> cave id, inverted to id -> screens."""
    by_id: dict[int, list[int]] = {}
    for scr in range(128):
        cid = (rom[OW_SCREEN_TABLE + scr] >> 2) & 0x3F
        if cid:
            by_id.setdefault(cid, []).append(scr)
    return by_id


def find_item_tables(rom: bytes) -> list[tuple[int, int]]:
    """(base, slot) windows where cave 16 slot `slot` holds a Wood Sword."""
    hits = []
    for base in range(0, len(rom) - N_CAVES * 3):
        for slot in range(3):
            if rom[base + slot] != 0x01:
                continue
            vals = [rom[base + i * 3 + slot] for i in range(N_CAVES)]
            if 0x02 in vals and 0x03 in vals and all(v <= 0x1F for v in vals):
                hits.append((base, slot))
    return hits


def main() -> int:
    rom = ROM.read_bytes()
    by_id = cave_ids(rom)
    print(f"wooden sword cave id 16 -> screens "
          f"{[hex(s) for s in by_id.get(16, [])]}")
    print(f"cave id 18 -> screens {[hex(s) for s in by_id.get(18, [])]}")

    hits = find_item_tables(rom)
    if not hits:
        print("\nno 3-slot cave table puts a Wood Sword in cave 16")
        return 1
    for base, slot in hits[:6]:
        print(f"\n== cave item table @ file 0x{base:05X} (PRG 0x{base - INES:05X}) "
              f"slot {slot} ==")
        for i in range(N_CAVES):
            cid = FIRST_CAVE_ID + i
            item = rom[base + i * 3 + slot]
            if item in (0x01, 0x02, 0x03) or cid in (16, 18, 19):
                screens = by_id.get(cid, [])
                print(f"   cave id {cid:3d}: item 0x{item:02X} "
                      f"{ITEM_NAMES.get(item, '?'):16s} screens "
                      f"{[hex(s) for s in screens]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

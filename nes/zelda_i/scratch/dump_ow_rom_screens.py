"""Find the White Sword cave's real overworld screen straight out of the ROM.

`route/item_gate_hops.py` puts it at 0x0A, decoded from a walkthrough letter
grid ("GameFAQs K-1"), and every live approach to 0x0A has now failed: west
off the Level 5 door 0x0B is sealed at every band (its north half is
mountain), and so is west off Lost Hills 0x1B. Before building a hop table
toward a screen that may simply be wrong, derive the cave screens from ROM.

Nothing is taken on faith. The overworld attribute table is *found* by
searching the ROM for a 128-byte window that reproduces the repo's live
dungeon-entrance anchors (L4 0x45, L5 0x0B, L6 0x22, L7 0x42, L8 0x6D,
L9 0x05 -> cave ids 4..9), the same self-validating trick
`dump_l9_rom_rooms.py` uses for the underworld.

    uv run python nes/zelda_i/scratch/dump_ow_rom_screens.py
"""
from __future__ import annotations

from pathlib import Path

ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
INES = 0x10

# Live anchors from zelda_i/anchors.py: screen -> dungeon number.
ENTRANCE_ANCHORS = {0x45: 4, 0x0B: 5, 0x22: 6, 0x42: 7, 0x6D: 8, 0x05: 9}
# The wooden sword cave, live since M3 (overworld/sword_cave.py).
WOODEN_SWORD_SCREEN = 0x77


def screens(rom: bytes, base: int) -> list[int]:
    return list(rom[base:base + 128])


def find_tables(rom: bytes) -> list[tuple[int, int, int]]:
    """Return (offset, shift, mask) windows matching every entrance anchor."""
    hits: list[tuple[int, int, int]] = []
    for shift, mask in ((0, 0x3F), (0, 0x1F), (0, 0x7F), (0, 0xFF), (2, 0x3F), (1, 0x3F)):
        for base in range(0, len(rom) - 128):
            ok = True
            for scr, want in ENTRANCE_ANCHORS.items():
                if ((rom[base + scr] >> shift) & mask) != want:
                    ok = False
                    break
            if ok:
                hits.append((base, shift, mask))
    return hits


def main() -> int:
    rom = ROM.read_bytes()
    print(f"rom={ROM.name} size={len(rom)}")
    hits = find_tables(rom)
    if not hits:
        print("no table reproduces all six entrance anchors")
        return 1
    for base, shift, mask in hits:
        prg = base - INES
        table = screens(rom, base)
        vals = [(v >> shift) & mask for v in table]
        print(f"\n== candidate table @ file 0x{base:05X} (PRG 0x{prg:05X}) "
              f"shift={shift} mask=0x{mask:02X} ==")
        for scr, want in sorted(ENTRANCE_ANCHORS.items()):
            print(f"   anchor screen 0x{scr:02X} -> {vals[scr]} (want {want}) OK")
        print(f"   wooden sword screen 0x{WOODEN_SWORD_SCREEN:02X} -> {vals[WOODEN_SWORD_SCREEN]}")
        nonzero = {s: v for s, v in enumerate(vals) if v}
        print(f"   {len(nonzero)} screens carry a nonzero id")
        by_id: dict[int, list[int]] = {}
        for s, v in nonzero.items():
            by_id.setdefault(v, []).append(s)
        for v in sorted(by_id):
            print(f"    id {v:3d} (0x{v:02X}): " +
                  " ".join(f"0x{s:02X}" for s in sorted(by_id[v])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

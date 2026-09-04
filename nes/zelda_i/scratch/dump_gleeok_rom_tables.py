"""Dump Zelda I ROM tables used by the rr-5eb2 Gleeok pre-fight model.

Offsets: ZeldaHacks / Data Crystal are PRG-ROM (add 16 for iNES).
Does not boot the emulator. Read-only.
"""

from __future__ import annotations

from pathlib import Path

ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
INES = 0x10

# Bank 7 is fixed at CPU $C000. ObjectTypeToAttributes is at 07:FAEF
# (aldonunez zelda1-disassembly Z_07.asm comment).
CPU_OBJ_ATTR = 0xFAEF
BANK7_PRG = 7 * 0x4000  # 0x1C000
OBJ_ATTR_PRG = BANK7_PRG + (CPU_OBJ_ATTR - 0xC000)  # 0x1FAEF

# ZeldaHacks (PRG offsets, not iNES)
L8_INFO_PRG = 105184  # 0x19B20, 252 bytes
L1_INFO_PRG = 103420
L4_INFO_PRG = 104176
L6_INFO_PRG = 104680
UW_L16_PRG = 100096
UW_L79_PRG = 100864
GLEEOK_PTR_PRG = 0x12809  # Data Crystal ROM map (PRG)


def hp_from_pair(pair: int, obj_type: int) -> int:
    """Match ExtractHitPointValue (Z_04.asm): even = high nibble as $x0."""
    if obj_type & 1:
        return (pair & 0x0F) << 4
    return pair & 0xF0


def main() -> None:
    data = ROM.read_bytes()
    assert data[:4] == b"NES\x1a", data[:16]
    prg = data[INES:]
    print(f"rom={ROM}")
    print(f"size={len(data)} prg={len(prg)} header={data[:16].hex()}")

    attrs = prg[OBJ_ATTR_PRG : OBJ_ATTR_PRG + 95]
    hp_pairs = prg[OBJ_ATTR_PRG + 95 : OBJ_ATTR_PRG + 95 + 38]
    print(f"\nObjectTypeToAttributes PRG 0x{OBJ_ATTR_PRG:05X} iNES 0x{OBJ_ATTR_PRG + INES:05X}")
    print(f"ObjectTypeToHpPairs     PRG 0x{OBJ_ATTR_PRG + 95:05X} iNES 0x{OBJ_ATTR_PRG + 95 + INES:05X}")
    print("type attr hp_pair extracted_hp")
    for t in range(0x40, 0x49):
        pair = hp_pairs[t // 2]
        print(
            f"  0x{t:02X}  0x{attrs[t]:02X}  pair[{t // 2}]=0x{pair:02X}  HP=0x{hp_from_pair(pair, t):02X}"
            f" ({hp_from_pair(pair, t)})"
        )

    print("\nGleeok pointer table Data Crystal 0x12809:")
    ptr = prg[GLEEOK_PTR_PRG : GLEEOK_PTR_PRG + 18]
    print(f"  PRG 0x{GLEEOK_PTR_PRG:05X} iNES 0x{GLEEOK_PTR_PRG + INES:05X} bytes={ptr.hex()}")

    def dump_info(name: str, off: int) -> None:
        block = prg[off : off + 252]
        entrance = block[47]
        triforce = block[48]
        level_n = block[51]
        boss = block[62]
        print(
            f"\n{name} info PRG 0x{off:05X} iNES 0x{off + INES:05X}"
            f" entrance=0x{entrance:02X} tf_room=0x{triforce:02X}"
            f" level#=0x{level_n:02X} boss=0x{boss:02X}"
        )
        print(f"  +0x2F..+0x40: {block[47:65].hex()}")

    dump_info("L1", L1_INFO_PRG)
    dump_info("L4", L4_INFO_PRG)
    dump_info("L6", L6_INFO_PRG)
    dump_info("L8", L8_INFO_PRG)

    print("\nUW L1-6 data PRG 0x18640 / L7-9 PRG 0x18A00 (768 bytes each)")
    for label, base in (("L1-6", UW_L16_PRG), ("L7-9", UW_L79_PRG)):
        blob = prg[base : base + 768]
        hits = []
        for i, b in enumerate(blob):
            if b in (0x42, 0x43, 0x44, 0x45, 0x46):
                hits.append((i, b))
        print(f"  {label} bytes matching Gleeok types 0x42-0x46: {len(hits)}")
        for i, b in hits[:40]:
            print(f"    off+{i:3d} (0x{base + i:05X}) = 0x{b:02X}")

    # Room extra: after doors (256) there are 128 more before the map at +384.
    # Dump hypothesized L8 boss room 0x3C and live L4 0x13 / L6 0x18 counterparts.
    print("\nPer-room bytes in UW data (doors N/S, W/E, extra@256, map@384):")
    for label, base, rooms in (
        ("L1-6", UW_L16_PRG, (0x13, 0x18)),
        ("L7-9", UW_L79_PRG, (0x3C, 0x2C, 0x7E, 0x1E, 0x2E)),
    ):
        blob = prg[base : base + 768]
        for room in rooms:
            ns = blob[room]
            we = blob[128 + room]
            extra = blob[256 + room]
            layout = blob[384 + room]
            print(
                f"  {label} room 0x{room:02X}: NS=0x{ns:02X} WE=0x{we:02X}"
                f" extra256=0x{extra:02X} map384=0x{layout:02X}"
            )
            for k, name in ((512, "+512"), (640, "+640")):
                if k + room < 768:
                    print(f"    {name}[room]=0x{blob[k + room]:02X}")


def find_mix_table(prg: bytes) -> None:
    """Search for a 4-byte object-list table indexed by UW extra256.

    L4 Gleeok room 0x13 extra256=0x03 and live type 0x43.
    L6 Gleeok 0x18 extra256=0x04 and live type 0x44.
    If mix 0x05 is L8, ROM would store type 0x45 there.
    """
    print("\nSearch 4-byte groups: idx3 has 0x43, idx4 has 0x44:")
    hits = 0
    for off in range(0, len(prg) - 4 * 8):
        g3 = prg[off + 3 * 4 : off + 4 * 4]
        g4 = prg[off + 4 * 4 : off + 5 * 4]
        if 0x43 in g3 and 0x44 in g4:
            g5 = prg[off + 5 * 4 : off + 6 * 4]
            print(
                f"  PRG 0x{off:05X} iNES 0x{off + INES:05X}"
                f"  [3]={g3.hex()} [4]={g4.hex()} [5]={g5.hex()}"
            )
            hits += 1
            if hits >= 12:
                break
    print(f"  (stopped after {hits} hits)" if hits else "  none")

    print("\nSearch 1-byte lists where table[3]=0x43, table[4]=0x44, table[5]=0x45:")
    hits = 0
    for off in range(0, len(prg) - 8):
        if prg[off + 3] == 0x43 and prg[off + 4] == 0x44 and prg[off + 5] == 0x45:
            ctx = prg[off : off + 8]
            print(f"  PRG 0x{off:05X} iNES 0x{off + INES:05X} bytes={ctx.hex()}")
            hits += 1
            if hits >= 8:
                break


if __name__ == "__main__":
    data = ROM.read_bytes()
    prg = data[INES:]
    main()
    find_mix_table(prg)


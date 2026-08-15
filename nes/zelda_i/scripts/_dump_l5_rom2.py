"""Dump Q1 level-info blocks and L5 cellar/path rooms."""
import zipfile
from pathlib import Path

zpath = Path("/home/v/01_projects/11_games/retro_rl/roms/Nintendo/NES/Legend of Zelda, The.zip")
for cand in (Path("roms/Nintendo/NES/Legend of Zelda, The.zip"), zpath):
    if cand.exists():
        zpath = cand
        break

with zipfile.ZipFile(zpath) as zf:
    data = zf.read(zf.namelist()[0])


def b(dc: int) -> int:
    return data[dc + 0x10]


CODES = {0: "open", 1: "wall", 2: "false", 3: "false2", 4: "bomb", 5: "key", 6: "key2", 7: "shutter"}

print("=== blocks with entrance in {0x73,0x76,0x7d} and sane cellar ===")
for base in range(0x19000, 0x1A000):
    ent = b(base + 0x0B)
    lvl = b(base + 0x0F)
    cell = [b(base + 0x10 + i) for i in range(6)]
    if ent not in (0x73, 0x76, 0x7D, 0x10) or lvl == 0 or lvl > 9:
        continue
    # cellar rooms should look like room ids (mostly < 0x80, not all FF)
    if all(c == 0xFF for c in cell):
        continue
    print(
        f"{base:05X} ent={ent:02X} tf={b(base+0x0C):02X} lvl={lvl:02X} "
        f"map={b(base+0x09):02X}/{b(base+0x0A):02X} "
        f"items={[hex(b(base+0x05+i)) for i in range(4)]} "
        f"cellar={[hex(c) for c in cell]} "
        f"stair4={[hex(b(base+0x16+i)) for i in range(4)]} boss={b(base+0x1A):02X}"
    )

print("\n=== raw L5-ish rooms ===")
for r in (0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17, 0x24, 0x25, 0x64, 0x65, 0x66):
    ns, ew = b(0x18700 + r), b(0x18780 + r)
    mon, dbyte = b(0x18800 + r), b(0x18880 + r)
    item, flags = b(0x18900 + r), b(0x18980 + r)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    print(
        f"0x{r:02X} ns={ns:02X} ew={ew:02X} N={CODES[n]} S={CODES[s]} "
        f"W={CODES[w]} E={CODES[e]} mon={mon:02X} D={dbyte:02X} "
        f"item={item:02X} flags={flags:02X} secret={flags & 7}"
    )

print("\n=== search entrance 0x73 (L1) ===")
for base in range(0x19000, 0x1A000):
    if b(base + 0x0B) == 0x73 and 1 <= b(base + 0x0F) <= 9:
        cell = [hex(b(base + 0x10 + i)) for i in range(6)]
        print(f"{base:05X} lvl={b(base+0x0F):02X} tf={b(base+0x0C):02X} cellar={cell} boss={b(base+0x1A):02X}")

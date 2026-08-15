
import zipfile
from pathlib import Path

zpath = Path("roms/Nintendo/NES/Legend of Zelda, The.zip")
with zipfile.ZipFile(zpath) as zf:
    data = zf.read(zf.namelist()[0])

def b(dc):
    return data[dc + 0x10]

print("L9 info block 19C00-19C2A")
for i in range(0x19C00, 0x19C2B):
    print(f"  {i:05X} = {b(i):02X}")

print("\n4 item tiles 19C05-19C08", [hex(b(0x19C05+i)) for i in range(4)])
print("map coord", hex(b(0x19C09)), "cursor", hex(b(0x19C0A)))
print("entrance", hex(b(0x19C0B)), "triforce room", hex(b(0x19C0C)))
print("level#", hex(b(0x19C0F)))
print("cellar/stair 6", [hex(b(0x19C10+i)) for i in range(6)])
print("next 4", [hex(b(0x19C16+i)) for i in range(4)])
print("boss", hex(b(0x19C1A)))

# UW square table DC 91928
print("\nUW square table 91928", [hex(b(0x91928+i)) for i in range(8)])

# 42 dungeon room layouts at 160DE, 12 bytes each
print("\nUnique layouts 0x00-0x29 (12 col bytes). Flag columns containing square 5/stairs-ish:")
base = 0x160DE
for uid in range(42):
    cols = [b(base + uid*12 + c) for c in range(12)]
    print(f"  uid 0x{uid:02X}: " + " ".join(f"{v:02X}" for v in cols))

# Also dump unique 0x3E if the table is actually 64
print("\nIf 64 layouts, uid 0x3E/0x3F:")
for uid in (0x3E, 0x3F, 0x3C, 0x3D):
    cols = [b(base + uid*12 + c) for c in range(12)]
    print(f"  uid 0x{uid:02X}: " + " ".join(f"{v:02X}" for v in cols))

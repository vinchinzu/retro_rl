
import zipfile
from pathlib import Path

zpath = Path("roms/Nintendo/NES/Legend of Zelda, The.zip")
with zipfile.ZipFile(zpath) as zf:
    data = zf.read(zf.namelist()[0])

def b(dc):
    return data[dc + 0x10]

CODES = {0: "open", 1: "wall", 2: "false", 3: "false2", 4: "bomb", 5: "key", 6: "key2", 7: "shutter"}
SECRETS = {
    0: "none",
    1: "all_dead",
    2: "ringleader",
    3: "last_boss",
    4: "block_door",
    5: "block_stairs",
    6: "money_or_life",
    7: "foes_item",
}

print("L9 stair list", [hex(b(0x19C10 + i)) for i in range(10)])
print("L9 entrance", hex(b(0x19C0B)), "boss", hex(b(0x19C1A)))
print()
rooms = [0x00, 0x4F, 0x52, 0x42, 0x32, 0x60, 0x62, 0x67, 0x70, 0x72, 0x75, 0x77]
print("room  NS N/S            EW W/E            mon  D    item F    secret")
for r in rooms:
    ns = b(0x18A00 + r)
    ew = b(0x18A80 + r)
    mon = b(0x18B00 + r)
    d = b(0x18B80 + r)
    item = b(0x18C00 + r)
    f = b(0x18C80 + r)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    secret = f & 7
    print(
        f"0x{r:02X}  {ns:02X} {n}/{s} {CODES.get(n,'?'):7}/{CODES.get(s,'?'):7}  "
        f"{ew:02X} {w}/{e} {CODES.get(w,'?'):7}/{CODES.get(e,'?'):7}  "
        f"{mon:02X}  {d:02X}  {item:02X}  {f:02X}  {secret} {SECRETS.get(secret,'?')}"
    )

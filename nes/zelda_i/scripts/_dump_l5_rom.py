"""Print first-quest L1-6 ROM room rows and L5 level-info / cellar list."""
import zipfile
from pathlib import Path

zpath = Path("/home/v/01_projects/11_games/retro_rl/roms/Nintendo/NES/Legend of Zelda, The.zip")
# Prefer cwd-relative if copied to repo
for cand in (
    Path("roms/Nintendo/NES/Legend of Zelda, The.zip"),
    zpath,
):
    if cand.exists():
        zpath = cand
        break

with zipfile.ZipFile(zpath) as zf:
    data = zf.read(zf.namelist()[0])


def b(dc: int) -> int:
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
ITEMS = {
    0x00: "bomb",
    0x01: "wood_boom",
    0x02: "magic_boom",
    0x03: "none_or_bow",
    0x04: "blue_candle",
    0x05: "WHISTLE",
    0x06: "food",
    0x07: "letter_or_potion",
    0x08: "rod",
    0x09: "raft",
    0x0A: "book",
    0x0B: "ring",
    0x0C: "ladder",
    0x0D: "magic_key",
    0x0E: "bracelet",
    0x0F: "letter",
    0x16: "compass",
    0x17: "map",
    0x19: "key",
    0x1A: "heart_container",
    0x1B: "triforce",
}

print("=== scan 0x19000-0x19FFF for entrance 0x76 or level# 5 ===")
for base in range(0x19000, 0x1A000):
    if b(base + 0x0B) == 0x76 or b(base + 0x0F) == 5:
        print(
            f"base {base:05X} ent={b(base+0x0B):02X} tf={b(base+0x0C):02X} "
            f"lvl={b(base+0x0F):02X} items={[hex(b(base+0x05+i)) for i in range(4)]} "
            f"cellar={[hex(b(base+0x10+i)) for i in range(6)]} "
            f"next4={[hex(b(base+0x16+i)) for i in range(4)]} boss={b(base+0x1A):02X}"
        )

print("\n=== stride hunt ending at L9=0x19C00 ===")
for size in (0x2B, 0x2C, 0x30, 0x40):
    start = 0x19C00 - 8 * size
    ents = [b(start + i * size + 0x0B) for i in range(9)]
    lvls = [b(start + i * size + 0x0F) for i in range(9)]
    print(f"size {size:02X} start {start:05X} ents={[hex(v) for v in ents]} lvls={lvls}")

print("\n=== L1-6 Q1 rooms (0x18700 family) ===")
rooms = [
    0x04, 0x05, 0x06, 0x07,
    0x13, 0x14, 0x15, 0x16, 0x17,
    0x23, 0x24, 0x25, 0x26, 0x27,
    0x33, 0x34, 0x35, 0x36, 0x37,
    0x43, 0x44, 0x45, 0x46, 0x47,
    0x53, 0x54, 0x55, 0x56, 0x57,
    0x63, 0x64, 0x65, 0x66, 0x67,
    0x73, 0x74, 0x75, 0x76, 0x77,
]
print(f"{'rm':5} {'N':8} {'S':8} {'W':8} {'E':8} {'secret':14} mon  D    item name")
for r in rooms:
    ns = b(0x18700 + r)
    ew = b(0x18780 + r)
    mon = b(0x18800 + r)
    dbyte = b(0x18880 + r)
    item = b(0x18900 + r)
    flags = b(0x18980 + r)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    secret = flags & 7
    print(
        f"0x{r:02X} {CODES.get(n,'?'):8} {CODES.get(s,'?'):8} "
        f"{CODES.get(w,'?'):8} {CODES.get(e,'?'):8} "
        f"{SECRETS.get(secret,'?'):14} {mon:02X}  {dbyte:02X}  "
        f"{item:02X} {ITEMS.get(item, '?')}"
    )

print("\n=== all L1-6 rooms with secret=block_stairs ===")
for r in range(128):
    flags = b(0x18980 + r)
    if (flags & 7) != 5:
        continue
    item = b(0x18900 + r)
    mon = b(0x18800 + r)
    ns = b(0x18700 + r)
    ew = b(0x18780 + r)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    print(
        f"  0x{r:02X} N={CODES[n]} S={CODES[s]} W={CODES[w]} E={CODES[e]} "
        f"mon={mon:02X} item={item:02X} {ITEMS.get(item,'?')} flags={flags:02X}"
    )

print("\n=== all L1-6 rooms with ROM item 0x05 (whistle) ===")
for r in range(128):
    item = b(0x18900 + r)
    if item != 0x05:
        continue
    flags = b(0x18980 + r)
    print(
        f"  0x{r:02X} secret={SECRETS.get(flags & 7)} flags={flags:02X} "
        f"mon={b(0x18800+r):02X}"
    )

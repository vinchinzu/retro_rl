"""Dump first-quest L7 stairway / cellar tables (no emulator).

Offsets: Data Crystal / ZeldaHacks are PRG-ROM (add 16 for iNES).
Calibrated against live L6 cellar 0x08 (AttrA=0x3A, AttrB=0x1D) and
L9 CellarRoomIdArray at DC 0x19C10.

CheckWarps (aldonunez Z_05.asm): scan LevelInfo_CellarRoomIdArray (6
ids); the first cellar whose LevelBlockAttrsA/B equals the play RoomId
is the dest, mode 9. In cellars those same bytes are dest room ids, not
door codes. InitMode9_EnterCellar: source==AttrA -> left ladder x=$30,
else right x=$C0. CheckSubroom: Y<$40 + UP, X<$80 -> AttrA else AttrB.
"""

from __future__ import annotations

from pathlib import Path

ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
INES = 0x10

# ZeldaHacks Level Info Blocks (PRG, 252 bytes). Stairway list is +52.
L1_INFO_PRG = 103420
L6_INFO_PRG = 104680
L7_INFO_PRG = 104932  # 0x199E4
L8_INFO_PRG = 105184
L9_INFO_PRG = 105436

# Data Crystal "beginning of stairway list" = info+52. L9 known: 0x19C10.
L7_STAIR_LIST_PRG = L7_INFO_PRG + 52  # 0x19A18
L9_STAIR_LIST_PRG = 0x19C10  # documented, not L9_INFO_PRG+52

UW_L16_PRG = 100096  # 0x18640
UW_L79_PRG = 100864  # 0x18A00
UW_SIZE = 768

DOOR_NAMES = {
    0: "open",
    1: "wall",
    2: "false",
    3: "false2",
    4: "bomb",
    5: "key",
    6: "key2",
    7: "shutter",
}
SECRET_NAMES = {
    0: "none",
    1: "all_dead",
    2: "stand_push",
    3: "secret_3",
    4: "secret_4",
    5: "block_stairs",
    6: "secret_6",
    7: "foes_item",
}


def _hex_list(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def _doors(ns: int, we: int) -> tuple[int, int, int, int]:
    north = (ns >> 5) & 7
    south = (ns >> 2) & 7
    west = (we >> 5) & 7
    east = (we >> 2) & 7
    return north, south, west, east


def _uw_room(blob: bytes, room: int) -> dict[str, int]:
    ns = blob[room]
    we = blob[128 + room]
    extra = blob[256 + room]
    layout = blob[384 + room]
    d = blob[512 + room]
    e = blob[640 + room]
    north, south, west, east = _doors(ns, we)
    return {
        "room": room,
        "ns": ns,
        "we": we,
        "attr_a": ns,
        "attr_b": we,
        "extra256": extra,
        "layout384": layout,
        "attr_d": d,
        "attr_e": e,
        "n": north,
        "s": south,
        "w": west,
        "e": east,
        "secret_e_hi": (e >> 5) & 7,
        "secret_e_lo": e & 7,
        "secret_d_hi": (d >> 5) & 7,
        "secret_d_lo": d & 7,
        "push_block": bool(d & 0x40),
        "item_lo5": e & 0x1F,
    }


def _fmt_play(row: dict[str, int]) -> str:
    n, s, w, e = row["n"], row["s"], row["w"], row["e"]
    return (
        f"  play 0x{row['room']:02X}: NS=0x{row['ns']:02X} WE=0x{row['we']:02X}"
        f" N={n}({DOOR_NAMES.get(n,'?')}) S={s}({DOOR_NAMES.get(s,'?')})"
        f" W={w}({DOOR_NAMES.get(w,'?')}) E={e}({DOOR_NAMES.get(e,'?')})"
        f" extra256=0x{row['extra256']:02X} layout=0x{row['layout384']:02X}"
        f" D=0x{row['attr_d']:02X} E=0x{row['attr_e']:02X}"
        f" push={int(row['push_block'])} item={row['item_lo5']:02X}"
        f" secretDhi={row['secret_d_hi']}({SECRET_NAMES.get(row['secret_d_hi'],'?')})"
        f" secretElo={row['secret_e_lo']}({SECRET_NAMES.get(row['secret_e_lo'],'?')})"
    )


def _fmt_cellar(row: dict[str, int]) -> str:
    return (
        f"  cellar 0x{row['room']:02X}: AttrA=0x{row['attr_a']:02X}"
        f" AttrB=0x{row['attr_b']:02X}"
        f" extra256=0x{row['extra256']:02X} layout=0x{row['layout384']:02X}"
        f" D=0x{row['attr_d']:02X} E=0x{row['attr_e']:02X}"
        f" item={row['item_lo5']:02X}"
    )


def _dump_info(name: str, off: int, prg: bytes) -> None:
    block = prg[off : off + 252]
    entrance = block[47]
    triforce = block[48]
    level_n = block[51]
    stairs = block[52:60]
    boss = block[62]
    print(
        f"\n{name} info PRG 0x{off:05X} iNES 0x{off + INES:05X}"
        f" entrance=0x{entrance:02X} tf_room=0x{triforce:02X}"
        f" level#=0x{level_n:02X} boss=0x{boss:02X}"
    )
    print(f"  +0x2F..+0x40: {_hex_list(block[47:65])}")
    print(f"  stairway +52 (8): {_hex_list(stairs)}  rooms={[f'0x{b:02X}' for b in stairs]}")


def _find_cellar_for_play(blob: bytes, cellars: bytes, play: int) -> list[tuple[int, str, int, int]]:
    hits: list[tuple[int, str, int, int]] = []
    for cellar in cellars:
        row = _uw_room(blob, cellar)
        a, b = row["attr_a"], row["attr_b"]
        if a == play:
            hits.append((cellar, "A", a, b))
        if b == play:
            hits.append((cellar, "B", a, b))
    return hits


def main() -> None:
    data = ROM.read_bytes()
    assert data[:4] == b"NES\x1a", data[:16]
    prg = data[INES:]
    print(f"rom={ROM}")
    print(f"size={len(data)} prg={len(prg)} header={data[:16].hex()}")

    _dump_info("L6", L6_INFO_PRG, prg)
    _dump_info("L7", L7_INFO_PRG, prg)
    _dump_info("L8", L8_INFO_PRG, prg)
    _dump_info("L9", L9_INFO_PRG, prg)

    print("\n=== Stairway list at documented DC offsets ===")
    for label, off in (
        ("L6 DC-arith", L6_INFO_PRG + 52),
        ("L7 DC-arith", L7_STAIR_LIST_PRG),
        ("L8 DC-arith", L8_INFO_PRG + 52),
        ("L9 documented", L9_STAIR_LIST_PRG),
        ("L9 ZeldaHacks+52", L9_INFO_PRG + 52),
    ):
        blob = prg[off : off + 8]
        print(f"  {label:20s} PRG 0x{off:05X} iNES 0x{off + INES:05X}  {_hex_list(blob)}")

    l16 = prg[UW_L16_PRG : UW_L16_PRG + UW_SIZE]
    l79 = prg[UW_L79_PRG : UW_L79_PRG + UW_SIZE]

    print("\n=== Calibration: L6 cellar 0x08 (live AttrA=0x3A AttrB=0x1D) ===")
    print(_fmt_cellar(_uw_room(l16, 0x08)))
    print(_fmt_play(_uw_room(l16, 0x3A)))
    print(_fmt_play(_uw_room(l16, 0x1D)))

    print("\n=== Calibration: L9 cellar 0x77 (live left 0x52 right 0x03) ===")
    print(_fmt_cellar(_uw_room(l79, 0x77)))
    print(_fmt_play(_uw_room(l79, 0x52)))
    print(_fmt_play(_uw_room(l79, 0x03)))
    print(_fmt_play(_uw_room(l79, 0x30)))  # known block_stairs secret=5

    l7_stairs = bytes(prg[L7_STAIR_LIST_PRG : L7_STAIR_LIST_PRG + 8])
    l7_cellars = l7_stairs[:6]
    print("\n=== L7 CellarRoomIdArray (first 6 of stairway list) ===")
    print(f"  PRG 0x{L7_STAIR_LIST_PRG:05X} iNES 0x{L7_STAIR_LIST_PRG + INES:05X}")
    print(f"  8-byte list: {_hex_list(l7_stairs)}")
    print(f"  6 cellars:   {[f'0x{b:02X}' for b in l7_cellars]}")

    print("\n=== L7 cellar AttrA/AttrB (dest play rooms) ===")
    real_cellars = bytes(b for b in l7_cellars if b != 0xFF)
    for cellar in real_cellars:
        row = _uw_room(l79, cellar)
        print(_fmt_cellar(row))
        kind = "treasure (A==B)" if row["attr_a"] == row["attr_b"] else "tunnel (A!=B)"
        print(
            f"    {kind}: left/A play 0x{row['attr_a']:02X} <-> right/B play 0x{row['attr_b']:02X}"
        )

    print("\n=== CheckWarps reverse map: which L7 cellar claims play rooms ===")
    for play in (0x0D, 0x29, 0x1A, 0x0C, 0x1C, 0x1B, 0x1D, 0x0E, 0x0B, 0x2A, 0x2B, 0x7B, 0x4A):
        hits = _find_cellar_for_play(l79, real_cellars, play)
        if not hits:
            print(f"  play 0x{play:02X}: NO cellar in L7 array")
            continue
        for cellar, side, a, b in hits:
            other = b if side == "A" else a
            print(
                f"  play 0x{play:02X}: cellar 0x{cellar:02X} as {side}"
                f" (A=0x{a:02X} B=0x{b:02X}); other endpoint 0x{other:02X}"
            )

    print("\n=== L7 play-room layout for nose / candle / boss path ===")
    for room in (
        0x0D,
        0x0C,
        0x0E,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1D,
        0x28,
        0x29,
        0x2A,
        0x2B,
        0x2C,
        0x39,
        0x3A,
        0x4A,
        0x7B,
        0x79,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x18,
    ):
        print(_fmt_play(_uw_room(l79, room)))

    print("\n=== All L7-9 rooms whose AttrA or AttrB is 0x0D (even outside the 6) ===")
    for room in range(128):
        row = _uw_room(l79, room)
        if row["attr_a"] == 0x0D or row["attr_b"] == 0x0D:
            in_list = room in l7_cellars
            print(
                f"  room 0x{room:02X} A=0x{row['attr_a']:02X} B=0x{row['attr_b']:02X}"
                f" in_L7_cellar_array={in_list}"
            )

    print("\n=== L7-9 rooms with AttrE secret nibble 5 (block_stairs) ===")
    for room in range(128):
        row = _uw_room(l79, room)
        if row["secret_e_lo"] == 5:
            print(
                f"  0x{room:02X} E=0x{row['attr_e']:02X} D=0x{row['attr_d']:02X}"
                f" N={row['n']} S={row['s']} W={row['w']} E={row['e']}"
            )


if __name__ == "__main__":
    main()

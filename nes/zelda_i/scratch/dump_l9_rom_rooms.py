"""Decode the first-quest underworld room tables straight out of the ROM.

Answers "where are the Level 9 Silver Arrows?" without booting the emulator.
Read-only; prints a report and optionally writes JSON.

Offsets below are **PRG** (iNES file offset = PRG + 0x10). Nothing here is
taken on faith: every field is re-derived and checked against the live
anchors in `nes/zelda_i/docs/LEVEL*_ROUTE.md` (see ANCHORS / main()).

Layout that validated (128-byte parallel tables, 6 per quest-1 level block):

    PRG 0x18700  levels 1-6 room attrs   (6 x 128 bytes)
    PRG 0x18A00  levels 7-9 room attrs   (6 x 128 bytes)

    t0[room]  bits 7-5 north door, bits 4-2 south door, bits 1-0 unknown
    t1[room]  bits 7-5 west  door, bits 4-2 east  door, bits 1-0 unknown
              (for a *cellar* room t0/t1 are instead the left/right
               stairway destination room ids -- no bit packing)
    t2[room]  bits 7-6 object-count index, bits 5-0 object type low 6 bits
    t3[room]  bit 7 = object type bit 6; bits 6-0 unidentified
    t4[room]  bits 4-0 room item id, bit 7 = dark room, bits 6-5 unknown
    t5[room]  bits 2-0 "secret" (item/door gating), bits 5-4 unknown

    object count = LevelInfo[+0x24 + count_index]  -> (3, 5, 6, 8)
    object type >= 0x62 is not a type: it indexes the underworld object-list
    table (pointers PRG 0x1473F, lists from PRG 0x14676) whose length is the
    room's object count.

Level info blocks: PRG 0x193FC + (level-1) * 252
    +0x24..+0x27 object-count table   +0x2F entrance room
    +0x30 triforce/goal room          +0x33 level number
    +0x34..     cellar room id array, 0xFF terminated
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

ROM = Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
INES = 0x10

UW_Q1_L16_PRG = 0x18700
UW_Q1_L79_PRG = 0x18A00
LEVEL_INFO_L1_PRG = 0x193FC
LEVEL_INFO_STRIDE = 252

OBJ_LIST_PTRS_PRG = 0x1473F  # 30 little-endian CPU pointers, bank 5 ($8000)
OBJ_LIST_BANK5_PRG = 0x14000  # CPU $8000 -> PRG 0x14000
OBJ_LIST_FIRST_TYPE = 0x62  # object type >= this indexes the pointer table

DOOR_NAMES = {
    0: "open",
    1: "wall",
    2: "walkthrough",  # unvalidated (no live anchor)
    3: "walkthrough2",  # unvalidated
    4: "bombable",
    5: "locked",
    6: "locked2",  # unvalidated
    7: "shutter",
}
DIRS = ("N", "S", "W", "E")
DELTA = {"N": -0x10, "S": 0x10, "W": -1, "E": 1}

SECRET_NAMES = {
    0: "none",
    1: "foes_open_door",
    2: "ringleader",
    3: "last_boss",
    4: "block_opens_door",
    5: "block_reveals_stairs",
    6: "money_or_life",
    7: "foes_drop_item",
}

ITEM_NAMES = {
    0x00: "bomb",
    0x01: "wood_sword",
    0x02: "white_sword",
    0x03: "NONE",
    0x04: "food",
    0x05: "recorder",
    0x06: "blue_candle",
    0x07: "red_candle",
    0x08: "wood_arrow",
    0x09: "SILVER_ARROW",
    0x0A: "bow",
    0x0B: "magic_key",
    0x0C: "raft",
    0x0D: "ladder",
    0x0E: "power_triforce",
    0x0F: "5_rupees",
    0x10: "magic_rod",
    0x11: "book",
    0x12: "blue_ring",
    0x13: "red_ring",
    0x14: "bracelet",
    0x15: "letter",
    0x16: "compass",
    0x17: "map",
    0x18: "rupee",
    0x19: "key",
    0x1A: "heart_container",
    0x1B: "triforce_piece",
    0x1C: "magic_shield",
    0x1D: "boomerang",
    0x1E: "magic_boomerang",
    0x1F: "blue_potion",
}

# Names for object types this repo has already live-verified (zelda_i.dungeon.ids)
# plus the few extra the ROM turns up.
OBJ_NAMES = {
    0x00: "-",
    0x05: "goriya_blue",
    0x06: "goriya",
    0x0B: "darknut",
    0x0C: "darknut_blue",
    0x12: "vire",
    0x13: "zol",
    0x15: "gel",
    0x16: "pols_voice",
    0x17: "like_like",
    0x18: "zol_or_gel_variant",
    0x1B: "keese",
    0x23: "wizzrobe_blue",
    0x24: "wizzrobe_orange",
    0x27: "wallmaster",
    0x28: "rope",
    0x29: "rope_red",
    0x2A: "stalfos",
    0x2B: "trap_invulnerable",
    0x2C: "bubble",
    0x2D: "bubble2",
    0x30: "gibdo",
    0x31: "moldorm_or_dodongo_variant",
    0x32: "dodongo",
    0x33: "gohma_a",
    0x34: "gohma_b",
    0x35: "rupee_stash_or_cluster",
    0x36: "grumble",
    0x37: "zelda",
    0x38: "digdogger",
    0x39: "digdogger_small",
    0x3A: "lanmola",
    0x3B: "lanmola_blue",
    0x3C: "manhandla",
    0x3D: "aquamentus",
    0x3E: "ganon",
    0x41: "moldorm",
    0x43: "gleeok_2head",
    0x44: "gleeok_3head",
    0x45: "gleeok_4head",
    0x47: "patra_1",
    0x48: "patra_2",
    0x49: "blade_trap",
    0x4A: "trap",
    0x4B: "old_man_or_stone",
    0x4C: "old_man_or_stone",
    0x4D: "old_man",
    0x4E: "old_man",
    0x4F: "old_man",
    0x50: "old_man",
}

# (block, room, field, expected) live-verified anchors from the repo docs.
ANCHORS: list[tuple[str, int, str, object]] = [
    # items
    ("L1-6", 0x0F, "item", 0x0C),  # L3 raft (LEVEL3_ROUTE.md)
    ("L1-6", 0x0D, "item", 0x1B),  # L2 triforce room (rr-n5i)
    ("L1-6", 0x4F, "item", 0x1E),  # L2 magic boomerang (rr-cjf)
    ("L7-9", 0x10, "item", 0x03),  # L9 silver-arrow *staircase* room: no item
    ("L7-9", 0x62, "item", 0x0F),  # L9 8-keese room drop
    ("L7-9", 0x2C, "item", 0x1B),  # L8 triforce room
    ("L7-9", 0x7E, "item", 0x03),  # L8 entry room
    ("L7-9", 0x3C, "item", 0x1A),  # L8 boss heart container
    # doors
    ("L7-9", 0x62, "doors", (1, 1, 0, 5)),  # LEVEL9_ROUTE.md live table
    ("L7-9", 0x52, "doors", (7, 1, 1, 1)),  # final Patra: north shutter only
    ("L7-9", 0x65, "doors_N", 4),  # bomb north -> 0x55
    ("L7-9", 0x06, "doors_W", 4),  # bomb west -> 0x05
    ("L7-9", 0x20, "doors_N", 4),  # bomb north -> 0x10
    ("L7-9", 0x31, "doors_W", 4),  # bomb west -> 0x30
    ("L7-9", 0x04, "doors_W", 4),  # bomb west -> 0x03
    ("L7-9", 0x14, "doors_E", 5),  # east key door -> 0x15
    ("L7-9", 0x15, "doors_E", 0),  # east open door -> 0x16
    ("L7-9", 0x42, "doors_N", 7),  # Ganon -> Zelda shutter
    ("L7-9", 0x32, "doors_S", 7),  # Zelda room shutter back to Ganon
    # objects
    ("L7-9", 0x52, "objtype", 0x47),  # live Patra body
    ("L7-9", 0x61, "objtype", 0x47),  # the other Patra
    ("L7-9", 0x42, "objtype", 0x3E),  # Ganon
    ("L7-9", 0x32, "objtype", 0x37),  # Zelda
    ("L7-9", 0x62, "objtype", 0x1B),  # 8 keese
    ("L7-9", 0x62, "objcount", 8),
    ("L7-9", 0x3C, "objtype", 0x45),  # L8 Gleeok 4-head
    ("L1-6", 0x13, "objtype", 0x43),  # L4 Gleeok 2-head
    ("L1-6", 0x18, "objtype", 0x44),  # L6 Gleeok 3-head
    ("L7-9", 0x05, "objlist", [0x23, 0x23, 0x24, 0x23, 0x24]),  # 5 wizzrobes
    # secrets
    ("L7-9", 0x62, "secret", 7),  # item appears after clearing foes
    ("L7-9", 0x52, "secret", 1),  # kill Patra -> north shutter opens
    ("L7-9", 0x42, "secret", 3),  # last boss
    ("L7-9", 0x05, "secret", 5),  # push block -> stairs (live)
    # cellars: dest rooms + the item that sits in them
    ("cellar", 0x4F, "L9", (0x10, 0x10, 0x09)),  # SILVER ARROWS
    ("cellar", 0x00, "L9", (0x07, 0x07, 0x13)),  # red ring
    ("cellar", 0x77, "L9", (0x52, 0x03, 0x03)),  # live: 0x03 stairs -> Patra
    ("cellar", 0x75, "L9", (0x20, 0x61, 0x03)),  # live
    ("cellar", 0x67, "L9", (0x30, 0x04, 0x03)),  # live
    ("cellar", 0x70, "L9", (0x63, 0x05, 0x03)),  # live
    ("cellar", 0x60, "L9", (0x14, 0x55, 0x03)),  # live
    ("cellar", 0x72, "L9", (0x71, 0x74, 0x03)),  # live
]


class Rooms:
    """Decoded 128-room attribute block for one quest-1 level group."""

    def __init__(self, prg: bytes, base: int, name: str) -> None:
        self.name = name
        self.base = base
        self.t = [prg[base + i * 128 : base + (i + 1) * 128] for i in range(6)]

    def doors(self, room: int) -> dict[str, int]:
        a, b = self.t[0][room], self.t[1][room]
        return {
            "N": (a >> 5) & 7,
            "S": (a >> 2) & 7,
            "W": (b >> 5) & 7,
            "E": (b >> 2) & 7,
        }

    def item(self, room: int) -> int:
        return self.t[4][room] & 0x1F

    def dark(self, room: int) -> bool:
        return bool(self.t[4][room] & 0x80)

    def item_flags(self, room: int) -> int:
        return (self.t[4][room] >> 5) & 0x03  # bits 6-5, purpose unidentified

    def secret(self, room: int) -> int:
        return self.t[5][room] & 0x07

    def objtype(self, room: int) -> int:
        return (self.t[2][room] & 0x3F) | ((self.t[3][room] & 0x80) >> 1)

    def count_index(self, room: int) -> int:
        return self.t[2][room] >> 6

    def cellar_dests(self, room: int) -> tuple[int, int]:
        return self.t[0][room], self.t[1][room]

    def raw(self, room: int) -> list[int]:
        return [self.t[i][room] for i in range(6)]


class Level:
    def __init__(self, prg: bytes, number: int, rooms: Rooms) -> None:
        off = LEVEL_INFO_L1_PRG + (number - 1) * LEVEL_INFO_STRIDE
        self.prg_off = off
        self.number = number
        self.rooms = rooms
        b = prg[off : off + LEVEL_INFO_STRIDE]
        self.info = b
        self.count_table = list(b[0x24:0x28])
        self.entrance = b[0x2F]
        self.goal_room = b[0x30]
        self.level_number_byte = b[0x33]
        cellars = []
        for x in b[0x34:0x3E]:
            if x == 0xFF:
                break
            cellars.append(x)
        self.cellars = cellars

    def objcount(self, room: int) -> int:
        return self.count_table[self.rooms.count_index(room)]


def obj_list(prg: bytes, objtype: int, count: int) -> list[int]:
    """Resolve an object-list index (type >= 0x62) into concrete object types."""
    idx = objtype - OBJ_LIST_FIRST_TYPE
    lo = prg[OBJ_LIST_PTRS_PRG + idx * 2] | (prg[OBJ_LIST_PTRS_PRG + idx * 2 + 1] << 8)
    off = OBJ_LIST_BANK5_PRG + (lo - 0x8000)
    return list(prg[off : off + count])


def objects(prg: bytes, level: Level, room: int) -> list[int]:
    t = level.rooms.objtype(room)
    n = level.objcount(room)
    if t == 0:
        return []
    if t >= OBJ_LIST_FIRST_TYPE:
        return obj_list(prg, t, n)
    return [t] * n


def obj_summary(types: list[int]) -> str:
    if not types:
        return "-"
    out, prev, n = [], None, 0
    for t in types:
        if t == prev:
            n += 1
        else:
            if prev is not None:
                out.append((prev, n))
            prev, n = t, 1
    out.append((prev, n))
    merged: dict[int, int] = {}
    for t, n in out:
        merged[t] = merged.get(t, 0) + n
    return " ".join(
        f"{n}x{OBJ_NAMES.get(t, f'0x{t:02X}')}(0x{t:02X})" for t, n in merged.items()
    )


def level_rooms(level: Level) -> tuple[set[int], list[tuple]]:
    """Rooms reachable from the entrance through decoded doors + stairs."""
    r = level.rooms
    cell = set(level.cellars)
    stair: dict[int, set[int]] = {}
    for c in cell:
        for d in r.cellar_dests(c):
            stair.setdefault(d, set()).add(c)
    seen = {level.entrance}
    q = deque([level.entrance])
    edges = []
    while q:
        room = q.popleft()
        if room in cell:
            continue
        d = r.doors(room)
        for k in DIRS:
            if d[k] == 1:
                continue
            if k == "N" and room < 0x10:
                continue
            if k == "S" and room >= 0x70:
                continue
            if k == "W" and (room & 0x0F) == 0:
                continue
            if k == "E" and (room & 0x0F) == 15:
                continue
            n = room + DELTA[k]
            edges.append((room, k, DOOR_NAMES[d[k]], n))
            if n not in seen:
                seen.add(n)
                q.append(n)
        for c in stair.get(room, ()):
            edges.append((room, "STAIRS", "cellar", c))
            for dd in (c, *r.cellar_dests(c)):
                if dd not in seen:
                    seen.add(dd)
                    q.append(dd)
    return seen, edges


DOOR_COST = {0: 1, 1: None, 2: 1, 3: 1, 4: 40, 5: 5, 6: 5, 7: 1}


def _neighbours(r: Rooms, cell: set[int], stair: dict, room: int, weighted: bool):
    out = []
    if room in cell:
        for side, d in zip(("left", "right"), r.cellar_dests(room)):
            out.append((d, f"cellar {side} mouth", 1))
        return out
    d = r.doors(room)
    for k in DIRS:
        if d[k] == 1:
            continue
        if (
            (k == "N" and room < 0x10)
            or (k == "S" and room >= 0x70)
            or (k == "W" and (room & 0x0F) == 0)
            or (k == "E" and (room & 0x0F) == 15)
        ):
            continue
        out.append((room + DELTA[k], f"{k} {DOOR_NAMES[d[k]]}", DOOR_COST[d[k]]))
    for c in stair.get(room, ()):
        out.append((c, f"stairs ({SECRET_NAMES[r.secret(room)]})", 1))
    return out


def shortest_path(level: Level, src: int, dst: int, weighted: bool = False) -> list[tuple]:
    """BFS (hop count) or Dijkstra weighted so bombable walls / locks cost more."""
    r = level.rooms
    cell = set(level.cellars)
    stair: dict[int, set[int]] = {}
    for c in cell:
        for d in r.cellar_dests(c):
            stair.setdefault(d, set()).add(c)
    prev: dict[int, tuple] = {src: ()}
    if weighted:
        import heapq

        dist = {src: 0}
        heap = [(0, src)]
        while heap:
            cost, room = heapq.heappop(heap)
            if cost > dist.get(room, 1 << 30):
                continue
            for n, how, w in _neighbours(r, cell, stair, room, weighted=True):
                if cost + w < dist.get(n, 1 << 30):
                    dist[n] = cost + w
                    prev[n] = (room, how)
                    heapq.heappush(heap, (cost + w, n))
        if dst not in prev:
            return []
        path, cur = [], dst
        while prev[cur]:
            room, how = prev[cur]
            path.append((room, how, cur))
            cur = room
        return list(reversed(path))
    q = deque([src])
    while q:
        room = q.popleft()
        if room == dst:
            break
        nxt = []
        if room in cell:
            for side, d in zip(("left", "right"), r.cellar_dests(room)):
                nxt.append((d, f"cellar {side} mouth"))
        else:
            d = r.doors(room)
            for k in DIRS:
                if d[k] == 1:
                    continue
                if (
                    (k == "N" and room < 0x10)
                    or (k == "S" and room >= 0x70)
                    or (k == "W" and (room & 0x0F) == 0)
                    or (k == "E" and (room & 0x0F) == 15)
                ):
                    continue
                nxt.append((room + DELTA[k], f"{k} {DOOR_NAMES[d[k]]}"))
            for c in stair.get(room, ()):
                nxt.append((c, f"stairs ({SECRET_NAMES[r.secret(room)]})"))
        for n, how in nxt:
            if n not in prev:
                prev[n] = (room, how)
                q.append(n)
    if dst not in prev:
        return []
    path, cur = [], dst
    while prev[cur]:
        room, how = prev[cur]
        path.append((room, how, cur))
        cur = room
    return list(reversed(path))


def check_anchors(prg: bytes, blocks: dict[str, Rooms], levels: dict[int, Level]) -> int:
    ok = fail = 0
    for block, room, field, want in ANCHORS:
        if block == "cellar":
            lvl = levels[9]
            r = lvl.rooms
            got = (*r.cellar_dests(room), r.item(room))
            label = f"L9 cellar 0x{room:02X} dests+item"
        else:
            r = blocks[block]
            lvl = levels[9] if block == "L7-9" else levels[4]
            label = f"{block} room 0x{room:02X} {field}"
            if field == "item":
                got = r.item(room)
            elif field == "doors":
                d = r.doors(room)
                got = (d["N"], d["S"], d["W"], d["E"])
            elif field.startswith("doors_"):
                got = r.doors(room)[field[-1]]
            elif field == "objtype":
                got = r.objtype(room)
            elif field == "objcount":
                got = lvl.objcount(room)
            elif field == "objlist":
                got = obj_list(prg, r.objtype(room), lvl.objcount(room))
            elif field == "secret":
                got = r.secret(room)
            else:
                raise AssertionError(field)
        good = got == want
        ok, fail = ok + good, fail + (not good)
        mark = "PASS" if good else "FAIL"
        print(f"  [{mark}] {label}: got {got!r} want {want!r}")
    print(f"  -> {ok} passed, {fail} failed")
    return fail


def main() -> None:
    data = ROM.read_bytes()
    assert data[:4] == b"NES\x1a"
    prg = data[INES:]

    blocks = {
        "L1-6": Rooms(prg, UW_Q1_L16_PRG, "L1-6"),
        "L7-9": Rooms(prg, UW_Q1_L79_PRG, "L7-9"),
    }
    levels = {
        n: Level(prg, n, blocks["L7-9"] if n >= 7 else blocks["L1-6"])
        for n in range(1, 10)
    }
    for n, lv in levels.items():
        assert lv.level_number_byte == n, (n, lv.level_number_byte)

    print(f"rom={ROM}")
    print(
        f"quest-1 UW attrs: L1-6 PRG 0x{UW_Q1_L16_PRG:05X} (iNES 0x{UW_Q1_L16_PRG + INES:05X}), "
        f"L7-9 PRG 0x{UW_Q1_L79_PRG:05X} (iNES 0x{UW_Q1_L79_PRG + INES:05X})"
    )
    print("\n== validation anchors ==")
    fails = check_anchors(prg, blocks, levels)

    print("\n== level info blocks ==")
    for n, lv in levels.items():
        cel = " ".join(f"{c:02X}" for c in lv.cellars) or "-"
        print(
            f"  L{n} PRG 0x{lv.prg_off:05X} entrance=0x{lv.entrance:02X} "
            f"goal=0x{lv.goal_room:02X} counts={lv.count_table} cellars={cel}"
        )
        for c in lv.cellars:
            a, b = lv.rooms.cellar_dests(c)
            it = lv.rooms.item(c)
            print(
                f"       cellar 0x{c:02X}: left->0x{a:02X} right->0x{b:02X} "
                f"item=0x{it:02X} {ITEM_NAMES.get(it, '?')}"
            )

    print("\n== level membership (BFS from entrance over decoded doors + stairs) ==")
    member = {}
    for n in (7, 8, 9):
        seen, _ = level_rooms(levels[n])
        member[n] = seen
        print(f"  L{n} ({len(seen)} rooms): " + " ".join(f"{r:02X}" for r in sorted(seen)))
    print(
        f"  overlaps: 7&8={sorted(member[7] & member[8])} 7&9={sorted(member[7] & member[9])} "
        f"8&9={sorted(member[8] & member[9])}"
    )
    unclaimed = [r for r in range(128) if r not in member[7] | member[8] | member[9]]
    print(f"  unclaimed slots: {unclaimed or 'none'}")

    l9 = levels[9]
    r9 = l9.rooms
    cell9 = set(l9.cellars)

    print("\n== rooms holding an item, all three levels ==")
    blk79 = blocks["L7-9"]  # levels 7, 8 and 9 all share this attribute block
    for n in (7, 8, 9):
        for room in sorted(member[n]):
            it = blk79.item(room)
            if it != 0x03:
                kind = "CELLAR" if room in set(levels[n].cellars) else "room"
                print(
                    f"  L{n} {kind} 0x{room:02X}: item 0x{it:02X} {ITEM_NAMES.get(it, '?')} "
                    f"secret={SECRET_NAMES[blk79.secret(room)]}"
                )

    print("\n== LEVEL 9 room table ==")
    print(
        "  id   N/S/W/E doors                       item          secret"
        "               dark objects"
    )
    rows = []
    for room in sorted(member[9]):
        if room in cell9:
            a, b = r9.cellar_dests(room)
            it = r9.item(room)
            print(
                f"  {room:02X}*  cellar: left->{a:02X} right->{b:02X}"
                f"{'':<21}{ITEM_NAMES.get(it, '?'):<14}-{'':<20}"
                f"{'Y' if r9.dark(room) else '.'}    -"
            )
            rows.append(
                {
                    "room": room,
                    "cellar": True,
                    "left_dest": a,
                    "right_dest": b,
                    "item": it,
                    "item_name": ITEM_NAMES.get(it, "?"),
                    "raw": r9.raw(room),
                }
            )
            continue
        d = r9.doors(room)
        ds = " ".join(f"{k}={DOOR_NAMES[d[k]]}" for k in DIRS)
        it = r9.item(room)
        objs = objects(prg, l9, room)
        print(
            f"  {room:02X}   {ds:<44}{ITEM_NAMES.get(it, '?'):<14}"
            f"{SECRET_NAMES[r9.secret(room)]:<21}"
            f"{'Y' if r9.dark(room) else '.'}    {obj_summary(objs)}"
        )
        rows.append(
            {
                "room": room,
                "cellar": False,
                "doors": {k: DOOR_NAMES[d[k]] for k in DIRS},
                "door_ids": {k: d[k] for k in DIRS},
                "item": it,
                "item_name": ITEM_NAMES.get(it, "?"),
                "secret": SECRET_NAMES[r9.secret(room)],
                "dark": r9.dark(room),
                "objtype": r9.objtype(room),
                "objcount": l9.objcount(room),
                "objects": objs,
                "raw": r9.raw(room),
            }
        )

    print("\n== door symmetry check (L9) ==")
    bad = 0
    for room in sorted(member[9] - cell9):
        d = r9.doors(room)
        for k, opp in (("N", "S"), ("S", "N"), ("W", "E"), ("E", "W")):
            if (
                (k == "N" and room < 0x10)
                or (k == "S" and room >= 0x70)
                or (k == "W" and (room & 0x0F) == 0)
                or (k == "E" and (room & 0x0F) == 15)
            ):
                continue
            n = room + DELTA[k]
            if n in cell9:
                continue
            if r9.doors(n)[opp] != d[k]:
                bad += 1
                print(
                    f"  0x{room:02X}.{k}={DOOR_NAMES[d[k]]} but "
                    f"0x{n:02X}.{opp}={DOOR_NAMES[r9.doors(n)[opp]]}"
                )
    print(f"  {bad} asymmetric door pairs")

    print("\n== Silver Arrows ==")
    silver = [c for c in l9.cellars if r9.item(c) == 0x09]
    for c in silver:
        a, b = r9.cellar_dests(c)
        print(f"  cellar room 0x{c:02X}: item 0x09 SILVER_ARROW, mouths -> 0x{a:02X} / 0x{b:02X}")
        for src in {a, b}:
            d = r9.doors(src)
            print(
                f"    entered from play room 0x{src:02X}: secret={SECRET_NAMES[r9.secret(src)]}, "
                f"doors {' '.join(f'{k}={DOOR_NAMES[d[k]]}' for k in DIRS)}, "
                f"objects {obj_summary(objects(prg, l9, src))}"
            )

    print("\n== decoded paths: L9 entrance -> Silver Arrows ==")
    for c in silver:
        for label, weighted in (("fewest hops", False), ("fewest bombs/keys", True)):
            print(f"  -- {label} --")
            bombs = keys = 0
            for room, how, nxt in shortest_path(l9, l9.entrance, c, weighted=weighted):
                bombs += "bombable" in how
                keys += "locked" in how
                print(f"    0x{room:02X} --{how}--> 0x{nxt:02X}")
            print(f"    (bombable walls: {bombs}, locked doors: {keys})")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(
            json.dumps(
                {
                    "level9_rooms": rows,
                    "level9_cellars": l9.cellars,
                    "level9_entrance": l9.entrance,
                    "membership": {str(n): sorted(member[n]) for n in member},
                },
                indent=1,
            )
        )
        print(f"\nwrote {out}")
    if fails:
        raise SystemExit(f"{fails} anchor(s) failed")


if __name__ == "__main__":
    main()

"""Chest slots and door slots for one loaded indoor room.

``chest_lines`` reads the RAM chest list, the ROM chest table, and the z3
item name at each record. ``door_lines`` reads the live WRAM door slots and
the ROM object list for a subtype-3 cell lock (object ``0x18``), which is
not a door slot. Neither function writes RAM.
"""

from __future__ import annotations

import numpy as np

from alttp.paths import INTEGRATION_DIR, REPO_ROOT
from alttp.ram import read_snapshot, read_u16, room_flags
from alttp.room_map import load_room_map
from alttp.z3_json_data import Z3JsonData

__all__ = ["chest_lines", "closed_chests", "door_lines"]


# snesrev get_chest_info. 168 records: u16 room (bit 15 = big chest), u8 item.
_CHEST_TABLE_SNES = 0x81E96E
_CHEST_RECORDS = 504 // 3
# dung_num_chests_x2, dung_chest_locations. Cell locks sit after this count.
_CHEST_COUNT = 0x0496
_CHEST_LOC = 0x06E0
# dung_savegame_state_bits. Chest i uses bit 0x100 << i while the room is up.
# save_dung_info stores that word >> 4, so chest 0 is room_flags bit 0x0010.
_SAVE_BITS = 0x0402
_OPEN_MASK0 = 0x0100
_SAVE_BIT0 = 0x0010

# Door arrays. ram_map.md Doors. Slots 8–15 are the other layer, not an exit.
_DOOR_TYPE = 0x1980
_DOOR_POS = 0x19A0
_DOOR_DIR = 0x19C0
_DOOR_SLOTS = 16
_LAYER_SLOT = 8
_DIR_NAME = ("north", "south", "west", "east")
_MAP_DIR = {"north": "UP", "south": "DOWN", "west": "LEFT", "east": "RIGHT"}
_FROM_MAP = {value: key for key, value in _MAP_DIR.items()}
# snesrev kDoorType. 0x40 is the high-range regular door. 0x18 here is a
# shutter, not the cell-lock object.
_TYPE_NAME = {
    0x00: "regular",
    0x02: "regular",
    0x04: "regular",
    0x06: "entrance",
    0x08: "waterfall",
    0x0A: "entrance",
    0x0C: "entrance",
    0x0E: "entrance",
    0x10: "entrance",
    0x12: "exit",
    0x14: "throne",
    0x16: "bg_change",
    0x18: "shutter",
    0x1A: "invisible",
    0x1C: "small_key",
    0x1E: "big_key",
    0x20: "stairs",
    0x22: "stairs",
    0x24: "stairs",
    0x26: "stairs",
    0x28: "bombable",
    0x30: "explosion",
    0x32: "slashable",
    0x36: "shutter",
    0x38: "regular",
    0x40: "regular",
    0x44: "shutter",
    0x46: "warp",
    0x48: "shutter",
    0x4A: "shutter",
}
# Room object pointer table. Subtype-3 index 0x18 is the cell lock.
_ROOM_PTR = 0x1F8000
_CELL_LOCK = 0x18
_ROOM_COUNT = 0x140


def _snes_file_offset(ea: int) -> int:
    """snesrev ``LoadedRom.get_byte`` map. Address must sit in a ROM window."""
    if (ea & 0x8000) == 0:
        raise SystemExit(f"snes address {ea:#x} is not in a ROM window")
    return ((ea >> 16) & 0x7F) * 0x8000 + (ea & 0x7FFF)


def _headerless_rom() -> bytes:
    path = REPO_ROOT / "roms" / "zelda3.sfc"
    if not path.is_file():
        path = INTEGRATION_DIR / "rom.sfc"
    if not path.is_file():
        raise SystemExit(f"missing ROM at {path}")
    data = path.read_bytes()
    if (len(data) & 0xFFFFF) == 0x200:
        data = data[0x200:]
    return data


def _chest_records(rom: bytes) -> dict[int, list[tuple[int, bool]]]:
    """Room id → chest-table records in ROM order: (file offset, big)."""
    by_room: dict[int, list[tuple[int, bool]]] = {}
    ea = _CHEST_TABLE_SNES
    for _ in range(_CHEST_RECORDS):
        off = _snes_file_offset(ea)
        room = rom[off] | (rom[off + 1] << 8)
        by_room.setdefault(room & 0x7FFF, []).append((off, bool(room & 0x8000)))
        ea += 3
    return by_room


def _z3_items_by_address() -> dict[int, str]:
    """z3 ``nodeAddress`` file offset → item name. Refs checkout, no download."""
    items: dict[int, str] = {}
    for room in Z3JsonData.load().rooms:
        for node in room.nodes:
            addr = node.node_address
            if not addr or not node.node_item or not addr.startswith("0x"):
                continue
            try:
                key = int(addr, 16)
            except ValueError:
                continue
            items[key] = node.node_item
    return items


def _chest_tile(loc: int) -> tuple[int, int, int] | None:
    """``(col, row, layer)`` from a ``$06E0`` word. Cleared slots are ``None``."""
    if loc == 0:
        return None
    tile = (loc & 0x7FFF) >> 1
    layer = tile >> 12
    tile &= 0xFFF
    return tile & 63, tile >> 6, layer


def _chest_place(loc: int) -> str:
    decoded = _chest_tile(loc)
    if decoded is None:
        return "loc=$0000 cleared"
    col, row, layer = decoded
    return f"tile=({col},{row}) layer={layer} loc=${loc:04X}"


def closed_chests(env: object) -> tuple[tuple[int, int, int, int], ...]:
    """Unopened chests in the loaded room: ``(index, col, row, layer)``.

    A cleared location is not a chest. Outdoors yields nothing. No writes.
    """
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if not snap.indoors:
        return ()
    count = read_u16(ram, _CHEST_COUNT) // 2
    bits = read_u16(ram, _SAVE_BITS)
    found: list[tuple[int, int, int, int]] = []
    for index in range(count):
        if bits & (_OPEN_MASK0 << index):
            continue
        decoded = _chest_tile(read_u16(ram, _CHEST_LOC + 2 * index))
        if decoded is None:
            continue
        col, row, layer = decoded
        found.append((index, col, row, layer))
    return tuple(found)


def chest_lines(env: object) -> list[str]:
    """Chests in the loaded room. Reads RAM and the ROM chest table. No writes."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if not snap.indoors:
        return ["chests (outdoors)"]
    count = read_u16(ram, _CHEST_COUNT) // 2
    if count == 0:
        return [f"chests room=0x{snap.room_id:04X} (none)"]
    bits = read_u16(ram, _SAVE_BITS)
    flags = room_flags(ram, snap.room_id)
    records = _chest_records(_headerless_rom()).get(snap.room_id, [])
    named = _z3_items_by_address()
    lines = [f"chests room=0x{snap.room_id:04X} n={count} live=${bits:04X}"]
    join = len(records) == count
    for i in range(count):
        loc = read_u16(ram, _CHEST_LOC + 2 * i)
        live_bit = _OPEN_MASK0 << i
        save_bit = _SAVE_BIT0 << i
        item = "unjoined"
        addr = ""
        big = ""
        if join:
            file_addr, is_big = records[i]
            item = named.get(file_addr, "unlisted")
            addr = f" z3=0x{file_addr:04X}"
            if is_big:
                big = " big"
        lines.append(
            f"  chest {i}{big} {_chest_place(loc)} "
            f"opened={int(bool(bits & live_bit))} "
            f"live=0x{live_bit:04X} save=0x{save_bit:04X}:"
            f"{int(bool(flags & save_bit))} item={item}{addr}"
        )
    if not join:
        lines.append(
            f"  table has {len(records)} records for this room; item not joined"
        )
    return lines


def _type_text(code: int) -> str:
    name = _TYPE_NAME.get(code)
    if name is None:
        return f"0x{code:02X}"
    return f"0x{code:02X} {name}"


def _occupied_slots(ram: np.ndarray) -> list[dict[str, int]]:
    """Occupied door slots. A slot is empty only when all three words are 0."""
    found: list[dict[str, int]] = []
    for slot in range(_DOOR_SLOTS):
        typ = read_u16(ram, _DOOR_TYPE + 2 * slot)
        pos = read_u16(ram, _DOOR_POS + 2 * slot)
        direc = read_u16(ram, _DOOR_DIR + 2 * slot)
        if typ == 0 and pos == 0 and direc == 0:
            continue
        found.append(
            {
                "slot": slot,
                "type": typ & 0xFF,
                "pos": pos,
                "dir": direc & 0xFF,
            }
        )
    return found


def _stair_map_door(door: object) -> bool:
    text = f"{getattr(door, 'label', '')} {getattr(door, 'notes', '')}".lower()
    return "stair" in text


def _dest_text(room_base: int, dir_name: str, type_code: int) -> str:
    """Map door in this direction, or ``unknown``. No invented exit.

    A stair note is not the destination of a key or regular slot. Room
    ``0x72`` slot 0 is the small-key door; the only UP map door is the stair.
    """
    try:
        room_map = load_room_map(f"0x{room_base & 0xFF:02X}")
    except FileNotFoundError:
        return "unknown"
    want = _MAP_DIR[dir_name]
    hits = [door for door in room_map.doors if door.direction == want]
    stair_slot = type_code in {0x20, 0x22, 0x24, 0x26}
    if stair_slot:
        stairs = [door for door in hits if _stair_map_door(door)]
        if stairs:
            hits = stairs
    else:
        plain = [door for door in hits if not _stair_map_door(door)]
        if plain:
            hits = plain
        elif hits:
            return "unknown"
    parts: list[str] = []
    for door in hits:
        if door.outdoors:
            if door.screen_id is None:
                parts.append("outdoors")
            else:
                parts.append(f"screen=0x{int(door.screen_id):02X}")
        elif door.to_room is None:
            parts.append("unknown")
        else:
            parts.append(f"0x{int(door.to_room) & 0xFF:02X}")
    return ",".join(parts) if parts else "unknown"


def _rom_u8(rom: bytes, ea: int) -> int:
    return rom[_snes_file_offset(ea)]


def _rom_u16(rom: bytes, ea: int) -> int:
    return _rom_u8(rom, ea) | (_rom_u8(rom, ea + 1) << 8)


def _rom_u24(rom: bytes, ea: int) -> int:
    return _rom_u16(rom, ea) | (_rom_u8(rom, ea + 2) << 16)


def _cell_lock_tiles(rom: bytes, room: int) -> list[tuple[int, int]]:
    """Subtype-3 object 0x18 tiles in this room. Empty when the ROM has none."""
    if not 0 <= int(room) < _ROOM_COUNT:
        return []
    ptr = _rom_u24(rom, _ROOM_PTR + int(room) * 3)
    if (ptr & 0x8000) == 0:
        return []
    locks: list[tuple[int, int]] = []
    addr = ptr + 2
    for _layer in range(3):
        steps = 0
        while steps < 2000:
            steps += 1
            try:
                word = _rom_u16(rom, addr)
            except SystemExit:
                return locks
            if word == 0xFFFF:
                addr += 2
                break
            if word == 0xFFF0:
                addr += 2
                while steps < 2000:
                    steps += 1
                    try:
                        door = _rom_u16(rom, addr)
                    except SystemExit:
                        return locks
                    if door == 0xFFFF:
                        addr += 2
                        break
                    addr += 2
                break
            p0 = word & 0xFF
            p1 = word >> 8
            try:
                p2 = _rom_u8(rom, addr + 2)
            except SystemExit:
                return locks
            if (word & 0xFC) != 0xFC and p2 >= 0xF8:
                index = (p2 & 7) << 4 | (p1 & 3) << 2 | (p0 & 3)
                if index == _CELL_LOCK:
                    dst = ((p1 >> 2) << 7) | ((p0 & 0xFC) >> 1)
                    locks.append(((dst >> 1) & 0x3F, (dst >> 7) & 0x3F))
            addr += 3
        else:
            return locks
    return locks


def _lock_line(room: int, tile: tuple[int, int]) -> str:
    return (
        f"  object lock room=0x{int(room) & 0xFF:02X} object=0x18 "
        f"tile=({tile[0]},{tile[1]}) not a door slot"
    )


def door_lines(env: object) -> list[str]:
    """Live door slots, plus a cell lock that is not a door slot. No writes."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if not snap.indoors:
        return ["doors (outdoors)"]
    slots = _occupied_slots(ram)
    lines = [f"doors room=0x{snap.room_id:04X}"]
    primary = [slot for slot in slots if slot["slot"] < _LAYER_SLOT]
    live_dirs = {
        _DIR_NAME[slot["dir"]]
        for slot in primary
        if slot["dir"] < len(_DIR_NAME)
    }
    if not slots:
        lines.append("  (no door slots)")
    for slot in slots:
        dir_name = (
            _DIR_NAME[slot["dir"]] if slot["dir"] < len(_DIR_NAME) else f"dir={slot['dir']}"
        )
        if slot["slot"] >= _LAYER_SLOT:
            lines.append(
                f"  slot {slot['slot']} {dir_name} type={_type_text(slot['type'])} "
                f"pos=${slot['pos']:04X} layer of slot {slot['slot'] - _LAYER_SLOT}"
            )
            continue
        dest = (
            "unknown"
            if dir_name not in _MAP_DIR
            else _dest_text(snap.room_base_id, dir_name, slot["type"])
        )
        lines.append(
            f"  slot {slot['slot']} {dir_name} type={_type_text(slot['type'])} "
            f"pos=${slot['pos']:04X} dest={dest}"
        )
    rom = _headerless_rom()
    seen: set[tuple[int, int, int]] = set()
    room_ids = [int(snap.room_id) & 0x1FF]
    try:
        room_map = load_room_map(f"0x{snap.room_base_id & 0xFF:02X}")
    except FileNotFoundError:
        room_map = None
    if room_map is not None:
        for door in room_map.doors:
            side = _FROM_MAP.get(door.direction)
            if side is None or side in live_dirs or door.to_room is None:
                continue
            room_ids.append(int(door.to_room) & 0xFF)
    for room in room_ids:
        for tile in _cell_lock_tiles(rom, room):
            key = (room & 0xFF, tile[0], tile[1])
            if key in seen:
                continue
            seen.add(key)
            lines.append(_lock_line(room, tile))
    return lines

#!/usr/bin/env python3
"""Disassemble Harvest Moon CC event scripts straight from the ROM.

Scripts live in banks $B3-$B5 (LoROM, ``0x33*0x8000`` is the file offset of
$B3:8000). ``$B38000`` is a table of 3-byte pointers, one per group; a group
points at a table of 2-byte script pointers, one per event
(``UNK_LoadCCDataLong`` at $848097). A script starts child scripts with
opcode ``09`` (``NewCC``), so one event reaches many CC scripts.

    uv run python -m harvest.scripts.event_dis at 0xB3D213
    uv run python -m harvest.scripts.event_dis group 5 --event 0
    uv run python -m harvest.scripts.event_dis sites eve

Operand lengths come from ``HM-Decomp/EventScript.txt`` except 00, which reads two
bytes (the festival scripts only decode that way). Opcodes at 0x51 and
above are unknown: the walk stops there and says so. Opcode 0x4F/0x50 (put
tools away) take no operand bytes.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

GROUP_TABLE = 0xB38000
GROUP_COUNT = 0x48
HEART_POINTERS = {"maria": 0x1F1F, "ann": 0x1F21, "nina": 0x1F23, "ellen": 0x1F25, "eve": 0x1F27}

# operand byte count per opcode (EventScript.txt)
OPERANDS: Dict[int, int] = {
    **dict.fromkeys((0x01, 0x02, 0x04, 0x08, 0x10, 0x11, 0x30, 0x31, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38), 0),
    **dict.fromkeys((0x3C, 0x40, 0x4A, 0x4C, 0x4E, 0x4F, 0x50), 0),
    **dict.fromkeys((0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0C, 0x0E, 0x0F, 0x17, 0x24, 0x25), 1),
    **dict.fromkeys((0x3B, 0x3D, 0x3E, 0x3F), 1),
    **dict.fromkeys((0x00, 0x12, 0x13, 0x1E, 0x1F, 0x2A, 0x2F, 0x4D), 2),  # 00 reads song and a byte
    **dict.fromkeys((0x09, 0x18, 0x19, 0x1B, 0x1C, 0x1D, 0x20, 0x2B, 0x49), 3),
    **dict.fromkeys((0x05, 0x0D, 0x21, 0x23, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x39, 0x3A, 0x47), 4),
    **dict.fromkeys((0x29, 0x2C, 0x32, 0x41, 0x4B), 5),
    **dict.fromkeys((0x14, 0x15, 0x22, 0x42, 0x48), 6),
    **dict.fromkeys((0x16, 0x1A, 0x43), 7),
    0x44: 8,
    0x45: 9,
    0x46: 11,
}
NAMES = {
    0x09: "NEWCC", 0x10: "STOP", 0x12: "JMP", 0x14: "JFLAG", 0x15: "JEQ", 0x16: "JBTW",
    0x17: "RNG", 0x18: "JRNG", 0x1C: "TEXT", 0x1D: "TEXTC", 0x1F: "APRESS1", 0x20: "JCHOICE",
    0x23: "SETFLAG", 0x26: "RSTFLAG", 0x2D: "APRESS2", 0x32: "STORE", 0x41: "ADD16",
    0x43: "JEQV", 0x45: "JBTWV", 0x47: "SETB", 0x4D: "CHK4D", 0x3D: "TRANSIT",
}


def rom_offset(addr: int) -> int:
    return ((addr >> 16) & 0x7F) * 0x8000 + (addr & 0x7FFF)


def _w(b: bytes, i: int) -> int:
    return b[i] | (b[i + 1] << 8)


def _a24(b: bytes, i: int) -> int:
    return b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)


def _signed(b: bytes, i: int) -> int:
    return int.from_bytes(b[i : i + 2], "little", signed=True)


def describe(op: int, b: bytes, bank: int) -> Tuple[str, List[int]]:
    """Note text and the addresses this instruction can continue at."""
    j = lambda i: bank | _w(b, i)  # noqa: E731 - same-bank jump word at operand i
    if op == 0x12 or op == 0x1F or op == 0x4D:
        return f"-> {j(1):06X}", [j(1)]
    if op == 0x09:
        return f"cc[{b[1]}] -> {j(2):06X}", [j(2)]
    if op == 0x14:
        return f"[{_a24(b, 1):06X}] bit {b[4]} -> {j(5):06X}", [j(5)]
    if op == 0x15:
        return f"[{_a24(b, 1):06X}] == {b[4]} -> {j(5):06X}", [j(5)]
    if op == 0x16:
        return f"[{_a24(b, 1):06X}] in [{b[4]},{b[5]}] -> {j(6):06X}", [j(6)]
    if op == 0x18:
        return f"rng == {b[1]} -> {j(2):06X}", [j(2)]
    if op == 0x20:
        return f"choice == {b[1]} -> {j(2):06X}", [j(2)]
    if op == 0x2D:
        return f"talk -> {j(1):06X} gift -> {j(3):06X}", [j(1), j(3)]
    if op in (0x1C, 0x1D):
        return f"text {_w(b, 1):#06x}", []
    if op == 0x41:
        return f"[{_a24(b, 1):06X}] += {_signed(b, 4)}", []
    if op == 0x43:
        return f"[{_a24(b, 1):06X}] == {_w(b, 4):#x} -> {j(6):06X}", [j(6)]
    if op == 0x45:
        return f"[{_a24(b, 1):06X}] in [{_w(b, 4)},{_w(b, 6)}] -> {j(8):06X}", [j(8)]
    if op == 0x47:
        return f"[{_a24(b, 1):06X}] = {b[4]:#x}", []
    if op == 0x32:
        return f"[{_a24(b, 1):06X}] = {_w(b, 4):#x}", []
    if op in (0x23, 0x26):
        return f"[{_a24(b, 1):06X}] bit {b[4]}", []
    return "", []


def decode(rom: bytes, start: int, max_insns: int = 400) -> Dict[int, Tuple[int, Optional[bytes], str]]:
    """Every instruction reachable from ``start``: {addr: (opcode, bytes, text)}."""
    bank = start & 0xFF0000
    out: Dict[int, Tuple[int, Optional[bytes], str]] = {}
    work = [start]
    while work and len(out) < max_insns:
        addr = work.pop()
        while addr not in out and len(out) < max_insns:
            off = rom_offset(addr)
            op = rom[off]
            n = OPERANDS.get(op)
            if n is None:
                out[addr] = (op, None, f"?? opcode {op:02X} has no known length")
                break
            raw = rom[off : off + 1 + n]
            note, targets = describe(op, raw, bank)
            name = NAMES.get(op, f"op{op:02X}")
            out[addr] = (op, raw, f"{name:8s} {raw[1:].hex(' '):22s} {note}".rstrip())
            work.extend(targets)
            if op in (0x12, 0x10):
                break
            addr += 1 + n
    return out


def group_pointers(rom: bytes) -> List[int]:
    base = rom_offset(GROUP_TABLE)
    return [_a24(rom, base + 3 * g) for g in range(GROUP_COUNT)]


def event_starts(rom: bytes, group_ptr: int) -> List[int]:
    """Script address of each event of a group (the word table ends at the lowest script)."""
    bank, low = group_ptr & 0xFF0000, group_ptr & 0xFFFF
    count, lowest = 0, 0xFFFF
    while low + 2 * count < lowest and count < 120:
        word = _w(rom, rom_offset(group_ptr) + 2 * count)
        if word < low:
            break
        lowest = min(lowest, word)
        count += 1
    return [bank | _w(rom, rom_offset(group_ptr) + 2 * e) for e in range(max(count, 1))]


def heart_sites(rom: bytes, girl: str) -> List[Tuple[int, int, List[str]]]:
    """(address, delta, [group.event that reach it]) for every add to ``girl``'s hearts."""
    ptr = HEART_POINTERS[girl]
    needle = bytes([0x41, ptr & 0xFF, ptr >> 8, 0x7F])
    sites: Dict[int, int] = {}
    for bank in (0xB3, 0xB4, 0xB5):
        lo = rom_offset(bank << 16 | 0x8000)
        i = lo
        while (i := rom.find(needle, i, lo + 0x8000)) >= 0:
            sites[(bank << 16) | (i - lo + 0x8000)] = _signed(rom, i + 4)
            i += 1
    reach: Dict[int, List[str]] = {a: [] for a in sites}
    for g, gptr in enumerate(group_pointers(rom)):
        for e, start in enumerate(event_starts(rom, gptr)):
            seen = decode(rom, start, 4000)
            for a in sites:
                if a in seen:
                    reach[a].append(f"G{g:02X}.E{e:02X}")
    return sorted((a, d, reach[a]) for a, d in sites.items())


def _load_rom() -> bytes:
    from harvest.runtime.retro_setup import find_harvest_rom

    path = find_harvest_rom()
    if path is None:
        raise SystemExit("Harvest Moon ROM not found (see ROM_SETUP.md)")
    return path.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    at = sub.add_parser("at", help="disassemble from a 24-bit address")
    at.add_argument("addr", type=lambda s: int(s, 16))
    grp = sub.add_parser("group", help="disassemble one event of a group")
    grp.add_argument("group", type=lambda s: int(s, 0))
    grp.add_argument("--event", type=lambda s: int(s, 0), default=0)
    for walker in (at, grp):
        walker.add_argument("--max", type=int, default=400, dest="max_insns")
    st = sub.add_parser("sites", help="every heart add for one girl and the events that reach it")
    st.add_argument("girl", choices=sorted(HEART_POINTERS))
    args = ap.parse_args()
    rom = _load_rom()
    if args.cmd == "sites":
        for addr, delta, events in heart_sites(rom, args.girl):
            print(f"{addr:06X} {delta:+4d} {' '.join(events) or '-'}")
        return 0
    if args.cmd == "group":
        start = event_starts(rom, group_pointers(rom)[args.group])[args.event]
    else:
        start = args.addr
    for addr, (op, _raw, text) in sorted(decode(rom, start, args.max_insns).items()):
        print(f"{addr:06X}: {op:02X} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

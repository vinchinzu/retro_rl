"""Locate category-preset payloads in the practice ROM (24-bit SNES pointers).

The IPS on disk can disagree with GitHub ``*_data.asm`` parent words. Walk the
ROM's sequential blobs instead: parent 0 starts a chain, ``dw $FFFF`` ends a
record, and inheritance uses the practice-hack bank-cross rule (parent_lo >=
current_lo means previous bank). Join catalog rows by ``data_label`` using
``snes`` / ``effective_state_sha256`` (full word hash), not a 5-tuple.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from super_metroid.paths import (
    PRACTICE_PRESET_ADDR_PATH,
    SHARED_PRACTICE_ROM,
)
from super_metroid.practice_repertoire.catalog import load_catalog

BANK_SIZE = 0x8000
_MAX_PAIRS = 256
_ADDR_MIN = 0x0070
_ADDR_MAX = 0xDFFF


def file_offset_to_snes(offset: int) -> int:
    """Headerless LoROM: bank $80+ file/32KiB, addr $8000 | (offset % 32KiB)."""

    bank = 0x80 + (offset // BANK_SIZE)
    addr = 0x8000 + (offset % BANK_SIZE)
    return (bank << 16) | addr


def snes_to_file_offset(snes: int) -> int:
    bank = (snes >> 16) & 0xFF
    addr = snes & 0xFFFF
    return ((bank - 0x80) * BANK_SIZE) + (addr & 0x7FFF)


def word_hash(words: dict[int, int]) -> str:
    digest = hashlib.sha256()
    for address, value in sorted(words.items()):
        digest.update(struct.pack("<HH", address, value))
    return digest.hexdigest()


def _payload(parent_lo: int, pairs: list[tuple[int, int]]) -> bytes:
    words = [parent_lo]
    for address, value in pairs:
        words.extend((address, value))
    words.append(0xFFFF)
    return struct.pack("<" + "H" * len(words), *words)


def _u16(rom: bytes, offset: int) -> int:
    return rom[offset] | (rom[offset + 1] << 8)


def _read_preset(
    rom: bytes, offset: int
) -> tuple[int, list[tuple[int, int]], int] | None:
    if offset + 4 > len(rom):
        return None
    parent_lo = _u16(rom, offset)
    pairs: list[tuple[int, int]] = []
    pos = offset + 2
    while pos + 2 <= len(rom):
        word = _u16(rom, pos)
        if word == 0xFFFF:
            if not pairs:
                return None
            return parent_lo, pairs, pos + 2
        if pos + 4 > len(rom):
            return None
        addr, value = word, _u16(rom, pos + 2)
        if not _ADDR_MIN <= addr <= _ADDR_MAX:
            return None
        pairs.append((addr, value))
        pos += 4
        if len(pairs) > _MAX_PAIRS:
            return None
    return None


def _parent_snes(snes: int, parent_lo: int) -> int | None:
    if parent_lo == 0:
        return None
    current_lo = snes & 0xFFFF
    bank = snes >> 16
    if parent_lo < current_lo:
        return (bank << 16) | parent_lo
    return ((bank - 1) << 16) | parent_lo


def _root_offsets(rom: bytes) -> list[int]:
    needle = bytes.fromhex("00008d07")  # parent 0, DDB $078D
    hits: list[int] = []
    start = 0
    while True:
        at = rom.find(needle, start)
        if at < 0:
            break
        hits.append(at)
        start = at + 1
    return hits


def walk_preset_blobs(rom: bytes) -> list[dict[str, Any]]:
    """Walk every parent-0 chain. Later roots stop an earlier overlapping walk."""

    roots = set(_root_offsets(rom))
    blobs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for start in sorted(roots):
        if start in seen:
            continue
        by_snes: dict[int, dict[int, int]] = {}
        offset = start
        while offset not in seen:
            parsed = _read_preset(rom, offset)
            if parsed is None:
                break
            parent_lo, pairs, nxt = parsed
            snes = file_offset_to_snes(offset)
            if parent_lo == 0:
                words: dict[int, int] = {}
            else:
                parent = _parent_snes(snes, parent_lo)
                if parent is None or parent not in by_snes:
                    break
                words = dict(by_snes[parent])
            words.update(pairs)
            by_snes[snes] = words
            seen.add(offset)
            blobs.append(
                {
                    "offset": offset,
                    "snes": snes,
                    "snes_hex": f"0x{snes:06X}",
                    "parent_lo": parent_lo,
                    "words": words,
                }
            )
            if nxt in roots and nxt != start:
                break
            offset = nxt
    return blobs


def join_blobs_by_label(
    sessions: list[dict[str, Any]],
    blobs: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Map ``data_label`` → ROM blob via catalog ``snes`` or full word hash."""

    by_snes = {int(blob["snes"]): blob for blob in blobs}
    by_hash = {word_hash(blob["words"]): blob for blob in blobs}
    found: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for rec in sessions:
        label = rec.get("data_label")
        if not label:
            continue
        blob = None
        snes = rec.get("snes")
        if snes is not None:
            blob = by_snes.get(int(snes))
        if blob is None:
            digest = rec.get("effective_state_sha256")
            if digest:
                blob = by_hash.get(str(digest))
            elif rec.get("words"):
                blob = by_hash.get(word_hash(rec["words"]))
        if blob is None:
            missing.append(str(label))
            continue
        found[str(label)] = {
            "label": label,
            "session_id": rec.get("id"),
            "offset": blob["offset"],
            "snes": blob["snes"],
            "snes_hex": blob["snes_hex"],
            "parent_lo": blob["parent_lo"],
        }
    return found, missing


def map_preset_addresses(
    practice_rom: Path = SHARED_PRACTICE_ROM,
) -> dict[str, Any]:
    """Match catalog sessions to ROM blobs by ``data_label``."""

    if not practice_rom.is_file():
        raise FileNotFoundError(f"practice ROM missing: {practice_rom}")
    rom = practice_rom.read_bytes()
    blobs = walk_preset_blobs(rom)
    found, missing = join_blobs_by_label(load_catalog()["sessions"], blobs)
    return {
        "practice_rom": str(practice_rom),
        "blobs": len(blobs),
        "mapped": len(found),
        "missing": missing,
        "presets": found,
    }


def write_address_map(
    report: dict[str, Any],
    path: Path = PRACTICE_PRESET_ADDR_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        "practice_rom": report["practice_rom"],
        "blobs": report.get("blobs"),
        "mapped": report["mapped"],
        "missing": report["missing"],
        "presets": {
            label: {
                "snes": row["snes"],
                "snes_hex": row["snes_hex"],
                "offset": row["offset"],
                "session_id": row.get("session_id"),
            }
            for label, row in report["presets"].items()
        },
    }
    path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    return path


def load_address_map(path: Path = PRACTICE_PRESET_ADDR_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "file_offset_to_snes",
    "join_blobs_by_label",
    "load_address_map",
    "map_preset_addresses",
    "snes_to_file_offset",
    "walk_preset_blobs",
    "word_hash",
    "write_address_map",
]

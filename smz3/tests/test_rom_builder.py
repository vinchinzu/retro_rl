"""ROM merge / IPS / seed-patch unit tests (no network, synthetic fixtures)."""

from __future__ import annotations

import struct

import pytest

from smz3.paths import COMBO_ROM_SIZE
from smz3.rom_builder import apply_ips, apply_seed_patch, merge_sm_z3


def _minimal_sm() -> bytes:
    # 3 MiB zeroed SM; paint first/last bank markers.
    sm = bytearray(0x300000)
    sm[0:4] = b"SMHI"
    sm[0x200000:0x200004] = b"SMLO"
    return bytes(sm)


def _minimal_z3() -> bytes:
    z3 = bytearray(0x100000)
    z3[0:4] = b"Z3HI"
    return bytes(z3)


def test_merge_size_and_layout() -> None:
    rom = merge_sm_z3(_minimal_sm(), _minimal_z3())
    assert len(rom) == COMBO_ROM_SIZE
    # First SM bank: lo at 0, hi at 0x8000
    assert bytes(rom[0:4]) == b"SMLO"
    assert bytes(rom[0x8000:0x8004]) == b"SMHI"
    # Z3 hi bank starts at 0x400000 + 0x8000
    assert bytes(rom[0x408000:0x408004]) == b"Z3HI"


def test_apply_ips_simple() -> None:
    rom = bytearray(256)
    # PATCH + dest 0x000010 + len 0x0004 + data ABCD + EOF
    ips = b"PATCH" + bytes([0x00, 0x00, 0x10, 0x00, 0x04]) + b"ABCD" + b"EOF"
    apply_ips(rom, ips)
    assert bytes(rom[0x10:0x14]) == b"ABCD"


def test_apply_ips_rle() -> None:
    rom = bytearray(64)
    # dest 0x000008, size 0 → RLE len 5 byte 0xEE
    ips = (
        b"PATCH"
        + bytes([0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x05, 0xEE])
        + b"EOF"
    )
    apply_ips(rom, ips)
    assert rom[0x08:0x0D] == bytes([0xEE] * 5)


def test_apply_seed_patch() -> None:
    rom = bytearray(1024)
    # two records: dest 0x10 len 3 "XYZ"; dest 0x20 len 2 "AB"
    patch = (
        struct.pack("<IH", 0x10, 3)
        + b"XYZ"
        + struct.pack("<IH", 0x20, 2)
        + b"AB"
    )
    apply_seed_patch(rom, patch)
    assert bytes(rom[0x10:0x13]) == b"XYZ"
    assert bytes(rom[0x20:0x22]) == b"AB"


def test_seed_patch_overrun_raises() -> None:
    rom = bytearray(16)
    patch = struct.pack("<IH", 0, 100) + b"x" * 100
    with pytest.raises(ValueError, match="past end"):
        apply_seed_patch(rom, patch)


def test_merge_rejects_tiny_roms() -> None:
    with pytest.raises(ValueError, match="Super Metroid"):
        merge_sm_z3(b"tiny", _minimal_z3())
    with pytest.raises(ValueError, match="Zelda"):
        merge_sm_z3(_minimal_sm(), b"tiny")

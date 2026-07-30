"""Build an SMZ3 combo ROM from vanilla SM + Z3 + base IPS + seed patch.

Ports the samus.link web client pipeline (``prepareRom`` / ``mergeRoms`` /
``applyIps`` / ``applySeed``) so seed patches from the public API apply cleanly.

Does **not** redistribute ROMs or IPS contents in git — callers supply local
vanilla ROMs; the base IPS is fetched into ``smz3/refs/`` (gitignored).
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import BinaryIO

from smz3.paths import (
    BASE_IPS_GZ,
    BASE_IPS_URL,
    COMBO_ROM_SIZE,
    SHARED_SM_ROM,
    SHARED_Z3_ROM,
)

# Super Metroid unheadered size (3 MiB). Headered dumps are not accepted.
_SM_SIZE = 0x300000
_Z3_SIZE = 0x100000  # ALttP JP 1.0 unheadered


def ensure_base_ips(*, path: Path = BASE_IPS_GZ, url: str = BASE_IPS_URL) -> Path:
    """Download the combo base IPS gzip if missing. Return the local path."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        urllib.request.urlretrieve(url, path)  # noqa: S310 — pinned public URL
    except Exception as exc:  # pragma: no cover - network
        raise FileNotFoundError(
            f"Missing base IPS {path} and download failed: {exc}\n"
            f"Fetch manually:\n  curl -L '{url}' -o {path}"
        ) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Base IPS download produced empty file: {path}")
    return path


def load_base_ips(path: Path | None = None) -> bytes:
    """Load and gunzip the combo base IPS (must start with ``PATCH``)."""
    ips_path = ensure_base_ips(path=path or BASE_IPS_GZ)
    raw = ips_path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        data = gzip.decompress(raw)
    else:
        data = raw
    if data[:5] != b"PATCH":
        raise ValueError(f"Not a valid IPS (missing PATCH header): {ips_path}")
    return data


def merge_sm_z3(sm_rom: bytes, z3_rom: bytes) -> bytearray:
    """Interleave Super Metroid + ALttP into a 6 MiB SMZ3 base image.

    Matches ``mergeRoms`` in tewtal SMZ3Randomizer web client.
    """
    if len(sm_rom) < _SM_SIZE:
        raise ValueError(f"Super Metroid ROM too small: {len(sm_rom)} < {_SM_SIZE}")
    if len(z3_rom) < _Z3_SIZE:
        raise ValueError(f"Zelda 3 ROM too small: {len(z3_rom)} < {_Z3_SIZE}")

    rom = bytearray(COMBO_ROM_SIZE)
    pos = 0
    for i in range(0x40):
        hi = sm_rom[i * 0x8000 : i * 0x8000 + 0x8000]
        lo = sm_rom[(i + 0x40) * 0x8000 : (i + 0x40) * 0x8000 + 0x8000]
        # Beyond SM file end: leave zeros (same as short Stream.Read in C#).
        rom[pos : pos + len(lo)] = lo
        rom[pos + 0x8000 : pos + 0x8000 + len(hi)] = hi
        pos += 0x10000

    pos = 0x400000
    for i in range(0x20):
        hi = z3_rom[i * 0x8000 : i * 0x8000 + 0x8000]
        rom[pos + 0x8000 : pos + 0x8000 + len(hi)] = hi
        pos += 0x10000
    return rom


def apply_ips(rom: bytearray, ips: bytes) -> None:
    """Apply a standard IPS patch in-place (no RLE edge cases beyond IPS spec)."""
    if ips[:5] != b"PATCH":
        raise ValueError("IPS missing PATCH header")
    offset = 5
    end = len(ips) - 3  # EOF marker
    while offset < end:
        dest = (ips[offset] << 16) | (ips[offset + 1] << 8) | ips[offset + 2]
        length = (ips[offset + 3] << 8) | ips[offset + 4]
        offset += 5
        if length > 0:
            if dest + length > len(rom):
                raise ValueError(
                    f"IPS write past end of ROM: dest={dest:#x} len={length}"
                )
            rom[dest : dest + length] = ips[offset : offset + length]
            offset += length
        else:
            rle_length = (ips[offset] << 8) | ips[offset + 1]
            rle_byte = ips[offset + 2]
            offset += 3
            if dest + rle_length > len(rom):
                raise ValueError(
                    f"IPS RLE past end of ROM: dest={dest:#x} len={rle_length}"
                )
            rom[dest : dest + rle_length] = bytes([rle_byte]) * rle_length


def apply_seed_patch(rom: bytearray, patch: bytes) -> None:
    """Apply a samus.link / SMZ3 seed patch (u32 LE dest, u16 LE length, data)."""
    offset = 0
    n = len(patch)
    while offset < n:
        if offset + 6 > n:
            raise ValueError(f"Truncated seed patch at offset {offset}")
        dest, length = struct.unpack_from("<IH", patch, offset)
        offset += 6
        if offset + length > n:
            raise ValueError(
                f"Seed patch record overruns buffer at dest={dest:#x} len={length}"
            )
        if dest + length > len(rom):
            raise ValueError(
                f"Seed patch write past end of ROM: dest={dest:#x} len={length}"
            )
        rom[dest : dest + length] = patch[offset : offset + length]
        offset += length


def decode_seed_patch_b64(patch_b64: str) -> bytes:
    """Decode the base64 ``worlds[0].patch`` field from the samus.link API."""
    import base64

    return base64.b64decode(patch_b64)


def read_vanilla_roms(
    sm_path: Path | None = None,
    z3_path: Path | None = None,
) -> tuple[bytes, bytes]:
    """Load shared vanilla Super Metroid + ALttP ROMs."""
    sm_path = sm_path or SHARED_SM_ROM
    z3_path = z3_path or SHARED_Z3_ROM
    if not sm_path.is_file():
        raise FileNotFoundError(f"Missing Super Metroid ROM: {sm_path}")
    if not z3_path.is_file():
        raise FileNotFoundError(f"Missing Zelda 3 ROM: {z3_path}")
    return sm_path.read_bytes(), z3_path.read_bytes()


def build_combo_rom(
    seed_patch: bytes,
    *,
    sm_rom: bytes | None = None,
    z3_rom: bytes | None = None,
    base_ips: bytes | None = None,
    sm_path: Path | None = None,
    z3_path: Path | None = None,
) -> bytes:
    """Full pipeline: merge vanilla → base IPS → seed patch → final ROM bytes."""
    if sm_rom is None or z3_rom is None:
        sm_loaded, z3_loaded = read_vanilla_roms(sm_path, z3_path)
        sm_rom = sm_rom or sm_loaded
        z3_rom = z3_rom or z3_loaded
    if base_ips is None:
        base_ips = load_base_ips()

    rom = merge_sm_z3(sm_rom, z3_rom)
    apply_ips(rom, base_ips)
    apply_seed_patch(rom, seed_patch)
    return bytes(rom)


def write_combo_rom(path: Path, rom: bytes) -> Path:
    """Write combo ROM bytes to ``path`` (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rom)
    return path


def rom_title_bytes(rom: bytes | bytearray) -> bytes:
    """Return the 21-byte SNES internal title at PC ``0x007FC0`` (mapped).

    SMZ3 writes the title at SNES ``$00FFC0`` / ``$80FFC0`` which map into the
    combo image; after ExHi remapping the seed patch writes both. We read the
    low mapping used by the seed patch helper (``Snes(0x00FFC0)`` → offset in
    the 6 MiB image). The web client / patch use PC offsets already applied.
    """
    # Seed patches write title at PC offsets produced by Snes(); after apply,
    # a common place is 0x007FC0 for the header mirror and 0x407FC0 area.
    # Prefer ASCII starting with ZSM if present.
    for off in (0x007FC0, 0x00FFC0, 0x407FC0, 0x40FFC0):
        if off + 21 <= len(rom):
            chunk = bytes(rom[off : off + 21])
            if chunk.startswith(b"ZSM") or chunk[:3].isalpha():
                return chunk
    return bytes(rom[0x007FC0 : 0x007FC0 + 21]) if len(rom) > 0x007FC0 + 21 else b""

"""Snes9x save-state parsing and WRAM helpers for Harvest Moon SNES."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

from harvest.paths import GAME_DIR

STATES_DIR = GAME_DIR
WRAM_ABSOLUTE_BASE = 0x7E0000
WRAM_SIZE = 0x20000


def wram_offset(address: int) -> int:
    """Normalize a WRAM address to the 0x00000..0x1FFFF snapshot RAM range."""
    if 0 <= address < WRAM_SIZE:
        return address
    if WRAM_ABSOLUTE_BASE <= address < WRAM_ABSOLUTE_BASE + WRAM_SIZE:
        return address - WRAM_ABSOLUTE_BASE
    raise ValueError(f"Address is outside Harvest Moon WRAM: 0x{address:X}")


@dataclass(frozen=True)
class SaveStateData:
    ram: bytes  # 128KB WRAM
    vram: bytes  # 64KB VRAM


@dataclass(frozen=True)
class SaveStateArchive:
    header: bytes
    block_order: tuple[str, ...]
    blocks: dict[str, bytes]
    size_fields: dict[str, str]
    compressed: bool
    source_path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "SaveStateArchive":
        return cls.from_bytes(path.read_bytes(), source_path=path)

    @classmethod
    def from_bytes(cls, raw: bytes, *, source_path: Path | None = None) -> "SaveStateArchive":
        compressed = raw[:2] == b"\x1f\x8b"
        if compressed:
            raw = gzip.decompress(raw)
        if not raw.startswith(b"#!s9xsnp"):
            origin = source_path if source_path is not None else "<bytes>"
            raise ValueError(f"Not a snes9x snapshot: {origin}")

        header_end = raw.index(b"\n") + 1
        header = raw[:header_end]
        blocks: dict[str, bytes] = {}
        size_fields: dict[str, str] = {}
        block_order: list[str] = []
        pos = header_end
        while pos < len(raw):
            colon1 = raw.index(b":", pos)
            tag = raw[pos:colon1].decode("ascii")
            colon2 = raw.index(b":", colon1 + 1)
            size_field = raw[colon1 + 1 : colon2].decode("ascii")
            size = int(size_field)
            payload = raw[colon2 + 1 : colon2 + 1 + size]
            blocks[tag] = payload
            size_fields[tag] = size_field
            block_order.append(tag)
            pos = colon2 + 1 + size

        return cls(
            header=header,
            block_order=tuple(block_order),
            blocks=blocks,
            size_fields=size_fields,
            compressed=compressed,
            source_path=source_path,
        )

    def require_block(self, tag: str) -> bytes:
        payload = self.blocks.get(tag)
        if payload is None:
            raise ValueError(f"Save state missing {tag} block")
        return payload

    def with_block(self, tag: str, payload: bytes) -> "SaveStateArchive":
        blocks = dict(self.blocks)
        blocks[tag] = payload
        size_fields = dict(self.size_fields)
        block_order = list(self.block_order)
        if tag not in block_order:
            block_order.append(tag)
            size_fields[tag] = f"{len(payload):06d}"
        return SaveStateArchive(
            header=self.header,
            block_order=tuple(block_order),
            blocks=blocks,
            size_fields=size_fields,
            compressed=self.compressed,
            source_path=self.source_path,
        )

    def to_bytes(self) -> bytes:
        raw = bytearray(self.header)
        for tag in self.block_order:
            payload = self.blocks[tag]
            width = max(1, len(self.size_fields.get(tag, "")) or 6)
            raw.extend(tag.encode("ascii"))
            raw.extend(b":")
            raw.extend(f"{len(payload):0{width}d}".encode("ascii"))
            raw.extend(b":")
            raw.extend(payload)
        output = bytes(raw)
        if self.compressed:
            return gzip.compress(output, mtime=0)
        return output

    def write(self, path: Path) -> Path:
        path.write_bytes(self.to_bytes())
        return path


@dataclass
class MutableSaveState:
    archive: SaveStateArchive
    ram: bytearray
    vram: bytearray

    @classmethod
    def load(cls, path: Path) -> "MutableSaveState":
        archive = SaveStateArchive.load(path)
        return cls(
            archive=archive,
            ram=bytearray(archive.require_block("RAM")),
            vram=bytearray(archive.require_block("VRA")),
        )

    def to_data(self) -> SaveStateData:
        return SaveStateData(ram=bytes(self.ram), vram=bytes(self.vram))

    def save(self, path: Path) -> Path:
        archive = self.archive.with_block("RAM", bytes(self.ram)).with_block("VRA", bytes(self.vram))
        archive.write(path)
        self.archive = archive
        return path

    def read_u8(self, address: int) -> int:
        return self.ram[wram_offset(address)]

    def read_u16(self, address: int) -> int:
        offset = wram_offset(address)
        return self.ram[offset] | (self.ram[offset + 1] << 8)

    def read_u24(self, address: int) -> int:
        offset = wram_offset(address)
        return self.ram[offset] | (self.ram[offset + 1] << 8) | (self.ram[offset + 2] << 16)

    def write_u8(self, address: int, value: int) -> None:
        self.ram[wram_offset(address)] = value & 0xFF

    def write_u16(self, address: int, value: int) -> None:
        offset = wram_offset(address)
        self.ram[offset] = value & 0xFF
        self.ram[offset + 1] = (value >> 8) & 0xFF

    def write_u24(self, address: int, value: int) -> None:
        offset = wram_offset(address)
        self.ram[offset] = value & 0xFF
        self.ram[offset + 1] = (value >> 8) & 0xFF
        self.ram[offset + 2] = (value >> 16) & 0xFF


def parse_save_state(path: Path) -> SaveStateData:
    """Parse a snes9x save state (.state) into RAM + VRAM."""
    archive = SaveStateArchive.load(path)
    return SaveStateData(
        ram=archive.require_block("RAM"),
        vram=archive.require_block("VRA"),
    )


def list_save_states() -> list[str]:
    """List available save state names (without .state extension)."""
    if not STATES_DIR.exists():
        return []
    return sorted(p.stem for p in STATES_DIR.glob("*.state"))


def resolve_state_path(state_name: str) -> Path:
    """Resolve a state name to its .state file path."""
    path = STATES_DIR / f"{state_name}.state"
    if not path.exists():
        raise FileNotFoundError(f"Save state not found: {path}")
    return path


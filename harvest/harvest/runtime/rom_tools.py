#!/usr/bin/env python3
"""ROM-first inspection utilities for Harvest Moon SNES.

This module treats the ROM as the source of truth and uses HM-Decomp only as a
comparison target. It is intentionally biased toward byte-accurate inspection so
we can build editing tools on top of verified structures instead of emulator
captures or hand-copied notes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from harvest.paths import DECOMP_DIR, GAME_DIR, PROJECT_DIR, ROMS_DIR, SHARED_ROMS_DIR

SCRIPT_DIR = PROJECT_DIR
MAPS_GRAPHICS_TABLE_ADDR = 0x80AA7C
MAPS_GRAPHICS_ASM_PATH = DECOMP_DIR / "src" / "maps" / "Maps_Graphics.asm"
BANK_80_ASM_PATH = DECOMP_DIR / "src" / "code_banks" / "bank_80.asm"
ROM_SHA_PATH = GAME_DIR / "rom.sha"
DEFAULT_ROM_CANDIDATES = (
    ROMS_DIR / "Harvest Moon.sfc",
    ROMS_DIR / "Harvest Moon.smc",
    SHARED_ROMS_DIR / "Harvest Moon.sfc",
    SHARED_ROMS_DIR / "Harvest Moon.smc",
    SHARED_ROMS_DIR / "Harvest Moon.zip",
)
GRAPHIC_PRESET_COUNT = 11
BACKGROUND_PALETTE_SLOTS_PER_MAP = 6
SPRITE_PALETTE_SLOTS_PER_MAP = 2


@dataclass(frozen=True)
class LabeledDataTable:
    label: str
    snes_address: int


GRAPHIC_PRESET_TABLES = (
    LabeledDataTable("Table_OBSEL_Presets", 0x808B44),
    LabeledDataTable("Table_BGMODE_Presets", 0x808B4F),
    LabeledDataTable("Table_BG1SC_Presets", 0x808B5A),
    LabeledDataTable("Table_BG2SC_Presets", 0x808B65),
    LabeledDataTable("Table_BG3SC_Presets", 0x808B70),
    LabeledDataTable("Table_BG4SC_Presets", 0x808B7B),
    LabeledDataTable("Table_BG12NBA_Presets", 0x808B86),
    LabeledDataTable("Table_BG34NBA_Presets", 0x808B91),
    LabeledDataTable("Table_TM_Presets", 0x808B9C),
    LabeledDataTable("Table_TS_Presets", 0x808BA7),
    LabeledDataTable("Table_TMW_Presets", 0x808BB2),
    LabeledDataTable("Table_TSW_Presets", 0x808BBD),
    LabeledDataTable("Table_CGWSEL_Presets", 0x808BC8),
    LabeledDataTable("Table_CGADSUB_Presets", 0x808BD3),
    LabeledDataTable("Table_SETINI_Presets", 0x808BDE),
    LabeledDataTable("Table_W12SEL_Presets", 0x808BE9),
    LabeledDataTable("Table_W34SEL_Presets", 0x808BF4),
    LabeledDataTable("Table_WOBJSEL_Presets", 0x808BFF),
    LabeledDataTable("Table_WH0_Presets", 0x808C0A),
    LabeledDataTable("Table_WH1_Presets", 0x808C15),
    LabeledDataTable("Table_WH2_Presets", 0x808C20),
    LabeledDataTable("Table_WH3_Presets", 0x808C2B),
    LabeledDataTable("Table_WBGLOG_Presets", 0x808C36),
    LabeledDataTable("Table_WOBJLOG_Presets", 0x808C41),
)

MAP_PALETTE_TABLES = (
    LabeledDataTable("PalettePointerTable", 0x80B9FD),
    LabeledDataTable("Time_Palette_Table", 0x80BB5C),
    LabeledDataTable("UNK_Table9", 0x80BD9C),
    LabeledDataTable("UNK_Table10", 0x80BDFC),
    LabeledDataTable("UNK_Table11", 0x80BE44),
)

VRAM_LAYOUT_TABLES = (
    LabeledDataTable("UNK_Table2", 0x80AA68),
    LabeledDataTable("UNK_Table3", 0x80AA72),
)

UNK_TABLE2_ADDR = 0x80AA68
UNK_TABLE3_ADDR = 0x80AA72
VRAM_LAYOUT_ENTRY_COUNT = 5

UNK_TABLE9_ADDR = 0x80BD9C
UNK_TABLE9_ENTRY_SIZE = 6
UNK_TABLE9_ENTRY_COUNT = 16

UNK_TABLE10_ADDR = 0x80BDFC
UNK_TABLE10_ENTRY_SIZE = 6
UNK_TABLE10_ENTRY_COUNT = 12


@dataclass(frozen=True)
class MapGraphicsBlock:
    kind: str
    vram_destination: int
    source_address: int
    compressed_offset: int
    decompressed_size: int
    vram_dma_size: int | None


@dataclass(frozen=True)
class MapGraphicsEntry:
    tilemap_id: int
    entry_label: str | None
    entry_address: int
    graphic_preset: int | None
    flag_0196_mask: int | None
    param_0181: int
    param_0182: int
    object_clamp_left: int | None
    object_clamp_right: int | None
    object_clamp_up: int | None
    object_clamp_down: int | None
    tilemaps: tuple[MapGraphicsBlock, ...]
    charactermaps: tuple[MapGraphicsBlock, ...]
    raw_bytes: bytes


@dataclass(frozen=True)
class ByteComparison:
    label: str
    source_path: str
    compared_bytes: int
    length_delta: int
    mismatch_count: int
    first_mismatches: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class GraphicPreset:
    preset_id: int
    obsel: int
    bgmode: int
    bg1sc: int
    bg2sc: int
    bg3sc: int
    bg4sc: int
    bg12nba: int
    bg34nba: int
    w12sel: int
    w34sel: int
    wobjsel: int
    wbglog: int
    wobjlog: int
    wh0: int
    wh1: int
    wh2: int
    wh3: int
    tm: int
    ts: int
    tmw: int
    tsw: int
    cgwsel: int
    cgadsub: int
    setini: int
    coldata: int


@dataclass(frozen=True)
class PaletteReference:
    palette_index: int | None
    source_address: int | None
    source_offset: int | None


@dataclass(frozen=True)
class VramLayout:
    param_0181: int
    charmap_row_height: int
    charmap_vram_size: int
    tilemap_dma_size: int


@dataclass(frozen=True)
class SpritePaletteOverride:
    table_name: str
    table_index: int
    colors: tuple[int, ...]
    target_slots: tuple[int, ...]


@dataclass(frozen=True)
class MapSceneModel:
    tilemap_id: int
    entry_label: str | None
    map_entry: MapGraphicsEntry
    graphic_preset: GraphicPreset | None
    background_palettes: tuple[PaletteReference, ...]
    sprite_palettes: tuple[PaletteReference, ...]
    vram_layout: VramLayout


def _expected_rom_sha1() -> str | None:
    if not ROM_SHA_PATH.exists():
        return None
    return ROM_SHA_PATH.read_text(encoding="utf-8").strip() or None


def _parse_hexish_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    token = token.split()[0]
    token = token.rstrip(",")
    token = token.lstrip("#")
    if token.startswith("$"):
        return int(token[1:], 16)
    if token.startswith("0x") or token.startswith("0X"):
        return int(token, 16)
    if token.isdigit():
        return int(token, 10)
    return None


def _directive_bytes(directive: str, operand_text: str) -> bytes:
    return _directive_bytes_with_fallback(directive, operand_text, fallback_values=None)


def _directive_bytes_with_fallback(
    directive: str,
    operand_text: str,
    fallback_values: list[int] | None,
) -> bytes:
    width = {"db": 1, "dw": 2, "dl": 3}[directive]
    out = bytearray()
    fallback_queue = list(fallback_values or [])
    for raw_operand in operand_text.split(","):
        value = _parse_hexish_int(raw_operand)
        if value is None and fallback_queue:
            value = fallback_queue.pop(0)
        if value is None:
            continue
        out.extend(value.to_bytes(width, "little"))
    return bytes(out)


def _directive_comment_fallback_values(raw_line: str, directive: str) -> list[int]:
    width = {"db": 1, "dw": 2, "dl": 3}[directive]
    comment_fields = raw_line.split(";")[1:]
    if len(comment_fields) < 2:
        return []

    values: list[int] = []
    for field in comment_fields[1:]:
        stripped = field.strip()
        if not stripped:
            continue
        token = stripped.split()[0].lstrip("$")
        if not re.fullmatch(r"[0-9A-Fa-f]+", token):
            continue
        value = int(token, 16)
        if value <= (1 << (width * 8)) - 1:
            values.append(value)
    return values


def parse_numeric_asm_bytes(path: Path) -> bytes:
    """Extract raw numeric bytes from asm files that use db/dw/dl data directives."""
    output = bytearray()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        directive = parts[0].lower()
        if directive not in {"db", "dw", "dl"}:
            continue
        operand_text = parts[1] if len(parts) > 1 else ""
        fallback_values = _directive_comment_fallback_values(raw_line, directive)
        output.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))
    return bytes(output)


def parse_maps_graphics_asm(path: Path = MAPS_GRAPHICS_ASM_PATH) -> tuple[list[str], dict[str, bytes]]:
    """Parse HM-Decomp's map graphics table into labels plus entry byte payloads."""
    table_labels: list[str] = []
    entry_bytes: dict[str, bytes] = {}
    current_label: str | None = None
    current_payload = bytearray()
    in_table = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        label_match = re.match(r"^([A-Za-z0-9_]+):", stripped)
        if label_match:
            if current_label is not None:
                entry_bytes[current_label] = bytes(current_payload)
                current_label = None
                current_payload = bytearray()

            label = label_match.group(1)
            in_table = label == "Maps_Graphics_Table"
            if not in_table:
                current_label = label
            continue

        if in_table:
            table_match = re.match(r"^dw\s+([A-Za-z_][A-Za-z0-9_]*)$", stripped)
            if table_match:
                table_labels.append(table_match.group(1))
            continue

        if current_label is None:
            continue

        parts = stripped.split(None, 1)
        directive = parts[0].lower()
        if directive not in {"db", "dw", "dl"}:
            continue
        operand_text = parts[1] if len(parts) > 1 else ""
        fallback_values = _directive_comment_fallback_values(raw_line, directive)
        current_payload.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))

    if current_label is not None:
        entry_bytes[current_label] = bytes(current_payload)

    return table_labels, entry_bytes


def parse_labeled_data_asm(path: Path) -> dict[str, bytes]:
    """Parse contiguous db/dw/dl blocks keyed by their label."""
    blocks: dict[str, bytes] = {}
    current_label: str | None = None
    current_payload = bytearray()
    collecting = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        label_match = re.match(r"^([A-Za-z0-9_]+):", stripped)
        if label_match:
            if current_label is not None and collecting:
                blocks[current_label] = bytes(current_payload)
            current_label = label_match.group(1)
            current_payload = bytearray()
            collecting = False
            stripped = stripped.split(":", 1)[1].strip()
            if not stripped:
                continue

        if current_label is None:
            continue

        parts = stripped.split(None, 1)
        directive = parts[0].lower()
        if directive in {"db", "dw", "dl"}:
            operand_text = parts[1] if len(parts) > 1 else ""
            fallback_values = _directive_comment_fallback_values(raw_line, directive)
            current_payload.extend(_directive_bytes_with_fallback(directive, operand_text, fallback_values))
            collecting = True
            continue

        if collecting:
            blocks[current_label] = bytes(current_payload)
        current_label = None
        current_payload = bytearray()
        collecting = False

    if current_label is not None and collecting:
        blocks[current_label] = bytes(current_payload)

    return blocks


class HarvestMoonRom:
    def __init__(self, path: Path, data: bytes, *, sha1: str, header_size: int) -> None:
        self.path = path
        self.data = data
        self.sha1 = sha1
        self.header_size = header_size

    @classmethod
    def load(cls, rom_path: Path | None = None) -> "HarvestMoonRom":
        path = rom_path or cls.find_default_path()
        if path is None:
            raise FileNotFoundError("Could not locate Harvest Moon ROM")
        raw = cls._read_container(path)
        header_size = 512 if len(raw) % 0x8000 == 512 else 0
        if header_size:
            raw = raw[header_size:]
        sha1 = hashlib.sha1(raw).hexdigest()
        return cls(path, raw, sha1=sha1, header_size=header_size)

    @staticmethod
    def find_default_path() -> Path | None:
        for candidate in DEFAULT_ROM_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _read_container(path: Path) -> bytes:
        if path.suffix.lower() != ".zip":
            return path.read_bytes()
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith((".sfc", ".smc"))]
            if not names:
                raise FileNotFoundError(f"No .sfc/.smc ROM found in {path}")
            with archive.open(names[0]) as handle:
                return handle.read()

    @staticmethod
    def lorom_to_offset(snes_address: int) -> int:
        bank = (snes_address >> 16) & 0xFF
        addr = snes_address & 0xFFFF
        return (bank & 0x7F) * 0x8000 + (addr & 0x7FFF)

    def read(self, snes_address: int, size: int) -> bytes:
        offset = self.lorom_to_offset(snes_address)
        return self.data[offset : offset + size]

    def read_u8(self, snes_address: int) -> int:
        return self.read(snes_address, 1)[0]

    def read_u16(self, snes_address: int) -> int:
        data = self.read(snes_address, 2)
        return data[0] | (data[1] << 8)

    def read_u24(self, snes_address: int) -> int:
        data = self.read(snes_address, 3)
        return data[0] | (data[1] << 8) | (data[2] << 16)

    def read_table_bytes(self, snes_address: int, size: int) -> bytes:
        return self.read(snes_address, size)

    def bank_bytes(self, bank: int) -> bytes:
        if not (0x80 <= bank <= 0xBF):
            raise ValueError("bank must be in 0x80..0xBF for Harvest Moon LoROM data")
        start = self.lorom_to_offset((bank << 16) | 0x8000)
        return self.data[start : start + 0x8000]

    def read_palette_pointer(self, palette_index: int) -> int:
        if palette_index < 0:
            raise ValueError("palette_index must be non-negative")
        return self.read_u24(0x80B9FD + palette_index * 3)

    def resolve_palette_reference(self, palette_index: int | None) -> PaletteReference:
        if palette_index is None:
            return PaletteReference(palette_index=None, source_address=None, source_offset=None)
        source_address = self.read_palette_pointer(palette_index)
        return PaletteReference(
            palette_index=palette_index,
            source_address=source_address,
            source_offset=self.lorom_to_offset(source_address),
        )

    def read_vram_layout(self, param_0181: int) -> VramLayout:
        if not (0 <= param_0181 < VRAM_LAYOUT_ENTRY_COUNT):
            raise ValueError(f"param_0181 must be in 0..{VRAM_LAYOUT_ENTRY_COUNT - 1}")
        charmap_row_height = self.read_u16(UNK_TABLE2_ADDR + param_0181 * 2)
        charmap_vram_size = self.read_u16(UNK_TABLE3_ADDR + param_0181 * 2)
        return VramLayout(
            param_0181=param_0181,
            charmap_row_height=charmap_row_height,
            charmap_vram_size=charmap_vram_size,
            tilemap_dma_size=0x2000,
        )

    def read_sprite_palette_overrides(
        self, tilemap_id: int, season: int, hour: int
    ) -> list[SpritePaletteOverride]:
        if not (0 <= season <= 3):
            raise ValueError("season must be 0..3")
        if not (0 <= hour <= 23):
            raise ValueError("hour must be 0..23")

        # Phase 1 time offset: after 6PM on non-seasonal outdoor maps → offset 4
        time_offset = 0
        if hour >= 18:
            if tilemap_id >= 0x31 or tilemap_id < 0x15:
                time_offset = 4
            # 0x15..0x30 are seasonal outdoor maps, no time offset

        # Category index from tilemap_id
        if tilemap_id < 0x04:
            # Farm maps → season
            category = season
        elif tilemap_id < 0x10 or tilemap_id >= 0x14:
            # Fork maps (0x04-0x0F) and mountain maps (0x14+) → season
            category = season
        else:
            # Maps 0x10-0x13 → tilemap - 8
            category = tilemap_id - 8

        base_index = category + time_offset
        table_offset = base_index * UNK_TABLE9_ENTRY_SIZE
        colors = tuple(
            self.read_u16(UNK_TABLE9_ADDR + table_offset + i * 2) for i in range(3)
        )
        return [
            SpritePaletteOverride(
                table_name="UNK_Table9",
                table_index=base_index,
                colors=colors,
                target_slots=(0x0A, 0x0B, 0x0C),
            )
        ]

    def read_graphic_preset(self, preset_id: int) -> GraphicPreset:
        if not (0 <= preset_id < GRAPHIC_PRESET_COUNT):
            raise ValueError(f"preset_id must be in 0..{GRAPHIC_PRESET_COUNT - 1}")
        values = {
            table.label: self.read_u8(table.snes_address + preset_id) for table in GRAPHIC_PRESET_TABLES
        }
        return GraphicPreset(
            preset_id=preset_id,
            obsel=values["Table_OBSEL_Presets"],
            bgmode=values["Table_BGMODE_Presets"],
            bg1sc=values["Table_BG1SC_Presets"],
            bg2sc=values["Table_BG2SC_Presets"],
            bg3sc=values["Table_BG3SC_Presets"],
            bg4sc=values["Table_BG4SC_Presets"],
            bg12nba=values["Table_BG12NBA_Presets"],
            bg34nba=values["Table_BG34NBA_Presets"],
            w12sel=values["Table_W12SEL_Presets"],
            w34sel=values["Table_W34SEL_Presets"],
            wobjsel=values["Table_WOBJSEL_Presets"],
            wbglog=values["Table_WBGLOG_Presets"],
            wobjlog=values["Table_WOBJLOG_Presets"],
            wh0=values["Table_WH0_Presets"],
            wh1=values["Table_WH1_Presets"],
            wh2=values["Table_WH2_Presets"],
            wh3=values["Table_WH3_Presets"],
            tm=values["Table_TM_Presets"],
            ts=values["Table_TS_Presets"],
            tmw=values["Table_TMW_Presets"],
            tsw=values["Table_TSW_Presets"],
            cgwsel=values["Table_CGWSEL_Presets"],
            cgadsub=values["Table_CGADSUB_Presets"],
            setini=values["Table_SETINI_Presets"],
            coldata=0xE0,
        )

    def compressed_block_size(self, snes_address: int) -> int:
        return self.read_u16(snes_address)

    def decompress_block(self, snes_address: int) -> bytes:
        offset = self.lorom_to_offset(snes_address)
        data = self.data[offset:]
        target_size = data[0] | (data[1] << 8)
        output = bytearray(target_size)
        ring = bytearray(target_size)
        output_i = 0
        ring_i = 2014
        data_i = 4
        flags = 0
        flags_i = -1

        while output_i < target_size:
            if flags_i < 0:
                flags = data[data_i]
                data_i += 1
                flags_i = 7
                continue

            literal = (flags >> flags_i) & 1
            flags_i -= 1
            if literal:
                value = data[data_i]
                data_i += 1
                output[output_i] = value
                ring[ring_i] = value
                output_i += 1
                ring_i = (ring_i + 1) % len(ring)
                continue

            offset_in_ring = data[data_i]
            special = data[data_i + 1]
            data_i += 2
            copy_count = (special & 0x1F) + 3
            offset_in_ring |= (special & 0xE0) << 3
            for _ in range(copy_count):
                value = ring[offset_in_ring]
                output[output_i] = value
                ring[ring_i] = value
                output_i += 1
                ring_i = (ring_i + 1) % len(ring)
                offset_in_ring = (offset_in_ring + 1) % len(ring)
                if output_i >= target_size:
                    break

        return bytes(output)

    def read_map_entry_bytes(self, tilemap_id: int) -> bytes:
        entry_address = self.read_map_entry_pointer(tilemap_id)
        cursor = entry_address
        payload = bytearray()
        is_background_entry = tilemap_id < 0x57
        if is_background_entry:
            payload.extend(self.read(cursor, 3))
            cursor += 3

        payload.extend(self.read(cursor, 4))
        param_offset = cursor
        cursor += 4
        tilemap_count = self.read_u8(param_offset + 2)
        charmap_count = self.read_u8(param_offset + 3)

        if is_background_entry:
            payload.extend(self.read(cursor, 8))
            cursor += 8

        block_count = tilemap_count + charmap_count
        payload.extend(self.read(cursor, block_count * 5))
        return bytes(payload)

    def read_map_entry_pointer(self, tilemap_id: int) -> int:
        if not (0 <= tilemap_id < 0x60):
            raise ValueError("tilemap_id must be in 0x00..0x5F")
        pointer = self.read_u16(MAPS_GRAPHICS_TABLE_ADDR + tilemap_id * 2)
        return 0x800000 | pointer

    def read_map_entry(self, tilemap_id: int, *, entry_label: str | None = None) -> MapGraphicsEntry:
        entry_address = self.read_map_entry_pointer(tilemap_id)
        cursor = entry_address
        graphic_preset: int | None = None
        flag_0196_mask: int | None = None
        clamp_left: int | None = None
        clamp_right: int | None = None
        clamp_up: int | None = None
        clamp_down: int | None = None
        raw = bytearray()

        is_background_entry = tilemap_id < 0x57
        if is_background_entry:
            graphic_preset = self.read_u8(cursor)
            raw.extend(self.read(cursor, 1))
            cursor += 1
            flag_0196_mask = self.read_u16(cursor)
            raw.extend(self.read(cursor, 2))
            cursor += 2

        param_0181 = self.read_u8(cursor)
        param_0182 = self.read_u8(cursor + 1)
        tilemap_count = self.read_u8(cursor + 2)
        charmap_count = self.read_u8(cursor + 3)
        raw.extend(self.read(cursor, 4))
        cursor += 4

        if is_background_entry:
            clamp_left = self.read_u16(cursor)
            clamp_right = self.read_u16(cursor + 2)
            clamp_up = self.read_u16(cursor + 4)
            clamp_down = self.read_u16(cursor + 6)
            raw.extend(self.read(cursor, 8))
            cursor += 8

        charmap_vram_size: int | None = None
        if 0 <= param_0181 < VRAM_LAYOUT_ENTRY_COUNT:
            charmap_vram_size = self.read_u16(UNK_TABLE3_ADDR + param_0181 * 2)

        tilemaps: list[MapGraphicsBlock] = []
        for _ in range(tilemap_count):
            vram_destination = self.read_u16(cursor)
            source_address = self.read_u24(cursor + 2)
            tilemaps.append(
                MapGraphicsBlock(
                    kind="tilemap",
                    vram_destination=vram_destination,
                    source_address=source_address,
                    compressed_offset=self.lorom_to_offset(source_address),
                    decompressed_size=self.compressed_block_size(source_address),
                    vram_dma_size=0x2000,
                )
            )
            raw.extend(self.read(cursor, 5))
            cursor += 5

        charactermaps: list[MapGraphicsBlock] = []
        for _ in range(charmap_count):
            vram_destination = self.read_u16(cursor)
            source_address = self.read_u24(cursor + 2)
            charactermaps.append(
                MapGraphicsBlock(
                    kind="charactermap",
                    vram_destination=vram_destination,
                    source_address=source_address,
                    compressed_offset=self.lorom_to_offset(source_address),
                    decompressed_size=self.compressed_block_size(source_address),
                    vram_dma_size=charmap_vram_size,
                )
            )
            raw.extend(self.read(cursor, 5))
            cursor += 5

        return MapGraphicsEntry(
            tilemap_id=tilemap_id,
            entry_label=entry_label,
            entry_address=entry_address,
            graphic_preset=graphic_preset,
            flag_0196_mask=flag_0196_mask,
            param_0181=param_0181,
            param_0182=param_0182,
            object_clamp_left=clamp_left,
            object_clamp_right=clamp_right,
            object_clamp_up=clamp_up,
            object_clamp_down=clamp_down,
            tilemaps=tuple(tilemaps),
            charactermaps=tuple(charactermaps),
            raw_bytes=bytes(raw),
        )

    def read_map_scene(self, tilemap_id: int, *, entry_label: str | None = None) -> MapSceneModel:
        entry = self.read_map_entry(tilemap_id, entry_label=entry_label)
        graphic_preset = None
        if entry.graphic_preset is not None:
            graphic_preset = self.read_graphic_preset(entry.graphic_preset)

        vram_layout = self.read_vram_layout(entry.param_0181)

        background_palette_indices = self.read(
            0x80BB5C + tilemap_id * BACKGROUND_PALETTE_SLOTS_PER_MAP,
            BACKGROUND_PALETTE_SLOTS_PER_MAP,
        )
        sprite_palette_indices = self.read(
            0x80BE44 + tilemap_id * SPRITE_PALETTE_SLOTS_PER_MAP,
            SPRITE_PALETTE_SLOTS_PER_MAP,
        )

        background_palettes = tuple(
            self.resolve_palette_reference(None if palette_index == 0xFF else palette_index)
            for palette_index in background_palette_indices
        )
        sprite_palettes = tuple(
            self.resolve_palette_reference(None if palette_index == 0xFF else palette_index)
            for palette_index in sprite_palette_indices
        )
        return MapSceneModel(
            tilemap_id=tilemap_id,
            entry_label=entry_label,
            map_entry=entry,
            graphic_preset=graphic_preset,
            background_palettes=background_palettes,
            sprite_palettes=sprite_palettes,
            vram_layout=vram_layout,
        )


# ---------------------------------------------------------------------------
# Save-state parsing & SNES graphics decoders
# ---------------------------------------------------------------------------

STATES_DIR = GAME_DIR
WRAM_ABSOLUTE_BASE = 0x7E0000
WRAM_SIZE = 0x20000

# RAM addresses
METATILE_GRID_ADDR = 0x09B6
METATILE_GRID_SIZE = 64 * 64
METATILE_LOOKUP_ADDR = 0x2000
METATILE_LOOKUP_SIZE = 256 * 16  # 256 entries x 16 bytes
CGRAM_MIRROR_ADDR = 0x10900
CGRAM_MIRROR_SIZE = 512  # 16 palettes x 16 colors x 2 bytes
TILEMAP_ID_ADDR = 0x22
PLAYER_X_ADDR = 0xD6
PLAYER_Y_ADDR = 0xD8

# ROM addresses for metatile rendering (from HM-Decomp analysis)
# DATA8_81B363: visual_tile_index -> (rom_offset, width, height) descriptor
_TILE_INDEX_TABLE_ADDR = 0x81B363
# ROM bank $A6 starting at $A096: actual tilemap word data for metatile visuals
_TILE_DATA_SNES_BANK = 0xA6
_TILE_DATA_BASE_ADDR = 0xA096
_TILE_DATA_ROW_STRIDE = 0x40  # bytes between rows in ROM tile data

# UNK_PointersTable: tilemap_id -> metatile lookup table SNES address
# Each entry is a DATA16 table in bank $82 mapping metatile_id -> visual_tile_index
_METATILE_LOOKUP_BY_MAP: tuple[int, ...] = (
    0x82B3B4, 0x82B3B4, 0x82B3B4, 0x82B3B4,  # 0-3: farm (all seasons)
    0x82BFB4, 0x82BFB4, 0x82BFB4, 0x82BFB4,  # 4-7: town variants
    0x82BFB4, 0x82BFB4, 0x82BFB4, 0x82BFB4,  # 8-11
    0x82BBB4, 0x82BBB4, 0x82BBB4, 0x82BBB4,  # 12-15: mountain
    0x82C3B4, 0x82C3B4, 0x82C3B4, 0x82C3B4,  # 16-19
    0x82C3B4, 0x82B7B4, 0x82B7B4, 0x82B7B4,  # 20-23
    0x82C7B4, 0x82C7B4, 0x82C7B4, 0x82C7B4,  # 24-27
    0x82C7B4, 0x82C7B4, 0x82CBB4, 0x82CBB4,  # 28-31
    0x82CBB4, 0x82CBB4, 0x82CBB4, 0x82CBB4,  # 32-35
    0x82C7B4, 0x82CBB4, 0x82BBB4, 0x82BBB4,  # 36-39
    0x82BBB4, 0x82C3B4, 0x82B7B4, 0x82C3B4,  # 40-43
    0x82B7B4,                                   # 44
)
# Fence/crop metatile IDs that use neighbor-based variants
_VARIANT_META_ID_MIN = 0x73
_VARIANT_META_ID_MAX = 0x79
# Variant lookup: neighbor bitmask -> variant offset (from DATA8_82B02A)
_VARIANT_OFFSETS = (9, 2, 7, 1, 7, 4, 7, 4, 9, 3, 8, 2, 9, 6, 8, 5)


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


def decode_4bpp_tile(data: bytes, offset: int = 0) -> np.ndarray:
    """Decode one 8x8 4bpp planar SNES tile into an (8,8) palette-index array."""
    pixels = np.zeros((8, 8), dtype=np.uint8)
    for row in range(8):
        # Bitplanes 0-1 in first 16 bytes, bitplanes 2-3 in next 16 bytes
        bp0 = data[offset + row * 2]
        bp1 = data[offset + row * 2 + 1]
        bp2 = data[offset + row * 2 + 16]
        bp3 = data[offset + row * 2 + 17]
        for col in range(8):
            shift = 7 - col
            pixels[row, col] = (
                ((bp0 >> shift) & 1)
                | (((bp1 >> shift) & 1) << 1)
                | (((bp2 >> shift) & 1) << 2)
                | (((bp3 >> shift) & 1) << 3)
            )
    return pixels


def bgr555_to_rgb(color: int) -> tuple[int, int, int]:
    """Convert a 15-bit BGR555 SNES color to (R, G, B) 8-bit."""
    r = (color & 0x1F) << 3
    g = ((color >> 5) & 0x1F) << 3
    b = ((color >> 10) & 0x1F) << 3
    return r, g, b


def build_palette_rgb(cgram_bytes: bytes) -> np.ndarray:
    """Build (16, 16, 3) RGB palette from 512-byte CGRAM mirror."""
    palette = np.zeros((16, 16, 3), dtype=np.uint8)
    for row in range(16):
        for col in range(16):
            offset = (row * 16 + col) * 2
            color = cgram_bytes[offset] | (cgram_bytes[offset + 1] << 8)
            palette[row, col] = bgr555_to_rgb(color)
    return palette


def _decode_vram_tile(
    vram: bytes,
    char_base_words: int,
    tile_num: int,
) -> np.ndarray:
    """Decode a single 4bpp tile from VRAM given char base and tile number."""
    byte_offset = ((char_base_words + tile_num * 16) & 0x7FFF) * 2
    if byte_offset + 32 > len(vram):
        return np.zeros((8, 8), dtype=np.uint8)
    return decode_4bpp_tile(vram, byte_offset)


def _render_8x8_tile(
    pixels: np.ndarray,
    palette_rgb: np.ndarray,
    palette_row: int,
    hflip: bool,
    vflip: bool,
) -> np.ndarray:
    """Render an 8x8 indexed tile to (8, 8, 3) RGB using the given palette row."""
    if hflip:
        pixels = pixels[:, ::-1]
    if vflip:
        pixels = pixels[::-1, :]
    result = np.zeros((8, 8, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            idx = int(pixels[r, c])
            result[r, c] = palette_rgb[palette_row, idx]
    return result


def _extract_ground_words(state: SaveStateData, bg1sc: int) -> tuple[int, int, int, int]:
    """Extract ground tile words from the VRAM BG1 tilemap.

    Reads the BG1 tilemap from the save state's VRAM and finds the most common
    2x2 tile block, which should be the ground/dirt pattern.
    """
    vram = state.vram
    tilemap_base = (bg1sc >> 2) * 0x800  # byte address in VRAM
    # 64x64 tilemap: 4 screens of 32x32
    # Read all tile entries and find the most common 2x2 block
    from collections import Counter

    def _read_vram_tile(col: int, row: int) -> int:
        """Read a single tilemap word from VRAM at (col, row) in 64x64 grid."""
        sx, sy = col // 32, row // 32
        screen = sx + sy * 2
        lc, lr = col % 32, row % 32
        offset = tilemap_base + screen * 0x800 + lr * 0x40 + lc * 2
        if offset + 2 > len(vram):
            return 0
        return vram[offset] | (vram[offset + 1] << 8)

    # Sample 2x2 blocks across the tilemap, count occurrences
    block_counter: Counter[tuple[int, int, int, int]] = Counter()
    for row in range(0, 62, 2):
        for col in range(0, 62, 2):
            block = (
                _read_vram_tile(col, row),
                _read_vram_tile(col + 1, row),
                _read_vram_tile(col, row + 1),
                _read_vram_tile(col + 1, row + 1),
            )
            # Skip all-zero blocks
            if block == (0, 0, 0, 0):
                continue
            block_counter[block] += 1

    if block_counter:
        return block_counter.most_common(1)[0][0]
    return (0, 0, 0, 0)


def _metatile_lookup_addr(tilemap_id: int) -> int:
    """Return SNES address of the metatile lookup table for a given map."""
    if tilemap_id < 4:
        return 0x82B3B4  # farm (all seasons use same table)
    if tilemap_id < len(_METATILE_LOOKUP_BY_MAP):
        return _METATILE_LOOKUP_BY_MAP[tilemap_id]
    return 0x82B3B4  # fallback


def _read_tile_words_from_rom(
    rom: "HarvestMoonRom",
    tile_index: int,
) -> tuple[int, int, int, int] | None:
    """Read 4 BG1 tilemap words (TL, TR, BL, BR) from ROM for a visual tile index.

    Returns None if the tile_index is 0 (empty).
    """
    if tile_index == 0:
        return None

    # Read descriptor from DATA8_81B363: 6 bytes per entry
    desc_addr = _TILE_INDEX_TABLE_ADDR + tile_index * 6
    rom_offset = rom.read_u16(desc_addr)       # bytes 0-1: offset into tile data
    # bytes 2-3: width/height (not needed for 1x1 metatile rendering)
    # bytes 4-5: extra data (unused for rendering)

    # Read 4 tilemap words from ROM bank $A6 at $A096 + rom_offset
    # Layout: 2 words per row, rows separated by 0x40 bytes
    base = (_TILE_DATA_SNES_BANK << 16) | (_TILE_DATA_BASE_ADDR + rom_offset)
    tl = rom.read_u16(base)
    tr = rom.read_u16(base + 2)
    bl = rom.read_u16(base + _TILE_DATA_ROW_STRIDE)
    br = rom.read_u16(base + _TILE_DATA_ROW_STRIDE + 2)
    return tl, tr, bl, br


def build_metatile_atlas(
    state: SaveStateData,
    bg12nba: int,
    *,
    rom: "HarvestMoonRom | None" = None,
    tilemap_id: int = 0,
    bg1sc: int = 0,
) -> np.ndarray:
    """Build a (256, 16, 16, 3) RGB atlas of all 256 metatiles.

    When *rom* is provided, uses the ROM lookup tables for correct metatile→tile
    mapping. Without ROM, falls back to the (incorrect) RAM-based method.

    Each metatile is 16x16 pixels (2x2 SNES 8x8 tiles).
    """
    vram = state.vram
    ram = state.ram
    bg1_base = (bg12nba & 0xF) * 0x1000

    cgram = ram[CGRAM_MIRROR_ADDR : CGRAM_MIRROR_ADDR + CGRAM_MIRROR_SIZE]
    palette_rgb = build_palette_rgb(cgram)

    atlas = np.zeros((256, 16, 16, 3), dtype=np.uint8)

    if rom is not None:
        lookup_addr = _metatile_lookup_addr(tilemap_id)

        for meta_id in range(256):
            # Read visual tile index (byte 0 of 4-byte entry in ROM lookup table)
            tile_index = rom.read_u8(lookup_addr + meta_id * 4)
            words = _read_tile_words_from_rom(rom, tile_index)
            if words is None:
                continue
            atlas[meta_id] = _render_metatile_from_words(
                words, vram, bg1_base, palette_rgb
            )
    else:
        # Legacy RAM-based fallback (reads from $7E:$2000 VRAM buffer -- incorrect
        # spatial interpretation, but kept for compatibility when ROM is unavailable)
        bg2_base = ((bg12nba >> 4) & 0xF) * 0x1000
        for meta_id in range(256):
            lookup_offset = METATILE_LOOKUP_ADDR + meta_id * 16
            entries = []
            for i in range(8):
                val = ram[lookup_offset + i * 2] | (ram[lookup_offset + i * 2 + 1] << 8)
                entries.append(val)
            quad_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]
            metatile = np.zeros((16, 16, 3), dtype=np.uint8)
            for q, (dx, dy) in enumerate(quad_offsets):
                val = entries[4 + q]
                tile_num = val & 0x3FF
                pal_row = (val >> 10) & 7
                hflip = bool(val & 0x4000)
                vflip = bool(val & 0x8000)
                pixels = _decode_vram_tile(vram, bg2_base, tile_num)
                rendered = _render_8x8_tile(pixels, palette_rgb, pal_row, hflip, vflip)
                metatile[dy : dy + 8, dx : dx + 8] = rendered
            for q, (dx, dy) in enumerate(quad_offsets):
                val = entries[q]
                tile_num = val & 0x3FF
                pal_row = (val >> 10) & 7
                hflip = bool(val & 0x4000)
                vflip = bool(val & 0x8000)
                pixels = _decode_vram_tile(vram, bg1_base, tile_num)
                rendered = _render_8x8_tile(pixels, palette_rgb, pal_row, hflip, vflip)
                if hflip:
                    raw_pixels = pixels[:, ::-1]
                else:
                    raw_pixels = pixels
                if vflip:
                    raw_pixels = raw_pixels[::-1, :]
                mask = raw_pixels != 0
                for ch in range(3):
                    metatile[dy : dy + 8, dx : dx + 8, ch] = np.where(
                        mask, rendered[:, :, ch], metatile[dy : dy + 8, dx : dx + 8, ch]
                    )
            atlas[meta_id] = metatile

    return atlas


def _render_metatile_from_words(
    words: tuple[int, int, int, int],
    vram: bytes,
    bg1_base: int,
    palette_rgb: np.ndarray,
) -> np.ndarray:
    """Render a 16x16 metatile from 4 BG1 tilemap words (TL, TR, BL, BR)."""
    metatile = np.zeros((16, 16, 3), dtype=np.uint8)
    # Quadrant positions: (dx, dy) for TL, TR, BL, BR
    positions = [(0, 0), (8, 0), (0, 8), (8, 8)]
    for (dx, dy), val in zip(positions, words):
        tile_num = val & 0x3FF
        pal_row = (val >> 10) & 7
        hflip = bool(val & 0x4000)
        vflip = bool(val & 0x8000)
        pixels = _decode_vram_tile(vram, bg1_base, tile_num)
        rendered = _render_8x8_tile(pixels, palette_rgb, pal_row, hflip, vflip)
        metatile[dy : dy + 8, dx : dx + 8] = rendered
    return metatile


def read_metatile_grid(ram: bytes) -> np.ndarray:
    """Read the 64x64 metatile grid from RAM as (64, 64) uint8."""
    data = ram[METATILE_GRID_ADDR : METATILE_GRID_ADDR + METATILE_GRID_SIZE]
    return np.frombuffer(data, dtype=np.uint8).reshape((64, 64)).copy()


def read_tilemap_id(ram: bytes) -> int:
    """Read the current tilemap ID from RAM."""
    return ram[TILEMAP_ID_ADDR]


def read_player_pos(ram: bytes) -> tuple[int, int]:
    """Read player pixel position (x, y) from RAM."""
    x = ram[PLAYER_X_ADDR] | (ram[PLAYER_X_ADDR + 1] << 8)
    y = ram[PLAYER_Y_ADDR] | (ram[PLAYER_Y_ADDR + 1] << 8)
    return x, y


def render_full_map(atlas: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Render the full 1024x1024 map from a metatile atlas and grid."""
    h, w = grid.shape
    tile_h, tile_w = atlas.shape[1], atlas.shape[2]
    canvas = np.zeros((h * tile_h, w * tile_w, 3), dtype=np.uint8)
    for ty in range(h):
        for tx in range(w):
            tile_id = int(grid[ty, tx])
            patch = atlas[tile_id]
            if not patch.any():
                patch = _semantic_fallback_tile(tile_id)
            y0 = ty * tile_h
            x0 = tx * tile_w
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = patch
    return canvas


def _semantic_fallback_tile(tile_id: int) -> np.ndarray:
    """Solid 16x16 fallback when ROM metatile art is unavailable."""

    colors = {
        0x00: (110, 90, 65),
        0x01: (90, 70, 50),
        0x02: (70, 55, 40),
        0x07: (70, 55, 40),
        0x08: (50, 45, 55),
        0x70: (80, 180, 60),
        0xA6: (30, 80, 180),
        0xFF: (40, 40, 40),
    }
    if 0x80 <= tile_id <= 0x85:
        rgb = (50, 160, 50)
    elif 0x1E <= tile_id <= 0x6F:
        rgb = (200, 180, 50)
    elif tile_id in (0xA0, 0xA2, 0xA3):
        rgb = (180, 160, 120)
    elif 0xF0 <= tile_id <= 0xFD:
        rgb = (30, 100, 200)
    else:
        rgb = colors.get(tile_id, (200, 50, 200))
    patch = np.zeros((16, 16, 3), dtype=np.uint8)
    patch[:, :] = rgb
    return patch


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


def compare_map_entry_to_hm_decomp(
    rom: HarvestMoonRom,
    tilemap_id: int,
    maps_asm_path: Path = MAPS_GRAPHICS_ASM_PATH,
) -> ByteComparison:
    table_labels, entry_bytes = parse_maps_graphics_asm(maps_asm_path)
    label = table_labels[tilemap_id]
    expected = entry_bytes.get(label, b"")
    actual = rom.read_map_entry_bytes(tilemap_id)
    compared = min(len(actual), len(expected))
    mismatches: list[tuple[int, int, int]] = []
    for idx in range(compared):
        if actual[idx] != expected[idx]:
            mismatches.append((idx, actual[idx], expected[idx]))
            if len(mismatches) >= 12:
                break
    return ByteComparison(
        label=label,
        source_path=str(maps_asm_path),
        compared_bytes=compared,
        length_delta=len(actual) - len(expected),
        mismatch_count=sum(1 for idx in range(compared) if actual[idx] != expected[idx]),
        first_mismatches=tuple(mismatches),
    )


def compare_labeled_data_to_hm_decomp(
    rom: HarvestMoonRom,
    *,
    label: str,
    snes_address: int,
    asm_path: Path = BANK_80_ASM_PATH,
) -> ByteComparison:
    blocks = parse_labeled_data_asm(asm_path)
    expected = blocks.get(label)
    if expected is None:
        raise KeyError(f"Label {label!r} not found in {asm_path}")
    actual = rom.read_table_bytes(snes_address, len(expected))
    compared = min(len(actual), len(expected))
    mismatches: list[tuple[int, int, int]] = []
    mismatch_count = 0
    for idx in range(compared):
        if actual[idx] != expected[idx]:
            mismatch_count += 1
            if len(mismatches) < 12:
                mismatches.append((idx, actual[idx], expected[idx]))
    mismatch_count += abs(len(actual) - len(expected))
    return ByteComparison(
        label=label,
        source_path=str(asm_path),
        compared_bytes=compared,
        length_delta=len(actual) - len(expected),
        mismatch_count=mismatch_count,
        first_mismatches=tuple(mismatches),
    )


def compare_data_bank_to_asm(
    rom: HarvestMoonRom,
    bank: int,
    asm_path: Path | None = None,
) -> ByteComparison:
    asm_path = asm_path or DECOMP_DIR / "src" / "data_banks" / f"bank_{bank:02X}.asm"
    expected = parse_numeric_asm_bytes(asm_path)
    actual = rom.bank_bytes(bank)
    compared = min(len(actual), len(expected))
    mismatches: list[tuple[int, int, int]] = []
    mismatch_count = 0
    for idx in range(compared):
        if actual[idx] != expected[idx]:
            mismatch_count += 1
            if len(mismatches) < 12:
                mismatches.append((idx, actual[idx], expected[idx]))
    mismatch_count += abs(len(actual) - len(expected))
    return ByteComparison(
        label=f"bank_{bank:02X}",
        source_path=str(asm_path),
        compared_bytes=compared,
        length_delta=len(actual) - len(expected),
        mismatch_count=mismatch_count,
        first_mismatches=tuple(mismatches),
    )


def _resolve_rom_path(arg: str | None) -> Path | None:
    if arg is None:
        return None
    return Path(arg).expanduser().resolve()


def _print_comparison(result: ByteComparison) -> None:
    print(
        f"{result.label}: compared={result.compared_bytes}"
        f" mismatches={result.mismatch_count}"
        f" length_delta={result.length_delta}"
    )
    if result.first_mismatches:
        print("  first mismatches:")
        for idx, actual, expected in result.first_mismatches:
            print(f"    +0x{idx:04X}: rom=0x{actual:02X} hm_decomp=0x{expected:02X}")


def _entry_to_jsonable(entry: MapGraphicsEntry) -> dict[str, object]:
    payload = asdict(entry)
    payload["raw_bytes"] = entry.raw_bytes.hex()
    return payload


def _scene_to_jsonable(scene: MapSceneModel) -> dict[str, object]:
    payload = asdict(scene)
    payload["map_entry"] = _entry_to_jsonable(scene.map_entry)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Moon ROM-first inspection tools")
    parser.add_argument("--rom", default=None, help="ROM path (.sfc/.smc/.zip). Defaults to known local candidates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Print ROM metadata and hash verification")

    dump_map = subparsers.add_parser("dump-map-entry", help="Decode one Maps_Graphics entry from the ROM")
    dump_map.add_argument("--tilemap-id", type=lambda value: int(value, 0), required=True)

    compare_map = subparsers.add_parser("compare-map-entry", help="Compare one ROM map entry to HM-Decomp")
    compare_map.add_argument("--tilemap-id", type=lambda value: int(value, 0), required=True)

    compare_all = subparsers.add_parser("compare-all-map-entries", help="Compare all Maps_Graphics entries to HM-Decomp")

    compare_labeled = subparsers.add_parser("compare-labeled-data", help="Compare one labeled db/dw/dl block to ROM bytes")
    compare_labeled.add_argument("--label", required=True)
    compare_labeled.add_argument("--address", type=lambda value: int(value, 0), required=True, help="SNES address for the label data")
    compare_labeled.add_argument("--asm-path", default=str(BANK_80_ASM_PATH), help="Asm file that defines the label")

    subparsers.add_parser("compare-graphic-preset-tables", help="Compare all graphic preset tables in bank_80.asm")
    subparsers.add_parser("compare-palette-tables", help="Compare palette-related tables in bank_80.asm")

    compare_bank = subparsers.add_parser("compare-data-bank", help="Compare one ROM data bank to HM-Decomp")
    compare_bank.add_argument("--bank", type=lambda value: int(value, 16), required=True, help="Hex bank number, e.g. A8")
    compare_bank.add_argument("--asm-path", default=None, help="Optional HM-Decomp asm path override")

    dump_preset = subparsers.add_parser("dump-graphic-preset", help="Decode one graphic preset from ROM")
    dump_preset.add_argument("--preset-id", type=lambda value: int(value, 0), required=True)

    dump_scene = subparsers.add_parser("dump-map-scene", help="Decode a map scene model from ROM")
    dump_scene.add_argument("--tilemap-id", type=lambda value: int(value, 0), required=True)

    export_scenes = subparsers.add_parser("export-map-scenes", help="Export all map scene models to JSON")
    export_scenes.add_argument("--output", required=True, help="Output JSON path")

    subparsers.add_parser("compare-vram-layout-tables", help="Compare UNK_Table2/3 VRAM layout tables in bank_80.asm")

    dump_sprite_pal = subparsers.add_parser("dump-sprite-palette-overrides", help="Decode sprite palette overrides for a map/season/hour")
    dump_sprite_pal.add_argument("--tilemap-id", type=lambda value: int(value, 0), required=True)
    dump_sprite_pal.add_argument("--season", type=int, required=True, help="0=spring, 1=summer, 2=fall, 3=winter")
    dump_sprite_pal.add_argument("--hour", type=int, required=True, help="0..23")

    block_info = subparsers.add_parser("block-info", help="Inspect one compressed block header and decompressed size")
    block_info.add_argument("--address", type=lambda value: int(value, 0), required=True, help="SNES address, e.g. 0x92D3AB")

    args = parser.parse_args()
    rom = HarvestMoonRom.load(_resolve_rom_path(args.rom))

    if args.command == "info":
        expected_sha = _expected_rom_sha1()
        payload = {
            "path": str(rom.path),
            "sha1": rom.sha1,
            "expected_sha1": expected_sha,
            "sha1_matches_expected": expected_sha == rom.sha1 if expected_sha else None,
            "header_size": rom.header_size,
            "size_bytes": len(rom.data),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "dump-map-entry":
        table_labels, _entry_bytes = parse_maps_graphics_asm()
        label = table_labels[args.tilemap_id] if args.tilemap_id < len(table_labels) else None
        entry = rom.read_map_entry(args.tilemap_id, entry_label=label)
        print(json.dumps(_entry_to_jsonable(entry), indent=2))
        return

    if args.command == "compare-map-entry":
        result = compare_map_entry_to_hm_decomp(rom, args.tilemap_id)
        _print_comparison(result)
        return

    if args.command == "compare-all-map-entries":
        table_labels, _entry_bytes = parse_maps_graphics_asm()
        mismatch_total = 0
        for tilemap_id in range(len(table_labels)):
            result = compare_map_entry_to_hm_decomp(rom, tilemap_id)
            mismatch_total += result.mismatch_count
            if result.mismatch_count or result.length_delta:
                _print_comparison(result)
        print(f"total_entry_mismatches={mismatch_total}")
        return

    if args.command == "compare-labeled-data":
        result = compare_labeled_data_to_hm_decomp(
            rom,
            label=args.label,
            snes_address=args.address,
            asm_path=Path(args.asm_path),
        )
        _print_comparison(result)
        return

    if args.command == "compare-graphic-preset-tables":
        mismatch_total = 0
        for table in GRAPHIC_PRESET_TABLES:
            result = compare_labeled_data_to_hm_decomp(
                rom,
                label=table.label,
                snes_address=table.snes_address,
            )
            mismatch_total += result.mismatch_count
            _print_comparison(result)
        print(f"total_graphic_preset_table_mismatches={mismatch_total}")
        return

    if args.command == "compare-palette-tables":
        mismatch_total = 0
        for table in MAP_PALETTE_TABLES:
            result = compare_labeled_data_to_hm_decomp(
                rom,
                label=table.label,
                snes_address=table.snes_address,
            )
            mismatch_total += result.mismatch_count
            _print_comparison(result)
        print(f"total_palette_table_mismatches={mismatch_total}")
        return

    if args.command == "compare-data-bank":
        asm_path = Path(args.asm_path) if args.asm_path else None
        result = compare_data_bank_to_asm(rom, args.bank, asm_path=asm_path)
        _print_comparison(result)
        return

    if args.command == "dump-graphic-preset":
        print(json.dumps(asdict(rom.read_graphic_preset(args.preset_id)), indent=2))
        return

    if args.command == "dump-map-scene":
        table_labels, _entry_bytes = parse_maps_graphics_asm()
        label = table_labels[args.tilemap_id] if args.tilemap_id < len(table_labels) else None
        scene = rom.read_map_scene(args.tilemap_id, entry_label=label)
        print(json.dumps(_scene_to_jsonable(scene), indent=2))
        return

    if args.command == "export-map-scenes":
        table_labels, _entry_bytes = parse_maps_graphics_asm()
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scenes = []
        for tilemap_id, label in enumerate(table_labels):
            scene = rom.read_map_scene(tilemap_id, entry_label=label)
            scenes.append(_scene_to_jsonable(scene))
        output_path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
        print(output_path)
        return

    if args.command == "compare-vram-layout-tables":
        mismatch_total = 0
        for table in VRAM_LAYOUT_TABLES:
            result = compare_labeled_data_to_hm_decomp(
                rom,
                label=table.label,
                snes_address=table.snes_address,
            )
            mismatch_total += result.mismatch_count
            _print_comparison(result)
        print(f"total_vram_layout_table_mismatches={mismatch_total}")
        return

    if args.command == "dump-sprite-palette-overrides":
        overrides = rom.read_sprite_palette_overrides(args.tilemap_id, args.season, args.hour)
        for override in overrides:
            print(json.dumps(asdict(override), indent=2))
        return

    if args.command == "block-info":
        address = args.address
        payload = {
            "address": f"0x{address:06X}",
            "rom_offset": f"0x{rom.lorom_to_offset(address):X}",
            "decompressed_size": rom.compressed_block_size(address),
            "first_16_bytes": rom.read(address, 16).hex(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()

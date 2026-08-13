"""ROM model, path constants, and map/graphics data types for Harvest Moon SNES."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

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

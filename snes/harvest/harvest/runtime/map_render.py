"""Metatile atlas and full-map render helpers from save-state + ROM data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from harvest.runtime.save_state_io import SaveStateData

if TYPE_CHECKING:
    from harvest.runtime.rom_model import HarvestMoonRom

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

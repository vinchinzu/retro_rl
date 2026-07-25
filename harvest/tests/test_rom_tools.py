from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from harvest.runtime.rom_tools import (
    BANK_80_ASM_PATH,
    MAPS_GRAPHICS_ASM_PATH,
    VRAM_LAYOUT_TABLES,
    HarvestMoonRom,
    SaveStateData,
    bgr555_to_rgb,
    build_metatile_atlas,
    build_palette_rgb,
    compare_labeled_data_to_hm_decomp,
    compare_map_entry_to_hm_decomp,
    decode_4bpp_tile,
    list_save_states,
    parse_labeled_data_asm,
    parse_maps_graphics_asm,
    parse_save_state,
    read_metatile_grid,
    read_player_pos,
    read_tilemap_id,
    render_full_map,
    resolve_state_path,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ROM_PATH = SCRIPT_DIR / "roms" / "Harvest Moon.sfc"
STATES_DIR = SCRIPT_DIR / "custom_integrations" / "HarvestMoon-Snes"


class LoRomMappingTests(unittest.TestCase):
    def test_lorom_to_offset_maps_known_addresses(self) -> None:
        self.assertEqual(HarvestMoonRom.lorom_to_offset(0x80AA7C), 0x2A7C)
        self.assertEqual(HarvestMoonRom.lorom_to_offset(0x92D3AB), 0x953AB)

    def test_load_strips_copier_header(self) -> None:
        rom_body = bytes(range(256)) * 128
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "headered.sfc"
            path.write_bytes(b"H" * 512 + rom_body)
            rom = HarvestMoonRom.load(path)

        self.assertEqual(rom.header_size, 512)
        self.assertEqual(rom.data[:32], rom_body[:32])


class MapsGraphicsAsmTests(unittest.TestCase):
    def test_parse_maps_graphics_asm_extracts_table_and_entry_bytes(self) -> None:
        table_labels, entry_bytes = parse_maps_graphics_asm(MAPS_GRAPHICS_ASM_PATH)
        self.assertGreaterEqual(len(table_labels), 0x60)
        self.assertEqual(table_labels[0], "MapFarmSpring")
        self.assertTrue(entry_bytes["MapFarmSpring"].startswith(bytes.fromhex("00E08004040301")))

    def test_parse_labeled_data_asm_extracts_bank_80_tables(self) -> None:
        blocks = parse_labeled_data_asm(BANK_80_ASM_PATH)
        self.assertEqual(blocks["Table_OBSEL_Presets"], bytes.fromhex("6060606060600303030363"))
        self.assertTrue(blocks["PalettePointerTable"].startswith(bytes.fromhex("0094A80096A8")))


class SnesDecoderTests(unittest.TestCase):
    def test_bgr555_to_rgb_converts_known_values(self) -> None:
        self.assertEqual(bgr555_to_rgb(0x0000), (0, 0, 0))
        self.assertEqual(bgr555_to_rgb(0x7FFF), (248, 248, 248))
        self.assertEqual(bgr555_to_rgb(0x001F), (248, 0, 0))  # max red
        self.assertEqual(bgr555_to_rgb(0x03E0), (0, 248, 0))  # max green
        self.assertEqual(bgr555_to_rgb(0x7C00), (0, 0, 248))  # max blue

    def test_decode_4bpp_tile_produces_8x8_indices(self) -> None:
        # All zeros = blank tile
        data = bytes(32)
        pixels = decode_4bpp_tile(data)
        self.assertEqual(pixels.shape, (8, 8))
        self.assertTrue(np.all(pixels == 0))

        # Bit pattern: bp0 = 0xFF (row 0), rest 0
        # Row 0 should have bit 0 set for all 8 pixels = index 1
        data = bytearray(32)
        data[0] = 0xFF  # bp0, row 0
        pixels = decode_4bpp_tile(bytes(data))
        np.testing.assert_array_equal(pixels[0], [1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(pixels[1], [0, 0, 0, 0, 0, 0, 0, 0])

    def test_build_palette_rgb_shape_and_values(self) -> None:
        # 512 bytes of zeros = all black
        cgram = bytes(512)
        palette = build_palette_rgb(cgram)
        self.assertEqual(palette.shape, (16, 16, 3))
        np.testing.assert_array_equal(palette[0, 0], [0, 0, 0])

    def test_render_full_map_shape(self) -> None:
        atlas = np.zeros((256, 16, 16, 3), dtype=np.uint8)
        atlas[1, :, :, 0] = 200  # red tile for ID 1
        grid = np.zeros((64, 64), dtype=np.uint8)
        grid[0, 0] = 1
        result = render_full_map(atlas, grid)
        self.assertEqual(result.shape, (1024, 1024, 3))
        self.assertEqual(result[0, 0, 0], 200)  # top-left pixel is red
        self.assertEqual(result[16, 0, 0], 0)   # next tile is black


@unittest.skipUnless(
    any(p.exists() for p in [STATES_DIR / "Y1_After_Buy_Potato.state"]),
    "Harvest Moon save states not available locally",
)
class SaveStateTests(unittest.TestCase):
    def test_parse_save_state_returns_ram_and_vram(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_After_Buy_Potato"))
        self.assertEqual(len(state.ram), 131072)
        self.assertEqual(len(state.vram), 65536)

    def test_read_metatile_grid_shape_and_values(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_After_Buy_Potato"))
        grid = read_metatile_grid(state.ram)
        self.assertEqual(grid.shape, (64, 64))
        self.assertGreater(len(np.unique(grid)), 5)

    def test_read_tilemap_id(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_After_Buy_Potato"))
        tilemap_id = read_tilemap_id(state.ram)
        self.assertEqual(tilemap_id, 0x00)  # Farm

    def test_read_player_pos(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_After_Buy_Potato"))
        px, py = read_player_pos(state.ram)
        self.assertGreater(px, 0)
        self.assertGreater(py, 0)

    def test_list_save_states_includes_known_state(self) -> None:
        states = list_save_states()
        self.assertIn("Y1_After_Buy_Potato", states)


@unittest.skipUnless(ROM_PATH.exists(), "Harvest Moon ROM not available locally")
class HarvestMoonRomIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rom = HarvestMoonRom.load(ROM_PATH)

    def test_map_farm_spring_entry_has_expected_block_sizes(self) -> None:
        entry = self.rom.read_map_entry(0x00, entry_label="MapFarmSpring")
        self.assertEqual(entry.entry_label, "MapFarmSpring")
        self.assertEqual(entry.graphic_preset, 0x00)
        self.assertEqual(len(entry.tilemaps), 3)
        self.assertEqual(len(entry.charactermaps), 1)
        self.assertEqual([block.decompressed_size for block in entry.tilemaps], [0x2000, 0x2000, 0x2000])
        self.assertEqual(entry.charactermaps[0].decompressed_size, 0x0968)

    def test_map_farm_spring_entry_matches_hm_decomp_bytes(self) -> None:
        result = compare_map_entry_to_hm_decomp(self.rom, 0x00)
        self.assertEqual(result.label, "MapFarmSpring")
        self.assertEqual(result.mismatch_count, 0)
        self.assertEqual(result.length_delta, 0)

    def test_graphic_preset_zero_matches_expected_table_values(self) -> None:
        preset = self.rom.read_graphic_preset(0)
        self.assertEqual(preset.obsel, 0x60)
        self.assertEqual(preset.bgmode, 0x09)
        self.assertEqual(preset.bg1sc, 0x63)
        self.assertEqual(preset.bg2sc, 0x72)
        self.assertEqual(preset.tm, 0x15)
        self.assertEqual(preset.cgadsub, 0x73)
        self.assertEqual(preset.coldata, 0xE0)

    def test_map_farm_spring_scene_has_expected_palette_links(self) -> None:
        scene = self.rom.read_map_scene(0x00, entry_label="MapFarmSpring")
        self.assertEqual(scene.entry_label, "MapFarmSpring")
        self.assertEqual(scene.graphic_preset.preset_id, 0x00)
        self.assertEqual([palette.palette_index for palette in scene.background_palettes], [0x00, 0x01, 0x02, 0x06, 0x07, None])
        self.assertEqual(scene.background_palettes[0].source_address, 0xA89400)
        self.assertEqual([palette.palette_index for palette in scene.sprite_palettes], [0x6B, 0x6C])

    def test_vram_layout_tables_match_hm_decomp(self) -> None:
        for table in VRAM_LAYOUT_TABLES:
            result = compare_labeled_data_to_hm_decomp(
                self.rom,
                label=table.label,
                snes_address=table.snes_address,
            )
            self.assertEqual(result.mismatch_count, 0, f"{table.label} mismatch")
            self.assertEqual(result.length_delta, 0, f"{table.label} length_delta")

    def test_vram_layout_farm_spring(self) -> None:
        layout = self.rom.read_vram_layout(4)
        self.assertEqual(layout.param_0181, 4)
        self.assertEqual(layout.charmap_row_height, 0x0100)
        self.assertEqual(layout.charmap_vram_size, 0x4000)
        self.assertEqual(layout.tilemap_dma_size, 0x2000)

    def test_sprite_palette_overrides_farm_spring_morning(self) -> None:
        overrides = self.rom.read_sprite_palette_overrides(
            tilemap_id=0x00, season=0, hour=10
        )
        self.assertEqual(len(overrides), 1)
        override = overrides[0]
        self.assertEqual(override.table_name, "UNK_Table9")
        self.assertEqual(override.table_index, 0)
        self.assertEqual(len(override.colors), 3)
        self.assertEqual(override.target_slots, (0x0A, 0x0B, 0x0C))
        # Verify first color matches ROM byte at UNK_Table9+0: $F8,$3B → 0x3BF8
        self.assertEqual(override.colors[0], 0x3BF8)

    def test_map_scene_includes_vram_layout(self) -> None:
        scene = self.rom.read_map_scene(0x00, entry_label="MapFarmSpring")
        self.assertIsNotNone(scene.vram_layout)
        self.assertEqual(scene.vram_layout.param_0181, scene.map_entry.param_0181)
        self.assertEqual(scene.vram_layout.charmap_vram_size, 0x4000)

    def test_graphic_preset_and_palette_tables_match_hm_decomp(self) -> None:
        preset_result = compare_labeled_data_to_hm_decomp(
            self.rom,
            label="Table_OBSEL_Presets",
            snes_address=0x808B44,
        )
        palette_result = compare_labeled_data_to_hm_decomp(
            self.rom,
            label="Time_Palette_Table",
            snes_address=0x80BB5C,
        )
        self.assertEqual(preset_result.mismatch_count, 0)
        self.assertEqual(palette_result.mismatch_count, 0)
        self.assertEqual(preset_result.length_delta, 0)
        self.assertEqual(palette_result.length_delta, 0)


@unittest.skipUnless(
    ROM_PATH.exists()
    and (STATES_DIR / "Y1_After_Buy_Potato.state").exists(),
    "Harvest Moon ROM and save states not available locally",
)
class MetatileAtlasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rom = HarvestMoonRom.load(ROM_PATH)
        cls.state = parse_save_state(resolve_state_path("Y1_After_Buy_Potato"))

    def test_build_metatile_atlas_shape_and_content(self) -> None:
        scene = self.rom.read_map_scene(read_tilemap_id(self.state.ram))
        atlas = build_metatile_atlas(self.state, scene.graphic_preset.bg12nba)
        self.assertEqual(atlas.shape, (256, 16, 16, 3))
        # At least some metatiles should have non-zero pixels
        non_zero = sum(1 for i in range(256) if atlas[i].any())
        self.assertGreater(non_zero, 10)

    def test_render_full_map_produces_1024x1024(self) -> None:
        scene = self.rom.read_map_scene(read_tilemap_id(self.state.ram))
        atlas = build_metatile_atlas(self.state, scene.graphic_preset.bg12nba)
        grid = read_metatile_grid(self.state.ram)
        full_map = render_full_map(atlas, grid)
        self.assertEqual(full_map.shape, (1024, 1024, 3))
        self.assertGreater(np.count_nonzero(full_map), 0)


if __name__ == "__main__":
    unittest.main()

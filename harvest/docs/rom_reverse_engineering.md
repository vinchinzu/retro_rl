# Harvest Moon ROM Reverse Engineering

This track is ROM-first. `HM-Decomp/` is a comparison target, not the source of
truth.

## Verified Local ROM

- Path: `roms/Harvest Moon.sfc`
- SHA1: `a64a5634429a4f5341868a40c220d7be89fda70a`
- Matches `custom_integrations/HarvestMoon-Snes/rom.sha`

## Tooling

Use `harvest/runtime/rom_tools.py` for direct ROM inspection.

```bash
uv run python -m harvest.runtime.rom_tools info
uv run python -m harvest.runtime.rom_tools dump-map-entry --tilemap-id 0x00
uv run python -m harvest.runtime.rom_tools dump-graphic-preset --preset-id 0x00
uv run python -m harvest.runtime.rom_tools dump-map-scene --tilemap-id 0x00
uv run python -m harvest.runtime.rom_tools export-map-scenes --output debug_alignment/rom_exports/map_scenes.json
uv run python -m harvest.runtime.rom_tools compare-map-entry --tilemap-id 0x00
uv run python -m harvest.runtime.rom_tools compare-all-map-entries
uv run python -m harvest.runtime.rom_tools compare-graphic-preset-tables
uv run python -m harvest.runtime.rom_tools compare-palette-tables
uv run python -m harvest.runtime.rom_tools compare-labeled-data --label Time_Palette_Table --address 0x80BB5C
uv run python -m harvest.runtime.rom_tools compare-data-bank --bank A8
uv run python -m harvest.runtime.rom_tools block-info --address 0x92D3AB
```

## Current Verified Facts

- `Maps_Graphics_Table` is present in the ROM at SNES address `0x80AA7C`.
- `MapFarmSpring` decodes directly from ROM and matches `HM-Decomp/src/maps/Maps_Graphics.asm` byte-for-byte.
- All `Maps_Graphics.asm` entries currently compare cleanly against the ROM.
- Data bank `A8` currently compares cleanly against `HM-Decomp/src/data_banks/bank_A8.asm`.
- The graphic preset tables used by `ManageGraphicPresets` can now be compared label-by-label against ROM bytes.
- The palette path for each map can now be decoded from ROM as:
  - 6 background palette slots from `Time_Palette_Table`
  - 2 sprite palette slots from `UNK_Table11`
  - direct palette pointers from `PalettePointerTable`
- The farm spring block at `0xA18000` decompresses to `0x0968` bytes, which is a hard ROM fact and a useful warning that labels/comments around “tilemap” vs “charmap” should not be trusted blindly.

## Current Working Context

- `HM-Decomp` is still only a comparison target. Each new claim should be backed by a ROM address, decoded bytes, and a repeatable command.
- Rendering is still blocked by incomplete VRAM role reconstruction, not by export plumbing.
- The immediate goal is an editor-facing ROM model: map entry, graphics preset, palette slots, sprite palette selection, then the remaining dynamic tables.

## Current Plan

1. Keep expanding labeled-data comparisons around the map load path in `bank_80.asm`.
2. Decode the remaining palette customization tables and state-driven branches used by `CustomiseSpritePalette`.
3. Reconstruct the exact VRAM write semantics of `BackgroundsManager` for tilemap and charactermap blocks.
4. Build a ROM-native editable scene model on top of those verified structures, then wire rendering/editor output to that model.

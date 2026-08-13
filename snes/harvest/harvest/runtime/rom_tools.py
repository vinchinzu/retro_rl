#!/usr/bin/env python3
"""ROM-first inspection utilities for Harvest Moon SNES.

This module treats the ROM as the source of truth and uses HM-Decomp only as a
comparison target. It is intentionally biased toward byte-accurate inspection so
we can build editing tools on top of verified structures instead of emulator
captures or hand-copied notes.

Public API is re-exported from focused submodules for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from harvest.paths import DECOMP_DIR, GAME_DIR, PROJECT_DIR, ROMS_DIR, SHARED_ROMS_DIR
from harvest.runtime.map_render import (
    CGRAM_MIRROR_ADDR,
    CGRAM_MIRROR_SIZE,
    METATILE_GRID_ADDR,
    METATILE_GRID_SIZE,
    METATILE_LOOKUP_ADDR,
    METATILE_LOOKUP_SIZE,
    PLAYER_X_ADDR,
    PLAYER_Y_ADDR,
    TILEMAP_ID_ADDR,
    bgr555_to_rgb,
    build_metatile_atlas,
    build_palette_rgb,
    decode_4bpp_tile,
    read_metatile_grid,
    read_player_pos,
    read_tilemap_id,
    render_full_map,
)
from harvest.runtime.rom_model import (
    BACKGROUND_PALETTE_SLOTS_PER_MAP,
    BANK_80_ASM_PATH,
    DEFAULT_ROM_CANDIDATES,
    GRAPHIC_PRESET_COUNT,
    GRAPHIC_PRESET_TABLES,
    MAP_PALETTE_TABLES,
    MAPS_GRAPHICS_ASM_PATH,
    MAPS_GRAPHICS_TABLE_ADDR,
    ROM_SHA_PATH,
    SCRIPT_DIR,
    SPRITE_PALETTE_SLOTS_PER_MAP,
    UNK_TABLE2_ADDR,
    UNK_TABLE3_ADDR,
    UNK_TABLE9_ADDR,
    UNK_TABLE9_ENTRY_COUNT,
    UNK_TABLE9_ENTRY_SIZE,
    UNK_TABLE10_ADDR,
    UNK_TABLE10_ENTRY_COUNT,
    UNK_TABLE10_ENTRY_SIZE,
    VRAM_LAYOUT_ENTRY_COUNT,
    VRAM_LAYOUT_TABLES,
    ByteComparison,
    GraphicPreset,
    HarvestMoonRom,
    LabeledDataTable,
    MapGraphicsBlock,
    MapGraphicsEntry,
    MapSceneModel,
    PaletteReference,
    SpritePaletteOverride,
    VramLayout,
    _expected_rom_sha1,
)
from harvest.runtime.rom_parse import (
    parse_labeled_data_asm,
    parse_maps_graphics_asm,
    parse_numeric_asm_bytes,
)
from harvest.runtime.save_state_io import (
    STATES_DIR,
    WRAM_ABSOLUTE_BASE,
    WRAM_SIZE,
    MutableSaveState,
    SaveStateArchive,
    SaveStateData,
    list_save_states,
    parse_save_state,
    resolve_state_path,
    wram_offset,
)

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

# Super Mario Bros NES ROM Editor

Kotlin multiplatform level editor for Super Mario Bros (NES), plus Python tools for programmatic ROM patching and emulator-based validation.

## Project Structure

```
super_mario_editor/
├── shared/          # KMP shared module: ROM parsing, level decoding, rendering
│   └── data/        #   Level, LevelRegistry, ObjectCatalog, EnemyCatalog, EditState
│   └── rom/         #   NesRomParser, ChrDecoder, LevelDecoder, LevelRenderer, etc.
├── cli/             # CLI: list levels, decode, render to PNG, export JSON
├── desktopApp/      # Compose Desktop GUI: level list, zoomable canvas, object palette
├── extend_level.py  # Programmatic level extension (double 1-1)
├── render_extended.py  # Render extended level replay to MP4
├── self_eval.py     # Structural validation (byte-level + page progression)
├── replay_eval.py   # Functional validation (replay recording on both ROMs)
└── smb_extended_1_1.nes  # Extended ROM: 1-1 at ~1.8x original length
```

## Quick Start

### Build (Kotlin)

```bash
./gradlew :shared:build     # shared module
./gradlew :cli:build        # CLI
./gradlew :desktopApp:build # desktop app
```

### CLI Usage

```bash
# List all 32 levels
./gradlew :cli:runCli -Pargs="--rom path/to/smb.nes levels"

# Decode a level
./gradlew :cli:runCli -Pargs="--rom path/to/smb.nes level 1-1"

# Render to PNG
./gradlew :cli:runCli -Pargs="--rom path/to/smb.nes render 1-1 -o level.png"

# Render with overlays (enemies, objects, grid, coins, powerups, pipes, warps)
./gradlew :cli:runCli -Pargs="--rom path/to/smb.nes render 4-2 --overlay enemies,objects,grid,coins,powerups,pipes,warps -o level_overlay.png"

# Export to JSON
./gradlew :cli:runCli -Pargs="--rom path/to/smb.nes export -o out/"
```

### Bulk Export (All 32 Levels)

```bash
ROM=/path/to/smb.nes
MAP_DIR=/path/to/retro_rl/super_mario_bros/maps
JSON_DIR=/path/to/retro_rl/super_mario_bros/maps/levels_json

# Export all level JSON
./gradlew :cli:runCli -Pargs="--rom $ROM export -o $JSON_DIR"

# Render all level maps
for lvl in 1-1 1-2 1-3 1-4 2-1 2-2 2-3 2-4 3-1 3-2 3-3 3-4 \
           4-1 4-2 4-3 4-4 5-1 5-2 5-3 5-4 6-1 6-2 6-3 6-4 \
           7-1 7-2 7-3 7-4 8-1 8-2 8-3 8-4; do
  ./gradlew :cli:runCli -Pargs="--rom $ROM render $lvl -o $MAP_DIR/smb_${lvl//-/_}.png"
done
```

## Current Status

- 32/32 world levels export to JSON.
- 32/32 world levels render to PNG maps.
- Object and enemy exports resolve all decoded IDs (no `"Unknown"` labels in current exports).
- Level editor/CLI header view now reports `Area Style`, `Terrain Control`, and `Cloud Override`.
- Overlay layers supported: `enemies`, `objects`, `grid`, `coins`, `powerups`, `pipes`, `warps`.
- Open follow-up work is tracked in `TODO.md`.

### Desktop GUI

```bash
./gradlew :desktopApp:run
```

## Level Extension Tool

`extend_level.py` creates a modified ROM where World 1-1 is duplicated: the gameplay section (pages 1-10) is repeated, then the ending (flagpole + castle) is appended.

```bash
python3 extend_level.py
```

**Output:** `smb_extended_1_1.nes` (94 objects, 23 pages vs original 49 objects, 13 pages = 1.8x longer)

### How It Works

1. Reads 1-1 object data at PRG $A68E (49 objects, 101 bytes)
2. Splits into gameplay (pages 1-10, 45 objects) and ending (pages 11-12, 4 objects)
3. Duplicates gameplay with page shift so second copy starts at page 11
4. Re-encodes page advance flags (SMB uses relative +1 page flags, not absolute)
5. Writes extended data in-place at $A68E, patches pointer table at Lo=$9D34 Hi=$9D55
6. Saves as `smb_extended_1_1.nes`

**Caveat:** Overwrites 90 bytes of adjacent level data (affects other levels that share ROM space).

## Video Rendering

```bash
uv run python render_extended.py
```

Renders `extended_1_1.mp4` (720x672, 60fps, ~105s):
- Phase 1 (REPLAY): Replays an optimized 1-1 hill-climb recording through the first copy
- Phase 2 (BOT): Simple run-right bot continues into the second copy of the level

HUD shows frame count, time, button inputs, page number, and pixel position.

## Validation

### Structural (`self_eval.py`)

```bash
python3 self_eval.py
```

Compares original vs extended ROM byte-by-byte:
- Header preserved ($50 $21)
- First half: 45 objects match original exactly
- Second half: 45 objects correctly shifted by 10 pages
- Ending: flagpole/castle preserved at shifted position
- Page progression: 1-10 (copy 1), 11-20 (copy 2), 21-22 (ending)

### Functional (`replay_eval.py`)

```bash
uv run python replay_eval.py
```

Replays optimized 1-1 recording on both original and extended ROM via stable-retro:
- Original: reaches page 12, max_x=3266 (flagpole)
- Extended: reaches page 16, max_x=4119 (deep into second copy)
- Confirms extended content is present and playable

## SMB ROM Technical Details

| Property | Value |
|----------|-------|
| Format | iNES (16-byte header + PRG + CHR) |
| PRG ROM | 32 KB at $8000-$FFFF (NROM mapper, flat) |
| CHR ROM | 8 KB = 512 tiles in 2bpp (16 bytes/tile) |
| Level objects | 2 bytes each: `(col<<4\|row)`, `(page_flag\|type)`, terminated by $FD |
| Enemy data | Mostly 2 bytes `(col<<4\|row)`,`(page_flag\|type)`; special row `$0e` entries consume 3 bytes; terminated by $FF |
| Page flag | Bit 7 of byte 1 = advance to next page (+1, relative) |
| Special page commands | Object row `$0d` with d6 clear sets absolute object page; enemy row `$0f` sets absolute enemy page |
| Header decode | Byte0: `time/entrance/fg-or-bgcolor`; Byte1: `area-style/bg-scenery/terrain-control` with cloud override when style=`3` |
| 1-1 data | PRG $A68E, 49 objects, 101 bytes, max page 12 |
| Pointer | Lo=$9D34 Hi=$9D55 → $A68E |

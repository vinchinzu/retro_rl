# External Tools And References

Fetched on 2026-04-29.

## Local Clones

| Path | Repository | Commit | Date | Role |
| --- | --- | --- | --- | --- |
| `refs/smw-port` | `https://github.com/snesrev/smw.git` | `eae20c65c58930c8b62c76188d259579ad4130f1` | 2023-08-17 | Reverse-engineered C reimplementation / native port target |
| `refs/SMWDisX` | `https://github.com/IsoFrieze/SMWDisX.git` | `3390ee1a094bce35defb51f456423d62017d28f9` | 2023-08-07 | Readable, buildable disassembly and RAM symbol source |
| `refs/smw-editor` | `https://github.com/smw-editor/smw-editor.git` | `2a732a65480b3c14b517b19871c5a4891506d260` | 2024-09-28 | Open-source editor architecture reference |
| `tools/external/pixi` | `https://github.com/JackTheSpades/SpriteToolSuperDelux.git` | `35901292e4d961e67368bdf20d74c8e1c65187e2` | 2026-04-21 | Custom sprite insertion and sprite editor/library reference |
| `tools/external/asar` | `https://github.com/RPGHacker/asar.git` | `5fd539cd510e4a26afef66fc14e35ca6d8ff0497` | 2025-09-21 | SNES assembler used by SMW tools/disassemblies |
| `tools/external/AddMusicKFF` | `https://github.com/KungFuFurby/AddMusicKFF.git` | `70e3caba90f6b2cb95e0cf3877e603f272c90e0c` | 2026-04-27 | Music insertion/editing toolchain reference |

These clone directories are gitignored. Refresh with:

```bash
git -C SMW/refs/smw-port pull --ff-only
git -C SMW/refs/SMWDisX pull --ff-only
git -C SMW/refs/smw-editor pull --ff-only
git -C SMW/tools/external/pixi pull --ff-only
git -C SMW/tools/external/asar pull --ff-only
git -C SMW/tools/external/AddMusicKFF pull --ff-only
```

Then update this table.

## Non-Cloned Required Tools

### Lunar Magic

Closed-source freeware. Treat it as the practical compatibility target for
levels, overworld, Map16, graphics, entrances, messages, and Lunar Magic's ROM
expansions. Do not vendor it. Document local install paths only in ignored
personal notes.

Official page: https://fusoya.eludevisibility.org/lm/

### GPS

Gopher Popcorn Stew is the standard custom block insertion tool. It is
distributed through SMW Central rather than maintained as a normal source clone
in this workspace. Use it through a local ignored install and keep block source
under `SMW/mods/` once that tree exists.

### UberASM Tool

Use for level, overworld, game-mode, status bar, and global ASM snippets. It
belongs in the patch/mod pipeline, not the autoplay baseline.

## Why These Sources

- Data Crystal identifies SMWDisX and `snesrev/smw` as the relevant public
  disassembly/reimplementation projects and lists Lunar Magic, GPS, PIXI, and
  Asar as core utilities.
- SnesLab classifies Lunar Magic, PIXI, GPS, AddMusicK, and UberASM Tool as
  standard SMW hacking tools.
- `snesrev/smw` is the closest match to the requested "actual port to C" track.
- SMWDisX is better for exact address/symbol lookup and ROM rebuilding.
- SMW Editor is not yet usable as a replacement for Lunar Magic, but its Rust
  code and symbols are useful for designing our editor model.

## Build Notes

### C Port

```bash
cd SMW/refs/smw-port
cp ../../roms/smw.sfc ./smw.sfc
make
```

The port requires locally extracting assets from a user-owned ROM. Do not commit
`smw.sfc`, `smw_assets.dat`, snapshots, or saves.

### SMWDisX

```bash
cd SMW/refs/SMWDisX
../../tools/external/asar/asar smw.asm smw.smc
```

This may need exact Asar version compatibility. Use the cloned Asar first; fall
back to the version requested by SMWDisX if assembly fails.

### PIXI

```bash
cd SMW/tools/external/pixi
cmake -S . -B build
cmake --build build --config Release
```

PIXI is both a command-line sprite inserter and a useful reference for sprite
JSON/CFG formats.

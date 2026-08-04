# SMW Architecture

## Goal

SMW should become the canonical layout for future platformer work in this repo:
emulator autoplay first, then speedrun chaining, then ROM/RAM decomposition,
then editor/modding and the native C-port track.

The design is deliberately split into layers so each layer can be validated
against the one below it.

## Source Policy

Use these sources, in this order:

1. Local emulator evidence: stable-retro RAM reads, states, recordings, and
   replay traces.
2. Public reverse-engineered references: SMWDisX, SMW Central RAM/ROM maps,
   Data Crystal, SnesLab, and the SMW Editor symbols/docs.
3. Public tool source: `snesrev/smw`, PIXI, Asar, AddMusicKFF, and SMW Editor.
4. User-owned ROM-derived assets generated locally.

Do not import leaked Nintendo source. Do not commit ROMs or ROM-derived binary
assets unless they are explicitly legal/generated metadata.

## Runtime Layers

### 1. Emulator truth

Purpose: make the game playable, recordable, and measurable.

Owned files:

- `custom_integrations/SuperMarioWorld-Snes-v0/data.json`
- `retro_harness/platformer/levels/super_mario_world.py`
- `recordings/`
- `optimizer/`

Responsibilities:

- Publish RAM fields that matter for control, progress, death, level exit, and
  trace fingerprinting.
- Keep stable-retro states local to `SMW/custom_integrations/...`.
- Use the shared `retro_harness.platformer` commands for play, verify, hillclimb,
  watch, and route evaluation.

Current first-pass command:

```bash
uv run python -m SMW list-levels
```

### 2. Autoplay and speedrun

Purpose: convert recordings and optimizer output into reliable segments, then
chain them into route manifests.

Initial target:

1. `smw_yoshi_island_1` segment smoke test.
2. Human recording with raw button trace.
3. Headless verify with RAM trace.
4. Hillclimb from the recording.
5. Repeat for stable-retro states already available.
6. Create missing any% states manually and promote them to published anchors.

Completion should be based on RAM invariants, not pixels. The first pass uses
`GameMode` changing from active level mode (`0x14`) toward overworld modes
after enough camera/player progress. If this proves too loose, add a shared
completion hook for `OWLevelExitMode`, `PlayerAnimation`, and boss/keyhole
signals.

### 3. RAM decomposition

Purpose: turn RAM from "addresses we read" into a complete live state model.

Expected local package shape:

```text
SMW/smw/
  core/
    ram_catalog.py       canonical addresses and typed fields
    world_snapshot.py    per-frame state object
    sprite_catalog.py    live sprite slots and names
    level_state.py       current level, exits, timers, powerups
  runtime/
    retro_setup.py       ROM/state registration and SHA checks
    trace.py             recording and replay trace format
    probes.py            RAM diff and transition probes
```

Do not build editor code directly on ad hoc emulator reads. Build it on
`WorldSnapshot` once the snapshot is backed by tests.

### 4. ROM and editor model

Purpose: expose SMW as editable data without losing compatibility with existing
ROM-hacking workflows.

Core editor surfaces:

- Level/scene editor: objects, sprites, entrances, exits, secondary exits,
  camera/level mode, midway, message index, music, palettes.
- Tile editor: GFX/ExGFX, 8x8 tiles, 16x16 Map16, acts-like behavior,
  animations, palette rows.
- Sprite editor: vanilla sprite slots, PIXI JSON/CFG/ASM definitions,
  extra bytes, display metadata.
- Overworld editor: submaps, paths/events, level nodes, exits, save prompts,
  star/pipe links.
- Patch/mod pipeline: Asar patches, GPS blocks, PIXI sprites, AddMusicKFF
  songs, UberASM snippets, Lunar Magic compatibility.

Editor data should be generated from a verified ROM model:

```text
ROM bytes -> decoded structures -> editable JSON/domain models -> patch/build
```

Pixel/editor displays are allowed to use emulator screenshots, but the saved
editor data must round-trip through decoded ROM structures.

### 5. C-port and native modding

Purpose: use `snesrev/smw` as the native, hackable port target once emulator
behavior is understood.

Track separately from the emulator route:

- Build `SMW/refs/smw-port` after placing a local `smw.sfc` in that clone.
- Extract `smw_assets.dat` locally; do not commit it.
- Use the port's built-in emulator comparison snapshots to detect behavior
  mismatches.
- Add wrapper scripts only after the baseline port builds reproducibly.

The C port is not the first source of truth for speedrun timing. It is the
future modding target after RAM and route behavior are validated in emulator.

## Canonical Directory Contract

```text
SMW/
  docs/                 active runbooks and architecture
  custom_integrations/  stable-retro data, states, ROM symlink
  smw/                  future local Python package
  scripts/              operational scripts only
  refs/                 ignored source references
  tools/external/       ignored third-party tools
  maps/                 decoded/rendered maps
  recordings/           human/TAS/action recordings
  optimizer/            route manifests and optimizer output
  models/               model checkpoints
  logs/                 run logs
  roms/                 local ROMs
```

## Shared Code Boundary

Use `retro_harness.platformer` when behavior is useful to other platformers:

- action tables
- evaluator hooks
- route manifests
- replay/hillclimb/chain commands
- trace rendering contracts

Keep SMW-specific behavior in `SMW/` when it depends on SMW RAM, ROM data, or
SMW editor/modding semantics.

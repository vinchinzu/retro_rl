# Super Mario World RL / Editor Workspace

SMW is the Super Mario World workspace for emulator autoplay, speedrun
optimization, RAM/ROM decomposition, ROM-hacking tools, and the eventual
native C-port/modding track.

## Quick Start

Commands below assume the monorepo root as the working directory. If you are
already inside `SMW/`, use `./play_speedrun.sh` for the launcher.

```bash
# Fresh-game speedrun practice/recording from ROM boot/title screen
./SMW/play_speedrun.sh
# equivalent module entrypoint:
uv run python -m SMW speedrun

# List the initial stable-retro level configs
uv run python -m SMW list-levels

# Human play / recording entrypoint once the ROM is installed
uv run python -m SMW -l smw_yoshi_island_1 play

# Verify a recorded action sequence
uv run python -m SMW -l smw_yoshi_island_1 verify --actions recording.json
```

Local ROM setup:

```bash
mkdir -p SMW/roms
# Place a legally owned Super Mario World USA ROM at:
# SMW/roms/smw.sfc
ln -sf ../../roms/smw.sfc SMW/custom_integrations/SuperMarioWorld-Snes-v0/rom.sfc
```

The stable-retro integration expects SHA1
`6b47bb75d16514b6a476aa0c73a683a2a4c18765` for the standard USA ROM.

## Fresh-Game Speedrun Recording

Use the launcher when you want to play from a clean boot rather than from a
published level state:

```bash
# from repo root
./SMW/play_speedrun.sh

# from SMW/
./play_speedrun.sh
```

By default this starts stable-retro with `state=NONE` and uses a temporary
custom integration without `rom.srm`, so the title screen sees blank SRAM even
when local fixture SRAM exists. Pass `--keep-sram` if you intentionally want the
current `rom.srm`.

Sessions are written under `SMW/recordings/speedrun/<timestamp>/`:

- `frames.jsonl`: sampled per-frame raw input and RAM trace.
- `events.jsonl`: reset, save/load, RAM transition, and marker events.
- `branches/branch_*.json`: replayable raw button branches after resets or
  hot-state loads.
- `states/`: session-local copies of hot states/checkpoints.

Useful controls:

- `F1`-`F4`: save memory checkpoints; `Shift+F1`-`Shift+F4`: load them.
- `F5`: save `QuickSave`; `F7`/`F8`: load `QuickSave`.
- `F6`: mark/save a hard spot; `F12`: save a named speedrun state.
- `F9`/`F10`: cycle local states; `F11`: load selected state.
- `R`: reset to the fresh start; `TAB`: turbo; `[`/`]`: speed; `ESC`: stop and save.

## Test Fixtures

```bash
# Build a checksum-valid 96-exit SRAM fixture.
uv run python -m SMW.scripts.create_all_exits_sram

# Build an overworld state with that save loaded into WRAM.
uv run python -m SMW.scripts.create_all_exits_state
```

The fixture state is
`SMW/custom_integrations/SuperMarioWorld-Snes-v0/AllExitsComplete.state`.
It starts on the overworld with `ExitsCompleted == 0x60`, all event flags set,
and all four switch flags set.

## Layout

```text
SMW/
  custom_integrations/SuperMarioWorld-Snes-v0/  stable-retro metadata + states
  docs/                                        architecture and runbooks
  refs/                                        ignored source refs and ports
  tools/external/                              ignored external tool clones
  optimizer/                                   run outputs and manifests
  recordings/                                  human/TAS/emulator recordings
  models/                                      trained checkpoints
  maps/                                        extracted/rendered map data
  roms/                                        local ROMs, ignored
```

## Current Priority

1. Establish emulator autoplay on stable-retro states.
2. Build a RAM-backed segment verifier and route manifest.
3. Create the first chained speedrun route.
4. Grow editor data models from verified ROM/RAM structures.
5. Build against the `snesrev/smw` C port after local assets are extracted.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/autoplay_speedrun_plan.md](docs/autoplay_speedrun_plan.md).

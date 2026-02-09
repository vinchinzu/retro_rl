# Donkey Kong Country RL

Local setup for Donkey Kong Country (SNES) with stable-retro integration files.

## Quick start

1. Place your legal ROM at:
   - `roms/DonkeyKongCountry.sfc`
2. Create the ROM symlink expected by stable-retro:
   ```bash
   ln -s "$(pwd)/roms/DonkeyKongCountry.sfc" "$(pwd)/custom_integrations/DonkeyKongCountry-Snes/rom.sfc"
   ```
3. Verify SHA1 (expected in `custom_integrations/DonkeyKongCountry-Snes/rom.sha`):
   ```bash
   sha1sum roms/DonkeyKongCountry.sfc
   ```
4. Start a Retro session (example):
   ```bash
   python - <<'PY'
import retro
retro.data.Integrations.add_custom_path('custom_integrations')
env = retro.make(game='DonkeyKongCountry-Snes', state='1Player.CongoJungle.JungleHijinks.Level1')
obs = env.reset()
print('obs', obs.shape)
PY
   ```

## States included

- `1Player.CongoJungle.JungleHijinks.Level1.state`
- `1Player.CongoJungle.RopeyRampage.Level2.state`
- `QuickSave.state` (created via `F5`, used as default if present; gzip-compressed)

These are sourced from the stable-retro integration set.

## Notes

- If the window does not appear on Wayland, `run_bot.sh` defaults to X11 (`SDL_VIDEODRIVER=x11`).
- Controls:
  - `Tab`: fast-forward (2x cap over current speed)
  - `[` / `]`: decrease / increase speed
  - `M`: save `ManualSave.state`
  - `F5`: save `QuickSave.state`
- Supports two controllers; if the env exposes 2 players (24 inputs), inputs map to P1/P2.
- Start button override:
  - Set `RETRO_START_OVERRIDE="0"` to force Start on button 0 (or `RETRO_START_OVERRIDE="0,7"` per controller).
  - Press `F6` / `F7` to assign Start to the last-pressed button for controller 1/2.
- Swap X/Y mapping per controller:
  - Default: controller 2 swaps X/Y (Xbox pad quirk).
  - `RETRO_SWAP_XY="1"` (all controllers) or `RETRO_SWAP_XY="0,1"` (only controller 2).
- Save states are written to the current directory and to `custom_integrations/<Game>/`.
- Shared helper: `retro_harness.save_state(...)` is the generic save path for all games.

## Autosplit + Best Times

Enable split tracking and best-time storage:

```bash
./run_bot.sh play --autosplit
```

Notes:
- Uses RAM `0x003E` (7E003E) as the current level ID.
- Splits trigger when the level ID changes.
- Best times are saved to `best_times.json` in this folder.
- Timer starts when the in-game timer begins moving (RAM `0x0046` frames + `0x0048` minutes).
- HUD shows current level time + best time. The in-game timer (frames+minutes) is shown for debugging when present.
- Split records are appended to `split_runs.jsonl` (per-session log). The log now includes
  `level_seen`, `level_start`, `level_end`, and `episode_end` entries with `run_id`,
  `level_name`, timing, and state metadata so every run is logged.
  
To rebuild `best_times.json` from the log:

```bash
./run_bot.sh refresh-best
```

To print a table of split times (optionally filtered):

```bash
# All levels
./run_bot.sh table

# Winky's Walkway (by level id 0xD9)
./run_bot.sh table --level-id 0xD9

# Winky's Walkway (by name)
./run_bot.sh table --level-name "Winky's Walkway"
```

Filters:
- `--min-seconds 5` skips tiny partial runs (default 5s).
- `--no-dedupe` shows multiple entries per run (default is one per run).
- Level id debugging:
  - Override the RAM level id offset with `--level-id-offset` or `RETRO_LEVEL_ID_OFFSET=0xNN`.
  - Live runs print `[LEVEL_ID] candidates: ...` when any candidate offset changes.

To rebuild split logs from all recordings (and regenerate missing replays):

```bash
./run_bot.sh replay-splits-all
```

## Replay + Split Recovery (Standardized)

Use this flow to reproduce timings from recordings:

```bash
# 1) Dump a replay JSONL from a .bk2 recording
./run_bot.sh replay --bk2 recordings/<session>/DonkeyKongCountry-Snes-QuickSave-000000.bk2 \
  --out recordings/<session>/DonkeyKongCountry-Snes-QuickSave-000000.replay.jsonl --no-ram

# 2) Extract split timings from the replay and append to the log
./run_bot.sh replay-splits --replay recordings/<session>/DonkeyKongCountry-Snes-QuickSave-000000.replay.jsonl
```

Notes:
- `replay-splits` appends `level_start`/`level_end` events to `split_runs.jsonl` so
  timing can be reconstructed even if the live autosplit missed it.

## Recording

Record `.bk2` gameplay:

```bash
./run_bot.sh play --record
```

Notes:
- Recordings are saved under `donkey_kong_country/recordings/`.

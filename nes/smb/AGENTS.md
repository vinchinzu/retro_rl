# Agent Instructions — smb

NES Super Mario Bros. (**M8** Clean power-on → 8-4 ending). Shared:
`retro_harness.platformer` (RLE / neuro).
[`docs/STATUS.md`](docs/STATUS.md) · [`docs/plan.md`](docs/plan.md).

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run python smb/scripts/boot_probe.py
./play smb                         # power-on → all_exits_v1
./play smb --list                  # F5=save  F6=pin  ESC=cancel
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3
uv run python -m pytest smb/tests -q
```

TAS / polish / oracle / human-tape: [`docs/plan.md`](docs/plan.md).
Tip + frame budget: [`docs/STATUS.md`](docs/STATUS.md).

## Traps

- Power-on: **exactly** 350 boot frames + **16** idle, then seed.
- Level1_1 continuous: **exactly 14** idle after `Level1_1`. Natural 1-1 alone:
  idle **1** (`NATURAL_SETTLE_FRAMES`).
- World 4 = `world` index **3**. Underground `level_id=2` ≠ completion
  (`$0760` AreaNumber; 32-exit clock uses `$075C` LevelNumber so 1-2 UG is not 1-3).
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames. Recordings hold
  **780f** post-ending through Peach (`--peach-hold-frames`; do not cut on Bowser-drop).
- **Do not** absolute-frame stitch a faster 1-1 into old 1-2 — use
  `smb.reactive_12` control gates. **Do not** W4 idle-pad; retime later legs
  from natural control (`--retime-4-1` / `--retime-4-2` / `--retime-8-2`).
  Trim time; never pad macros to an old phase.
- 1-1-stairs polish window = frames **1050–1311** (wall-slam), not castle idle.
- 1-2 polish mutates only `underground_from_control` in
  `smb_1_2_reactive_fragments.json` (surface stays reactive RIGHT/DOWN).

## Layout

`./play smb` · `ram.py` / `obs.py` / `policy.py` · `reactive_12|late|route.py` ·
`scripts/run_warp_finish.py` · `tas/` · `retro_harness.platformer`.

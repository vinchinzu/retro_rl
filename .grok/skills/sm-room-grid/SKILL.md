---
name: sm-room-grid
description: >
  Super Metroid 4x4 autobot room-grid demo (~30s): 16 independent hops,
  ffmpeg mosaic, sequential or subprocess-parallel. Use when the user
  says "4x4 grid", "room mosaic", "16 rooms at once", "demo reel",
  "16 emulators", or runs /sm-room-grid.
---

# SM room grid

Showcase clip. Not a spine dual. Tiles are `KPDR_SEGMENTS` from catalog
pins in `super_metroid.demo.room_grid.DEFAULT_TILES`.

## This turn

1. **Do not dual. Do not STATUS.** Output is
   `recordings/room_grid.mp4` (gitignored). A tile RED is still a clip.
2. **Inventory first.**
   ```bash
   uv run python snes/super_metroid/scripts/export/room_grid_demo.py --list
   uv run python snes/super_metroid/scripts/export/room_grid_demo.py \
     --probe-parallel --workers 16
   ```
   `--list` must show 16 pins on disk. `--probe-parallel` prints the
   load/RAM verdict. Stop if pins are missing.
3. **Record only when the user wants the mp4 and the verdict is ok.**
   Default is `--workers 1` (one env). Parallel is
   `--workers 16` (one **process** per tile, spawn, not
   `EmulatorPool` lockstep). Refuse `workers>1` unless
   `--probe-parallel` is ok **or** the user passed `--force` after
   saying CPUs are free.
4. **Composite.** The recorder writes 16 tile mp4s, loops shorts, trims
   longs, `xstack` 4×4, ~30s, silent. Glance the file duration, not hop
   leave.

```bash
# sequential (safe under load)
uv run python snes/super_metroid/scripts/export/room_grid_demo.py --seconds 30

# later, idle machine
uv run python snes/super_metroid/scripts/export/room_grid_demo.py \
  --workers 16 --seconds 30
```

## Parallel

16 cores of `make_dev_env` **is** possible: each tile is its own process
(same pattern as `SubprocVecEnv` on this stack). In-process
`EmulatorPool` is the wrong tool — hops are not lockstep, and libretro
is not a 16-lane thread pool.

`--probe-parallel` is the machine check (ncpus, load, MemAvailable).
Do not hard-code host sizes.

## Non-claims

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
treat a grid clip as hop GREEN or continuous evidence.

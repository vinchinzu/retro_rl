# Agent Instructions — sm (lsnes Lua through Gravity)

In-emulator Clean KPDR through Gravity Suit. Lua 5.1 loaded by lsnes
rr2-β25. Not the Python gym in `snes/super_metroid/`. Not `../sm_ceres`.
No Python files in this tree. Do not STATUS-promote.

## Launch

```bash
./sm/run.sh                 # Gravity Suit (default)
TIP=ceres ./sm/run.sh       # Ceres prefix → Landing
TURBO=1 ./sm/run.sh
```

Native ELF `~/.local/opt/lsnes-rr2-beta25/lsnes`, ROM
`roms/SuperMetroid.sfc` (SHA-256 `12b77c4b…`), `--rom-a=` + `--lua=`.
No startup movie.

## Leave

RAM: room, game state, xy, pose, health, items, timer_type.
Ceres prefix: `0x91F8` gs=8 → `sm/recordings/ceres_leave.json`.
Gravity: `0xCE40` gs=8, `collected_items & 0x0020` →
`sm/recordings/gravity_leave.json`. No MP4. No STATUS.

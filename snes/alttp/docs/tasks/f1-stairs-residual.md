# Residual — F1 stairs after 0x01 (`rr-ccxt.1`)

Planner owns `docs/STATUS.md`. This sitting is **state-load / isolated** only.
Did not STATUS-promote. Did not treat a save-state pin as power-on. Did not
write `$F3CC` / sword / keys. Did not copy approach xy into Python. Did not
overwrite `recordings/verified_tip_run.json`. Did not edit `maps/*.json` (no
new door: no `room_base_id` in `0x70–0x82`).

**Result: no F1→B1 stairs in rooms `0x01` / `0x51` / `0x52` / `0x62`.**

JSON: `recordings/probe_b1_from_f1/`. Probe: `scratch/probe_b1_from_f1.py`.

## Commands

```bash
uv run python snes/alttp/scripts/room_engine.py show room_01
uv run python snes/alttp/scripts/room_engine.py show room_51
uv run python snes/alttp/scripts/room_engine.py show room_52
uv run python snes/alttp/scripts/room_engine.py show room_62
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --followup
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --north62
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_51 \
  --edge south_to_0x61 --state CastleRoom51 --no-clear \
  --json-out snes/alttp/recordings/probe_b1_from_f1/room_51_south_to_0x61.json
```

Every pin below: module `$10=0x07`, submodule `$11=0x00`, sword `$F359=1`,
follower `$F3CC=0`, keys `$F36F=0`, indoors, `has_control`.

## Rooms scanned

| State | Room | Spawn xy | After clear xy | B1? |
|-------|------|----------|----------------|-----|
| `CastleRoom01` | `0x01` | 960, 120 | 960, 120 (no hostiles) | no |
| `CastleRoom51` | `0x51` | 760, 2985 | 711, 2971 (3 soldiers) | no |
| `CastleRoom51` no-clear | `0x51` | 760, 2985 | (skipped) | no |
| `CastleRoom52` | `0x52` | 1144, 2993 | 1128, 2701 (2 knights) | no |
| `CastleRoom62` | `0x62` | 1072, 3320 | 1072, 3320 (followup; no extra chase) | no |
| `CastleRoom62North` | `0x62` | 1272, 3104 | 1272, 3104 | no |

## Exits found (not B1)

### 0x01 — north connector

Independent cardinals from spawn + map extents. Only two doors:

| Dir | Approach | Dest | Landing |
|-----|----------|------|---------|
| LEFT | ~(522, 120) | `0x50` | ~(511, 2680) |
| RIGHT | ~(1002, 120) | `0x52` | ~(1011, 2680) |

UP stuck y=96; DOWN stuck y=144. Tiny hallway. Matches `maps/room_01.json`.

### 0x51 — throne / mantle

Clear pulls off the south seam (DOWN from x≠760 hits the wall at y=2984).
`--no-clear` from spawn: UP walks the center carpet to the mantle
`(760, 2620)` **without** a room change. `room_engine` south door:

| Dir | Approach | Dest | Landing |
|-----|----------|------|---------|
| DOWN | ~(760, 2997) | `0x61` | (760, 3000) then settle (760, 3118) |

West/east walls x=640 / x=880. Mantle LEFT/RIGHT stay ~712–808 at y=2624.
No B1, no sewer drop (mantle does not fire without follower).

### 0x52 — NE chamber

Clear walks north to (1128, 2701). Exits only:

| Dir | Approach | Dest | Landing |
|-----|----------|------|---------|
| DOWN | ~(1144, 3040) | `0x62` | (1144, 3060) |
| LEFT | ~(1034, 2680) | `0x01` | (1023, 120) |

North wall ~y=2664 (`north_extent` 2608 not quite reached). Corridor x=1120–1168.

### 0x62 — main east

West door and north seam only. Corridor y=3320 walks RIGHT to x=1328 (wall),
LEFT → `0x61`. DOWN from most corridor x stuck at y=3352–3360; at x≈1135 a
south alcove to ~(1168, 3392) that does **not** change room. UP at x≈1144
and from `CastleRoom62North` (1272, 3104) → `0x52`.

| Dir | Approach | Dest | Landing |
|-----|----------|------|---------|
| LEFT | ~(1036, 3320) | `0x61` | (1023, 3320) |
| UP | ~(1144, 3093) or (1272, 3092) | `0x52` | (1144, 3088) / (1272, 3089) |

NE east wall has a **stair graphic** at ~(1352, 3120) (`CastleRoom62North`
RIGHT ray end). Cardinal holds + `A`/`B` there stay in `0x62`
(RIGHT wall 1352, UP stuck, DOWN to y=3160, no text). Not a measured B1
door. Keys were 0; a key/shutter stair is untested.

## RAM leftover (spawn pins)

| State | room | module | sub | xy | `$F359` | `$F3CC` | `$F36F` |
|-------|------|--------|-----|----|---------|---------|---------|
| `CastleRoom01` | `0x01` | `0x07` | 0 | 960, 120 | 1 | 0 | 0 |
| `CastleRoom51` | `0x51` | `0x07` | 0 | 760, 2985 | 1 | 0 | 0 |
| `CastleRoom52` | `0x52` | `0x07` | 0 | 1144, 2993 | 1 | 0 | 0 |
| `CastleRoom62` | `0x62` | `0x07` | 0 | 1072, 3320 | 1 | 0 | 0 |
| `CastleRoom62North` | `0x62` | `0x07` | 0 | 1272, 3104 | 1 | 0 | 0 |

`room_engine` 0x51 south dest leftover: room `0x61`, module `0x07`, sub 0,
xy=(760, 3118), sword=1, `$F3CC=0`, keys=0.

## Next

F1→B1 stair is **not** in the 0x01 chain rooms this sitting scanned. Reverse
from B1 (`CastleB2Landing` / `maps/room_70.json`) to find the matching F1
landing. West-wing `0x60` was out of this room list (its south edge is already
the outdoor courtyard door, not a B1 stair).

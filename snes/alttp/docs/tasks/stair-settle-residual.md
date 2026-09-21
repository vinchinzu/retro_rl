# Residual — stair dest settle past 240f (`rr-ccxt.14`)

Planner owns `docs/STATUS.md`. Isolated only. Did not STATUS-promote. Did not
promote hops. Did not poke `$F3CC` / sword / keys. Did not copy approach xy
into Python. Did not overwrite `verified_tip_run.json`. Did not change
`primitives.settle_control` default 240.

**Fix:** `run_room_edge` dest settle after `at_door_destination` uses
`DEST_SETTLE_MAX_FRAMES=480`. Other `settle_control(env)` calls stay 240.
Already-at-dest doors still skip dest settle (first settle returns on
`has_control`).

## Commands

```bash
uv run pytest snes/alttp/tests/test_room_engine_edges.py -q
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_01 \
  --edge down_to_0x72 --state CastleRoom01 --no-clear \
  --json-out snes/alttp/recordings/probe_b1_from_f1/room_01_down_to_0x72.json
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_72 \
  --edge north_to_0x01 --state CastleB1Guard --no-clear \
  --json-out snes/alttp/recordings/probe_b1_reverse/room_72_north_to_0x01.json
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_70 \
  --edge north_to_0x71 --state CastleB2Landing --no-clear \
  --json-out snes/alttp/recordings/probe_b1_reverse/room_70_north_to_0x71.json
```

Opt-in live pins: `uv run pytest snes/alttp/tests/test_room_engine_edges.py -m rom`
(skip when ROM / pin missing). Default pytest still `-m 'not rom'`.

## room_engine (this sitting)

| Edge | State | ok | frames | dest settle | leftover |
|------|-------|----|--------|-------------|----------|
| `room_01` `down_to_0x72` | `CastleRoom01` | **True** | 430 | 268f | `0x72` (1273, 3665) sub 0 ctrl=1 |
| `room_72` `north_to_0x01` | `CastleB1Guard` | **True** | 291 | 264f | `0x01` (760, 100) sub 0 ctrl=1 |
| `room_70` `north_to_0x71` | `CastleB2Landing` | **True** | 291 | 264f | `0x71` (672, 3659) sub 0 ctrl=1 |

Last sitting red: dest room reached, `ok=False` `settle_destination` still
submodule 14 (240f cap). Dest settle here is **264–268f** — just over the old
cap. Independent hold ~292–400f counted door-push + anim.

## RAM leftover (every row: sword `$F359=1`, `$F3CC=0`, keys `$F36F=0`)

| When | room | module/sub | xy | ctrl |
|------|------|------------|----|------|
| `CastleRoom01` spawn | `0x01` | `0x07` / 0 | 960, 120 | 1 |
| `down_to_0x72` final | **`0x72`** | `0x07` / **0** | **1273, 3665** | **1** |
| `CastleB1Guard` spawn | `0x72` | `0x07` / 0 | 1273, 3665 | 1 |
| `north_to_0x01` final | **`0x01`** | `0x07` / **0** | **760, 100** | **1** |
| `CastleB2Landing` spawn | `0x70` | `0x07` / 0 | 161, 3640 | 1 |
| `north_to_0x71` final | **`0x71`** | `0x07` / **0** | **672, 3659** | **1** |

Acceptance `at_door_dest=true` on all three. State-load / isolated only.

## Tests

Offline: dest settle mock stays submodule 14 for 292f then `ok=True` with
`settle_destination` frames `> 240`; already-at-dest still 0f / no dest
phase; `settle_control` default still 240; `room_50` east stays
`natural_entry`.

## Next

Compose 0x50 → well → 0x72 from a real predecessor (`rr-ccxt.9`). Glance
bands for 0x01/0x72 leftover (`rr-ccxt.15`). Do not STATUS from these pins.

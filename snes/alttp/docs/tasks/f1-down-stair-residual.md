# Residual — F1 down stair 0x01 → 0x72 (`rr-ccxt.7`)

Planner owns `docs/STATUS.md`. **Isolated** only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not copy
approach xy into Python. Did not overwrite `verified_tip_run.json`.

**Door:** `maps/room_01.json` `down_to_0x72` (trigger is **UP** into the
north-wall well; DOWN from the landing walks south to y=144 and stays in
`0x01`). Reverse of `room_72` `north_to_0x01`.

## Commands

```bash
uv run python snes/alttp/scripts/room_engine.py show room_01
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --down01
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --down01 --from50
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_01 \
  --edge down_to_0x72 --state CastleRoom01 --no-clear \
  --json-out snes/alttp/recordings/probe_b1_from_f1/room_01_down_to_0x72.json
uv run pytest snes/alttp/tests/test_escape_graph.py \
  snes/alttp/tests/test_room_sense.py snes/alttp/tests/test_room_engine_edges.py -q
```

JSON: `recordings/probe_b1_from_f1/down01_CastleRoom01.json`,
`down01_from50.json`, `room_01_down_to_0x72.json`.

## Path (CastleRoom01)

Spawn `(960, 120)` → west on y=120 to `(762, 120)` → north to `(762, 97)`
(map approach `(760, 99)`). **DOWN** from there → y=144, still `0x01`.
**UP** ~16f → room-change `0x72` (last F1 pixel ~(760, 81); toXy (760, 78)
during submodule 14). Independent settle ~400f → control in `0x72`.

North-band y≈104 hits a wall at x=832; do not cut northwest off the corridor.

## RAM leftover

Every row: sword `$F359=1`, follower `$F3CC=0`, keys `$F36F=0`.

| When | room | module/sub | xy | ctrl |
|------|------|------------|----|------|
| `CastleRoom01` spawn | `0x01` | `0x07` / 0 | 960, 120 | 1 |
| Approach (tight) | `0x01` | `0x07` / 0 | 762, 97 | 1 |
| Settled dest (probe) | **`0x72`** | `0x07` / **0** | **1273, 3665** | **1** |
| `CastleRoom50` east land | `0x01` | `0x07` / 0 | 560, 120 | 1 |
| from50 settled dest | **`0x72`** | `0x07` / 0 | 1273, 3665 | 1 |
| `room_engine` final | `0x72` | `0x07` / **14** | 1273, 3638 | 0 |

`room_engine` `at_door_dest=true` (room `0x72`) but `ok=False`
`settle_destination` — same submodule-14 stair animation vs 240f settle cap
as `room_72 north_to_0x01`. Probe leftover after longer idle is the leave
proof: **room `0x72`, module `0x07`, sub `0`, xy=(1273, 3665)**.

## Map / graph

- `down_to_0x72`: `direction=UP`, `toRoom=0x72`, `approachXy=[760,99]`,
  `landingXy=[1273,3665]`, path `stair_west_corridor` → `down_stair_approach`.
- Graph hop `room_01_down_to_0x72` verification=`isolated`, `map_id`+`door_label`
  only, `paths={}` (not on Sanctuary plan). Pair of `room_72_north_to_0x01`.
- `room_01_to_zelda_cell` stays **planned**.

from50 is **isolated** (state-load 0x50 east leftover → stairs), not
natural_entry, not continuous.

## Next

Compose 0x01 → stairs → 0x72 toward Zelda from a real 0x50 predecessor
(`natural_entry` only after that chain is green). Optional: lengthen
`settle_control` so stair edges go `room_engine` green.

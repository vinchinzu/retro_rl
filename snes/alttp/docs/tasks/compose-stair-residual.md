# Residual — compose 0x50 east → well → 0x72 (`rr-ccxt.9`)

Planner owns `docs/STATUS.md`. **Natural entry** for `room_01_down_to_0x72`
only. Not continuous. Reverse `room_72_north_to_0x01` stays isolated.
Did not poke `$F3CC` / sword / keys. Did not copy approach xy into Python.
Did not overwrite `verified_tip_run.json`. Did not STATUS.

## Commands

```bash
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/setup_rom.py
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_50 \
  --edge east_to_0x01 --state CastleRoom50
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py \
  --down01 --from50
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_compose_stair.py
uv run pytest alttp/tests/test_escape_graph.py alttp/tests/test_room_sense.py -q
```

JSON: `recordings/probe_compose_stair/compose.json`.

## Path (CastleRoom50)

Spawn `0x50` (448, 2680) → `room_50` `east_to_0x01` (1301f, ok) lands
`0x01` **(560, 120)**. Corridor walk is **east** on y=120 (leftover is west
of the well; CastleRoom01 spawn at 960 is the west-walk case). (758, 120)
then (758, 97). Hold **UP** 16f → room-change `0x72` at (760, 78) submodule
**14**, no control. Independent idle (not 240f `settle_control`): sub14 last
at 264f, control at **268f**.

DOWN from the well still walks south and stays in `0x01`.

## RAM leftover

Every row: sword `$F359=1`, follower `$F3CC=0`, keys `$F36F=0`.

| When | room | module/sub | xy | ctrl |
|------|------|------------|----|------|
| `CastleRoom50` spawn | `0x50` | `0x07` / 0 | 448, 2680 | 1 |
| `east_to_0x01` land | `0x01` | `0x07` / 0 | **560, 120** | 1 |
| Approach (tight) | `0x01` | `0x07` / 0 | 758, 97 | 1 |
| Edge (UP 16f) | **`0x72`** | `0x07` / **14** | 760, 78 | 0 |
| Settled dest | **`0x72`** | `0x07` / **0** | **1273, 3665** | **1** |

Leave proof is the settled dest glance. Map `landingXy` `[1273, 3665]` matches
(`landingDelta=[0,0]`); not rewritten. 240f `settle_control` would still be
in submodule 14 (last sub14 at 264f after the edge).

## Map / graph

- Graph hop `room_01_down_to_0x72` verification=`natural_entry`, not
  continuous. `map_id`+`door_label` only. Reverse `room_72_north_to_0x01`
  stays `isolated`.
- `paths=frozenset()` (off Sanctuary plan). Putting it on `_BOTH_PATHS`
  would insert `0x01→0x72` between `room_50_east_to_0x01` and planned
  `room_01_to_zelda_cell` (`0x01→0x80`) and break
  `test_escape_route_legs_are_contiguous`. Did not rewrite the planned
  `room_01_to_zelda_cell` note (ownership is this hop only).
- `room_50_east_to_0x01` stays `natural_entry`.

## Next

Compose 0x72 leftover toward Zelda (ledge drop / water / cell). Optional:
lengthen `settle_control` so stair edges go `room_engine` green (`rr-ccxt.14`).
Not continuous until the 0x50 east hop itself is continuous.

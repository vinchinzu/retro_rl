# B1 reverse residual (rr-ccxt.2)

Planner owns `docs/STATUS.md`. Isolated graph only. Not natural_entry, not
continuous, not STATUS. Follower RAM not poked.

## F1 stair (measured)

**0x72 north-wall column x≈1272, hold UP → F1 room 0x01.**

| Field | Value |
|-------|--------|
| Pin | `CastleB1Guard` room 0x72 ~(1273,3665) |
| Approach | (1272, 3656) |
| Trigger | UP, straight stairs, dungeon submodule 14 |
| Room-change | ~28f → 0x01 (1272,3638) ctrl=0 |
| Settled landing | **0x01 (760, 99)** submodule 0 ctrl=1 ~292f |
| Map | `maps/room_72.json` door `north_to_0x01` |
| Graph | `room_72_north_to_0x01` verification=`isolated` (map_id+door_label only; off Sanctuary plan) |

`CastleB1Key` ~(1320,3656) is the same north wall **48px east** of the stair
column — cardinal UP from the key pin misses the stairs.

F1-side door on `room_01` is **not** this bead (other agent owns F1 maps).

## room_engine

```
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py show room_72
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_70 \
  --edge north_to_0x71 --state CastleB2Landing --no-clear
SDL_VIDEODRIVER=dummy uv run python snes/alttp/scripts/room_engine.py run room_72 \
  --edge north_to_0x01 --state CastleB1Guard --no-clear
```

`room_70 north_to_0x71`: dest 0x71 reached (~27f at 160,3614) but
`settle_destination` red — submodule 14 animation needs ~292f; settle cap
240f. Settled landing after idle is **(672, 3658)** in 0x71 (spiral, not F1).

`room_72 north_to_0x01`: `at_door_dest=true` (final room 0x01) but
`ok=False` `settle_destination` at 267f xy=(759,78) submodule 14.
Independent hold-watch settled ctrl=1 at **(760,99)** ~292f
(`recordings/probe_b1_reverse/stair_72_to_01.json`).

## Pins scanned (26 CastleB1* + CastleB2Landing)

RAM leftover: **no pin has `$F3CC==1`**. Sword `$F359==1` on all. Lamp `$F34A==0`.
Keys `$F36F==0` except `CastleB1Key` / `CastleB1SecondKey` (`==1`).
Did not poke follower / sword / keys.

| State | Room | xy | module/sub | ctrl | keys | F3CC |
|-------|------|-----|------------|------|------|------|
| CastleB2Landing | 0x70 | 161,3640 | 07/0 | 1 | 0 | 0 |
| CastleB1Guard | 0x72 | 1273,3665 | 07/0 | 1 | 0 | 0 |
| CastleB1Key | 0x72 | 1320,3656 | 07/0 | 1 | 1 | 0 |
| CastleB1Pit | 0x72 | 1272,4032 | 07/0 | 1 | 0 | 0 |
| CastleB1SecondKey | 0x71 | 904,3988 | 07/0 | 1 | 1 | 0 |
| CastleB1SingleGreen | 0x71 | 632,4008 | 07/0 | 1 | 0 | 0 |
| CastleB1West | 0x82 | 1184,4108 | 07/0 | 1 | 0 | 0 |
| CastleB1WestRoom | 0x81 | 912,4496 | 07/0 | 1 | 0 | 0 |
| CastleB1FarDoor | 0x81 | 632,4416 | 07/0 | 1 | 0 | 0 |
| CastleB1FarWest | 0x81 | 996,4496 | 07/0 | 1 | 0 | 0 |
| CastleB1UpperCleared | 0x82 | 1252,4233 | 07/0 | 1 | 0 | 0 |
| CastleB1Bridge | 0x82 | 1255,4392 | 07/0 | 1 | 0 | 0 |
| CastleB1South | 0x82 | 1312,4456 | 07/0 | 1 | 0 | 0 |
| CastleB1GuardLamp | **0x52** | 1120,2656 | **12/1** | 0 | 0 | 0 |
| CastleB1IslandCleared | **0x61** | 760,3520 | 07/0 | 1 | 0 | 0 |
| CastleB1PitCleared | 0x72 | 1272,3872 | **12/2** | 0 | 0 | 0 |

`CastleB1GuardLamp` is a **death** pin already on F1 0x52, not a stair
measurement. `CastleB1IslandCleared` is F1 main hall spawn, not B1.

Full census: `recordings/probe_b1_reverse/pin_census.json`.

## B1 exits observed (not F1)

| From | Dir | To | Notes |
|------|-----|-----|-------|
| 0x70 | UP @ x≈160 | 0x71 | spiral; settle (672,3658) |
| 0x70 | UP @ x≈80 | 0x80 | Zelda cell (state-local) |
| 0x70 | DOWN/LEFT | stuck in 0x70 | small chamber |
| 0x71 spiral landing | UP | 0x70 | reverse of the spiral |
| 0x71 SingleGreen/SecondKey | cardinals | no room change | |
| 0x81 WestRoom | RIGHT | 0x82 | already mapped |
| 0x81 FarDoor | UP | 0x71 | already mapped |
| 0x82 West | UP | 0x72 | already mapped |
| 0x82 UpperCleared | UP | 0x72 | |

## Glance leftover (stair dest)

After 0x72 UP settle in 0x01: module `$10==0x07` submodule `$11==0`
xy=(760,99) sword `$F359==1` keys `$F36F==0` follower `$F3CC==0`.

## Files

- `docs/tasks/b1-reverse-residual.md` (this file)
- `scratch/probe_b1_reverse.py`
- `recordings/probe_b1_reverse/`
- `maps/room_72.json` door `north_to_0x01`
- `maps/room_70.json` / `room_71.json` spiral landing notes
- `opening_route/escape_graph.py` isolated hop `room_72_north_to_0x01` (not on plan)
- `tests/test_room_sense.py` / `tests/test_escape_graph.py` map+isolated assertions

## Non-claims

Did not STATUS-promote. Did not treat a save-state pin as power-on. Did not
write `$F3CC` / sword / keys. Did not copy approach xy into Python. Did not
edit F1 maps (`room_01` etc.). Did not overwrite `verified_tip_run.json`.
`room_01_to_zelda_cell` stays **planned**.

## Next action

F1 agent: add the **down** stair on `maps/room_01.json` from natural 0x01
entry (landing here is (760,99); 0x50 east still lands farther east). Then
compose 0x01 → stairs → 0x72 toward Zelda. Optional: lengthen
`settle_control` so spiral/straight-stair edges go `room_engine` green
(240f cap vs ~264f after dest).

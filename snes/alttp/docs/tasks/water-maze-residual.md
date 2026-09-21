# 0x82 water maze residual (rr-ccxt.12)

Planner owns `docs/STATUS.md`. Isolated chain only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / keys. Did not add a graph hop.

## Result: **ok** — 0x72-south **does** chain to 0x81

`CastleB1PitGuardCleared` `room_72` `south_to_0x82` leftover in 0x82
(1198,4108) `$F3CC==0` walks around the pit to `west_to_0x81` and settles in
**0x81 (996,4495)** `$F3CC==0` keys=0 sword=1 `$10==0x07` `$11==0`.

Direct walk north→west still pits (submodule 20, respawn north door). The
walkable route hugs the **east** wall then the **south** walkway.

## Path (one-axis; halt was south-from-bridge then x-drift off east wall)

| Step | xy | Note |
|------|----|------|
| land | (1198, 4108) | 0x72-south leftover; door alcove (LEFT/RIGHT walls) |
| floor_south | (1198, 4160) | onto north floor |
| east_lane | (1272, 4160) | east; west of here is pit / 1-tile west ledge dead-end |
| island_y | (1272, 4253) | south toward island |
| corridor_x | (1255, 4253) | align with bridge x; diagonal pits |
| bridge | (1255, 4393) | N-S corridor; south/west = pit |
| east_hug | (1324, 4393) | **x≥1312**; x=1310 then south → pit sub 20 |
| south_walk | (1324, 4492) | south walkway y-band |
| east_b1 | (1036, 4492) | west along y=4492 (CastleZeldaB1East RIGHT reaches x=1360 here) |
| LEFT | → 0x81 | transition (1031,4495) sub=2 |

## 0x81 leftover (RAM glance)

| Field | Value |
|-------|--------|
| Pin chain | `CastleB1PitGuardCleared` → `south_to_0x82` → maze → LEFT |
| Room | **0x81** (996, 4495) |
| Module | `$10==0x07` `$11==0` ctrl=1 |
| HP | `$F36D==4` / `$F36C==24` (soldiers on the north floor) |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | `$F36F==0` |
| Follower | **`$F3CC==0`** |

`west_to_0x81` landingXy left at [960, 4496]; this leftover matches the
isolated hop ~(996,4492), not 960.

## Blockers that are **not** the chain

- North-door alcove: LEFT stuck x=1184, RIGHT stuck x=1200.
- Direct south/west from north floor or from bridge: pit sub 20, respawn
  (1198,4108), −8 HP.
- West ledge (1136,4160): N/S/W all pit. Not a west corridor.

## Files

- `docs/tasks/water-maze-residual.md` (this file)
- `scratch/probe_82_water.py`
- `recordings/probe_82_water/` (`probe.json`, `leftover.json`)
- `maps/room_82.json` waypoints + `west_to_0x81` path / notes

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not add a graph hop.
Did not treat a save-state pin as power-on. Did not rewrite `west_to_0x81`
landingXy.

## Next

0x81 leftover is at the east door (996,4495), keys=0. Cell door
`west_to_0x80` is still locked (rr-ccxt.13).

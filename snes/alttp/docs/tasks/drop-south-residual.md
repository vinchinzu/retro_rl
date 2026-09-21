# 0x72 key-door drop leftover → south_to_0x82 (`rr-ccxt.17`)

Planner owns `docs/STATUS.md`. Isolated only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not copy
approach xy into Python hops. Did not add a graph hop.

## Result: **ok** — drop leftover **does** reach 0x82

`CastleB1Key` south key door spends the small key and lands on the lower
floor **(1272, 3945)** keys=0. From that leftover, west onto `zelda_pit`
then south through `south_door_approach` settles in **0x82 (1190, 4108)**
`$10==0x07` `$11==0` ctrl=1 `$F3CC==0` keys=0 sword=1.

Same north-door alcove as the already-isolated `CastleB1PitGuardCleared`
hop ~(1198, 4108). LandingXy untouched.

## Path

| Step | xy | Note |
|------|----|------|
| Pin | (1320, 3656) | `CastleB1Key` 0x72 keys=1 hp=20 |
| East lip | (1320, 3776) | DOWN, still keys=1 |
| Unlock | (1287, 3776) | DOWN+LEFT; sub=4, keys 1→0 |
| Drop | **(1272, 3945)** | DOWN through; lower floor `$5D==6` keys=0 hp=20 |
| Off rim | (1272, 3984) | south toward pit_guard y; pit is north |
| Fight | (1214, 4009) | one soldier hit, hp 20→16 |
| zelda_pit | (1190, 4009) | west-first onto west-wall corridor |
| Door | (1190, 4060) | south; room-change sub=2 |
| Land | **(1190, 4108)** | 0x82 north alcove, `$11==0` ctrl=1 |

South-first at pit_guard x≈1224 hits the **south railing** at (1224, 4032)
(overlay `stuck.png`). Isolated PitGuard hop diagonals SW; this leftover
must go **west** before south.

## 0x82 leftover (RAM glance)

| Field | Value |
|-------|--------|
| Pin chain | `CastleB1Key` → south key door → drop → west/south |
| Room | **0x82** |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| xy | **(1190, 4108)** |
| HP | `$F36D==16` / `$F36C==24` |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | **`$F36F==0`** (spent on 0x72 south key door) |
| Follower | **`$F3CC==0`** |

## Files

- `docs/tasks/drop-south-residual.md` (this file)
- `scratch/probe_drop_south.py`
- `recordings/probe_drop_south/` (`leftover.json`, `land.png`, `drop.png`)
- `maps/room_72.json` notes on `key_ledge_drop` / `south_to_0x82` (door path
  still `pit_guard_cleared` → `south_door_approach`)

## Non-claims

Did not STATUS-promote. Did not treat a pin as power-on. Did not write
`$F3CC` / sword / keys. Did not copy approach xy into Python. Did not edit
`escape_graph.py` / `room_engine.py` / `primitives.py` / `screen_glance.py` /
`run_opening_spine.py`. Did not add a graph hop. Did not keep keys>=1.

## Next action

0x82 leftover is the north alcove, keys=0. `west_to_0x81` already chains
from this alcove (rr-ccxt.12) still keys=0. Cell door `west_to_0x80` still
wants keys>=1 (the key this drop spent).

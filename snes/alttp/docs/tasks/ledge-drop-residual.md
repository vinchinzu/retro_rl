# 0x72 north-ledge drop residual (`rr-ccxt.11`)

Planner owns `docs/STATUS.md`. Isolated only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not copy
approach xy into Python hops. Did not add a graph hop.

## Drop that exists (spends the key)

**South key door on the north ledge**, then DOWN onto the lower floor.

| Field | Value |
|-------|--------|
| Pin | `CastleB1Key` 0x72 (1320, 3656) keys=1 hp=20 `$10==0x07` `$11==0` `$F3CC==0` |
| East lip | DOWN → (1320, 3776) still keys=1, railing holds |
| Unlock | DOWN+LEFT into ~(1287, 3776): submodule **4**, `$F36F` 1→0, `$0401` bit7 |
| Through | DOWN after settle → **(1272, 3945)** lower floor |
| Landing | room **0x72** `$10==0x07` `$11==0` ctrl=1 hp=20 **keys=0** `$F3CC==0` `$5D==6` |
| Map | `south_key_door` (1287, 3776), `key_ledge_drop` (1272, 3945) |

LEFT-only along y=3776 walks past the door to x=1224 and does **not** spend
the key. Must press DOWN into the door column.

Landing is `north_extent` (1271, 3944) +1px; south of the pit, toward
`pit_guard_cleared` (1230, 3989) / `entry_spawn` (1272, 4032). Overlay:
`recordings/probe_ledge_drop/keydoor_through.png`.

## No key-preserving drop

Grab / star / walk-off all stay on the ledge with keys=1:

| Act | Claim | RAM |
|-----|-------|-----|
| DOWN to east lip | reach y≈3776 keys>=1 | **miss drop** (1320, 3776) keys=1 sub=0 |
| A+DOWN at lip | grab-drop to y>=3944 or pit sub=20 | **miss** still (1320, 3776) keys=1 |
| Star ~(1272, 3712) | warp/drop | **miss** stuck (1296, 3720) keys=1 (decorative tile) |
| DOWN+LEFT (no halt) | pit or lower floor keys>=1 | **miss** slides to (1272, 3776); if it nicks the door, **keys=0** still on the ledge |

Pit submodule 20 never fired. Module 0x12 never fired on these acts. Walking
into lower-floor soldiers from the **closed** railing is the old death (Guard
sitting); the key door hops over the pit onto the floor.

## Glance leftover (after through)

| Field | Value |
|-------|--------|
| Room | **0x72** |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| xy | **(1272, 3945)** |
| HP | `$F36D==20` / `$F36C==24` |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | **`$F36F==0`** (spent on the south key door) |
| Follower | `$F3CC==0` |
| `$5D` | 6 (recoil/hop leftover; overlay is standing on the lower floor) |
| `$0401` | 128 (door bit) |

## Files

- `docs/tasks/ledge-drop-residual.md` (this file)
- `scratch/probe_ledge_drop.py` (`--key-door`)
- `recordings/probe_ledge_drop/` (`leftover.json`, `grab_star_diag.json`, overlays)
- `maps/room_72.json` waypoints `south_key_door` / `key_ledge_drop` (existing door landings untouched)

## Non-claims

Did not STATUS-promote. Did not treat a pin as power-on. Did not write
`$F3CC` / sword / keys. Did not copy approach xy into Python. Did not edit
`escape_graph.py` / `room_engine.py` / `primitives.py` / `screen_glance.py` /
`run_opening_spine.py`. Did not add a graph hop. Did not keep keys>=1 onto
the lower floor.

## Next action

Lower floor is reachable from `CastleB1Key` only by **spending** the small
key. That key is the one 0x81 `west_to_0x80` wants. Need a keys>=1 pin on
the 0x81 west wall (or a second key before the 0x72 south key door) to
isolate the cell door.

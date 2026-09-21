# Palace of Darkness entry pin (`rr-9zkv`)

Planner owns `docs/STATUS.md`. Dev pin, not continuous, not power-on.
Session: `.grok/skills/alttp-session/SKILL.md`.
Handoff: `docs/tasks/HANDOFF-dungeon-starts.md`.

## Result

`PalaceOfDarknessEntry.state` exists. Indoor room on ROM is **0x4A**
(hypothesis confirmed). Isolated first hop **0x4A east cage north stairs →
0x09 shooter room**. `room_engine.py run` is fail-closed: `in_room` requires
`not dark_world` and `$0FFF==1` on this pin. Probe `--isolate` leftover is
the hop proof. Not natural_entry, not continuous, not STATUS.

Kiki / overworld walk did not finish (PoD overworld door is Kiki-locked).
Indoor pin is a **real room-load**: kit pokes, then `$010E=0x26` + module
`$10=0x06` (underworld load). Controllable overlay is PoD lobby, `$0FFF==1`.

## Pin glance (`PalaceOfDarknessEntry`)

| Field | Value |
|-------|--------|
| Room | **0x4A** (5368, 2520) south door spawn |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 indoors |
| Dark world | **`$0FFF==1`** `$F3CA==0x40` `$040C==0x0C` |
| Entrance id | `$010E==0x26` (PoD exit/load; 0x25 is Swamp 0x28) |
| Sword | `$F359==2` (master) |
| Bow | `$F340==2` arrows `$F377==30` |
| Lamp | `$F34A==1` |
| Hammer | **`$F34B==0`** (prize not granted) |
| Gloves | `$F354==1` |
| Boots | `$F355==1` `$F379==0xFC` (dash bit) |
| Moon pearl | `$F357==1` |
| Shield | `$F35A==1` |
| Progress | `$F3C5==3` |
| HP | `$F36D==56` / `$F36C==56` (7 hearts) |
| Keys | `$F36F==0` |
| Follower | `$F3CC==5` (HyruleCastleGrounds leftover; **not poked**) |
| Layer | `$00EE==0` |

## Pokes

All via `env.unwrapped.data.memory.assign`. High WRAM needs bank `$7E`
(`0xF359` → `IndexError: No known mapping`; retry `0x7EF359`). Read-after-write
on `get_ram()[wram_index(offset)]`.

Source: `HyruleCastleGrounds` (2386, 2528) screen `0x1B` module `0x09`.

| Offset | Item | Before | After | Via |
|--------|------|--------|-------|-----|
| `$F359` | sword master | 0 | 2 | `memory.assign 0x7ef359` |
| `$F35A` | shield | 0 | 1 | `0x7ef35a` |
| `$F34A` | lamp | 0 | 1 | `0x7ef34a` |
| `$F340` | bow+arrows | 0 | 2 | `0x7ef340` |
| `$F354` | power gloves | 0 | 1 | `0x7ef354` |
| `$F355` | boots | 0 | 1 | `0x7ef355` |
| `$F357` | moon pearl | 0 | 1 | `0x7ef357` |
| `$F3C5` | Agahnim 1 / DW | 0 | 3 | `0x7ef3c5` |
| `$F3CA` | world DW | 0 | 0x40 | `0x7ef3ca` |
| `$F377` | arrows | 0 | 30 | `0x7ef377` |
| `$F36C` / `$F36D` | max/HP | 24 | 56 | `0x7ef36c` / `0x7ef36d` |
| `$F379` | ability dash | 0xF8 | 0xFC | `0x7ef379` |
| `$0FFF` | DW flag | 1 (after `$F3CA`) | 1 | `memory.assign 0xfff` |
| `$010E` | entrance | 0 | 0x26 | `memory.assign 0x10e` |
| `$10` | module load | 0x09 | 0x06 | `memory.assign 0x10` |

Did not poke `$F3CC`. Did not grant `$F34B` hammer. Load frames: module 6 →
7/sub 15 → 7/sub 0 ctrl at (5368, 2520) room 0x4A.

## First hop leftover — `east_stairs_to_0x09`

Path from spawn (map `room_4a.json`): `east_lane` (5504, 2424) → `east_green`
(5504, 2300) → `east_cage` (5512, 2192) → `east_stair_approach` (5512, 2104)
arrives ~(5504, 2111), hold **UP**. Dest **0x09** shooter room B1.

Replay: `SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --isolate`

| Field | Value |
|-------|--------|
| Hop | `east_stairs_to_0x09` |
| Room | **0x09** (4985, 55) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 indoors |
| Sword | `$F359==2` |
| Hammer | `$F34B==0` |
| Pearl | `$F357==1` |
| Keys | `$F36F==0` |
| Follower | `$F3CC==5` (unchanged) |
| DW | `$0FFF==1` `$F3CA==0x40` `$040C==0x0C` |
| HP | `$F36D==40` / `$F36C==56` (helmasaur chip) |
| Verification | **isolated** (probe). `run_room_edge` red: `in_room` rejects DW |

Mini-helmasaurs (`0x13`) on the east cage are not `CASTLE_HOSTILE_TYPES`.
South-east / south-west palace shutters at y≈2496 stay closed. North key
door not this hop. DOWN from spawn is overworld backtrack ~(3920, 1576)
module 0x08 (PoD door; screen 0x5E).

## Map

`maps/room_4a.json` — spawn, east-cage stair path, south overworld door.
`z3Label`: Palace of Darkness (Entrance).

## Next

Make `room_engine.in_room` DW-safe (not this sitting; `ram.py` /
`room_engine.py` are not owned). Then `run_room_edge` for
`east_stairs_to_0x09`. Helmasaur skirmish (type `0x13`) so the hop is less
RNG. Shooter-room key is the following hop.

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke
`$F3CC`. Did not copy approach xy into Python (geometry is map JSON). Did
not overwrite `verified_tip_run.json`. Did not claim `rr-ccxt.20` / `.21`.
Did not touch Sanctuary maps or `escape_graph.py`.

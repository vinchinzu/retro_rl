# Plan — Super Metroid assisted full clear

Shared workflow:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../../snes_oneshot/docs/FULL_RUN_PROCESS.md).
Assist semantics: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).

## Strategy

Unlimited energy and ammo make combat and hazard attrition secondary. The hard
problem remains long-horizon navigation: room identity, door/elevator
transitions, item requirements, movement abilities, boss/event state,
backtracking, and recovery from positional stalls.

Build a route graph and room policies. Do not start with a monolithic
coordinate script or a full-run loop.

## Phase 0 — contract and scaffold

- [x] Record ROM path and hash.
- [x] Define allowed resource writes and forbidden progression writes.
- [x] Define continuous completion at ending/credits, not final-boss HP zero.
- [x] Create the integration files, ROM link, typed state, and tests.
- [x] Choose and document the initial start condition: `retro.State.NONE`,
  fresh file A selected through the title flow.

Acceptance met: the integration boots the expected ROM and the contract is
represented in tests and report fields.

## Phase 1 — boot and core RAM

Map with probe evidence:

- game/menu/control mode
- area, room, door/elevator transition
- player X/Y, velocity, pose, grounded/control flags
- current/max energy and reserves
- current/capacity for each ammo type
- equipment/item bitsets
- boss/event/collected-item bits
- death/game over
- ending/credits state

Use the continuous reset boot trace as acceptance evidence. Development states
may be added later, but are not part of the accepted route.

Acceptance met: repeated reset runs reach the same Ceres control predicate at
frame 10,860 without a state load.

## Phase 2 — route graph and first natural suffix

Represent milestones as data:

```text
milestone
  entry predicate
  required inventory/events
  room/door target
  policy owner
  completion predicate
  timeout
  recovery state
```

Start with:

1. power-on/menu → first controllable Ceres room
2. Ceres traversal → escape/transition
3. Zebes arrival → first required upgrade
4. first upgrade → first ammo unlock
5. first ammo unlock → next route gate

The current verified prefix ends at Morph Ball. The imported full-game route
manifest remains a research baseline rather than accepted progression data.

Prefix acceptance met through Morph Ball from the state produced by every real
predecessor. Clean-state development evidence remains available in the sibling
project; this repository's claim is the stronger continuous natural-entry run.

## Phase 3 — navigation primitives

Build only primitives demonstrated by two or more rooms:

- approach and activate door/elevator
- run/jump across a room
- recover from wall, ledge, and platform stalls
- aim/shoot a door or obstacle
- traverse vertical shafts
- select and use naturally unlocked ammo
- fight or bypass an enemy
- boss-specific policy

Watchdogs use room/door/inventory/event progress, not player coordinates alone.
Every recovery action has a bounded budget and a regression state.

## Phase 4 — route expansion

Grow verified suffixes through:

- early required movement/combat upgrades
- early bosses and major area transitions
- midgame traversal/backtracking
- late-game access requirements
- final area and bosses
- endgame escape
- ending/credits

Maintain a route-requirement table. An item or boss flag is considered
required only when a real transition demonstrates the dependency.

## Phase 5 — assist validation

Before long chains:

- verify energy refill never changes maximum energy or item flags
- verify every ammo type stays locked at zero capacity until collected
- verify refill stops during transitions, menus, death, and scripted sequences
- verify damage and ammo use are measured before refill
- verify progression-write count remains zero

Test ordinary combat, environmental damage, an ammo door/obstacle, a room
transition, a boss transition, and a scripted sequence.

## Phase 6 — chain and full dry runs

Promotion order:

1. segment from clean state
2. segment from natural entry
3. two-milestone suffix
4. area suffix
5. late-game suffix through ending
6. full power-on dry run
7. final capture

Candidate reports and logs must not overwrite the last successful baseline.
Abort early on milestone timeout, route regression, forbidden write, invalid
assist write, or prolonged no-progress.

## Initial metrics

- completion milestone and furthest room
- total frames and split time per milestone
- room/door transitions
- item and boss/event acquisition frames
- deaths
- energy restored and write count
- ammo restored/writes by type
- action-reason counts by room/segment
- maximum no-progress interval
- state loads and progression writes

## First implementation slice

1. [x] Scaffold the integration around `roms/SuperMetroid.sfc`.
2. [x] Boot headlessly and identify the first controllable frame.
3. [x] Populate `docs/ram_map.md` with source and live-route evidence.
4. [x] Implement phase-guarded, capacity-preserving unlimited ammo.
5. [x] Clear all of Ceres continuously from power-on.
6. [x] Continue from the natural Zebes entry through Morph Ball.
7. [x] Extend through both early Missiles, Climb return, and Bomb Torizo/Bombs.
8. [x] Extend post-Torizo through Terminator/Green Brinstar, defeat Spore
   Spawn, and exit naturally to the Spore Super room.
9. [x] Merge full reference topology and editor geometry into 262 canonical
   room-development problems.
10. [x] Validate save-state teleport and natural target-room settlement on two
    queue-1 door clears plus Flyway.
11. [ ] Collect Spore Supers naturally, then clear
    `0xA0A4 → 0x9D19 → 0x9E11` and collect Power Bombs.
12. [ ] Work queues 1–2 before sequencing queue-3 rooms and queue-4 bosses.

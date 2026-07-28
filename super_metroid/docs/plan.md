# Plan — Super Metroid assisted full clear

Shared workflow:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../../snes_oneshot/docs/FULL_RUN_PROCESS.md).
Assist semantics: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).

## Strategy

Unlimited energy and ammo make combat and hazard attrition secondary. The hard
problem remains long-horizon navigation: room identity, door/elevator
transitions, item requirements, movement abilities, boss/event state,
backtracking, and recovery from positional stalls.

**Clear rooms by play.** Each hop on the completion path must be crossed with a
controller or room policy (natural door exit). Door-warps are topology
diagnostics only — never route evidence. Living inventory:

**[PATH_ROOM_BOARD.md](PATH_ROOM_BOARD.md)** (regenerate with
`scripts/export_path_room_board.py`).

Do not start with a monolithic full-run coordinate script. Grow one hop at a
time from the furthest played room.

**Boss fights stay deferred** until natural *entry* to that boss room exists on
the played chain. Continuous acceptance still requires natural boss flags and
zero progression writes.

---

## Current inventory (2026-07-27)

### Verified continuous (M5)

| Artifact | Coverage |
|----------|----------|
| `recordings/start_to_spore_spawn.mp4` | Power-on → Ceres → Morph → Missiles → Bombs/Torizo → Spore Spawn → Super room `0x9B5B` |
| Frames | 91,220 @ 60 fps (~25.3 min) |
| Integrity | 0 state loads; 0 progression/capacity writes; natural item/boss flags |

### Research topology (not continuous)

| Piece | Status |
|-------|--------|
| `maps/full_room_graph.json` | 261 rooms, 583 directed edges, 22/22 completion legs have room paths |
| `maps/full_route_hops.json` | **199 door hops** across all 22 legs (~**107 unique rooms**) |
| `maps/late_game_route_hops.json` | Late 9 legs only (subset of full; identical hop data) |
| `route_dev.py` + `probe_route.py` | **Full 22-leg hop runner proven** (dev); late subset still available; fights skipped |
| Null door substitute | Ceres ship `0xDF45 → 0x91F8` → door `0x896A` (Parlor→Landing Site) |
| Tour video path | `probe_route.py full-tour` → `full_route_tour.{mp4,json}`; hybrid `full-hybrid` → `full_route_hybrid.{mp4,json}` (continuous Super + warps, bosses skipped) |
| Mid/late dev states | `dev_power_bombs_collected`, `dev_phantoon_entry`, `dev_route_*` anchors through finish |

### Continuous gap (first missing natural progress)

```text
[VERIFIED] power-on ──► Spore Super collect 0x9B5B (capacity 0→5)
                              │
                    ★ GAP: natural Super exit → PB → …
                              │
[DEV ONLY] hybrid video: continuous Super + door-warp rest (bosses skipped)
                              │
[DEV ONLY] door-warp PB ──► ship ──► Phantoon ──► … ──► Landing Site
                              (boss bits written; loadout granted)
```

### Room-policy maturity

| Layer | Count / note |
|-------|----------------|
| Curated continuous segments | start_to_morph + early_game + spore_spawn_controller |
| Verified room_clears | **3** of 262 catalog problems |
| Bulk scaffolds | 262 templates exist; almost none promoted |
| Boss policies | Spore + Bomb Torizo continuous; Kraid dev spray only |

---

## Two product tracks

Keep these separate so topology probes do not pollute continuous acceptance.

| Track | Goal | Integrity rules | Bosses |
|-------|------|-----------------|--------|
| **A — Topology probe** | Hop table / door-warp walk for connectivity only | Dev warps allowed; **label** `developmentOnly`; not route evidence | Skip bits OK for topology |
| **B — Played room spine** | Every path hop crossed by controller/policy; grow continuous chain | Assist contract for continuous claims; natural entries preferred | Entry by play first; fights later |

**Primary product path is Track B.** Track A already proved 22-leg connectivity;
do not invest further in warp tours as a substitute for playing rooms.

How far we are (play, not warps):

| Layer | Furthest |
|-------|----------|
| Continuous | Super collect `0x9B5B` |
| Controller dev | Big Pink main shaft `0x9D19` ~(746,1465) |
| ★ Next hop | `0x9D19 → 0x9E11` (PB door climb) |

Full room list + waves: [PATH_ROOM_BOARD.md](PATH_ROOM_BOARD.md).

---

## Track A — end-to-end room-tour video (A0–A2 core done)

**Deliverable:** `recordings/full_route_tour.mp4` (+ JSON report) that:

1. Starts from a known state (power-on continuous prefix *or* Ceres/Landing boot).
2. Walks **all 22 completion legs** using `maps/full_route_hops.json`.
3. Visits ~107 rooms; holds each room briefly so the video is watchable.
4. Skips boss fights via `skip_boss` / `mark_all_major_bosses`.
5. Ends at Landing Site finish `0x91F8` (and, if cheap, idle into ship/credits
   probe — not required for “all rooms”).
6. Manifest marks every progression write and every state load.

### A0 — Wire full hop runner (1–2 days) — **done**

- [x] Extend `route_dev.py` to load `full_route_hops.json` (not only late).
- [x] Define `FULL_LEG_ORDER` = all 22 legs from `completionSequence`.
- [x] Handle the **one null door hop**: Ceres ship `0xDF45 → 0x91F8`
  substituted with door `0x896A` (Parlor→Landing Site) via
  `NULL_DOOR_SUBSTITUTES`.
- [x] At each item/boss anchor, apply progressive flag/loadout via
  `apply_anchor_progress` (early legs) + full loadout after Morph.
- [x] CLI: `probe_route.py full` / `full-tour [--video PATH]
  [--frames-per-room N] [--report PATH]`.
- [x] Tests: hop chain room continuity for all 22 legs; null-door documented
  (`tests/test_route_dev.py`, 12 tests).

**Acceptance met:** `probe_route.py full` returns success through
`landing_site_finish` with hop success on every non-null door (full emulator
run of all 22 legs).

### A1 — Early/mid loadout gates on the tour (1 day) — **done**

Door colors and elevators need inventory/events even when fighting is skipped.
Implemented as progressive `apply_anchor_progress`:

| Anchor leave | Grant / set (dev) |
|--------------|-------------------|
| morph_ball | Morph bit |
| first_missile | Missile capacity ≥ 5 |
| bomb_torizo | Bombs + Torizo bit |
| spore_spawn | Spore bit + Super capacity |
| early_power_bombs | PB capacity |
| kraid | Kraid bit + Varia if needed for heat later |
| speed_booster / ice_beam | Speed, Ice (and beams as hop table needs) |
| phantoon…ridley | Existing `ROUTE_ITEMS` / `ROUTE_BEAMS` + boss bits |
| mother_brain | Event 0x0E + escape-room placement |

### A2 — Record the tour video (1 day) — **done** (core path + hybrid)

- [x] Frame writer on every hop settle + short walk/idle in-room
  (`--frames-per-room`, default 36).
- [x] Hybrid splice: continuous `start_to_supers` prefix *then* resume
  from Super room with warps for the rest
  (`probe_route.py full-hybrid` → `recordings/full_route_hybrid.{mp4,json}`).
- [x] Report: per-leg room list, hop success, frames, flags written, label
  `developmentOnly: true`.

**Acceptance:** `probe_route.py full-tour` writes
`recordings/full_route_tour.mp4` + `.json` by default. Hybrid path:
`full-hybrid` concatenates continuous Super prefix + warp suffix. Both are
development-only — not continuous acceptance.

### A3 — Escape / credits glance (optional, 1–2 days)

Not required for “all rooms,” but cheap polish for the same video:

- [ ] After MB skip: place in Escape 1 pipes, warp Escape 1–4 → Climb → Parlor → LS.
- [ ] Probe ship interaction / game-state ending; if credits RAM is reachable
  without a real MB fight, append a short credits clip. If not, stop at LS
  and document credits as Track B.

---

## Track B — continuous room spine (after A0 skeleton)

Replace door-warps with natural room policies **in order**. Bosses remain
skippable in a “spine dry run” mode until fight scripts land; final acceptance
requires real fights.

### B1 — Continuous Super → Power Bombs (first continuous growth)

Natural suffix from verified Spore Super entry:

```text
0x9B5B (Spore Super) → collect Supers
  → 0xA0A4 → 0x9D19 (Big Pink) → 0x9E11 → collect Power Bombs
```

- [x] Super shaft descent + Chozo collect (capacity 0 → 5) from
  `natural_post_spore_spawn` — `post_spore_controller.play_super_room_collect`.
- [x] Continuous power-on → Super collect dry report
  (`recordings/start_to_supers.json`, 92,424 frames, integrity green).
- [x] Bottom gate bomb + door shot → farming `0xA0A4` (dev from post-Spore).
- [x] Farming green Super door → Big Pink `0x9D19` (dev).
- [x] Big Pink farm-pocket **crest** (run-right + spin-jump-left → ~1125,1387
  standing) — `play_big_pink_crest_pocket` (dev).
- [x] Identify 3b blocks: Super-only shot (69,87), bomb (62–63,87), scroll (3,5).
- [x] Crouch-Super clear of (69,87) from crest — `play_big_pink_clear_super_block`.
- [x] Double-tap DOWN morph (standing y≈1387 = morph y≈1401 pose height).
- [x] Tunnel morph-west + X bombs — `play_big_pink_tunnel_west`.
- [x] Green `play_big_pink_into_main_shaft` → main x≲750 (no place/WRAM).
- [ ] Continuous power-on through farming / Big Pink / crest / main.
- [ ] Big Pink → Pink PB room `0x9E11` natural — **open**:
  - climb from shaft ~y=1550 up to door (520, 1144);
  - standing blue-door touch (warp+place collect known).
- [ ] Natural PB collect in `0x9E11` after natural entry.
- [ ] Continuous power-on through PB.

Route-by-route board (Supers → Phantoon): [ROUTE_SUPERS_TO_PHANTOON.md](ROUTE_SUPERS_TO_PHANTOON.md).

### B2 — PB → Red Tower → ship → Phantoon entry (natural)

Research hop path already in `full_route_hops.json`
(`early_power_bombs__kraid` is Brinstar-heavy; ship route is the
`ice_beam__phantoon` / `phantoon_dev.SHIP_ROUTE` family). Practical continuous
order after PB:

```text
Big Pink → GHZ / Noob Bridge → Red Tower
  → Hellway → Caterpillar → elev → Crateria Kihunter
  → Moat → West Ocean → WS → Phantoon room
```

- [ ] One room policy (or short multi-room controller) per hard room:
  Red Tower climb, Moat, WS basement.
- [ ] Kraid detour is on the research path for Varia/Speed/Ice; either:
  - **B2a (any% spine):** ship route first, defer Kraid/Norfair until needed; or
  - **B2b (research path):** natural Kraid → Varia → Speed → Ice → ship.

Prefer **B2b** only if the hop table is the single source of truth for the
tour video; for continuous clear, **any% ordering can differ** as long as
capabilities and doors match. Document the chosen continuous order in
`progression.py` as a typed graph (like start_to_spore_spawn).

### B3 — In-room policy factory (parallel)

Stop hand-authoring only:

1. Scaffold from catalog waypoints (`run_room_problem.py scaffold`).
2. Replay from natural entry state captured from predecessor.
3. Promote to `verified_development_state` then to continuous graph edge.
4. Priority queue = **rooms on the completion path first** (~107), not all 262.

High-value path rooms (from hop table + known hard geometry):

| Priority | Rooms / legs |
|----------|----------------|
| P0 | Super room, PB room, Red Tower, Moat, West Ocean, WS shaft |
| P1 | Warehouse / Kraid approach, Norfair heat halls if on continuous path |
| P2 | Maridia sand / Botwoon hall, LN exit, Statues, Tourian metroids |
| P3 | Escape pipes / Climb / Parlor return |

### B4 — Boss scripts (deferred until B1–B3 spine exists)

| Boss | Dev status | Continuous need |
|------|------------|-----------------|
| Bomb Torizo | Continuous | done |
| Spore Spawn | Continuous | done |
| Kraid | Super-spray dev clear | natural entry + rear door + Varia |
| Phantoon | entry state only | fight + WS power restore |
| Botwoon | skip bit only | fight |
| Draygon | skip bit only | fight + Space Jump collect |
| Ridley | skip bit only | fight |
| Mother Brain | room entry + spray probes | zebetite kill, phases, escape init |
| Escape → credits | warp hop chain | timer, ship, ending/credits RAM |

### B5 — Full continuous dry run → verified capture

Promotion order (same as historical Phase 6):

1. Segment from natural entry
2. Multi-milestone suffix
3. Power-on dry run with boss skips (spine integrity)
4. Power-on dry run with real bosses
5. Credits evidence + video (M8)

---

## Gap checklist (what blocks “done”)

| Gap | Blocks | Track |
|-----|--------|-------|
| ~~Full hop runner not wired~~ | — | **A0 done** |
| ~~Ceres ship null door~~ (sub `0x896A`) | — | **A0 done** |
| ~~Progressive loadout on early tour legs~~ | — | **A1 done** |
| ~~Tour video recorder~~ (`full-tour`) | — | **A2 done** |
| ~~Hybrid continuous-prefix + warp tour~~ | — | **A2 done** (`full-hybrid`) |
| ~~Natural Super collect~~ | — | **B1 done** (continuous) |
| Crest → main shaft + PB door | Continuous past Big Pink | B1 |
| Natural PB + ship rooms | Continuous to Phantoon | B1–B2 |
| ~100 path room policies | Continuous room running | B3 |
| Boss fight scripts | True clear | B4 |
| Escape timer + credits predicate | M8 ending | B4–B5 |

---

## Recommended execution order (next 2 weeks)

```text
Track A topology — DONE (stop expanding warp product work)
  ✓  full hop runner + full-tour / full-hybrid diagnostics

Track B — play every path room (PRIMARY) — NEXT
  Day 1–3  W1: Big Pink climb → PB by play; continuous power-on → PB
  Day 4–7  W2: next open hops after PB (GHZ / Noob / Red Tower or ship branch)
  Ongoing  For each hop: natural entry → attempt → promote on PATH_ROOM_BOARD
  Later    W9 boss scripts only after natural boss-room entry exists
```

Do **not** door-warp past open hops to fake progress. Measure furthest played
room; fix that hop; repeat. Super/PB climb is the current bottleneck.

---

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

Prefix acceptance met through Morph Ball from the state produced by every real
predecessor. Continuous acceptance later extended through Bomb Torizo and
Spore Spawn (see STATUS).

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

## Implementation checklist

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
11. [x] **Late route skeleton (dev, fights skipped):** Phantoon → Gravity →
    Botwoon → Draygon → Ridley → Statues → Tourian → MB → Escape → Landing
    Site (`route_dev.py`, `maps/late_game_route_hops.json`).
12. [x] **Kraid defeated (dev):** Super spray; state `dev_kraid_defeated`.
13. [x] **Power Bombs (dev):** door-warp collect → `dev_power_bombs_collected`.
14. [x] **Ship route → Phantoon entry (dev):** `dev_phantoon_entry`.
15. [x] **A0** Full 22-leg hop runner from `full_route_hops.json` (Ceres null
    door → `0x896A`).
16. [x] **A1–A2** Progressive loadout + `full-tour` + `full-hybrid`
    (continuous Super prefix splice; bosses skipped; `developmentOnly`).
17. [~] **B1** Continuous Super collect done; farming→Big Pink→main shaft
    dev-proven; climb to PB + continuous still open.
18. [ ] **B2–B3** Natural ship/Norfair path rooms; path-priority room policies.
19. [ ] **B4** Boss fights (Kraid natural → Phantoon → … → MB) + escape.
20. [ ] **B5** Continuous dry run → credits video (M7/M8).

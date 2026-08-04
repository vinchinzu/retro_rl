# Full-room graph and problem catalog

## Result

The development catalog covers every room in the available editor export:

| Measure | Count |
|---|---:|
| Editor rooms / canonical room problems | 262 |
| Vanilla reference rooms | 261 |
| Editor-only unused rooms | 1 |
| Physical connections | 300 |
| Directed traversals | 583 |
| Bidirectional connections | 283 |
| Forward-only connections | 17 |
| Capability-aware completion legs | 22 / 22 |

The 261 vanilla rooms form one connected physical topology. The extra editor
room is unused and intentionally remains isolated. Each editor room gets one
canonical development problem with an entry, an exit or return objective,
item capabilities, door requirements, collision-grid waypoints, difficulty
tier, queue, state filename, policy filename, and report filename.

The generated files are `maps/full_room_graph.json` and
`maps/room_problems.json`. They are ROM-derived local artifacts and remain
gitignored; regenerate them from the source data with:

```bash
uv run python snes/super_metroid/scripts/export/room_problems.py
```

The default source locations are the sibling editor export and the game-local,
gitignored [`sm-json-data`](https://github.com/vg-json-data/sm-json-data) clone
at `refs/sm-json-data`. The latter is currently pinned at commit
`d49da689b2620aa1a4223ebf505d4b7791d88662`. Override either source with
`--editor-nav` and `--reference-root`, or with `SUPER_METROID_EDITOR_NAV` and
`SUPER_METROID_JSON_DATA`.

Of the 262 canonical problems, 157 have a connected air-cell waypoint path,
104 have entry/exit endpoints but require runtime movement or dynamic-block
planning, and the one unused editor room has no endpoints. Thus every real
room has a pre-calculated structural problem, while the catalog does not
pretend that collision connectivity alone solves every room.

## Evidence layers

The full graph is a research graph, not continuous-run acceptance evidence.
Its layers have different strengths:

1. The reference corpus supplies all physical connections and preserves true
   forward-only doors, sand pits, elevators, morph tunnels, and story markers.
2. The editor room export supplies room geometry, collision grids, enemies,
   items, and endpoint blocks.
3. The generator computes a static air-cell path and compresses it into
   waypoints. It does not solve jumps, physics, enemies, breakable blocks,
   resource floors, or dynamic room events.
4. Emulator room replay promotes one isolated policy to
   `verified_development_state`.
5. Only a no-state-load continuous run can promote the corresponding
   progression edge to accepted full-run evidence.

Door color and explicit flag requirements participate in route search. Local
locks are retained on edges for runtime work. Conditional escape locks do not
block ordinary traversal, and permanent one-way connections are never
silently reversed.

## Problem queues

The initial split is:

| Queue | Meaning | Current count |
|---:|---|---:|
| 0 | State and policy ready | 3 |
| 1 | Easy / small room | 67 |
| 2 | Standard traversal | 38 |
| 3 | Tough, scripted, or unresolved static geometry | 143 |
| 4 | Boss held for later | 11 |

Queue 3 combines 117 `tough` problems with 27 `late_special` problems, minus
the promoted Flyway policy. Two of the 69 easy rooms are also promoted to
queue 0. Counts are classifications, not estimates of emulator success.

List or execute work with:

```bash
uv run python snes/super_metroid/scripts/room/run_problem.py list --tier easy
uv run python snes/super_metroid/scripts/room/run_problem.py list --queue 1 --easiest-first --limit 20
uv run python snes/super_metroid/scripts/room/run_problem.py queue --limit 30
uv run python snes/super_metroid/scripts/export/room_work_queue.py
uv run python snes/super_metroid/scripts/room/run_problem.py bootstrap --queue 1 --max 5
uv run python snes/super_metroid/scripts/room/run_problem.py scaffold PROBLEM_ID
uv run python snes/super_metroid/scripts/room/run_problem.py run PROBLEM_ID --promote
uv run python snes/super_metroid/scripts/room/run_problem.py ready
uv run python snes/super_metroid/scripts/room/run_problem.py ready --run
```

**Easiest-first board:** `docs/routes/ROOM_WORK_QUEUE.md` (+ CSV/JSON). Ranks all
262 room problems by rough difficulty so small stations and low-enemy rooms
come before large shafts and bosses. Percent-complete focuses on classes 0–2
before investing in class 3 geometry or class 4 bosses. This board is
**practice metrics only** — not continuous evidence and not product next-work
(`docs/STATUS.md` + `docs/tasks/QUEUE.md`).

**Teleport fixtures:** `bootstrap` door-warps through the catalog **entry door**,
then settles Samus **just inside** that doorway (`method: doorway_natural`).
Do not mid-room teleport — segments start on a real door boundary so future
enemy/door RNG re-rolls can re-enter the same door (`--boot-idle-frames` is
recorded on `EntryContract`). Uses a controllable mid-game boot (default
`natural_post_spore_spawn`) rather than late full-loadout anchors that can
freeze input. Door pointers come from the reference connection graph
(`PhysicalEndpoint.door_ptr` / baked `entry.doorPtr`), not a parallel JSON map.
Teleport/run still require that fixture; continuous power-on evidence remains
separate.

`scaffold` creates an explicitly unverified starter policy using the
pre-calculated entry, exit, orientation, and static waypoints (frame budgets
from `staticPlan.pathBlocks`). Use `scaffold --all --output-dir PATH` to
materialize templates for the entire catalog without mixing them with curated
policies. A scaffold never enters queue 0 merely because its file exists; only
`run … --promote` (or `promote`) after a green report matching state/policy
sha256 marks `verified_development_state`.

The bulk command was exercised against this catalog and produced all 262
templates.

## Teleport workflow

Capture validates that a source snapshot is ordinary gameplay in the expected
room, rewrites it into the current stable-retro state format, and records a
provenance sidecar:

```bash
uv run python snes/super_metroid/scripts/room/run_problem.py capture \
  room_9879_from_9804_to_92fd \
  "/path/to/Flyway [from Bomb Torizo Room].state"
```

Teleport verifies the room after loading and can save a screenshot:

```bash
uv run python snes/super_metroid/scripts/room/run_problem.py teleport \
  room_9879_from_9804_to_92fd \
  --screenshot super_metroid/debug/room_clears/flyway_start.png
```

Run replays the compact JSON policy, requires a natural room-ID crossing, and
waits for ordinary gameplay in the target room:

```bash
uv run python snes/super_metroid/scripts/room/run_problem.py run \
  room_9879_from_9804_to_92fd
```

Reports go under `recordings/room_clears/`. States remain under
`custom_integrations/SuperMetroid-Snes/`. Both are development artifacts and
are explicitly excluded from continuous-run acceptance.

For `collect_*` objectives, a door crossing is not sufficient. The verifier
also requires the expected ammo capacity, energy/reserve capacity, equipment,
or beam delta. This rejected a First Missile attempt that reached the exit
without collecting the expansion; that policy was not promoted.

## Verified setup samples

Two queue-1 rooms and one extra simple traversal passed end to end on
2026-07-25:

| Problem | Result | Crossing | Settled |
|---|---|---:|---:|
| Green Brinstar Missile Station `0x9C89` → Fireflea `0x9C5E` | leave the one-screen station through its only door | frame 173 | frame 295 |
| Brinstar Map Room `0x9C35` → Pre-Map `0x9B9D` | leave the one-screen station through its only door | frame 173 | frame 292 |
| Flyway `0x9879` → Parlor `0x92FD` | hold left with bounded jump/shoot cycles until the door crossing | frame 293 | frame 424 |

All final states are ordinary gameplay in the expected target room. None of
the runs uses the progression assist or proves a continuous prefix; each
starts from an imported development snapshot.

## Full-game research sequence

The catalog expands the objective-level route into 23 anchors and 22
door-level legs:

```text
Ceres elevator → Ceres Ridley → Zebes landing → Morph Ball → First missiles
→ Bomb Torizo → Spore Spawn → Spore Spawn Super Missiles
→ Early Power Bombs → Kraid → Varia Suit → Speed Booster → Ice Beam
→ Phantoon → Gravity Suit → Botwoon → Draygon → Ridley
→ Golden Four statues → Tourian elevator → Mother Brain
→ Tourian escape 4 → Landing Site escape finish
```

All 22 legs currently have a capability-aware topology path. This means there
are no known room-graph gaps; it does not mean the in-room policies or overall
sequence are solved.

The immediate post-Spore route is:

```text
Spore Spawn Super Room 0x9B5B
→ Spore Spawn Farming Room 0xA0A4
→ Big Pink 0x9D19
→ Pink Brinstar Power Bomb Room 0x9E11
```

After naturally collecting Supers and Power Bombs, the proposed Kraid leg is:

```text
0x9E11 → 0x9D19 → 0x9E52 → 0x9FBA → 0xA253 → 0xA3DD
→ 0xA408 → 0xCF54 → 0xCEFB → 0xCF80 → 0xA6A1 → 0xA471
→ 0xA4DA → 0xA521 → 0xA56B → 0xA59F
```

Inspect any capability-aware route directly:

```bash
uv run python snes/super_metroid/scripts/room/run_problem.py route \
  0x9B5B 0x9E11 \
  --capability morph_ball \
  --capability bombs \
  --capability missiles \
  --capability spore_spawn_defeated \
  --capability super_missiles
```

# Architecture — A Link to the Past (`alttp/`)

Agent-facing map of **where code lives** and **which contracts are source of
truth**. Aligns with root `AGENTS.md`, local `AGENTS.md`, and `STATUS.md`.

## Goals

1. **Clean package boundaries** — opening route vs gauntlet vs romhack.
2. **Truthful continuous claim** — only natural-entry / continuous edges.
3. **Multi-truth anchors** — RAM + map/Yaze + visual; route ≠ approach ≠ trigger.
4. **Graph + Segment contracts** — escape graph drives plans; scripts adapt to
   Segment; work queue prioritizes continuous-spine blockers.
5. **Efficient observation** — sparse/selective RAM (`AlttpSnapshot`), not full
   bank dumps on hot loops.

## Layer map

```text
scripts/                          CLI entrypoints (boot, segments, export)
        │
session.py                        AlttpSession façade (snapshot, caps, segments)
        │
opening_route/                    Continuous trunk (ownership)
  escape_graph.py                 Capability graph grounds → Sanctuary
  segment.py                      Segment / natural-entry contract + registry
  runner.py                       Shared run_phases + run_from_state (pre-room-engine)
  anchors.py                      Multi-truth anchors + measured constants + tip resolve
  castle_to_sword.py              Live: grounds → uncle sword
  secret_entrance_clear.py        Live: sword → south → stairs outdoor clear
  pocket_to_main_hall.py          Live: pocket bush-cut → main door → 0x61
  full_tip.py                     Clean power-on composition through verified room_50 tip
  castle_dungeon.py               Typed continuous prefix: 0x61 → 0x60 → 0x50
  main_hall_to_zelda.py           Planned scaffold + helpers only (not in registry)
  room_engine.py                  Generic clear + door exit (ok=True on edge)
  escort_to_sanctuary.py          Planned scaffold: escort → Sanctuary
  catalog.py + data + validate    z3-backed opening catalog
  work_queue.py + work_queue_data.yaml  Discover/rank + curated practice data
        │
primitives.py + route_report.py   Preferred low-level control + evidence shape
room_map.py                       maps/*.json load/save (geometry schema)
room_sense.py                     Sprite boxes, edge detect, overlay
maps/room_XX.json                 Geometry authority (doors / points / clear)
startup.py + overworld.py         Boot / OW BFS
ram.py                            Sparse snapshot (gameplay authority)
        │
gauntlet/                         Combat experiments (shell; not continuous)
romhack/                          Editor/asset experiments (shell)
```

### Ownership split

| Package | Owns | Does not own |
|---------|------|----------------|
| `alttp/` root | RAM, primitives, startup, overworld, session, paths, game | Continuous route scripts |
| `alttp/opening_route/` | Escape graph, segments, catalog, work queue, live hops | Gauntlet RL, romhack sprites |
| `alttp/gauntlet/` | Arena/combat experiments (when revived) | Sanctuary continuous claims |
| `alttp/romhack/` | Editor/asset experiments (when revived) | Opening-route evidence |

**Layer rule (Sanctuary finish):**

| Layer | Owns | Does not own |
|-------|------|----------------|
| Graph | Coarse capability hops + verification ladder | Every B1 door as a node |
| Segment | Multi-room measured phases (`RoutePhaseResult`) | Alternate-path continuous claims |
| Work queue | Isolated practice for **open continuous-spine blockers** | Second route truth |
| Anchors | Measured approach/trigger windows + tip resolution | Filename heuristics |

Import continuous trunk as ``alttp.opening_route.*`` (no root re-export shims).

### Source of truth

| Concern | Authority |
|---------|-----------|
| Live position / inventory / room | `alttp.ram.AlttpSnapshot` (stable-retro RAM) |
| Continuous progress | `opening_route.escape_graph` edges with `verification=continuous` |
| Capability plan | `plan_escape_to_sanctuary` / `continuous_spine_legs` (single hop table; path tags) |
| Segment play + evidence | `opening_route.segment` + `route_report.SegmentResult` (entry enforced; `report_kind` / `diagnostics`) |
| Env typing | `startup.BootEnv` protocol (`reset` / `close` / `step` / `get_ram`) |
| Multi-truth anchors + tip node | `opening_route.anchors` (`resolve_continuous_tip_node`) |
| Measured room geometry | `maps/room_XX.json` via `room_map.load_room_map` |
| Approach/trigger windows | `anchors.py` (door approach derived from map; no copy) |
| z3 / Yaze labels | Association only — **not** screen coordinates |
| Save-state practice order | `opening_route.work_queue` + `work_queue_data.yaml` → `docs/routes/ROOM_WORK_QUEUE.md` |

### Continuous tip (2026-08-01)

Verified continuous spine:

`castle_grounds` → `room_55_uncle` → `room_55_sword` → `room_55_south` →
`courtyard_secret_pocket` → `room_61` (main hall) → `room_60` → `room_50`
(NW chamber)

`full_tip.run_to_verified_tip` owns the one-environment power-on composition;
`castle_dungeon_prefix` owns its measured west/north room-edge suffix.

Next planned hop: **after 0x50 → Zelda cell → escort → Sanctuary**
(map seeds for 0x01/51/52/62 and B1 0x71–0x82 under `maps/`).

Alternate internal key/shutter path remains on the graph (`path: internal_key`)
for practice only — **not** the default Sanctuary plan and **not** work-queue
primary blockers.

### Segment contract

```text
Segment:
  id
  entry (room/screen + inventory + optional anchors)
  exit  (acceptance keys + graph node + verification)
  play(env) → SegmentResult / SegmentEvidence
  play_checked: enforces entry (phase entry_rejected if mismatch)
```

Registered segments:

- continuous: `castle_to_sword`, `sword_to_secret_entrance_clear`,
  `pocket_to_main_hall`
- continuous power-on composition through `room_50`: `full_tip.run_to_verified_tip`
- continuous first-dungeon prefix: `castle_dungeon_prefix` (`0x61→0x60→0x50`)
- planned scaffold: `escort_to_sanctuary` (Zelda rescue not yet a segment)

Natural-entry rule: a hop is route-ready only from the real predecessor
continuous state (no privileged warps in published evidence).

### Session façade

```python
from alttp.session import bind_env

session = bind_env(env, source="natural_boot")
session.snapshot()           # selective RAM
session.capabilities()       # escape-graph tokens
session.anchor_report()      # multi-truth matches
session.continuous_tip_node()  # anchors.resolve_continuous_tip_node
session.play_segment("castle_to_sword")
session.plan_to_sanctuary()
```

### Primitives (preferred low-level API)

Use `alttp.primitives`: `settle_control`, `run_script`, `move_to` / `move_path`,
`fight_nearby`, `interact_until`. Prefer these over new open-loop mega-macros.
Promote to `retro_harness.adventure` only after a second game adopts them.

### Docs

| Doc | Role |
|-----|------|
| `docs/STATUS.md` | Verified facts + maturity gate |
| `docs/plan.md` | Future work |
| `docs/TRIGGER_HANDOFF.md` | Remaining trigger/hitbox problems |
| `docs/routes/ROOM_WORK_QUEUE.md` | Save-state practice queue (tip = room 0x61) |
| `docs/ram_map.md` | WRAM field notes |
| `docs/Z3_JSON_DATA.md` | Optional local z3 refs |

### Extension recipe (next continuous hop)

Prefer **room engine** for B1 doors (low agent context) — `docs/ROOM_ENGINE.md`:

1. Measure → write `maps/room_XX.json` (points + doors + path). Do not invent.
2. `scripts/room_engine.py show|run` until isolated green.
3. Graph: expand node when isolated; promote `planned` → `isolated` →
   `natural_entry` → `continuous` only with evidence.
4. Add the typed edge to `castle_dungeon.DungeonRoomEdge` when it composes
   with the first-dungeon prefix; keep a thin aggregate only when a multi-room
   acceptance needs spine registration.
5. Anchors for approach/trigger windows; geometry stays in JSON.
6. STATUS facts; never claim Zelda until `$F3CC == 1` on real RAM.

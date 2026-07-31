# Architecture — A Link to the Past (`alttp/`)

Agent-facing map of **where code lives** and **which contracts are source of
truth**. Aligns with root `AGENTS.md`, local `AGENTS.md`, `STATUS.md`, and
`ARCHITECTURE_AND_CLEANUP_PLAN.md` (ALTTP P1 cleanup).

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
  anchors.py                      Multi-truth route/approach/trigger anchors
  castle_to_sword.py              Live: grounds → uncle sword
  sword_to_zelda.py               Live: sword → south → stairs outdoor clear
  pocket_to_main_hall.py          Live: pocket bush-cut → main door → 0x61
  catalog.py + data + validate    z3-backed opening catalog
  work_queue.py                   Save-state Sanctuary work queue
        │
primitives.py + route_report.py   Preferred low-level control + evidence shape
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

Compat shims at the old top-level module names
(`alttp.escape_graph`, `alttp.castle_to_sword`, …) re-export
`opening_route` for existing imports.

### Source of truth

| Concern | Authority |
|---------|-----------|
| Live position / inventory / room | `alttp.ram.AlttpSnapshot` (stable-retro RAM) |
| Continuous progress | `opening_route.escape_graph` edges with `verification=continuous` |
| Capability plan | `plan_escape_to_sanctuary` / `continuous_spine_legs` |
| Segment play + evidence | `opening_route.segment` + `route_report.SegmentResult` |
| Multi-truth anchors | `opening_route.anchors` |
| z3 / Yaze labels | Association only — **not** screen coordinates |
| Save-state practice order | `opening_route.work_queue` + `docs/routes/ROOM_WORK_QUEUE.md` |

### Continuous tip (2026-07-30)

Verified continuous spine:

`castle_grounds` → `room_55_uncle` → `room_55_sword` → `room_55_south` →
`courtyard_secret_pocket` → `room_61` (main hall)

Next planned hop: main hall B1 → Zelda cell → escort → Sanctuary.

Alternate internal key/shutter path remains on the graph for the work queue
but is **not** the default Sanctuary plan.

### Segment contract

```text
Segment:
  id
  entry (room/screen + inventory + optional anchors)
  exit  (acceptance keys + graph node + verification)
  play(env) → SegmentResult / SegmentEvidence
```

Registered segments: `castle_to_sword`, `sword_to_secret_entrance_clear`,
`pocket_to_main_hall`. Natural-entry rule: a hop is route-ready only from
the real predecessor continuous state (no privileged warps in published
evidence).

### Session façade

```python
from alttp.session import bind_env

session = bind_env(env, source="natural_boot")
session.snapshot()           # selective RAM
session.capabilities()       # escape-graph tokens
session.anchor_report()      # multi-truth matches
session.play_segment("castle_to_sword")
session.plan_to_sanctuary()
```

### Primitives (preferred low-level API)

Use `alttp.primitives`: `settle_control`, `run_script`, `move_to` / `move_path`,
`fight_nearby`, `interact_until`. Prefer these over new open-loop mega-macros.
Promote to `adventure_common` only after a second game adopts them.

### Docs

| Doc | Role |
|-----|------|
| `docs/STATUS.md` | Verified facts + maturity gate |
| `docs/plan.md` | Future work |
| `docs/TRIGGER_HANDOFF.md` | Remaining trigger/hitbox problems |
| `docs/routes/ROOM_WORK_QUEUE.md` | Save-state practice queue |
| `docs/ram_map.md` | WRAM field notes |
| `docs/Z3_JSON_DATA.md` | Optional local z3 refs |

### Extension recipe (next continuous hop)

1. Measure approach + trigger with multi-truth anchors (do not invent coords).
2. Pure play function in `opening_route/` using `primitives`.
3. Graph edge: start `planned` or `isolated`; promote to `natural_entry` then
   `continuous` only with evidence.
4. Register `ScriptSegment` in `segment.py`.
5. Drive work queue from the continuous-spine blocker only.
6. Update STATUS with RAM facts + recording paths; never claim Zelda until
   `$F3CC == 1` on real RAM.

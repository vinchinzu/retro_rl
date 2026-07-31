# Architecture — Super Metroid

Maps the live package layout for continuous KPDR (and full-game spine)
work. Aligns with root `AGENTS.md`, local `AGENTS.md`, `STATUS.md`, and
`plan.md`. This is the agent-facing map of **where code lives** and
**which contracts are source of truth**.

## Goals

1. **Continuous spine** — power-on → ending with natural entries, assist
   contract only, zero progression writes.
2. **Graph + Segment contracts** — progression graph and hop/segment
   interfaces declare progress; controllers/policies implement them.
3. **Low fragility** — one tip-extension recipe; no new `start_to_*.py`
   scripts; natural-entry evidence only.
4. **Efficiency** — avoid full-bank WRAM copies on hot loops; prefer
   selective peeks + session-cached state.

## Layer map

```text
scripts/record|verify|probe|export|room   CLI entrypoints
        │
routes/continuous.py + catalog.py         tip registry + hop composition
routes/runtime.py                         RouteSession, integrity, reports
routes/segment.py                         Segment / HopExecutor contracts
        │
routes/kpdr/*.py                          pure room controllers (no env ownership)
combat/*.py                               boss fight policies (after natural entry)
policy.py + policies/**                   JSON raw-button PolicySegments
        │
progression.py                            RoomNode / DoorEdge / milestones / graphs
ram.py + assist.py                        state parse + resource assists
        │
rooms/*                                   isolated practice (EntryContract, queue)
dev/*                                     door-warp topology (developmentOnly)
legacy/*                                  frozen vision/RL remnants
```

### Runtime entrypoints

| Path | Role |
|------|------|
| `routes/continuous.py` | Power-on chain; `play_*` / `run_*` / `run_to` |
| `routes/catalog.py` | `ContinuousTip`, split tuples, `NamedRoute` |
| `scripts/record/continuous.py` | One CLI for all continuous tips (`--to`) |
| `scripts/verify/` | Replay/verify accepted baselines |
| `scripts/probe/` | Dev probes (KPDR pure, kraid combat, route tour) |
| `scripts/export/` | Path board, KPDR tracker, room queue, graphs |
| `scripts/room/run_problem.py` | Isolated room practice bootstrap/run |

### Navigation / progression stack

| Piece | Role |
|-------|------|
| `progression.RoomProgressionGraph` | Rooms, directed edges, capability BFS |
| `DoorEdge.verification` | `unverified` / `controller_dev` / `continuous` |
| `ProgressCondition` / `ProgressionMilestone` | Live RAM stop predicates |
| Staged graphs | `START_TO_MORPH` ⊂ … ⊂ `START_TO_WAREHOUSE` ⊂ `START_TO_HIJUMP` ⊂ `START_TO_KRAID` ⊂ `START_TO_VARIA` |
| `routes/catalog.CONTINUOUS_TIPS` | CLI tip order; `DEFAULT_CONTINUOUS_TIP` is furthest integrity-green tip |
| `routes/continuous.RouteHop` + `play_hops` | Ordered controller legs after Supers |

**Source of truth for continuous progress:** graph edges + tip hop tables +
integrity report (`0` state loads, `0` progression writes, required splits).
The path board (`maps/path_room_board.json`) is **topology**, not KPDR order.

### Policy vs controller vs combat

| Layer | Module(s) | Contract |
|-------|-----------|----------|
| High | graph + tip milestones | inventory-aware path / tip id |
| Mid | `RouteHop` / `ControllerSegment` / `PolicySegment` | entry → play → exit evidence |
| Low | `routes/controller_common` | `hold`, `wait_until`, morph/weapon, door exit |
| Boss | `combat/*` via `BossStrategy` / `BossSegment` | only after natural boss-room entry |

- **Controllers** (`routes/kpdr/`): pure movement/combat on
  `ControllerSession`; registered in `KPDR_SEGMENTS`.
- **Policies** (`policy.py` + `policies/`): hash-pinned raw button
  sequences with `StateRequirement` entry/exit checks.
- **Combat**: approach controllers enter the room; fight policies clear
  the boss (Route / Approach / Trigger split). Full pipeline:
  [`docs/BOSS_PIPELINE.md`](BOSS_PIPELINE.md) — catalog → natural entry →
  strategy → optional structured RL → continuous promote.

Both controllers and policies adapt to the same **Segment** surface in
`routes/segment.py` (thin wrappers over existing callables). Boss
strategies adapt via `combat.protocol.BossSegment`.

### RAM / state, assists, recording

| Piece | Role |
|-------|------|
| `ram.SuperMetroidState` | Compact nav/inventory/boss vector |
| `parse_env_state(..., mode=)` | `nav` (low WRAM) vs `full` (bank $7E) |
| `read_wram_u8/u16`, `peek_wram` | Selective peeks without 128 KiB copy |
| `StateCache` | Optional per-frame parse reuse |
| `assist.py` | Energy/ammo restore only; telemetry |
| `docs/ASSIST_CONTRACT.md` | Forbidden progression/capacity writes |
| `routes/runtime` | Continuous integrity + report |
| `video.py` / recordings | Optional video; machine JSON is authority |
| `room_timer.py` | Opt-in per-room dwell timing |

**Trap:** `env.get_ram()` is fine below `$7E:2000`; event/boss flags at
`$7E:D820+` need bank `$7E` (`mode="full"` or peeks).

### Maps / room catalog / work queue

| Piece | Role |
|-------|------|
| `rooms/room_graph.py` | sm-json-data topology |
| `rooms/room_catalog.py` | 262 problems export |
| `rooms/segment_contract.py` | Doorway-natural **practice** entry contract |
| `rooms/work_queue.py` | Easiest-first + continuous-spine blockers |
| `rooms/entry_bootstrap.py` | Door-warp fixtures for practice |
| `maps/*` | Generated graphs/trackers (mostly gitignored) |
| `docs/research/PATH_ROOM_BOARD.md` | 107 rooms / 199 hops topology |
| `docs/routes/KPDR_TRACKER.*` | Continuous KPDR status board |

### Legacy vs active

| Active | Frozen (`legacy/`) |
|--------|---------------------|
| `routes/`, `rooms/`, `combat/`, `ram`, `assist`, `policy`, `progression` | `legacy/models.py`, `legacy/visual_models.py` |
| Feature-vector boss scaffolding in `combat/` | Imported vision BC/PPO under `models/imported/` |
| Continuous JSON policies in `policies/early_game/` | Vision BC parked until gold |

Top-level `models.py` / `visual_models.py` are **compat shims** only.
Do not add new imports of legacy vision policies into continuous routes.

### Tooling / editor

- Shared SNES facade: `retro_harness.snes` (`GameSpec`, `StartupPlan`, named actions).
- Editor integration: `retro_harness.editor` + game registration via
  `retro_harness.editor_launcher`.
- Probe/dev modules under `dev/` are **not** continuous evidence
  (`developmentOnly` in reports).

## Continuous tip extension recipe

Do **not** add a new `start_to_*.py` script.

1. Pure controller in `routes/kpdr/` (+ `KPDR_SEGMENTS`).
2. Graph rooms/edges/milestones in `progression.py` (verification starts
   as `controller_dev`, promote to `continuous` after evidence).
3. Split ids + `ContinuousTip` + `NamedRoute` in `catalog.py`.
4. `RouteHop` rows + thin `play_*` / `run_post_supers_tip` in
   `continuous.py`.
5. Wire tip in `run_to()` + `register_continuous_segments`.
6. Record: `scripts/record/continuous.py --to <tip> --no-video`.
7. Promote STATUS / tracker only after integrity green.

## Segment / hop contracts

See `routes/segment.py`:

- **`Segment` Protocol** — `id`, optional entry/exit predicates, `play(session)`.
- **`ControllerSegment` / `PolicySegmentAdapter`** — adapt existing callables.
- **`HopExecutor`** — run `RouteHop` sequences with room asserts.
- **`ContinuousSession`** — thin facade: `run_to`, `execute_hop`,
  `current_state`, `verify_milestone`, `progress_vector`.

Practice uses a **different** contract: `rooms.segment_contract.EntryContract`
(doorway-natural bootstrap). Do not conflate with continuous power-on hops.

## Hierarchical control (product)

```text
High   RoomProgressionGraph + ContinuousTip sequence
Mid    RouteHop / PolicySegment / BossSegment / composed packages
Low    controller_common + combat.primitives
Boss   combat.* after natural entry only (see BOSS_PIPELINE.md)
```

Composed packages (e.g. `play_warehouse_hijump_kraid`) stay valid for
probes; continuous tips prefer **one hop per door** for split clarity
while still registering all intermediate graph edges for integrity.

## Efficiency notes

1. Continuous `RouteSession` already parses low WRAM via `get_ram()`;
   use `mode="full"` only when event/boss bits matter.
2. Controllers with tight `wait_until` loops should prefer peeks or
   cached state over repeated `parse_env_state` full-bank copies.
3. Manifest-driven recording + `--no-video` for long dry-runs.
4. Promote shared primitives only after a **second consumer**
   (`adventure_common` / `platformer_common` / `snes_oneshot`).

## Package boundaries (target)

```text
super_metroid/
  ram.py assist.py policy.py progression.py paths.py room_timer.py video.py
  routes/          continuous product spine + kpdr controllers
  rooms/           isolated practice product
  combat/          boss strategies
  scripts/         CLI
  docs/            STATUS, plan, ARCHITECTURE, contracts, boards
  maps/            generated artifacts
  policies/        JSON segments
  custom_integrations/  emulator integration + anchors
  dev/             developmentOnly probes
  legacy/          frozen vision/RL
  tests/
```

## Related docs

- Local rules: [`../AGENTS.md`](../AGENTS.md)
- Gate / verified tip: [`STATUS.md`](STATUS.md)
- Forward work: [`plan.md`](plan.md)
- Assists: [`ASSIST_CONTRACT.md`](ASSIST_CONTRACT.md)
- KPDR board: [`routes/ROUTE_KPDR.md`](routes/ROUTE_KPDR.md)
- Path topology: [`research/PATH_ROOM_BOARD.md`](research/PATH_ROOM_BOARD.md)
- Full-run process: [`../../snes_oneshot/docs/FULL_RUN_PROCESS.md`](../../snes_oneshot/docs/FULL_RUN_PROCESS.md)

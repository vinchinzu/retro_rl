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
3. **Low fragility** — one tip-extension recipe; no new per-tip runner modules
   scripts; natural-entry evidence only.
4. **Efficiency** — avoid full-bank WRAM copies on hot loops; prefer
   selective peeks + session-cached state.

## Layer map

```text
scripts/record|verify|probe|export|room   CLI entrypoints
        │
routes/continuous.py + catalog.py         tip registry + hop composition
routes/runtime.py                         RouteSession, integrity, reports
routes/segment.py                         practice Segment adapters only
        │
routes/kpdr/*.py                          pure room controllers (no env ownership)
combat/*.py                               boss fight policies (after natural entry)
policy.py + policies/**                   JSON raw-button PolicySegments
        │
progression/                              RoomNode / DoorEdge / milestones / graphs
ram.py + assist.py                        state parse + resource assists
        │
rooms/*                                   isolated practice (EntryContract, queue)
dev/*                                     door-warp topology (developmentOnly)
legacy/*                                  frozen vision/RL remnants
```

### Runtime entrypoints

| Path | Role |
|------|------|
| `routes/continuous.py` | Super+ harness, `run_to`, public re-exports |
| `routes/early_continuous.py` | Morph→supers play/run (seeds + policies) |
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
| Staged graphs | `START_TO_MORPH` ⊂ … ⊂ `START_TO_VARIA` ⊂ `START_TO_SPEED` (Business return + K4 scaffold) |
| `routes/catalog.CONTINUOUS_TIPS` | CLI tip order; `DEFAULT_CONTINUOUS_TIP` is furthest integrity-green tip (`bat_cave`) |
| `routes/tips.play_hops` + `SpineHop` | Ordered controller legs (only hop runner) |
| `source_states.py` | Code twin of `SOURCE_STATES.md` — pure entry fingerprints |

**Source of truth for continuous progress:** graph edges + tip hop tables +
integrity report (`0` state loads, `0` progression writes, required splits).
The path board (`maps/path_room_board.json`) is **topology**, not KPDR order.

### Policy vs controller vs combat

| Layer | Module(s) | Contract |
|-------|-----------|----------|
| High | graph + tip milestones | inventory-aware path / tip id |
| Mid | `SpineHop` / `ControllerSegment` / `PolicySegment` | entry → play → exit evidence |
| Low | `routes/controller_common` | hybrid primitives (below) |
| Boss | `combat/*` via `BossStrategy` / `BossSegment` | only after natural boss-room entry |

- **Controllers** (`routes/kpdr/`): pure movement/combat on
  `ControllerSession`; registered in `KPDR_SEGMENTS`.
- **Policies** (`policy.py` + `policies/`): hash-pinned raw button
  sequences with `StateRequirement` entry/exit checks. **Keep raw JSON for
  timing-critical segments**; compose primitives around hard slices.
- **Combat**: approach controllers enter the room; fight policies clear
  the boss (Route / Approach / Trigger split). Full pipeline:
  [`docs/BOSS_PIPELINE.md`](BOSS_PIPELINE.md) — catalog → natural entry →
  strategy → optional structured RL → continuous promote.

**Hybrid primitives** (`routes/controller_common.py`) reduce blank-JSON
poking without dropping evidence:

| Primitive | Role |
|-----------|------|
| `wait_until` / `wait_requirement` | Idle until pred / `StateRequirement` |
| `require_state` | Fail fast with requirement failure strings |
| `hold_until` | Hold buttons while polling |
| `wait_ordinary_room` | Multi-truth settle (room + phase + optional x/y) |
| `play_run_shoot_exit` / `traverse_door` | Horizontal door exit (+ entry window) |
| `collect_item_mask` | Wait for PLM item bit |
| `ensure_morph` / `select_weapon` | Pose / weapon helpers |

**Graph-driven next hop** (`progression.RoomProgressionGraph`):

- `outgoing(room, capabilities=, verification=)`
- `suggest_edges(room, prefer=, exclude_verifications=)` — **single** suggest surface
- `suggest_next_hops` / `suggest_pure_work` — thin wrappers over `suggest_edges`
- `path_summary(src, dst, caps, min_verification=)` — **single** path summary
- `path_verification` / `pure_gate` — thin wrappers (stable dict shapes)
- `VERIFICATION_RANK` — one rank table for path + pure gate
- `capabilities_from_state(state)` — live RAM → capability tokens

**Offline dwell ranking** (after continuous green, before extending):

```bash
uv run python snes/super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/varia.json --top 15
uv run python snes/super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/varia.json --reasons --top 20
```

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
| `rooms/room_graph.py` | sm-json-data **practice** topology (never continuous evidence) |
| `rooms/room_catalog.py` | 262 problems export |
| `rooms/segment_contract.py` | Doorway-natural **practice** entry contract |
| `rooms/work_queue.py` | Easiest-first practice board (dual-track metrics only) |
| `rooms/entry_bootstrap.py` | Door-warp fixtures for practice |
| `progression/` + `routes/kpdr/spine.py` / `early_spine.py` | **continuous** product graph + SpineHop orchestration |
| `maps/*` | Generated graphs/trackers (mostly gitignored) |
| `docs/research/PATH_ROOM_BOARD.md` | 107 rooms / 199 hops topology |
| `docs/routes/KPDR_TRACKER.*` | Continuous KPDR status board |

**Dual graph (intentional):** sm-json-data practice topology and continuous
`RoomProgressionGraph` stay separate. Practice greens never promote continuous
verification; continuous tips never import sm-json path planning.

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

Sub-agent process (pure-first, stabilize waves, residual schema):
[`docs/tasks/PROCESS.md`](tasks/PROCESS.md). Source states:
[`docs/SOURCE_STATES.md`](SOURCE_STATES.md).

1. Pure controller in `routes/kpdr/` (+ `KPDR_SEGMENTS`).
   **Gate:** pure-green from a continuous-like source state before step 2.
2. Graph rooms/edges/milestones in `progression/data.py` (verification starts
   as `controller_dev`, promote to `continuous` after evidence). Still hand-
   maintained this pass; spine does not yet emit DoorEdges.
3. Append `SpineHop` (+ `TipSegment` if new tip) in `routes/kpdr/spine.py`.
   Hop groups, `TipSpec` rows, and Super+ split suffixes are **generated**
   (`routes/tips.py` + `routes/kpdr/hops.py`). First Super+ tip parents to
   `supers`.
4. Thin `ContinuousTip` + `NamedRoute` metadata in `catalog.py`
   (capability flags: `supports_room_timing` / `supports_unlimited_energy` /
   `supports_checkpoint` — never hard-coded `run_to` allowlists).
5. `run_to()` dispatches **all** tips from the unified `TipSpec` table.
   (Steps 2–5 may be one executor card **after** pure green; keep integrity
   judgment with the planner.)
6. Record: `scripts/record/continuous.py --to <tip> --no-video`
   (optional `--state-output` for integrity-green checkpoints only).
   **Stabilize:** if live spine knobs changed in a prior stress wave, re-record
   before stacking more interacting knobs.
7. Promote STATUS / tracker only after integrity green (planner; Flash may
   only propose via `SM-ROLLUP-STATUS`).

Successful pure+continuous sequences should be promoted into
`routes/controller_common` primitives with unit tests before the next similar
geometry card.

### Controller lineage rule

When a hop has two natural entry poses (e.g. Warehouse left elevator vs
right Zeela-ledge return), **prefer separate segment callables or an
explicit entry mode chosen once at hop start**. Do not keep discovering
lineage inside the mid-frame loop via magic thresholds (`samus_x > 400`)
or ad-hoc mid-climb success escapes unless the phase is named and
documented. Continuous composition may still chain both lineages.

## Segment / hop contracts

### Ownership (continuous vs adapters)

**Live continuous composition is `SpineHop` + `TipSpec`**, not the
Segment protocol stack:

| Surface | Role |
|---------|------|
| `routes/tips.py` `TipSpec` + `play_hops` / `play_tip` / `run_tip` | **Source of truth** for all continuous tips; `run_to` dispatches here |
| `routes/kpdr/spine.py` (`spine_types` / `spine_hops` / `tip_segments`) | Product hop order + Super+ tip metadata; DoorEdges generate; public API re-exported from `spine` |
| `routes/kpdr/hops.py` | Builds Super+ `TipSpec` rows + named hop groups from the spine |
| `routes/segment.py` | **Practice / tests only** — not the continuous tip runner |
| `combat/protocol.py` `BossStrategy` / `BossSegment` | Boss fight adapters after natural entry |

Continuous hop execution is **only** `tips.play_hops` (on `SpineHop` /
`TipSpec.hops`). Do not add a second hop runner in `segment.py` or elsewhere.

See `routes/segment.py` (tests / practice adapters only):

- **`Segment` Protocol** — `id`, `play(session)`.
- **`ControllerSegment` / `PolicySegmentAdapter`** — adapt KPDR callables and
  hash-pinned policies.
- **`segment_from_kpdr`** — build a `ControllerSegment` from the KPDR registry.

Practice uses a **different** contract: `rooms.segment_contract.EntryContract`
(doorway-natural bootstrap). Do not conflate with continuous power-on hops.

## Hierarchical control (product)

```text
High   RoomProgressionGraph + ContinuousTip sequence
Mid    TipSpec / SpineHop / PolicySegment / BossSegment
Low    controller_common + combat.primitives
Boss   combat.* after natural entry only (see BOSS_PIPELINE.md)
```

Composed packages (e.g. `play_warehouse_hijump_kraid`) stay valid for
probes; continuous tips prefer **one hop per door** for split clarity
while still registering all intermediate graph edges for integrity.

## Efficiency & code plan (whole-game length)

Long continuous runs will grow past multi-hour frame counts. Keep practice
Segment adapters when useful; continuous hop execution stays on
`tips.play_hops` only. Make the spine cheaper to extend and run. Full
prioritization lives in [`plan.md`](plan.md); this section is the
architecture map for those workstreams.

### 1. Selective RAM + StateCache enforcement (highest leverage)

| Prefer | Avoid in hot loops |
|--------|--------------------|
| `read_wram_u8` / `read_wram_u16` / `peek_wram` | Repeated full-bank copies |
| `StateCache` (per-frame reuse) | Uncached `parse_env_state(..., mode="full")` every wait tick |
| `parse_env_state(..., mode="nav")` when a struct is needed | Accidental default full parse in controllers |

Continuous `RouteSession` already parses low WRAM via `get_ram()`; use
`mode="full"` only when event/boss bits matter.

**Planned enforcement:**

- Base controller helper / decorator (or lightweight linter) that forces
  cache/peek inside `wait_until` / settle / climb loops.
- Profile frame time on long `--to varia` and future full runs; optionally
  report WRAM-copy rate.
- Controllers must not call full-bank parse on every frame of a tight loop.

### 2. Declarative continuous composition (**unified TipSpec**)

**Today:** one `TipSpec` table in `routes/tips.py` for morph → bat_cave.
Super+ tips are spine-driven (`parent_tip_id` + `SpineHop` delta; first Super+
parents to `supers`). Early tips share the same runner: `assist_mode` +
`final_conditions_fn` plugins (no `custom_run`). `run_to` → `run_to_tip` →
`run_tip`. Named hop tables live on `routes/kpdr/hops` (not re-exported from
`continuous`). Super+ `play_<tip>` / `run_<tip>` aliases still bind on
`continuous` for the segment registry. Capability flags on `ContinuousTip`
gate optional kwargs — no tip-id allowlists.

| Work item | Intent | Status |
|-----------|--------|--------|
| Tip-extension scaffold script | Stub + residual + printed checklist | **landed** (`scaffold_tip.py`) |
| Unified TipSpec | One type for early + Super+; no Early/PostSupers split | **landed** (`routes/tips.py`) |
| SpineHop-only hops | Delete RouteHop projection layer | **landed** (no RouteHop alias) |
| Shared `play_hops` | One hop runner | **landed** (`tips.play_hops`) |
| Hop tables out of continuous | `routes/kpdr/hops.py` | **landed** |
| Checkpoint on `ContinuousTip` | `supports_checkpoint` like room timing / energy | **landed** |
| Early custom_run fold-in | Same generic runner as Super+ | **landed** (assist + condition plugins) |

### 3. Source-state & pure-probe diagnostics

Index: [`SOURCE_STATES.md`](SOURCE_STATES.md) + code twin
`source_states.py`. Process: [`tasks/PROCESS.md`](tasks/PROCESS.md).

| Work item | Intent | Status |
|-----------|--------|--------|
| Fingerprint validation | Room + optional pose/x/y on pure load | **landed** |
| `suggest-source` / `--expect-room` / pin JSON | Fail loud on wrong source | **landed** |
| Provenance fields | Command, parent continuous tip, capabilities | **open** |
| Pure RED auto-capture | Short video clip + PLM/door RAM snapshot | **open** |
| Dispatch source suggest | Card schema → recommended `--source` pre-dispatch | **open** |

### 4. Primitive library + promotion discipline

Grow shared helpers in layers:

| Layer | Home | Owns |
|-------|------|------|
| Generic session helpers | `routes/controller_common.py` | hold/wait/morph/door traverse, `WallJumpTiming` |
| Policy-driven micro-skills | `routes/skills/` | geometry, walljump, runway, knockback escape, Super-door pressure |
| Hop policies | `routes/skills/policies/` | room geometry + frame budgets for one hop (e.g. `bubble_to_bat`) |
| Product compose | `routes/kpdr/<hop>.py` | thin `play_*` that wires skills + policy (hop-named modules) |
| Continuous graphs | `progression/stages/` | staged rooms/edges/milestones → `*GRAPH` via thin `data.py` |

Bubble Mountain pilot: skills under `routes/skills/`; product hop
`routes/kpdr/to_bat_cave.py` (`play_bubble_to_bat_cave`). Prefer **hop-named**
product modules and **skill-named** libraries over room megamodules.

Promote into `controller_common` / `skills` **after a second consumer** with
unit tests + pure evidence. Combat primitives under `BossStrategy` follow the
same rule.

**Lineage / stack hygiene:** Warehouse `entry_mode` + shared
`_open_warehouse_stack(face=)`; Zeela reverse has named sub-phases (shotblock
clear / wall replant / wall-spin climb) and continuous
docs. Hop tables live in `routes/kpdr/hops.py`. Prefer `wait_ordinary_room`
handoff bands when extending neighbors.

### 5. Graph first-class (**API collapse landed**)

`RoomProgressionGraph` (capability/inventory BFS) is the source of truth for:

- next-hop suggestions for pure cards (`suggest_edges` / wrappers)
- verification promotion (`unverified` → `controller_dev` → `continuous`)
- integration with dwell analysis (`split_dwell.py`) and residual metrics

Practice ranking (`rooms/work_queue.py`) is dual-track metrics only — it does
**not** inject continuous-spine priority. Product next-work is STATUS + tasks
QUEUE.

**Landed:** `VERIFICATION_RANK` + `path_summary(min_verification=)` +
`suggest_edges(prefer=, exclude_verifications=)`. Wrappers keep stable dict
shapes for cards/tests. **Open:** typed path-summary model; further split
`progression/data.py` staging tables if line count remains the pain.

### 6. Hygiene

- Fence `legacy/` and `dev/` (door-warps) — never continuous evidence.
- Semantic state names (route anchor meaning), not opaque labels.
- Promote shared adventure patterns to `retro_harness.adventure` only after
  **SM + ALTTP** both prove the abstraction.
- Keep controller docstrings aligned with graph `verification` (no
  “not continuous evidence” on hops locked `continuous`).
- Delete completed task cards and residuals; do not keep archive trees.

### Known structural debt snapshot (2026-08-01 review)

Prioritized for maintainability, not product tip order:

| # | Issue | Preferred remedy | Status |
|---|--------|------------------|--------|
| 1 | `continuous.py` clone tip runners | Tip-spec table + generic runner; extract hops | **landed** (`TipSpec` / `SUPER_TIP_*` + `hops.py` + thin alias bind) |
| 2 | Twin graph planner APIs + soft dict contracts | Collapse + typed path summary | **partial** (collapse landed; typed model open) |
| 3 | Multi-registry tip wire (graph/catalog/hops/run_to/probe/__init__) | Single tip definition drives the rest | open (catalog flags + tip-spec help) |
| 4 | Lineage special-cases in dense frame loops | Explicit entry mode / separate segments | **landed** (Warehouse + Zeela phases) |
| 5 | Global `parse_counts` / probe report field growth | Session- or cache-scoped counters; keep pin schema stable | **partial** (cache-local stats) |
| 6 | File-size growth past 1k without decomposition | Decompose before next tip tax | **partial** (`continuous` ~600 lines; early still ~800) |

Product tip (Bat Cave → Speed Hall …; Frog Save remains a side tip) may
proceed in parallel; structure debt is **planner-serial** when it touches
`continuous.py` / `progression/` / `catalog.py` hot modules. Todo list:
[`plan.md`](plan.md) Structure & API + [`tasks/QUEUE.md`](tasks/QUEUE.md)
architecture cards.

### Runtime efficiency checklist (today)

1. Continuous `RouteSession` — low WRAM via `get_ram()`; full bank only when needed.
2. Pure probes (`scripts/probe/kpdr.py pure`) — **`mode="nav"`** every frame;
   report includes `parseCounts` + `probePin` / `residualPinLine` on RED.
3. Tight `wait_until` loops — peeks or `StateCache`, not full parse.
4. `StateCache.stats()` (hits/misses + local nav/full parses) + process
   `ram.parse_counts()` for long-run profiles.
5. Source catalog: `super_metroid/source_states.py` (+ `kpdr.py suggest-source`).
6. Tip scaffold: `scripts/scaffold_tip.py` (stub + residual + checklist).
7. Graph helpers: `path_summary` / `suggest_edges` (+ thin pure_gate wrappers).
8. Integrity-green `--state-output` via `ContinuousTip.supports_checkpoint`.
9. Manifest-driven recording + `--no-video` for long dry-runs.
10. Promote shared primitives only after a **second consumer**.
11. Offline dwell rank before live tighten (`split_dwell.py`).
12. Post-Supers tips: `SUPER_TIP_SPECS` in `routes/kpdr/hops.py`; harness in `continuous`.

## Package boundaries (target)

```text
super_metroid/
  ram.py assist.py policy.py progression/ paths.py room_timer.py video.py
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
- Forward work + structure plan: [`plan.md`](plan.md)
- Executor process: [`tasks/PROCESS.md`](tasks/PROCESS.md)
- Assists: [`ASSIST_CONTRACT.md`](ASSIST_CONTRACT.md)
- KPDR board: [`routes/ROUTE_KPDR.md`](routes/ROUTE_KPDR.md)
- Path topology: [`research/PATH_ROOM_BOARD.md`](research/PATH_ROOM_BOARD.md)
- Boss pipeline: [`BOSS_PIPELINE.md`](BOSS_PIPELINE.md)
- Full-run process: [`../../docs/FULL_RUN_PROCESS.md`](../../../docs/FULL_RUN_PROCESS.md)
- Hygiene: [`../../../docs/REPO_HYGIENE.md`](../../../docs/REPO_HYGIENE.md)

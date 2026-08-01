# Harvest Planning Stack

Structure and API direction for the day-plan + task system. Harvest is the
**pioneer planning/simulation trunk**: keep the deterministic control loop, make
the task graph legible and composable for human and agent-assisted rewrites.

Facts and gates: [STATUS.md](STATUS.md). Future work: [plan.md](plan.md).
Commands and traps: [../AGENTS.md](../AGENTS.md).

## Current architecture (strengths + friction)

```text
DayPlanTask (orchestrator)
  → PhaseSpec sequence from build_day_phases() / registry
  → DayTaskFactory + TaskBuildContext
  → concrete Task (CoopChores, CowChores, Harvest, CropWater, FarmClear, Nav, Recorded, …)
```

| Layer | Role | Notes |
|-------|------|-------|
| **Task protocol** (`retro_harness.protocol`) | `name`, `reset`, `can_start`, `step → TaskResult` | Minimal and good; extend via `progress_snapshot()` + child tree |
| **Planner** | Phase catalog + registry of builders, decision/advisor, `local_llm.py` | Advisor is advisory by default; gated apply path for validated optional rewrites |
| **Primitives** (`tasks/primitives.py`) | `TaskSequence`, `PressAndVerifyTask`, `RamCondition`, `RetryTask` | Prefer composing these over new 50–100 KB phase machines |
| **Skills** (`tasks/skills.py`) | Thin Task-protocol skills (nav, interact, verify, sequence) | Domain tasks should become skill composers over time |
| **Pathfinding** | `Pathfinder` / `Navigator` in `farm_clearer` | Viewport-limited BFS; use `densify_waypoints` for long same-map hops |
| **Observation** | `WorldSnapshot` + `WorldContext` cache | Prefer batched reads over re-inspecting RAM every sub-step |

**State of the art (see STATUS):** M3 verified (Spring D2 → Summer D1 continuous,
~29 overnights, Clean/Bronze). Crop economy still open (money stuck ~$100).
Large monolithic task files (coop / cow) with repeated nav/interact/verify FSMs
are the main maintainability tax.

## Design principles

1. **Determinism is sacred** — any LLM or agent rewrite must validate against
   contracts and produce a new `PhaseSchedule` (orchestrator already supports
   splice/append). Do not silently mutate running phase machines.
2. **Skills over giant enums** — domain tasks are thin composers of reusable
   skills; progress trees stay precise for stall detection.
3. **Promote only after a second consumer** — keep harvest-specific code here;
   planner primitives, Pathfinder, and catalogs graduate to shared packages once
   another game needs them (`planning_common` when ready).
4. **Recording → skill** — human recording, replay for tiles/RAM deltas, then
   autonomous skill/task. CLI extraction is a first-class goal.

## Hierarchical composition (skill-level tasks)

Domain tasks should look like sequences, not one class with dozens of `_phase`
values:

```python
# Target shape (CoopChores as skill composition)
CoopChoresTask ≈ TaskSequence(
    NavigateToFeedBin,
    FeedAdultsSkill(slots=...),
    CollectEggsSkill,
    DecideEggDisposition,  # incubate / ship / gift
    ExitStaging,
)
```

| Building block | Status | Location |
|----------------|--------|----------|
| `TaskSequence` / `RetryTask` | Done + `progress_snapshot` | `tasks/primitives.py` |
| `PressAndVerifyTask` / `RamCondition` | Done | `tasks/primitives.py` |
| `NavSkill` / interact helpers | Thin wrappers | `tasks/skills.py` |
| `Pathfinder` / `Navigator` | Exists; promote later | `tasks/farm_clearer.py` |
| `RecoveryTask` | Exists | `core/recovery.py` |
| Full coop/cow skill split | In progress | Extract before growing more mono files |

`ProgressSnapshot.child` already supports a tree — lean into it so agents can
inspect stall points at the skill level and propose rewrites there.

## Richer Task / Phase API

### `TaskBuildContext`

Shared by all phase builders. Fields include seed type, tasks dir, state name,
optional policy, and optional `WorldContext` for cached observations. Builders
should stay pure functions of `(ctx, spec, world)`.

### Contracts on `PhaseSpec`

Optional declarative fields for validation of agent proposals:

| Field | Purpose |
|-------|---------|
| `required_ram` | Named RAM fields that must be readable / meaningful |
| `required_maps` | Tilemap IDs the phase expects |
| `required_tools` | Tool / inventory preconditions (string tags) |
| `estimated_frames` | Budget hint for soak / watchdog |
| `failure_modes` | Documented failure reasons |

Keep the core `phase` / `kind` / `params` / `failure_policy` as today. Contracts
are additive and optional.

### LLM advisor (`local_llm.py`)

- **Default:** notes + deferred only; executable phase lists ignored
  (`advisor_phase_changes_ignored`).
- **Gated apply:** `HARVEST_PLAN_LLM_APPLY=1` or `apply_validated=True` enables
  a schema-checked path that may reorder optional phases or append known
  deferred work. Unknown kinds / required-phase deletion are rejected.
- Env: `HARVEST_PLAN_LLM_URL`, `HARVEST_PLAN_LLM_MODEL`, `HARVEST_PLAN_LLM_API`,
  `HARVEST_PLAN_LLM_TIMEOUT`, `HARVEST_PLAN_LLM_APPLY`.

## Efficiency wins

| Win | Approach |
|-----|----------|
| Observation caching | `WorldContext` batches RAM / tilemap / entity reads per frame or phase |
| Navigation | `densify_waypoints(max_hop_tiles=7)` auto-inserts intermediate hops; long soaks pay most for viewport BFS failures |
| State machines | Small explicit phases + skills; reuse primitives aggressively |
| Recording → autonomous | Slice + replay + assert RAM deltas + emit Task skeleton (CLI TBD; see Way Forward in AGENTS.md) |
| File size | Do not grow new 50–100 KB task files; compose skills instead |

## Repo-level structure

- Keep game-specific code, states, recordings, and docs under `harvest/`.
- Normalize artifact paths and semantic state names (`Y1_*`).
- Align STATUS / plan / AGENTS with the monorepo spine.
- When a second game needs day-plan primitives, extract `planning_common`
  the same way `platformer_common` / `fighters_common` were promoted.

## Roadmap alignment (accelerate past pure Phase 6)

Program ROADMAP places Harvest under longer-term Phase 6. Because infrastructure
is already strong (M3 continuous calendar, planner, editor, recordings), treat
Harvest as the **planning trunk** and pull domain depth forward:

1. **Immediate** — same-day water after plant; harvest + ship; money > $100;
   multi-adult coop fix.
2. **M4** — natural-entry summer from `Y1_Summer_D1_Morning`; Sunday/festival;
   hot-spring stamina gate in day plan.
3. **M5** — cow/barn chores; rainy-day ordering; multi-seed; gifts; stamina/tools.
4. **Campaign** — multi-year planner; hierarchical day → week/season goals;
   power-on → Spring 1; Bronze → Silver observation once the route is stable.

Success metrics live in [STATUS.md](STATUS.md). Concrete next tasks in
[plan.md](plan.md).

## Suggested implementation order

1. Extract reusable skills (Nav, Feed, CollectEgg, Ship) from `CoopChoresTask` + tests.
2. Enrich `TaskBuildContext` + contract fields on `PhaseSpec` (scaffolded).
3. Viewport-aware auto-waypoint helper + eventually shared Pathfinder.
4. Gated apply path for plan advisors with JSON schema (scaffolded).
5. Close crop loop acceptance test in plan.md.
6. Distill high-value recordings (`spring_festival.json`, `fix_rainy_day.json`).

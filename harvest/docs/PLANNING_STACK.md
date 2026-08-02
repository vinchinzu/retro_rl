# Harvest Planning Stack

Structure and API direction for the day-plan + task system. Harvest is the
**pioneer planning/simulation trunk**: keep the deterministic control loop, make
the task graph legible and composable for human and agent-assisted rewrites.

Facts and gates: [STATUS.md](STATUS.md). Future work: [plan.md](plan.md).
Commands and traps: [../AGENTS.md](../AGENTS.md).
Layer migration history: [bot_architecture_plan.md](bot_architecture_plan.md).
Older recovery/smoke plan: [autonomous_play_phased_plan.md](autonomous_play_phased_plan.md).

**Last architecture pass:** 2026-08-01 — production `TaskContract`s wired on crop
/ coop / sleep / hot-spring phases; `evaluate_task_contract()` for soft pre-checks;
skill factories expanded for feed / ship / talk boundaries.

## Current architecture

```text
DayPlanTask (orchestrator)
  → PhaseSpec sequence from build_day_phases() / registry
       each PhaseSpec may carry TaskContract (maps/tools/ram/estimates/modes)
  → DayTaskFactory + TaskBuildContext
  → concrete Task (CoopChores, CowChores, Harvest, CropWater, FarmClear, Nav, …)
       target: thin composers of skills (Nav / Interact / Verify / Sequence)
```

| Layer | Role | Status |
|-------|------|--------|
| **Task protocol** (`retro_harness.protocol`) | `name`, `reset`, `can_start`, `step → TaskResult` | Stable; extend via `progress_snapshot()` + child tree |
| **Planner** | Phase catalog + registry builders, decision/advisor, `local_llm.py` | Stable; advisor advisory by default; gated apply for optional reorder/append |
| **Contracts** (`TaskContract` on `PhaseSpec`) | Soft preconditions for advisors/tests/probes | **Wired** on crop establish/water, harvest, coop, ensure tools, exit/sleep, hot spring; evaluate helper exists; builders do **not** hard-abort on fail yet |
| **Primitives** (`tasks/primitives.py`) | `TaskSequence`, `PressAndVerifyTask`, `RamCondition`, `RetryTask` | Prefer composing these over new 50–100 KB phase machines |
| **Skills** (`tasks/skills.py`) | `NavSkill`, interact helpers, coop/farm/talk factories | Boundary factories exist; production coop/cow/harvest still mono FSMs |
| **Pathfinding** | `Pathfinder` / `Navigator` in `farm_clearer` | Viewport-limited BFS; `densify_waypoints` for long same-map hops |
| **Observation** | `WorldSnapshot` + `WorldContext` cache | Prefer batched reads over re-inspecting RAM every sub-step |
| **Scene / recovery** | `core/scene.py`, `core/recovery.py` | Classifies maps/dialogue/locks/endings; morning stability gate |

**State of the art (see STATUS):** M3 verified (Spring D2 → Summer D1 continuous,
~29 overnights, Clean/Bronze). Crop plant path ROM-ok; water refill flaky; money
stuck ~$100 (no harvest income). Power-on → D1 town gate clean; D1 handoff
auto via rest recording; natural power-on→D2 + shed on `house_size=0` still open.

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
5. **Contracts are soft first** — document failure modes and preconditions in
   the catalog; soft-evaluate in tests/probes; only later hard-gate builders if
   false starts dominate soak logs.

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
| Coop feed/ship nav factories | Done (boundary) | `tasks/skills.py` |
| Farm ship + talk factories | Done (boundary) | `tasks/skills.py` |
| `Pathfinder` / `Navigator` | Exists; promote later | `tasks/farm_clearer.py` |
| `RecoveryTask` | Exists | `core/recovery.py` |
| Full coop/cow/harvest skill split | Open | Extract before growing more mono files |
| `TownDay1HandoffTask` skill split | Open | 900+ line FSM; talk/nav skills ready to host |

`ProgressSnapshot.child` already supports a tree — lean into it so agents can
inspect stall points at the skill level and propose rewrites there.

## Richer Task / Phase API

### `TaskBuildContext`

Shared by all phase builders. Fields include seed type, tasks dir, state name,
optional policy, and optional `WorldContext` for cached observations. Builders
should stay pure functions of `(ctx, spec, world)`.

### Contracts on `PhaseSpec`

Optional declarative fields for validation of agent proposals and soak
diagnostics:

| Field | Purpose |
|-------|---------|
| `required_ram` | Named RAM fields that must exist in the catalog (and optionally be readable) |
| `required_maps` | Tilemap IDs the phase expects |
| `required_tools` | Tool / inventory tags (`hoe`, `seed`, `watering_can`, …) |
| `estimated_frames` | Budget hint for soak / watchdog |
| `failure_modes` | Documented failure reasons (strings for logs/advisors) |

API:

```python
from harvest.planner.day_phase_types import evaluate_task_contract

ok, reasons = evaluate_task_contract(
    phase.contract,
    tilemap=0x00,
    tools=("hoe", "seed"),
    ram=world.ram,  # optional
)
```

Keep the core `phase` / `kind` / `params` / `failure_policy` as today. Contracts
are additive. Production crop/coop/sleep/hot-spring specs declare them; empty
contract still means "no contract declared."

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
| Recording → autonomous | Slice + replay + assert RAM deltas + emit Task skeleton (CLI TBD) |
| File size | Do not grow new 50–100 KB task files; compose skills instead |
| Contract diagnostics | Include `contract` in decision JSON; soft-evaluate in probes when debugging false starts |

## Architecture workstreams (ordered)

| # | Work | Done when |
|---|------|-----------|
| A1 | Wire contracts on critical phases + evaluate helper | **Done 2026-08-01** |
| A2 | Expand skill factories (feed/ship/talk/farm bin) | **Boundary done**; production still mono |
| A3 | Close crop loop (water refill → harvest → ship → money > $100) | Domain acceptance in STATUS |
| A4 | Extract CoopChores feed/collect/ship into skill composers + multi-adult fix | Unit + replay green; coop back on daily plan |
| A5 | Soft-evaluate contracts in day-plan probe / optional preflight notes | Probe JSON shows contract ok/fail reasons |
| A6 | D1 town handoff: pure skill routes from power-on (reduce rest-recording dependency) | power-on → D2 without AnnEve fixture |
| A7 | Distill `spring_festival.json` + `fix_rainy_day.json` into phase ordering | Documented sequences in planner |
| A8 | Promote Pathfinder / skill primitives after second game consumer | Shared package or leave in harvest |

## Repo-level structure

```text
harvest/harvest/
  core/       # RAM, tiles, scene, recovery, world snapshot/context
  maps/       # map_config routes/landmarks, farm_map
  planner/    # day plan, phases, registry, decision, local_llm, crop_planner
  tasks/      # domain tasks + primitives + skills
  runtime/    # bot, autoplay, power_on, retro_setup, rom_tools
  scripts/    # boot_probe, run_to_day2, town_day1_recon, spa probes
  tools/      # editor, presets, ending probe
```

- Keep game-specific code, states, recordings, and docs under `harvest/`.
- Normalize artifact paths and semantic state names (`Y1_*`).
- When a second game needs day-plan primitives, extract `planning_common`
  the same way `platformer_common` / `fighters_common` were promoted.

## Roadmap alignment

Program ROADMAP places Harvest under longer-term Phase 6. Because infrastructure
is already strong (M3 continuous calendar, planner, editor, recordings), treat
Harvest as the **planning trunk** and pull domain depth forward:

1. **Immediate** — same-day water after plant; harvest + ship; money > $100;
   multi-adult coop fix; power-on D1 close.
2. **M4** — natural-entry summer from `Y1_Summer_D1_Morning`; Sunday/festival;
   hot-spring stamina gate in day plan.
3. **M5** — cow/barn chores; rainy-day ordering; multi-seed; gifts; stamina/tools.
4. **Campaign** — multi-year planner; hierarchical day → week/season goals;
   power-on → full Spring 1; Bronze → Silver observation once the route is stable.

Success metrics live in [STATUS.md](STATUS.md). Concrete next tasks in
[plan.md](plan.md).

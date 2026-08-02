# Autonomous Play Phased Plan

Updated: 2026-04-29 · **Status banner: 2026-08-01**

> **Superseded for day-to-day planning.** Use [STATUS.md](STATUS.md) (facts),
> [plan.md](plan.md) (queue), and [PLANNING_STACK.md](PLANNING_STACK.md)
> (architecture). This document remains as the recovery/smoke phase history
> from the April reliability push. Phases 0–2 are largely landed (scene
> classifier, recovery, failure policy, day-plan decision); Phase 3–5 remain
> open as skill migration + domain depth; Phase 6 package move is done; Phase 7
> long-run smoke matrix is still open. M3 spring calendar is verified separately
> (see STATUS).

Goal: make the bot able to start from durable save states and play the full game without drifting into unrecoverable map, cutscene, dialogue, or task states.

This plan extends the existing architecture notes in `docs/bot_architecture_plan.md`. The main diagnosis is that the core gameplay tasks exist, but the runtime does not yet have strong enough scene classification, recovery, failure policy, and smoke coverage to survive full autonomous play.

## Current Diagnosis

- The unit test suite is healthy enough to refactor against: `uv run python -m unittest discover -s tests -v` ran 477 tests successfully after adding scene-classifier, shared-primitive, recovery, pinned morning fixture, planner-decision, deferral, and local-advisor coverage.
- Replay coverage that used mutable `latest.state` should be replaced with pinned-state coverage.
- A manual probe from mutable `latest` slept successfully, then failed the next morning while trying to exit the remodeled house.
- After waking, the planner entered `EXIT_TO_FARM`, drifted through unexpected tilemaps, observed invalid coordinates, then repeatedly inserted more exit phases from unknown map states.
- The core problem is not only missing chores. The bot needs a global notion of scene, recovery, and fatal versus optional phase failure.

## Completion Audit

Audited on 2026-04-29 against the current code and tests.

- Phase 0: partial. `utils/day_plan_probe.py` exists and emits phase/tilemap/RAM JSONL diagnostics, and `Y1_After_Sleep` is now pinned in targeted scene/runtime tests. The full smoke-state matrix is not checked in.
- Phase 1: partial. `harvest/core/scene.py` now classifies normal maps, known locations, dialogue/menu/input locks, transitions, sleep/wake, endings, unknown tilemaps, and invalid coordinates; the morning rebuild gate uses it; `harvest/core/recovery.py` adds scene-level recovery; and `ExitToFarmTask` blocks unknown/ending scenes. ROM-backed smoke coverage remains missing.
- Phase 2: partial. `PhaseSpec.failure_policy` now makes required phases recover once before abort and optional phases skip explicitly; `harvest/planner/day_plan_decision.py` exposes rule-built plan facts, phases, notes, and tomorrow-facing deferrals. Future social/gift policies and typed phase contracts remain open.
- Phase 3: partial. `harvest/tasks/primitives.py` now provides shared button sequences, queue draining, dialogue dismissal, press-and-verify, RAM waits, scene waits, task sequencing, and bounded retry; broad navigation and domain-task migration remain incomplete.
- Phase 4: partial. Many routes/exits are in `harvest/maps/map_config.py`, including remodeled-house route handling, but navigation is still split across task-local implementations.
- Phase 5: partial. Cow talk/brush/milk/feed support and dynamic day ordering exist; coop replay hardening, festivals, gift/social routes, and seasonal rotations remain incomplete.
- Phase 6: package move done. The package layout migration is complete and tested; later file splitting remains intentionally deferred.
- Phase 7: not done. Long-run summaries, checkpoint bundles, and curated multi-day smoke tests remain future work.

## Phase 0: Baseline And Reproducibility

Status: partial. Keep open until the full pinned smoke matrix replaces mutable `latest` diagnosis.

Objective: make current failures easy to reproduce before changing architecture.

Work:

- Promote `utils/day_plan_probe.py` into the standard smoke diagnostic tool.
- Do not add automated tests that depend on mutable `latest.state`.
- Use `latest` only for manual diagnosis, then copy or back up the relevant state into a pinned named fixture before adding regression coverage.
- Add a checked or documented probe command for pinned states that records phase changes, tilemaps, coordinates, input lock, dialogue state, and failures.
- Completed on 2026-04-29: add targeted pinned-state coverage for `Y1_After_Sleep` as a normal 06:00 morning house scene and valid auto-day-plan rebuild point.
- Remaining: preserve the current morning-after-sleep exit failure as a ROM-backed smoke case if it still reproduces after recovery.
- Define a small smoke-state matrix:
  - pinned morning-after-sleep state
  - pinned rainy day with animals
  - pinned coop with two adults and eggs
  - pinned barn with cows
  - pinned festival day
  - pinned summer or fall crop day
  - pinned final-day bedtime state
  - pinned ending or credits state

Done when:

- A developer can run one command and see where the bot failed, which phase failed, what scene it was in, and the last useful RAM/map facts.
- The current morning exit path is captured from a pinned state as either a known failing smoke case or a passing recovery regression.

## Phase 1: Scene Classification And Recovery

Status: partial. Scene classification, scene-level recovery, and the morning stability gate exist; ROM-backed smoke coverage is still missing.

Objective: stop running normal day tasks while the game is in a cutscene, transition, dialogue, invalid position, or unknown map state.

Work:

- Completed on 2026-04-29: add a `Scene` classifier built from `WorldSnapshot` or live/save RAM using `harvest/core/ram_catalog.py`.
- Completed on 2026-04-29: classify at least:
  - normal map
  - farm
  - house by level or variant
  - barn
  - coop
  - shop
  - town
  - mountain
  - festival
  - dialogue
  - menu
  - input locked
  - map transition
  - sleep or wake transition
  - ending or credits
  - unknown tilemap
  - invalid coordinates
- Completed on 2026-04-29: add `RecoveryTask` in `harvest/core/recovery.py`.
- Completed on 2026-04-29: recovery waits through transitions, dismisses dialogue/menu/input-lock scenes, routes out of normal non-target scenes through an injected route task, and blocks cleanly on unknown maps, invalid coordinates, or endings.
- Completed on 2026-04-29: add a scene stability gate after sleep before rebuilding the day plan.
- Completed on 2026-04-29: update `ExitToFarmTask` to use scene facts for unknown/terminal scenes instead of falling back to blind house-exit assumptions.

Done when:

- From a pinned morning-after-sleep state, the bot can wait for a stable morning scene and either exit to farm or fail with a clear recovery diagnostic.
- Unknown tilemaps no longer cause repeated insertion of `EXIT_TO_FARM`.

## Phase 2: Explicit Phase Contracts

Status: partial. Required/optional failure policies and recovery-before-abort are implemented; future phase typing and social/gift policy work remain open.

Objective: make the planner know which failures can be skipped and which require recovery or abort.

Work:

- Completed on 2026-04-29: extend phase specs with a failure policy:
  - `required`: recover, then abort if still failing.
  - `optional`: log and continue.
  - `opportunistic`: run only when preconditions are clearly true.
- Completed on 2026-04-29: mark core transitions and required chores as required by default.
- Completed on 2026-04-29: mark money/berry extras as optional, including legacy `OPTIONAL_MONEY_PHASES`.
- Completed on 2026-04-29: add `DayPlanDecision`, `PlanningFacts`, and `DeferredPlan` so a planned day can be serialized, inspected, and handed to external advisors without scraping log text.
- Completed on 2026-04-29: optional and opportunistic failures now record deferred phase intentions for tomorrow instead of only skipping in-place.
- Completed on 2026-04-29: add a disabled-by-default local LLM advisory adapter in `harvest/planner/local_llm.py`; it can add notes/deferrals through `HARVEST_PLAN_LLM_URL`, but executable phase rewrites are ignored until a stricter validator exists.
- Remaining: mark future social/gift extras as optional or opportunistic as those phases are added.
- Replace stringly phase handling where practical with typed phase specs or enums.
- Keep planner orchestration in `harvest/planner/day_plan.py`, with behavior behind clearer package APIs.

Done when:

- A failed `EXIT_TO_FARM`, `ENTER_BARN`, `ENTER_COOP`, sleep, or required animal phase cannot be silently skipped into unrelated work.
- The final probe summary explains whether the bot aborted, recovered, skipped optional work, or completed the day.

## Phase 3: Verified Task Primitives

Status: partial. Shared primitive helpers exist; broad task migration and navigation/map-transition primitives remain open.

Objective: reduce duplicated task logic and make interactions retryable and verifiable.

Work:

- Completed on 2026-04-29: add shared primitives for button sequences, queue draining, press and verify, wait for RAM condition, wait for scene, and dismiss dialogue.
- Completed on 2026-04-29: add composable `TaskSequence` and `RetryTask` primitives so new tasks can share orchestration and bounded retry behavior.
- Remaining: add or consolidate shared primitives for navigate to tile, face direction, map transition, interact at landmark, and verify inventory, item, animal, crop, or map delta.
- Started on 2026-04-29: convert fragile planner task paths to these primitives first, including shared dialogue dismissal and queued action handling in navigation/sleep/shed interactions.
- Gradually migrate coop, cow, crop, harvest, shop, sleep, and social tasks.
- Avoid broad rewrites until the smoke probes protect the behavior.

Done when:

- New autonomous tasks can be built from common primitives instead of each task reimplementing navigation, A presses, retries, and verification.
- Failed interactions report the expected condition and the observed condition.

## Phase 4: Navigation And Map Cleanup

Status: partial. Route catalogs exist, but task-local navigation implementations remain.

Objective: make navigation route-driven, scene-aware, and based on catalog data.

Work:

- Consolidate `NavTask` and `MultiMapNavTask` behavior behind one map-aware navigation layer.
- Move house, barn, coop, shed, shop, church, town, and mountain exits into `harvest/maps/map_config.py`.
- Add house-level variants for exit and bed routes.
- Replace task-local walkable constants with `harvest/core/tile_catalog.py` and `harvest/maps/map_config.py`.
- Add intermediate waypoints for routes longer than the viewport-safe distance.
- Extract route facts from recordings instead of relying on static save-state tile IDs.

Done when:

- Building exits and long-distance routes are represented as map facts, not hard-coded inside task modules.
- Routes fail because a precondition is false, not because stale tiles were read outside the viewport.

## Phase 5: Domain Completion

Status: partial. Keep adding domain systems only behind pinned replay or smoke coverage.

Objective: finish the missing game systems after recovery and navigation are reliable.

Work:

- Coop:
  - fix the Spring 22 two-adult/two-egg state.
  - feed adults in separate feed slots.
  - treat visible egg object tiles as dynamic collision.
  - restore coop chores to the daily plan after replay coverage passes.
- Barn:
  - keep the verified multi-cow feed loop from `cow_chores_fix`.
  - add brushing and milking.
  - verify per-cow and per-feed-slot state.
- Rainy day:
  - extract the ideal routine from `fix_rainy_day.json`.
  - improve barn, coop, crop, town, and shed ordering.
- Festivals:
  - extract Spring 23 festival facts from `spring_festival.json`.
  - add dialogue and route handling for festival scenes.
- Social:
  - promote candidate NPC sprite IDs into named schedules.
  - add talk and gift tasks using dynamic NPC objects.
- Crops:
  - expand summer and fall crop rotations.
  - keep harvest, watering, planting, and seed purchase policy seasonal.
- Endings:
  - integrate ending scene and credits handling with the scene classifier.
  - treat going to bed on the final day as a game-over or ending transition, not as a normal next-day sleep.
  - prevent the multi-day planner from rebuilding normal morning chores after final-day bedtime.
  - keep ending probe presets as regression fixtures.

Done when:

- Each major system has at least one unit test with fake RAM and one ROM-backed replay or smoke path where practical.
- Domain tasks can be disabled independently without breaking the day planner.

## Phase 6: Runtime And Repository Reorganization

Status: done for the package move; open for later module splitting.

Objective: keep runtime, planner, task, map, core, and tool code separated so autonomy work lands in the right layer.

Target structure:

```text
harvest/
  core/
    animal_probe.py
    animal_status.py
    harvest_state.py
    npc_catalog.py
    ram_catalog.py
    tile_catalog.py
    world_snapshot.py
    scene.py
    task_protocol.py
    recovery.py
  maps/
    map_config.py
    farm_map.py
  tasks/
    coop_task.py
    cow_task.py
    crop_planter.py
    farm_clearer.py
    harvest_task.py
    recorded_task.py
  planner/
    crop_planner.py
    day_plan.py
    day_plan_phases.py
    day_plan_status.py
    day_task_factory.py
  runtime/
    harvest_bot.py
    harness_runtime.py
    probe_utils.py
    recording_trace.py
    retro_setup.py
    rom_tools.py
  tools/
    editor_app.py
    ending_probe.py
    livestock_builder.py
```

Migration order:

- Completed on 2026-04-29: moved the flat top-level modules into `harvest/core`, `harvest/maps`, `harvest/tasks`, `harvest/planner`, `harvest/runtime`, and `harvest/tools`.
- Completed on 2026-04-29: removed flat-module import targets from tests and runtime code instead of adding compatibility shims.
- Completed on 2026-04-29: `run_bot.sh` now invokes `python -m harvest.runtime.harvest_bot`.
- Remaining: split `harvest/planner/day_plan_tasks.py`, `harvest/tasks/cow_task.py`, `harvest/tasks/crop_planter.py`, and `harvest/runtime/harvest_bot.py` into smaller domain modules after scene/recovery contracts are in place.
- Completed on 2026-04-29: added `harvest/core/scene.py` and `harvest/tasks/primitives.py`.
- Completed on 2026-04-29: added `harvest/core/recovery.py`.
- Completed on 2026-04-29: added `harvest/planner/day_plan_decision.py` and `harvest/planner/local_llm.py`.
- Remaining: continue migrating domain tasks onto shared primitives.

Done when:

- The repo has a clear separation between core state, maps, task primitives, domain tasks, planner policy, runtime, and tools.
- The full test suite stays green after the package move.

Status:

- Done for the package move: `uv run python -m unittest discover -s tests -v` ran 439 tests successfully on 2026-04-29.
- Done for explicit phase-failure contracts: `uv run python -m unittest discover -s tests -v` ran 442 tests successfully on 2026-04-29.
- Done for scene-classifier/shared-primitive slice: `uv run python -m unittest discover -s tests -v` ran 459 tests successfully on 2026-04-29.
- Done for recovery and pinned morning fixture slice: `uv run python -m unittest discover -s tests -v` ran 469 tests successfully on 2026-04-29.
- Done for structured planner decisions, deferrals, retry/sequence primitives, and local advisor hook: `uv run python -m unittest discover -s tests -v` ran 477 tests successfully on 2026-04-29.
- Not done for runtime hardening: broader pinned smoke coverage and final-day ending flow remain separate implementation phases.

## Phase 7: Long-Run Autonomy Hardening

Status: not done. This starts after scene/recovery and curated smoke fixtures are reliable.

Objective: make full-game play observable, resumable, and debuggable.

Work:

- Add structured run summaries for every abort and day completion.
- Track day, season, year, money, inventory, animals, crops, relationship state, current plan, last phase, last scene, and last recovery action.
- Add periodic save-state checkpoints before risky transitions and major tasks.
- Add a replay/debug bundle format that captures state name, plan policy, final RAM snapshot, phase history, and recent input history.
- Run multi-day smoke tests from curated states.
- Keep automated smoke tests on curated pinned states only; do not use mutable `latest`.
- Add a "continue after recoverable failure" policy only after diagnostics are trustworthy.

Done when:

- A failed long run leaves enough information to reproduce the failure without watching the whole session.
- The bot can complete many consecutive days without planner drift.

## First Implementation Slice

Status after this audit:

1. Done: add the scene classifier.
2. Done: add `RecoveryTask`.
3. Done for unit coverage: gate morning-after-sleep planning on scene stability through `AutoClearBot._auto_day_plan_rebuild_ready()`.
4. Done: required phase failures now try recovery once before aborting; optional phase failures still skip explicitly.
5. Done for targeted RAM coverage: add pinned-state morning-after-sleep regression coverage using `Y1_After_Sleep`; ROM-backed exit smoke remains Phase 0 work.
6. Done: fix `ExitToFarmTask` and sleep routing for known remodeled house variants `0x16` and `0x17`; unknown/ending scenes now block instead of inserting blind exit work.
7. Skipped for now: add final-day bedtime handling so sleep transitions into ending/game-over flow instead of normal day planning.
8. Done: expose structured planner decisions, tomorrow deferrals, and a safe local-LLM advisory hook; the runtime remains rule-based unless the advisor environment is configured.

This slice gives the bot the missing safety layer before more game content is added.

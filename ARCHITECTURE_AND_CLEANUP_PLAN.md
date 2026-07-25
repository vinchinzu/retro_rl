# Architecture And Cleanup Plan

> Historical cleanup plan. Prefer current program docs:
> [`docs/VISION.md`](docs/VISION.md),
> [`docs/PROGRAM_STATUS.md`](docs/PROGRAM_STATUS.md).
> Authoritative paths: `super_metroid/`, `SMW/`. No `alttp/` in this checkout.
> Older prose may still say `super_metroid_rl` as a historical alias.

## Intent

Use ALTTP as a proving-ground case study, then shift active architecture work
to `super_metroid/` and harden shared library boundaries instead of continuing
to accrete one-off scripts.

This plan has three purposes:

1. Capture lessons learned from the ALTTP push.
2. Prioritize what should be improved and implemented next.
3. Clean up the repo so distinct problem domains stop bleeding into each other.

## Lessons Learned From ALTTP

### 1. Screen-level success is too weak

Reaching the right screen is not the same as reaching the right route anchor.

- The current castle start-state issue showed that "first frame on screen `0x1B`" was not the same as "right after the bridge, turn east".
- Future benchmarks must use honest anchor conditions:
  - route checkpoint
  - RAM position window
  - room/screen plus local coordinate bounds
  - screenshot artifact for verification

### 2. RAM truth, map truth, and visual truth all matter

Any one of them alone is insufficient.

- RAM gave us live `link_x/link_y` and wrapped local coordinates.
- Yaze/map data gave us the real hole tile and entrance target.
- Screenshots exposed when the route anchor itself was wrong even though the screen id was technically correct.

The rule going forward: any nontrivial navigation task should have all three.

### 3. Route, approach, and trigger are separate problems

We repeatedly blurred them together.

- Route problem: get to the correct area/checkpoint.
- Approach problem: get into the correct local pocket/lane.
- Trigger problem: find the exact interaction/hitbox/transition condition.

These should be separate benchmark tiers, separate start states, and separate scripts.

### 4. Published states need stronger semantics

A saved state should say what it means, not just where it was captured.

Bad example:

- `HyruleCastleGrounds` meaning "somewhere on the castle screen"

Good examples:

- `HyruleCastle_BridgeTurn_East`
- `HyruleCastle_SecretPassageApproach`
- `HyruleCastle_SecretPassageExactTile`

### 5. One-off experiments are cheap; mixed folders are expensive

The codebase can tolerate many experiments. It does not tolerate unclear ownership.

The current pain is not that there were too many probes. The pain is that gauntlet work, romhack work, and opening-route work all landed in the same ALTTP folder without clear boundaries.

## Priority Order

## P0: Pivot Active Architecture Work To Super Metroid

Super Metroid is the next target for extracting a decent reusable architecture library.

Why first:

- It already has enough real complexity to force good abstractions.
- It has route, segment, optimizer, recording, replay, navigation, and evaluation concerns already present.
- It is closer to a reusable library shape than continuing to grow ALTTP ad hoc.

What to improve first:

1. Define package boundaries inside `super_metroid/`.
2. Extract stable shared primitives into `retro_harness/` and `platformer_common/`.
3. Freeze or isolate legacy code instead of mixing it with current paths.
4. Add tests at the boundary seams before adding features.

## P1: Leave ALTTP In A Clean, Truthful State

Do not keep pushing ALTTP forward in a messy shape.

Minimum completion bar before deprioritizing it:

1. Keep the Yaze-backed map export/search helpers.
2. Fix the misleading route/state semantics around the castle approach.
3. Split opening-route work from gauntlet and romhack work.
4. Leave a small handoff doc for the remaining trigger-search problem.

## P2: Repo Hygiene

Stop treating the repo root as a flat staging area for unrelated experiments.

The cleanup should reduce ambiguity around:

- shared library code
- active game projects
- legacy/archive work
- editor/romhack side projects
- generated artifacts and debug leftovers

## Super Metroid Implementation Plan

## Phase 1: Make The Current Shape Explicit

Create a short internal architecture map for `super_metroid/`:

- runtime entrypoints
- navigation stack
- optimizer stack
- recording/replay stack
- training stack
- legacy code
- editor/tooling

Expected result:

- one page describing what is current, what is legacy, and what is shared-candidate code

## Phase 2: Extract Stable Interfaces

Target interfaces first, not behavior rewrites.

Candidate extraction targets:

- emulator/session lifecycle
- input scripting and replay
- state loading/saving metadata
- artifact output paths
- route evaluation protocol
- optimizer run manifest format
- benchmark case definition shape

Preferred homes:

- `retro_harness/` for emulator/session/recording primitives
- `platformer_common/` for route, segment, and platformer-level abstractions
- `super_metroid/` for game-specific map data, room logic, and route content

## Phase 3: Fence Off Legacy

`super_metroid/legacy/` should remain available but not leak into current imports by accident.

Concrete steps:

- mark legacy modules as frozen in docs
- remove current entrypoints that still import through legacy paths unless strictly needed
- add one compatibility shim layer if required, instead of scattered direct imports

## Phase 4: Test The Seams

Before major new features:

- config loading tests
- route/segment resolution tests
- artifact path tests
- replay/recording manifest tests
- optimizer input/output contract tests

## ALTTP Carry-Forward Notes

ALTTP should be parked in a shape that is honest and maintainable.

### Keep

- Yaze export helpers
- overworld feature tables
- RAM coordinate plumbing
- deterministic script playback
- benchmark runner integration

### Fix Before Parking

1. Rename or replace misleading start states/benchmarks that only mean "first screen hit".
2. Separate:
   - route benchmark
   - bridge-turn benchmark
   - approach benchmark
   - trigger benchmark
3. Record the remaining hidden-hole issue as a trigger/hitbox problem, not a route-discovery problem.

### Do Not Do

- Do not keep adding broad search scripts without stronger route anchors.
- Do not keep mixing opening-route work with gauntlet or romhack workflows.

## Repo Cleanup: The Three Distinct Problems

## Problem 1: Shared Library Work Vs Game-Specific Work

The repo root mixes shared harness code and game project code too loosely.

Need:

- clearer rule for what belongs in `retro_harness/`
- clearer rule for what belongs in `platformer_common/`
- fewer root-level operational docs that are really subproject docs

Cleanup steps:

1. Audit root markdown files and move game-specific docs into the owning project.
2. Keep only truly cross-project docs at root.
3. Add a short root map of active shared libraries vs active game projects.

## Problem 2: ALTTP Contains Three Subprojects

Right now `alttp/` contains at least three distinct problem sets:

1. Opening-route / benchmark / navigation proving-ground work
2. Gauntlet combat work from two weeks ago
3. Romhack/editor experiments

These should not share the same top-level namespace casually.

Recommended split:

- `alttp/opening_route/`
- `alttp/gauntlet/`
- `alttp/romhack/`

Concrete moves:

1. Move gauntlet work:
   - `alttp/gauntlet_env.py` -> `alttp/gauntlet/gauntlet_env.py`
   - `alttp/tests/test_gauntlet.py` -> `alttp/gauntlet/tests/test_gauntlet.py`
2. Move romhack work:
   - `alttp/romhack_sprites.py` -> `alttp/romhack/romhack_sprites.py`
   - `alttp/test_romhack_reentry.py` -> `alttp/romhack/tests/test_romhack_reentry.py`
   - `alttp/state_screenshots/romhack_test/` -> `alttp/romhack/artifacts/romhack_test/`
3. Leave only opening-route and core runtime files at the current `alttp/` top level.

## Problem 3: Super Metroid Mixes Current App, Legacy App, And Tooling

`super_metroid/` currently contains:

- active navigation/optimizer/runtime code
- legacy RL/runtime code
- embedded editor/tooling project
- debug output folders

That is too much in one flat project boundary.

Recommended split:

- keep `super_metroid/` for the active runtime/library
- keep `super_metroid/legacy/` but document it as frozen
- move editor/tooling under a clearer tools namespace if it remains in-repo

Concrete cleanup steps:

1. Write one `super_metroid/docs/ARCHITECTURE.md` with:
   - current
   - legacy
   - external/editor/tooling
2. Decide whether `super_metroid/super_metroid_editor/` is:
   - an embedded tool project to keep in place, or
   - a separate subproject to move under `tools/`
3. Move transient debug/artifact directories under a single `artifacts/` policy where practical:
   - `debug_frames/`
   - `debug_screens/`
   - temp demo output

## Suggested Execution Order

## Week 1

1. Write `super_metroid/docs/ARCHITECTURE.md`.
2. Split ALTTP gauntlet and romhack files into subfolders.
3. Move obviously game-specific root docs into owning folders.

## Week 2

1. Extract the first stable Super Metroid interfaces into shared libs.
2. Add tests for those interfaces.
3. Freeze legacy paths and remove accidental current-path imports.

## Week 3

1. Normalize artifact/output layout across active projects.
2. Add one short handoff doc per active game project:
   - current focus
   - active benchmarks
   - known blockers
   - what is frozen

## Definition Of Better

The repo is in a better state when:

- shared code is obviously shared
- game-specific code is obviously game-specific
- experiments are fenced off instead of mixed into active runtime paths
- published states and benchmarks encode real route meaning
- new work in `super_metroid/` can be done by extending stable interfaces instead of adding more one-off scripts

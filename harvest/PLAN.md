# Harvest Bot Skill Tech Tree & Tests

This plan defines the skill progression needed for the Harvest Moon bot to scale in complexity, plus the minimal tests to validate each skill level.

## Tech Tree

### Level 0: Core IO
- Skill: Controller/keyboard input mapping and action sanitization.
- Skill: Emulator step loop stability (no crash, no input lock deadlock).
- Skill: Basic RAM access (player pos, map, item).
- Tests:
  - Start game from a known state; hold no inputs for 300 frames; verify no crash.
  - Verify player position updates over 60 frames while moving.
  - Verify item/tool RAM reads match HUD for 10 random frames.

### Level 1: Deterministic Tasks
- Skill: Replay recorded tasks reliably from a known state.
- Skill: Task start-state gating/reset.
- Skill: Task success check (end state signals success).
- Tests:
  - Replay `ship_berry` 5x from its start state; succeed 5/5.
  - Replay `get_sickle` 5x from its start state; tool id becomes sickle (0x01).
  - Replay `toss_bush_pond` 3x; end position within 2 tiles of pond target.

### Level 2: Navigation
- Skill: Grid-based pathfinding with collision from RAM tilemap.
- Skill: Repathing on stasis and blocked-tile avoidance.
- Skill: Area bounds and exit prevention.
- Tests:
  - Navigate to 10 random tiles on farm without leaving bounds.
  - Artificially block a tile and verify replanning avoids it.
  - Path to shed entrance from farm in under N frames.

### Level 3: Target Detection
- Skill: Debris detection via tile IDs (weed, stone, rock, stump, fence).
- Skill: Dynamic tile learning (weed variations).
- Skill: Priority selection and filtering.
- Tests:
  - Scan the map and list all ROI tiles; verify counts match visual inspection on a fixed state.
  - Verify weed removal changes tile ids and is learned.
  - Priority-only mode clears only requested debris type.

### Level 4: Tooling
- Skill: Tool selection via inventory cycling.
- Skill: Tool use for each debris type (sickle, hammer, axe).
- Skill: Tool acquisition (get_* tasks) when missing.
- Tests:
  - For each tool, verify equip within 1 cycle.
  - For each debris type, clear 5 targets in a row.
  - Missing tool triggers correct `get_*` task and then succeeds.

### Level 5: Lift & Toss (Pond Disposal)
- Skill: Identify liftable targets (stone/bush/fence tile ids).
- Skill: Pick up, carry, and throw into nearest pond.
- Skill: Carry-state recovery on blocked paths.
- Tests:
  - Pick up and toss 5 stones into pond.
  - Pick up and toss 5 bushes into pond.
  - Confirm carry state returns to false after toss.

### Level 6: Multi-Objective Farm Clear
- Skill: Mix tool use and pond disposal based on target type.
- Skill: Avoid repeating failed targets.
- Skill: Progress tracking (cleanliness).
- Tests:
  - Clear 25 debris of mixed types without human input.
  - Ensure no target is attempted more than max_failures.
  - Verify farm bounds adjust to target set.

### Level 7: Robustness & Recovery
- Skill: Detect stuck loops and recover (repath, retarget, reset).
- Skill: Dialog dismissal without breaking tasks.
- Skill: Pause/continue safe mode.
- Tests:
  - Force a stuck scenario and verify recovery in under 10 seconds.
  - Trigger a dialog mid-nav and verify dismissal then resume.
  - Run 1 in-game day at 8x without disable.

### Level 8: Skill Composition
- Skill: Scripted day plan (wake → tasks → bed).
- Skill: Conditional branching (weather, stamina, time).
- Skill: Task chaining with checkpoints.
- Tests:
  - Run a scripted day and reach bed by night.
  - Skip watering on rain.
  - Resume from a checkpoint after failure.

### Level 9: Generalization
- Skill: Tool shed navigation from any farm location.
- Skill: Map transitions (farm ↔ town ↔ shed) with lock recovery.
- Skill: Multi-pond selection on different maps.
- Tests:
  - Start from 3 random farm locations and reach shed.
  - Traverse farm→town→farm 3x without deadlock.
  - Toss debris into different ponds based on proximity.

## Test Harness Ideas
- Headless runner that:
  - Loads a state.
  - Runs a bot or task for N frames.
  - Asserts conditions (tool id, position, tile changes).
- Save per-test artifacts:
  - Logs (frame, pos, tool id, map id).
  - Optional end state snapshot.

## Next Steps
- Shift default priority to hammer/axe targets (stone/rock/stump) and de-prioritize sickle/weeds.
- Improve carry movement so pond toss completes reliably (no run, avoid hotswap chord).
- Add hammer + axe clearing tests using fixed states, then re-enable L6 as a mixed-tool test.
- Record or generate a state with nearby stones/stumps for repeatable hammer/axe coverage.

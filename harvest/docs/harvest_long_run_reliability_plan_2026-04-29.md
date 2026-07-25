# Harvest auto-bot long-run reliability plan (2026-04-29)

## Reproduction

Command used:

```bash
HEADLESS=1 ./run_bot.sh play --autoplay --state latest --until-season 1 --until-day 17 --save-end \
  > logs/long_runs/harvest_until_day17_20260429_175950.log 2>&1
```

Artifacts:
- `logs/long_runs/harvest_until_day17_20260429_175950.log`
- `custom_integrations/HarvestMoon-Snes/latest_day_plan_end.state`

## What failed

The bot fails on **day 8, season 1**, during the final crop phase.

Key log sequence:

- `HARVEST_ROUTE -> SUCCESS`
- `ENSURE_WATERING_CAN -> SUCCESS`
- `ENSURE_CROP_SEEDS -> SUCCESS`
- `NAV_CROP -> SUCCESS`
- `CROP_WATER` immediately fails with `watering can not in carry pair`
- planner performs a generic recovery
- `CROP_WATER` is retried and fails the same way
- bot disables day-plan mode and drops into endless `scanning`

Relevant log lines:
- `logs/long_runs/harvest_until_day17_20260429_175950.log:197-217`
- scanning continues long after disable, e.g. `:740-779`

## Primary failure mode

### 1) Tool orchestration bug between seed retrieval and watering

Evidence:
- `harvest/tasks/crop_planter.py:1715-1717` hard-fails `CROP_WATER` on dry days when the watering can is not in the active 2-slot carry pair.
- `harvest/planner/day_plan_tasks.py:1098-1205` shows `EnsureCropSeedsTask` is best-effort. It can return success if stock exists or after shed work even when the seed tool is not actually carried, and it does not guarantee restoration of the watering can afterward.
- Runtime RAM snapshot from `latest_day_plan_end.state`:
  - `tool_selected=15`
  - `tool_backpack=14`
  - `water_can=12`
  - `potato_seeds=58`
- Interpretation: the watering can still exists in inventory/state, but it is **not** in the current carry pair when `CROP_WATER` begins.

Most likely chain:
1. `ENSURE_WATERING_CAN` makes the can available.
2. `ENSURE_CROP_SEEDS` changes the carry pair / shed state while fetching or checking seeds.
3. Control returns as `SUCCESS` without re-establishing the can+seed pair required by crop work.
4. `CROP_WATER` checks once on step 1, sees no watering can in the active pair, and aborts.

## Secondary failure modes

### 2) Recovery is generic and does not repair the actual missing precondition

Evidence:
- `harvest/planner/day_plan.py:188-209` uses a generic `RecoveryTask` that routes through `ExitToFarmTask`.
- It resets position, but it does **not** restore the missing carry pair.
- The log proves this: recovery completes, then the very next retry fails for the exact same reason.

### 3) Disabled bot keeps spinning instead of exiting with a terminal failure

Evidence:
- After the failure, runtime logs:
  - `[BOT] Day plan: stopped (...)`
  - `[BOT] Disabled: Day plan stopped (...)`
- But the process then keeps printing `scanning` for hundreds of frames instead of exiting.

Impact:
- For a 10–20 day unattended run, a supervisor could wrongly treat the process as healthy/alive while no progress is happening.
- That is a nasty little gremlin because “still running” is not the same as “still working.”

## Test coverage gap

Existing tests cover pieces, not the failure chain:
- `tests/test_crop_planter_logic.py:44-55` already asserts crop watering fails fast when the can is missing from the carry pair.
- `tests/test_day_plan_sequences.py:1566-1663` covers shed routing and seed retrieval behavior.
- There is no clear end-to-end test proving that after `ENSURE_CROP_SEEDS`, the carry pair is valid for `CROP_WATER` across repeated multi-day outdoor plans.

## Concrete plan to make 10–20 day runs reliable

### Phase 1 — Fix the immediate carry-pair bug

1. **Make crop preconditions explicit in one place**
   - Add a helper that defines the required carry pair for crop work on dry days.
   - For dry crop work, require:
     - watering can in carry pair
     - seed item in carry pair when planting is still needed

2. **Repair carry pair after seed retrieval**
   - After `EnsureCropSeedsTask` succeeds, explicitly re-establish the desired crop carry pair before returning success.
   - Do not let `ENSURE_CROP_SEEDS` report success if it leaves the run in a state that guarantees `CROP_WATER` failure.

3. **Add a dedicated pre-crop ensure task**
   - Either:
     - add `ENSURE_CROP_LOADOUT` before `CROP_WATER`, or
     - teach `CROP_WATER` to request/perform loadout repair before hard-failing on step 1.
   - Preferred: planner-level explicit phase. Easier to log, test, and reason about.

### Phase 2 — Make recovery actually recover

4. **Use targeted recovery for tool/loadout failures**
   - When failure reason is `watering can not in carry pair`, recovery should restore the carry pair, not just walk back to the farm.
   - Add reason-aware recovery that can:
     - ensure watering can
     - ensure seed tool if planting remains
     - return to crop tile only after loadout is valid

5. **Downgrade some failures to deferred work where safe**
   - If planting cannot proceed because seeds are unavailable, defer planting to tomorrow instead of disabling the whole bot.
   - If watering is impossible only because of bad loadout state, attempt one deterministic repair path before considering a hard stop.

### Phase 3 — Make long runs observable and supervisor-friendly

6. **Promote disable-to-idle into disable-to-exit (or fatal status)**
   - On unrecoverable day-plan failure during autoplay, return a non-zero exit or create a clear terminal state.
   - Do not sit there “scanning” like a Roomba trapped under a couch.

7. **Emit richer failure snapshots**
   - On day-plan abort, log:
     - day / season / time
     - phase name
     - selected tool / backpack tool / item in hand
     - seed counts
     - map / tilemap / position
   - Save a labeled state snapshot automatically on first required-phase failure and on post-recovery retry failure.

8. **Add per-day progress checkpoints**
   - Write one compact summary line per in-game day:
     - day, weather, money, active plan, completed phases, deferred phases, current carry pair
   - That will make 10–20 day forensic review much less miserable.

### Phase 4 — Add soak-test coverage

9. **Unit tests for the exact regression**
   - Add a test where:
     - can is initially ensured,
     - seed retrieval mutates carry state,
     - crop phase begins,
     - planner restores valid loadout before crop work.

10. **Planner-level regression test**
   - Add a `DayPlanTask` test covering:
     - `ENSURE_WATERING_CAN -> ENSURE_CROP_SEEDS -> NAV_CROP -> CROP_WATER`
     - verify no hard failure when can exists but is displaced from the pair.

11. **Headless multi-day soak target in CI/local script**
   - Add a repeatable script for a 10+ day autoplay soak with:
     - fixed starting state
     - headless mode
     - log capture
     - pass/fail based on terminal state and day advancement

12. **Add a “stalled but alive” watchdog**
   - Fail the soak if the bot remains in `scanning` or no day/time progress occurs for too long.

## Recommended implementation order

1. Fix crop loadout restoration after `ENSURE_CROP_SEEDS`
2. Add targeted recovery for carry-pair failures
3. Make fatal autoplay failures terminate clearly
4. Add focused regression tests
5. Run a 10-day soak
6. Run a 20-day soak

## Success criteria

A run is not “reliable” until all of these are true:
- completes 10 consecutive in-game days without required-phase abort
- completes 20 days from the same class of starting save
- no silent disable-then-scan state
- failures, if any, produce actionable logs + save snapshots
- the carry pair before each crop phase is visible in logs and validated in tests

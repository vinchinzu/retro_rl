## Residual — rr-20w.3 D3 spring second plot + daily grape

**Status:** planner GREEN + unit-locked, ROM end-to-end RED (grape nav).
Do not STATUS. No `--video`. No full-day soak.

### User spec (2026-09-09)

Spring D3+ every day: ship **2 mountain grapes**, buy **1 potato bag**
before the shop shuts, hoe + sow a **fresh 8-tile plot** beside the D2
rows (16 planted total), water all. Berry forage is a **recurring daily**
day-phase for the whole first spring.

### Done this session

- `_berry_run_phases` (`day_phase_berry.py`): spring (`season == 0`) now
  always schedules `MOUNTAIN_BERRY`; the farm-bush `ship_berry_phases`
  loop is dropped for spring (sealed — rr-w14t). `MOUNTAIN_BERRY_PHASE`
  carries `count: 2`.
- `MountainGrapeShipTask.target_count` (`mountain_grape_ship.py`): after a
  grape reaches the bin it loops back to forage again; a failed 2nd
  pick/return still ends **SUCCESS** once one grape shipped (best-effort).
- `_planting_today` (`day_plan_phases.py`): a same-day seed buy counts as
  plant intent, so `ENSURE_CROP_SEEDS → NAV_CROP → CROP_ESTABLISH` +
  water schedule for the new bag. Excluded on S0D2 (its own
  `D2_FARM_CLEAR` tactic already plants).
- `restock_berries_first`: on a restock day the grape run is ordered
  before the shop hop even when keep-alive crops need water.
- Unit: `tests.test_day_plan_crop_phases.Day3SecondPlotTests`,
  `tests.test_mountain_berry` grape-loop + best-effort tests. Full suite
  green (3 unrelated missing-fixture fails predate this).
- Pin `Y1_D3_Morning` — from `Y1_D2_PowerOn_FarmClear` + one sleep
  (~4.8k f, Clean, `recordings/d2farmclear_to_d3.json`). Spring D3 06:00
  house, $250, 8 D2 potatoes in the field.
- ROM: `run_to_day2 --state Y1_D3_Morning --days 1` → DYNAMIC_OUTDOOR_PLAN
  expands to `BERRY_RUN_WINDOW, MOUNTAIN_BERRY, BUY_SEEDS_WINDOW,
  NAV_FARM_EXIT, BUY_SEEDS, ENSURE_CROP_SEEDS, NAV_CROP, CROP_ESTABLISH,
  ENSURE_WATERING_CAN, NAV_CROP, CROP_WATER` — the exact D3 sequence.

### RED — exact blocker

Mountain-grape nav dies from `Y1_D3_Morning`:
`[MULTI_NAV] No BFS path from (10,422) toward (232, 128)` — the player
leaks onto path `0x0C` at the **far west edge** after the farm→path exit
and BFS cannot recover to the crossroads. The route is only proven from
`Y1_Inside_House` (ships 0→150 in ~3122f). This is the
rr-20w.2.4 / rr-20w.2.5 pixel-stuck / leaked-pose gap — the whole D3
day plan then burns to 18:00 and cascade-skips the shop.

`rr-20w.2.15`'s `pose_is_leaked` / force_run attempt at this made farm
nav worse elsewhere (walked onto `0x10` mid-`CROP_ESTABLISH`) and was
backed out (commit after 3160ba6f).

### Next action

1. Fix the farm→path exit so the player lands near the crossroads (or
   slice the leaked west-edge pose forward) — rr-20w.2.4. Re-run
   `mountain_berry_probe --state Y1_D3_Morning --ship`.
2. ROM-verify a 2nd grape actually spawns/routes same day; until then the
   daily target is effectively 1.
3. Tune `CROP_ESTABLISH` target tiles for a plot beside the D2 rows.

### Non-claims

- No STATUS promotion. Grape route not proven from D3.
- 2-grape daily income not ROM-verified (loop is best-effort).
- Did not power-on to D2 end (used `Y1_D2_PowerOn_FarmClear` + one sleep).

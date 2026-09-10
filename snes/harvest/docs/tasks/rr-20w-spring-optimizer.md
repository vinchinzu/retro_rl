# rr-20w — Spring-1 max-shipping: optimiser + campaign unblock

Session 2026-09-10. Do not STATUS. `Y1_D3_Morning --end-of-spring`.

## Delivered

- **`docs/SPRING_ECONOMY.md`** — ROM crop/economy model:
  - `NightlyFarmTilesCheck` @82A811: watered (odd tile) → +1 stage/night;
    dry (even) → no change in spring (stalls, no regression); rain → free
    grow; summer → spring crops `DEC`/die. Potato = 6 watered nights to
    mature; harvest leaves `0x08` (replant needs no re-hoe).
  - Prices: potato/turnip bag 200 G; potato 80 G/unit, turnip 60; a bag is
    **9** plants (`$096B`→9); grape ~150 G.
  - **No forced bedtime** (user confirmed) — water all night, refill can +
    stamina freely. Plot count is not daily-frame-capped.
- **`harvest.planner.spring_opt`** + `harvest.scripts.spring_plan` — infinite-
  evening beam search. `--sweep` saturates ~18–20 potato rings
  (~$22 k final / $31 k gross) at the harvest-pile-up point vs the current
  reactive campaign ~$1.5 k. `CostModel` is calibratable from run logs.
  `recordings/spring_plan.json` = the 4-ring plan (bump `--rings`).
- Tests: `tests/test_spring_opt.py` (8).

## Bugs fixed

1. **Grape `return_to_bin` pin (rr-20w.3.1 residual).** `960585b3` put
   `run_direction="down"/force_run` on the shared `(520,712)` mountain-exit
   waypoint; the grape *return* slices that list and force-ran into the
   carpenter terrace wall at ~(505,633) → `return_to_bin timeout` → farmer
   stranded on 0x10 → BUY_SEEDS skipped → `CROP_ESTABLISH select_carry_0x07`
   **every day** (run8/run10). Fix: plain waypoint + `MultiMapNavTask`
   run_direction stall guard (90 f no-progress → BFS that wp). Commit `2204e37b`.
   **Verified run11**: grape ships + returns D3–D8 every day, money climbs.
2. **`SwapCarrySlotsTask` farm timeout (rr-20w.3.2).** The `[hoe,can]→[can,hoe]`
   preserve-hoe swap timed out on the farm after BUY_SEEDS (walk-settle eats
   the X tap). Fix: gate on `player_action==0`, 2-frame X pulse, timeout
   480. Commit `7d07444d`. Not yet in a full run.

## Evidence in flight

`logs/spring_d3_30/run11_grapefix.log` (grape fix, no WIP crop_skills):
D3 establish still lost to carry-swap (self-recovers D4); D4 ring 1 planted;
D5–D8 water+grape, money $350→$800. Watch for the D10 harvest + replant.

## Open / next

- **run12**: HEAD + carry-swap fix + WIP `crop_skills.py` (rr-20w.3.2 ring
  nav) → does replant now cycle? Target: 2+ potato cycles, money ≫ $1540.
- Campaign day-plan is erratic (run11 D7 skipped CROP_WATER entirely) — the
  reactive `_berry_run_phases` / `_planting_today` heuristics should be
  replaced by a `spring_opt.SpringPlan` schedule fed to `multi_day_planner`.
- Only 1 grape/day actually ships (`stopped early (shop window)`); the 2nd
  grape is never picked. Either fix the 2-grape loop or accept $150/day.
- **Top lever**: a catalogue of nav-executable ring sites in the cleared
  farm (>2). `crop_planner.plan_crop_field` scores them; each needs proven
  hoe/water/harvest stands. Every extra ring ≈ +1700 G/spring.
- Calibrate `spring_opt.CostModel` from run11/run12 measured frame costs.

## Non-claims

No STATUS. Did not start from `Y1_D2_Morning_After_D1`. Did not record a
BFS-closable walk. Optimiser numbers are model output, not ROM-measured
end-to-end. Grape sustained-yield past D6 still unmeasured.

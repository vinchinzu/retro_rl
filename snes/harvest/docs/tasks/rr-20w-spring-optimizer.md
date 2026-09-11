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
- Campaign day-plan is erratic (run11 D7 skipped CROP_WATER entirely).
  Do not replace `DayPlanTask` with a `SpringPlan` dispatcher;
  `optimize_spring` stays a wallet search. Feed quantities (grape count,
  rings to plant/water/harvest) into the existing phase table.
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

## 2026-09-10 — ROM model corrections (research pass, no emulator)

Full writeup: [SPRING_ECONOMY.md](../SPRING_ECONOMY.md) §0-§10. Summary:

- **The "summer wipes spring crops" claim above (line 10) was wrong.**
  `NightlyFarmTilesCheck`'s season dispatch (`bank_82.asm:2440-2458`) only
  kills a growing crop in **Fall** (`season==2`); Winter freezes it;
  Spring *and* Summer both grow it. The only wholesale wipe is
  `MonthlyFarmTilesCheck`, called once a year on **Winter D1**
  (`bank_82.asm:2526-2534`). A spring-planted potato ring left unharvested
  at Spring D30 simply keeps maturing in Summer — it is not destroyed.
- **New, more important constraint found:** potato/turnip can only be
  *planted* (produce a real growing tile) when `season==0` (Spring);
  planting them any other season silently drops a dead placeholder tile
  (`$00EB`, never processed by the nightly check) and still burns the bag
  charge. Corn/Tomato are the mirror image — Summer-only plantable.
  (`bank_82_toolused_subrutines.asm:742-936`.) Net effect for this
  optimizer's Spring-only, ship-by-Summer-D1 objective: the practical
  "stop planting new ground ≈D24" cutoff survives, but for a *different,
  weaker* reason (can't mature+ship in time) rather than crop death — so
  it should be revisited if the objective horizon is ever pushed into
  Summer.
- **Hurricane risk is real but Spring-safe.** Hurricanes can only occur in
  Summer (`Hurricane_Chance_Table`, `bank_82.asm:1580`, zero for every
  other season), ~1/30 per eligible summer night (~1/60 with a Turtle
  Shell), and when one hits, every tilled/crop tile independently has a
  **25% chance of being wiped** (`ClimateFarmDamageCheck` Hurricane row,
  `bank_82.asm:346`, `2273-2286`). Irrelevant to the Spring-D1-horizon
  objective as currently scoped; a hard blocker for any future
  push-into-Summer extension.
- **Rain never damages crops** (Rain/Snow damage-table Crops column = 0,
  `bank_82.asm:344`) and **weeds can never appear on or destroy a
  tilled/crop tile** (`.addrandomtrash` only reachable for raw tile values
  `<3`, i.e. bare ground, `bank_82.asm:2335-2337, 2470-2490`).
- **ROM-confirmed, previously-asserted-without-citation facts:** watering
  can capacity = 20 (`bank_82_toolanimation_subrutines.asm:163`); seed bag
  = exactly 9 plants, a hard game limit via a RAM counter shared across
  all four crop types (`$096B`, `bank_82_toolused_subrutines.asm:769-929`);
  hoe swing and watering-can use each cost 2 stamina, max stamina 100,
  fully refilled every `NightReset` regardless of prior spend
  (`bank_81.asm:7590`, `bank_83.asm:4549-4550`, `bank_82.asm:459-461`) —
  reinforces "no forced bedtime."
- **Prices cross-validated:** `Items_Price_Table` (`bank_81.asm:3060`,
  confirmed live via the shipping-bin credit path,
  `bank_81.asm:2563-2597`) has a contiguous 4-value block (items 16-19 =
  120/100/80/60 G) matching Corn/Tomato/Potato/Turnip exactly against
  external guides (fogu.com). Potato 80G / Turnip 60G, both 200G seed,
  unchanged from the prior doc. Corn/Tomato 120G/100G ship, 300G seed
  (seed cost external-only, not ROM-decoded).
- **Miscitation fixed:** the prior "shop hours 7-17, closed weekends"
  citation to `DATA16_B9CE24` was wrong — that address is plain dialogue
  text (`bank_B9.asm:1464`), not an hours table. The real shop-hours gate
  is still unlocated; treat the hours/weekend-closure claim as
  community-sourced, not ROM-verified.
- **Known wrong constants in code (not edited — file owner's call):**
  - `harvest/planner/spring_opt.py:24` — `SUMMER_KILLS_SPRING_CROPS = True`
    encodes the disproven belief directly.
  - `harvest/planner/spring_opt.py:58` — docstring says "cannot mature
    before the D30 summer wipe"; there is no D30 summer wipe (only a
    Winter-D1 wipe). Same fix needed as the constant above.
  - `harvest/planner/crop_planner.py` was checked too and is **already
    correct**: `CropSpec` already scopes potato/turnip to `SEASON_SPRING`
    and corn/tomato to `SEASON_SUMMER` with matching ROM-confirmed prices
    (60/80/100/120) and seed costs — no change needed there.
    `require_harvest_before_season_end` (line 231) is a legitimate
    horizon-boundary choice for the "ship by Summer D1" objective, not a
    factual error.
- **Open questions honestly unresolved this pass:** shop/shipping open
  hours and weekend closure gating code; grass/fodder, egg, milk, and
  forage (grape/mushroom) ROM ship prices; corn/tomato ROM seed cost;
  exhaustion's actual gameplay effect; hot-spring stamina restore rate.
  See SPRING_ECONOMY.md §8 for the full list.

## 2026-09-10 (session 2) — cash-flow ledger, sowing-season gate, chicken stub

No emulator this pass. All model work; full writeup in
[SPRING_ECONOMY.md](../SPRING_ECONOMY.md) §11-§13.

### Bug fixed: the optimiser was sowing potatoes in Summer

The previous pass *documented* the ROM rule (potato/turnip only sow a live
tile when `season == 0`; out of season the bag is burned for a dead `$00EB`
placeholder, `bank_82_toolused_subrutines.asm:742-936`) but never encoded it.
With the default horizon at Summer D30 the search was establishing rings on
D31/D32 and booking the harvests — which is where the previous headline
"~18-20 rings / ~$22 k" came from. `_plantable()` now gates both the
establish loop and the seed-buy branch; `_rank` also stops crediting bags
that can no longer be sown (200 G of dead stock was being ranked as an
asset). **Corrected spring-only figure: ~10-13 k G, saturating at ~4 rings
at 18:00 sleep / ~12 at 22:00.**

### Search fix: money-free beam dedup

`State.key()` included `wallet`/`pending_ship`, so the beam filled with
wallet-variants of one farm configuration and crowded out genuinely
different farms — visibly, a *cheaper* cost model scored below a dearer one.
Money is monotone here (every action is a choice bounded by the wallet,
never forced by it), so same-day/same-bags/same-farm states dominate by
wallet and the key drops money. Tightens the search as well as fixing the
inversion.

### Delivered

- **Cash-flow ledger.** `DayAction` carries `wallet_start`,
  `crop_income_g` (posts next morning), `berry_income_g` (same-day),
  `seed_spend_g`, `livestock_spend_g`; `SpringPlan.ledger()` prints the
  input/output table, `SpringPlan.cashflow` rolls it up, `to_dict()` carries
  both. Every row balances exactly (asserted per-day in tests).
- **Binding-constraint diagnostic.** `cash_blocked` (empty ring + free
  evening + no 200 G) vs `frame_blocked` (empty ring + bag in pocket +
  evening full). **Result: zero cash-blocked days at every farm size
  tested; 8 rings is frame-blocked on 12 days.** Cash is not the
  constraint — which independently corroborates `rr-20w-idle-day.md`.
- **Berries are 50-65 % of income at every size** (4 rings/18:00: 3 840 G
  potato vs 7 200 G grape). `grape_value_g = 150` is still a GUESS, so it
  is now the highest-value number left to measure; added a grape-price row
  to `--sensitivity`.
- **`min_cash_reserve`** (`--reserve G`) — a floor the plan may not spend
  through — and `SpringPlan.earliest_day_affording()`. Both exist for the
  chicken.
- **`harvest/planner/livestock_econ.py`** — chicken/cow purchase stub.
  Decision mechanism complete and tested; inputs deliberately `None` so
  `plan_purchase` returns "unknown + what to measure" instead of a guess.
  `--chicken` prints it, plus the half the ledger *can* answer already
  ("if a chicken costs 1500 G, this plan first affords it on D9").
- **Egg/milk ship prices ROM-corroborated.** `Items_Price_Table` continues
  `12,10,8,6 | 5,15,25,35` — egg 50 G, milk S/M/L 150/250/350, one block
  past the already-pinned corn/tomato/potato/turnip. Upgrades four §3.2
  rows from "ext, UNVERIFIED-BY-ROM". Chicken *purchase* price is not in
  that table (ship prices only) and was not found; pointer left at
  `ReplaceTilesAnimalShop` (`bank_81.asm:4844`).
- **Horizon default is now Spring D30**, not Summer D30. `--through-summer`
  still projects, behind a printed caveat banner. §13 is the written plan
  for actually measuring summer: corn/tomato regrow branch, their
  ROM seed cost, hurricane scenarios, summer frame calibration — in that
  order, and quote a distribution not a number.
- Tests: `test_spring_opt.py` 8→27, new `test_livestock_econ.py` (6).

### Open / next (unchanged priorities, re-ranked by the ledger)

1. **Frames, not money.** More nav-executable ring sites (§10.1) and the
   idle-afternoon work loop (`rr-20w-idle-day.md`) are the only levers the
   ledger says are binding. Seed capital never is.
2. **Measure `grape_value_g`.** It is over half the modelled income and is
   a guess. Also fix the 1-of-2-grape shop-window cutoff.
3. **run12** (HEAD + carry-swap fix + WIP `crop_skills.py`) — still the
   outstanding ROM proof that the replant cycle cycles at all. Calibrate
   `CostModel` from it.

### Non-claims

No STATUS. No emulator run this session — every number above is model
output. The chicken is *not* priced; nothing in the ledger spends on it.
Summer figures are projections and are labelled as such in the tool.

### Correction, same session — calibrated against run12, numbers moved

The figures above ("~10-13 k G", "berries are 50-65 % of income", "zero
cash-blocked days") were model output from an **uncalibrated** cost model,
posted before `logs/spring_d3_30/run12_short.log` — which already existed —
was read. Calibrating against it (`spring_plan --calibrate`) moved several
defaults and two of the three headline claims:

- **Day length was under-counted ~27 %.** Measured day-change deltas: n=9,
  median 12 859 f. The model's 06:00-18:00 policy budgeted 10 800.
  `sleep_hour` default 18 → 20.
- **Two runtime caps were not modelled at all**, and they dominate:
  `max_bags_per_day = 1` (every `BUY_SEEDS` in every log reads
  `0->1`) and `grape_max_per_day = 1` (run12 ships 1/2 on 4 of 6 berry
  days; D7-D9 earned nothing — one `pick: farm_to_path: pixel_stuck`, one
  15:03 `BERRY_RUN_WINDOW` cutoff). Neither is a ROM limit.
- **Corrected figure: ~8 300 G**, saturating at ~4-6 rings — not 10-13 k.
- **Lifting the two runtime caps is worth ~+3 200 G (+38 %)** → ~11 500 G.
  That is now the top-ranked fix, ahead of new ring sites.
- **"Berries dominate" was regime-dependent, not a fact.** Under the real
  caps crops lead (7 040 vs 3 000 at 6 rings); with caps lifted berries lead
  again (7 050 vs 5 760). `grape_value_g = 150` is now *confirmed*
  (`shipping_money=150->300`), so the open question is the count, not price.
- **"Zero cash-blocked days" no longer holds** — the bag cap prevents
  front-loading seed buys, so cash-blocking reappears (1 day at 3 and 6
  rings). Frames still bite only at 6+.

Also exposed and **not** fixed: `RING_TILES = 8` books 1 280 G for two rings
where run12 actually shipped 15 tiles / 1 200 G (~5 % optimistic), and
**nothing past D13 has ever been executed** — run12 is a `until=(0,13)` short
run, so the D14-D30 replant steady state the whole plan rests on is still
unproven. Full provenance table: SPRING_ECONOMY.md §11.4.

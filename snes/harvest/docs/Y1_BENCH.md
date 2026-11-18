# Year 1 benchmark

## Score policy comparison (2026-11-12)

The first frozen `score_herd` attempt was interrupted at the owner's request on
Spring D2: **47,734 frames / 72.8 seconds**, no failed days, loads, writes or
assists. It is a stopped prefix, not a failed annual gate. Its report and
423-file runtime manifest remain in
`recordings/y1_bench/score_policy_20261112/`.

After the owner resumed work, a fresh power-on comparison was started from a
new frozen tree in `recordings/y1_bench/score_policy_final_20261112/`.
That manifest is authoritative. The later player-happiness festival and
whole-field purchase fixes only add behavior from Year 2 onward; they cannot
affect the Year 1 input timeline.

**Result (finished 11:41):** power-on to **Y2 Spring D1**, 2,610,023 frames / 3,197.6 s, CLEAN (one initial load, zero
mid-run loads, RAM writes or assists), 119 days, **1 bad day** (Winter D24 Star Night, cosmetic), 0 rescues, 0 floor trips.
Against `run16_fixes_y1.json` (2,030G, 36 crops shipped, 31 bad days): net worth **79,850G** (79,350 at resale), wallet
68,850G, 2 cows and 1 hen, **522 crops shipped** (corn 190, tomato 164, potato 168, turnip 0), bin gold 99,340G, first hen
day 38, first cow day 46. Affection at the Year 2 start: Maria 196, Ann 196, Nina 192, Ellen 222, Eve 120. **Berries 7 of
10, 170 max stamina**; scorecard in `score_policy_final_20261112/scorecard`.

Faults the year exposed (each reproduced from its morning pin, `PR_score_policy_final_20261112_Y1_*`):

- **Harvest Festival berry lost (Fall D12):** `pixel_stuck pos=(566,593)` on mountain map 0x10 with the fish in hand. The
  farmer stood 7 px above a tile centre with a solid below, so the vertical centring before a sideways step could never
  finish; each pixel-stuck replan reset `stasis` (centring yields at 30) before the 48-frame replan timer, and four
  replans failed the walk. Carrying (`player_state` 3) also counted as "buttons dropped", hiding the wall for 240 frames.
  Fix: an opt-in stalled-centring detector (`Navigator.center_stall_limit`, `CENTER_STALL_FRAMES` 30 live-pad frames on one
  pixel, counted across replans), switched on only for the festival-entry walk (`FestivalEntryTask`). Replay from the Fall
  D12 pin: berry 6 -> 7 in 6,552 frames, zero writes or loads; the six festival ROM tests pass. Two earlier, broader versions
  failed and were dropped: skipping centring after any pixel-stuck replan broke the Y1 Spring D2 opening plant
  (`test_opening_day_two_plants_before_any_courting_detour`), and the detector on every walk killed fresh power-on runs on
  Y1 Spring D2 (`compose3`, `compose4`) although every pin ROM test passed. Opt-in leaves every other walk, and Year 1 up to
  Spring D23, frame-identical to this bench.
- **Spring D23 Flower Festival missed (dance, snow-flower seed, Eve hearts):** the capped `CROP_WATER` (`stop_hour` 12)
  ran 06:04 to 18:12 (17,017 frames) because the main-pond refill from the south stand (32,34) looped through eight
  "south-lip charge" detours; `stop_hour` only limited how many tiles were armed. Fix: a runtime deadline,
  `stop_hour` + 1 hour (`STOP_HOUR_GRACE_HOURS`), after which the watering hands the day back as SUCCESS. A replay from
  the pin does not reproduce the loop (it waters 27 tiles by 12:00 and buys the seed), but the power-on timeline does,
  twice (the bench and `compose_20261112`), so only a power-on run exercises the guard.
- **Spring D23 on the power-on timeline with the guard (`compose5_20261112`):** the guard fired at 13:00, the Flower Festival
  completed and `snow_seed_bought` / `snow_seed_owned` are True at the D24 morning (the bench had neither); money 3,840G against
  5,840G (the 2,000G seed). The cost: the capped watering ended `failure: dry_crops=33 watered=0` (the bench watered 27 tiles
  late in the day), so Spring D23's field went unwatered for a night. The loop starts at the pond's south stand (32,34): on the
  same morning terrain a static `Pathfinder` finds a 2-step path from (31,35) to it, so the blocker at that moment is dynamic
  (a sprite is the suspect; the log names none). It never reproduces from a pin.
- **Courting took the Summer corn trip (found in `compose5`):** with the Flower Festival working, the field was one night
  behind, so 58 potatoes ripened together on Summer D2. D2's harvest ran to the 17:00 shop close (47 shipped), and D3's
  last 13 were picked by mid-morning, but the plan was `HARVEST_ROUTE, SOCIAL_TOWN, DYNAMIC_OUTDOOR_PLAN`: Nina's court
  ran to 16:01, the dynamic plan was a no-op, and the weekend shut the shop. No corn went into the rings, and Summer D17-D30
  had no harvest income (wallet $1,460 against the bench's $23,500 at Summer D29, no cow before Fall). Fix:
  `social_calendar.ring_corn_trip_first` (Summer D1-D3, shop open, ripe score-family crops due) makes `social_phase`
  stand down, so the day is `BUY_SEEDS, HARVEST_ROUTE, ENSURE_CROP_SEEDS, CROP_ESTABLISH x5, CROP_WATER`. Pin replay from
  `PR_e2ecompose5_20261112_Y1_Summer_D3`: five rings of corn planted on D3, wallet 12,620G at the end of the day.
- **Refill stand measurement (all three Year 1 logs: bench, `compose`, `compose6`):** a refill that targets the pond's south
  stand **(32,34) succeeded 91 of 199 logged attempts (46%)**; (32,30) 70 of 77, **(33,34) 65 of 67 (97%)**, (34,30) 37 of 37.
  Each failed (32,34) approach starts the multi-hour "south-lip charge" loop; `compose6` lost four whole watering days to the
  24,000-frame watchdog (Spring D26, Summer D2, D7, D14). The adjacent (33,34) is the stand that finally works every time.
  (32,34) is wired into `map_config` (verified south-lip fill), `pond_hop` and `fence_corridor`, so the next step is to make
  the selector prefer (33,34) there and re-check the hops, not to blacklist the stand.
- **Year 1 close-out of `compose6_20261112` (power-on, `score_herd`, all fixes above):** Y2 Spring D1 at frame 3,077,316
  (the bench: 2,610,023), **9 berries / 190 max stamina** (bench 7 / 170): statue, both field berries, fish, stump, Harvest
  Festival (Fall D12, the festival-entry fix), Egg Festival (this run's coin flip paid the berry, not the turtle shell), snow
  flower (the Spring D23 seed, eaten in Winter) and the cave. Missing: the hen trade, which the calendar offers only in Fall
  with two adult hens (this run had one); the Y2 Fall trade is its second chance. **Eve is married** (proposal and wedding on
  Y1 Summer D18, driven by the working Spring D23 dance), wallet 55,670G at Winter D30 (bench 68,850G). Unfinished: the
  turtle shell (Y2 Egg Festival), the hen trade, children, development, crop quotas and the Y2 happiness top-up.
- **Y2 Spring D8 wedge (`compose6`, 2M frames, run killed):** `HOUSE_WOOD` timed out, then `ENSURE_ANIMAL_TOOLS` never
  returned. Cause: the whole-field cover's young grass (`0x75`/`0x76`, 389 tiles outside the pasture) made
  `farm_map_loaded` read the loaded farm as unloaded, so `EnsureCarryToolTask._needs_leave_shed_door` stayed True and its
  leave-the-door walk (which succeeds at once) restarted forever; with the day floor off (owner: no per-day cap) only CROP and
  HARVEST had a watchdog. Fixes: `farm_map_loaded` / `stale_under_player` use the cover bounds under the cover, and every other
  phase gets a 120,000-frame backstop (`phase_backstop_frames`). The pin replay now plays the whole day in 19,449 frames, Clean,
  `ENSURE_ANIMAL_TOOLS` 3,490 frames. Known failing test, not caused by these changes (it fails against the pre-edit runtime
  too): `test_power_berry_planner_rom.py::test_planner_visits_the_missing_stump_after_the_early_berries_are_done` (the stump
  trip's first-attempt `nav timeout`, as on every run's Spring D14).
- **Refill loop, second mechanism (Summer D14 replays from `PR_e2ecompose6_20261112_Y1_Summer_D14`; reproduces):**
  CROP_WATER starts ~f=13,000 (evening), the first refill at (32,30) works (~f=15,500), the second asks for (32,34) with the
  farmer at (8,42) in the south-west; the "South-lip densify (8,42) -> (12,42)" hop thrashes ("Densify thrash ... n=6"), the
  west->south-lip charge macros then run to (62,42), (29,22), (9,40) ... until the 24,000-frame watchdog. A fresh `Pathfinder`
  from the farmer's tile finds a 27-step path to (32,34) in the same state, and the pond rim is open, so the suspect is the
  densify hop target (12,42): in Summer the south-west block holds growing crops, i.e. an unwalkable hop target. Save a state with
  `--max-frames 17950 --save-end-state` on that pin to continue. The Spring D23 loop (farmer at (31,35), "no path" to (32,34))
  is a different trigger (a sprite) and does not reproduce from a pin.
- **Y2 Spring D16 wedge (`compose7`, died after 8 rescues; the run before the fix got past D8):** the 6th cow is bought at the
  animal shop, then `RETURN_FARM_AFTER_COW_PURCHASE` pixel-stuck four times at the shop front and the rescue chain re-entered
  the shop, where `ExitToFarmTask` (farm buildings only) timed out every time. Cause: sprite `0x023C` is **Ellen**, who stands at
  town (648,904) on weekday mornings (`calendar_schedule`), and `animal_shop_town_to_farm` began with a radius-4 waypoint on
  exactly that pixel. Fixes: (1) that route now stops at (632,904) r10, swings south of her spot ((656,936) r10, then (696,920)
  r8) and joins the x=696 column from below; (2) `ReturnHomeTask` handles the animal shop interior and the plaza outside it with
  the shop-exit and town routes instead of `ExitToFarmTask` (two attempts, shared with the mountain return); (3) a sprite whose
  90-frame yield timed out is "parked" and its ring is padded for the rest of that waypoint on any map
  (`nav_corridor.pad_entity_blocks(force=...)`, reset on the next waypoint). Pin replay from
  `PR_e2ecompose7_20261112_Y2_Spring_D16`: 35,690 frames, Clean, no bad day, no rescue, the cow bought and all chores done.
  The walk-home trace from the phase-start state takes ~575 frames to the town gate.
- **Open, not fixed:** the same refill loop hit Summer D14 (`CROP_WATER` watchdog, 24,001 frames, a day of corn
  watering) and the Egg Festival paid the turtle shell (the berry needs the other prize; Year 2 gives the second try).
  Fall D4's hen trade also timed out (22,792 frames) and succeeded the next day. The south-stand refill
  (`tasks/pond_policy.py`, `crop_refill.py`) is the next reliability item.

## Berry calendar power-on prefix (2026-11-11)

`berryfix_20261111` tests `score_herd` from power-on with the berry repairs in a
frozen tree (`recordings/y1_bench/berryfix_20261111.source.json`). It was stopped
at **Y1 Spring D3**, 68,646 frames / 501.5 seconds, after Day 2's `CLEAR_PLOT`
repeated the blocked approach `(201,526)` and failed. Wallet 300G, no animals,
no planted grass, one bad day. The report's terminal reason is the requested
interrupt; this is an incomplete prefix, not a completed year or berry campaign.
Zero initial/mid-run loads, RAM writes, assists or rescues.

Report/log: `recordings/y1_bench/berryfix_20261111.{json,log}`. Both scorecards
were compared with `run16_fixes_y1.json` in `berryfix_20261111_comparison.txt`;
different end dates prevent an annual delta claim. Native berry-source and
planner checks are recorded separately under `recordings/berries/` and in
[plan.md](plan.md#power-berry-route-repair-2026-11-11).

Next power-on gate: repair the Day 2 farm approach, then repeat the year. An
earlier verified Spring checkpoint can exercise the berry calendar separately.

## Twelve-cow checkpoint ending: 851 (2026-11-11)

`recordings/y23_sweep/ending12_20261111_s1.json` reaches Y3 Summer D30 credits index `0x25`, ROM score **851**, from
the natural `PR_winter12c_20261111_s1_Y3_Spring_D2` checkpoint. **All twelve cows finish healthy (status 0x09),
happiness 255 each**, paying 336/336 count-and-affection points. Twelve chickens, 494,490G, 60 hay. Relationships
154/155 (Eve 494; the other four are 496–500), berries 6/10. Scorecard: `recordings/y23_sweep/ending12_20261111_FINAL.md`.

3,708,687 frames / 4,370.1 seconds wall: 72.8 minutes, 849 fps, 14.1 times real time. Audit: one initial checkpoint
load, zero mid-run loads, RAM writes, assists or rescues. Required-phase failures: Spring D4 coop navigation and D6
cure map lock. The score is 82 above the earlier eight-cow 769 ending, but the starting dates and routes differ;
neither an isolated A/B nor a fresh continuous power-on result is claimed. `STATUS.md` remains unchanged.

Its preceding `winter12c_20261111_s1` replay proves the hay repair across Winter: Fall D28 to Y3 Spring D2,
all twelve cows retained at happiness 255, 224 hay left, 1,116,390 frames / 1,357.8 seconds. The original twelve-cow
attempt had reached maximum affection, then lost nine cows after starting Winter with only 115 hay. Full cause,
native cutter receipts and remaining score gaps are in [plan.md](plan.md#twelve-cows-and-the-harvest-festival-berry-2026-11-11).

Next: repair Eve's last-day gift from the new Summer D29 pin, then collect the missing berries from earlier dates.

## Barn verification follow-up (2026-11-06)

`barn_complete` uses the `score` strategy and the completed barn fixes. It was stopped on
**Y1 Spring D4**, after **639,020 frames / 1,080.1 seconds**, because return-home repeatedly
failed to clear held item `0x08` near `(518,550)`. The journal has four `return_home/rescue`
failure records. The run is **partial and not comparable as a completed year**: 250G, no cows,
no hens, nine pasture tiles, no hay. Audit: zero initial loads, mid-run loads and RAM writes,
with empty assists. The scorecard's zero rescue total must not erase the journal's failed
return-home rescue attempts.

Reports: `recordings/y1_bench/barn_complete.json`, `barn_complete_scorecard.json` / `.txt`,
and `barn_complete_comparison.txt`. The comparison uses `run16_fixes_y1.json` (completed
Y2 Spring D1: 2,030G, no animals); different end dates prevent an annual wallet/performance
claim. The new prefix's cow gate never opened and waited on season for its two morning reads.
It cannot establish barn economics before a cow is bought.

Barn component verification is separate: **165/165** barn trials from the original 200 seeds,
**32/32** independent knock trials, and **41 ROM pytest cases** passed. The strict outside-cow
roundup profile remains **7/35**. Receipts, source hashes and test logs are under
`recordings/barn_fuzz/barn_complete/`; the full breakdown is in `docs/plan.md`.

Reproduce from the repository root:

```bash
HEADLESS=1 QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 \
  --power-on --strategy score --until-year 2 \
  --out snes/harvest/recordings/y1_bench/<new-tag>.json
```

Next campaign check: repair the clear-hands/return-home loop on Spring D4, then repeat the year.

Run power-on through Winter D30 from `snes/harvest` with its venv (`uv sync` once; PyYAML is a declared dependency, the root venv works too): `HEADLESS=1 .venv/bin/python -m harvest.scripts.run_to_day2 --power-on --until-year 2 --out recordings/y1_bench/<name>.json`.
Score it: `.venv/bin/python -m harvest.tools.y1_bench recordings/y1_bench/<name>.json [other.json] [--out card.md|card.json]`; two reports add a B-A delta.
Every economy number (prices, reserves, pasture and hay targets, the cow target) is in `harvest/planner/livestock_econ.py`. Journal rows carry `gates` (`ranch_gates`, the same morning read as `economy`), and the scorecard prints the first morning the cow gate opened and how many mornings each gate held it (`cow waited on`). Reports older than 2026-10-28 have no `gates`.
Headline is net worth = wallet + 1,000 x chickens + 5,000 x cows, both parts stated. Only a power-on, zero-write, zero-assist run is marked comparable.
Journal rows carry `economy` (morning read, same moment as `money`) and `eod` (pre-sleep read). `shipping_money` is read after the night payout and is always 0. Older reports use `eod.bin_gold`; new reports use the frame ledger described below.
Milk and egg gold is bin gold minus the lifetime crop counters' delta at ship prices minus forage `shipping_deposit`s, on days with cow or coop chores. Milk tier and egg count are unrecorded, so it is an estimate.
Reports older than the `economy` fields print `n/a` for brush, milker, bin gold and animal gold, and mark flock sizes `~` (from purchase receipts). `planted` is `n/a` while the establish phase reports "sequence complete". Chickens include chicks and incubating eggs.

## Spring frame measurements

The existing `run_to_day2` runner now records `work_metrics` alongside the
sales ledger. `DayPlanTask` remains the only work dispatcher. The observer in
`core/task_progress.py` sends no inputs and cannot select a route.

- `phase_results[].work` contains before/after clock, position, stamina, can
  level, seed stocks, wallet and bin value; crop counts are read only on a
  loaded farm. Counts survive partial failures and day-floor interruptions.
- `work_metrics.segments` preserves task states, target/stand coordinates and
  visit order. Frame intervals are `[start_frame, end_frame)` and charge only
  actual emulator steps. Boot, D1 handoff and terminal calls without an action
  are excluded. `phase_frames` totals include navigation, equipment and sleep,
  so an optimizer can account for travel outside the crop phase too.
  Segment counters are cumulative task snapshots; sum the completed phase
  receipts for quantities, never the counters repeated across segments.
- The scorecard's `farm_work` section reports frames per watered tile, newly
  planted tile, or confirmed shipped crop. Rates use the emulator-step ledger
  where available; `receipt_frames` retains the Composer's phase-clock spans.
  Plant counts come from live crop-count
  increases, including a partially occupied ring, rather than "sequence
  complete" prose. Plant gates require new tiles plus successful skill
  completion; water gates require zero remaining dry crops. Indoors or with
  an unloaded map, a gate is unknown. Grass is excluded from crop watering.
- Harvest quantities for frame rates come from the native lifetime-counter
  ledger. `shipment_delta` exposes the task's reported count minus native
  receipts. Empty hands after a throw do not prove a sale: late shipping can
  report success without a bin increase. A counter decrease invalidates the
  confirmed rate; missing native evidence stays unknown.
- A successful phase can still leave dry crops: always read the water gate
  and skipped count. Per-phase rates include the work that failed or was
  interrupted, and exclude separately reported tool-fetch/field approaches.
  Old reports without these receipts retain unknown metrics.

An older day-floor bug could copy a whole previous day's journal into a
morning skipped before plan activation. `farm_work.duplicate_receipts` lists
repeated nonzero frame intervals and excludes them from its quantities. Use
the native sales ledger for those older runs; their headline journal totals
are not corrected. The planner now activates and resets a new day before
checking its work budget.

Future routing changes belong in `maps/farm_pond.py` (plant centers),
`crop_geometry.build_water_steps` (stands/order), `harvest_task.build_harvest_steps`
(pick/ship order), and the existing phase/policy tables. Compare the same
natural pin through the Composer and this report before a full Spring replay.
Preserve the shop, shipping and last-sow deadlines when changing layouts;
compare actual crop receipts as well as faster frame counts.

Reproduce from the accepted D2 save, from `snes/harvest` (use a new pin prefix
and output name for each comparison):

```bash
HEADLESS=1 QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 \
  --state LN_d2_perfect_poweron --until-season 1 --until-day 1 \
  --save-day-pins --pin-prefix PR_spring_compare_ \
  --save-end-state LN_spring_compare_summer_d1 \
  --out recordings/spring_y1/compare.json
uv run python -m harvest.tools.y1_bench recordings/spring_y1/compare.json \
  --out recordings/spring_y1/compare_scorecard.json
```

An initial checkpoint load is recorded explicitly. Such a continuation is
useful for Spring reliability and routing; it does not replace the separate
continuous power-on economy proof. Daily `PR_` pins retain planner lineage.

Same-state watering check, natural `PR_spring_check_Y1_Spring_D7`: the old
field pre-nav plus water path took **11,221 frames**, watered seven and skipped
one. The revised path takes **2,263 frames** (water phase **1,435**), waters
all eight and skips none: **8,958 frames saved**. The can swap must not consume
the facing window. The old fixed field approach also ends in a fragile pose;
the per-tile watering skill owns the stand approach now. This is a bounded
same-state comparison, not a monthly throughput claim.

ROM checks from the repo root:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests/test_floor.py \
  snes/harvest/tests/test_buy_seeds.py snes/harvest/tests/test_crop_skills.py -q -m rom
```

## Day-3 pin: nine rings and spa watering (2026-10-31)

All runs start from `LN_d2_perfect_poweron` (the accepted D2 evening; one
initial load, zero writes and assists) and stop on Fall D1. One draw each, on
frozen scratch copies of the tree; reports and gzipped logs are in
`recordings/spring_y1/d3pin/` (gitignored). Weather follows the RNG path, so read
Summer with its storm days: base and spa drew Summer D10, D14 and D16; rings
D6 and D27; v2 none. `sumbase` replays the base tree from its own Summer D1
pin and drew none, so it is the stormless Summer reference for v2.

| run | change on top of the row above | Spring-end wallet | Spring crops | Summer crops | Summer storms | bad days Sp/Su | Fall D1 net worth | frames |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| base | tree at the start of the session | 7,470 | 131 | 60 | 3 | 1/3 | 7,980 | 989k |
| spa | `CROP_WATER` reports `stamina_low`, one hot-spring soak, rerun | 7,470 | 131 | 63 | 3 | 1/2 | 9,780 | 997k |
| rings | + four `WEST_STRIP_PLANT_CENTERS` (nine rings) | 7,280 | 140 | 263 | 2 | 7/2 | 17,080 | 1,421k |
| v2 | + ring access rule, 17:00 ship window, festival-morning water, sow to D26 | 9,480 | 183 | 296 | 0 | 3/0 | 30,040 | 1,607k |
| v5 | + festival-eve spa pool, west-steps gate route, yard exit, establish skip | 10,720 | 195 | 326 | 0 | 1/2 | 34,690 | 1,525k |
| final | + 12-minute ship margin, centred exit-row hop (the tree as left) | 9,830 | 198 | 343 | 0 | 0/1 | 36,130 | 1,746k |
| sumbase | base, replayed from its Summer D1 pin | | | 171 | 0 | -/3 | 17,950 | 549k (Summer) |

Spring-end wallet is the Summer D1 06:00 read, before the corn buy. v2's
Summer D1 bought seven corn bags, not nine: two rings still held the D25-D26
potatoes. v5 shipped 521 crops for 54,620G of bin gold, bought the hen on
Summer D10, the first cow on D14 and the second on D20, and paid the first
house upgrade (5,000G). It spliced 31 soaks and lost none; the all-wet gate
passed on 22 of 75 watering phases (v2 12 of 95). Its day cap tripped twice
(Spring D11, Summer D3). The final tree spliced 40 soaks, lost none, left 3
crops unpaid (v5 4), deferred 80 ripe crops past the ship window, and had no
capped day or rescue: 541 crops, 57,360G of bin gold. Copies with a 64k day cap (`v2b64`, `v5b64`) and the
D24 cutoff (`v2c24`) all wedged before Fall (below), so neither cap nor
cutoff has a clean A/B. `v5b64` also showed that a longer day hands the
evening `CLEAR_FIELD` time to reach the far south-east corner.

What each change fixed:

- **Stamina, not the clock, caps the field.** 40 tiles cost ~80 of the 100
  daily stamina; v2 spliced 44 soaks (`spliced hot spring`). The soak is ~660
  frames for 95 stamina plus a ~3,000-frame walk each way.
- **Rings** (rings row): Summer corn 60 -> 263 with nine rings. Five aborted
  days (D19-D24, `nav_pocket_plant: nav timeout`) came from a bag sent to a
  partly harvested ring whose open cells were corners only, which walls the
  sowing stand in. `ring_center_reachable` plus a three-open-cell minimum
  removed them, and a `CROP_ESTABLISH` that fails after its recovery retry is
  skipped instead of aborting the day (`SKIP_AFTER_RECOVERY_PHASES`).
- **The bin pays nothing at 17:00+** (`Dropedonsaleplace`, bank_81
  `$8193BF`). sumbase threw 27 corn (3,240G) in late; v2 left the ripe crops
  for the next morning 64 times (`ship window closed`) and still lost 5
  west-strip crops picked just before 17:00, so `SHIP_OVERHEAD_MINUTES` went
  from 8 to 12 after the v5 launch (not yet benched).
- **Flower Festival**: the dance rolls the day, so D23 never watered. v2
  watered 25 tiles before noon and still danced.
- **Sow to D26**: a potato ripens five watered nights after sowing, so D25 and
  D26 bags ripen by Summer D1 and the ring takes corn the same morning.

Wedges found by the v2 copies and fixed (pins are local):

- `v2b64` Spring D22: the eve of the Flower Festival makes the scene's
  location FESTIVAL on the mountain, and the soak's `player_state` 0x21 read
  as a cutscene; the spa task pressed dismiss for 20k frames, then RETURN_HOME
  failed eight rescues. `scene.SPA_POOL_TILE_BOX` excludes the pool. The D22
  replay now soaks 4 -> 100 in 660 frames. Pin `LN_spa_whatif_soak_stall_FAIL`.
- v2 lost 18 of 44 soaks at (223,425): `_FARM_SOUTH_FIELD_TO_WEST_GATE`
  climbs x=13 through the grown (13,28) ring. South-field starts now take
  `_FARM_SOUTH_WEST_TO_WEST_GATE`, the west steps at (3-4,30-31); ROM probes
  from (6,37), (12,36) and (30,40) reach the spa in 3,000-3,300 frames.
- `v2c24` Summer D7: a hen chase deferred at (510,794), just outside
  `yard_free`, took the town route from inside the fenced yard and RETURN_HOME
  wedged. `in_animal_yard` covers the whole fence box for the route choice and
  RETURN_HOME. The D7 replay now ends the day normally, but the hen is still
  not delivered: both `to_farm` walks step back into the shop at the town door
  (600,874), and the chase targets a bird outside the yard. Pin
  `LN_hen_whatif_yard_strand_FAIL`.
- `v5b64` Summer D2 18:09 (still open): the west exit refused every walk left
  of x=35 with nothing in hand, so that soak was skipped (the day went on).
  The weather byte read 1 (rain) while the `$0196` rain flags were clear. Pin
  `LN_spa_whatif_exit_walk_FAIL`; the row-26 centring hop is in the tree but
  did not change that replay.
- `v5b64` Summer D16 (still open, not this session's code): after the spa and
  watering, the evening `CLEAR_FIELD` ended at (777,937); RETURN_HOME's
  far-east escape entered map `0x2A` with text `0x031A`/`0x0328` up and failed
  eight rescues.

## Forage first, from the Day-3 pin (2026-11-02, later the same day)

`blocks` layout, Day-3 pin, frozen tree copies, one draw each, Clean, 0 rescues. `y1base` is the tree before the forage change (it is also the
full-year baseline below); `fa` puts the mountain trip ahead of the crop work on every day with nothing ripe to ship (`planner/forage_calendar.py`).

| run | change | Spring D30 wallet | Summer D5 wallet | grape trips (phases / grapes) |
| --- | --- | ---: | ---: | --- |
| `y1base` | forage last, after all crop work | 13,060 | 13,120 | 3 / 3 |
| `fa` | forage first on no-harvest days | **14,130** | 12,830 | 9 / 12 (D4-D8, D10, D11, D13, D15) |

- `fa` leads by 3,570G on D20 and by 1,070G on D30: from D9 the daily potato sowings ripen daily, so most days have a harvest and keep their
  morning for the bin. Summer D5 is level because `fa` spent more on seed bags.
- Summer D9 and D12 replays (`PR_s2base_Y1_Summer_D9/D12`, live tree): two green fruit shipped (400G) before the watering, and D9's 103 tiles
  were watered in three passes with two spliced soaks, nothing unfunded. Before, every Summer night left 20-50 tiles dry
  (`stamina_low unfunded=N` after the one allowed refill).
- Item facts and the ROM table: [plan.md](plan.md#forage-calendar-spa-refills-year-2-levers-2026-11-02-handoff).

### Full Year 1 from the Day-3 pin, all four changes (2026-11-02)

`HARVEST_RING_LAYOUT=blocks`, `--until-year 2`, Clean, 0 rescues, one draw each. `y1base` is the tree before this session's changes, `y1new` has
forage first (Spring and Summer, Fall poison mushrooms), up to four spa refills for watering, the sowing gate counting three, and the grape landing pull.
The run bundles the four, so it does not say which one earned what. Wallets are the last day of each season.

| | `y1base` | `y1new` | |
| --- | ---: | ---: | --- |
| Spring D30 | 13,060 | **16,800** | +3,740 |
| Summer D30 | 18,540 | **29,400** | +10,860 |
| Fall D30 | 28,740 | **48,000** | +19,260 |
| Winter D30 (Year 2 Spring D1) | 53,540 | **66,550** | +13,010 |
| net worth (wallet + hens + cows) | 70,540 | **78,550** | 2 cows vs 3 |
| crops shipped | 485 | 591 | corn 287 vs 207, potato 304 vs 278 |
| bin gold | 88,230 | 102,660 | |
| forage trips / gold | 33 / 4,950 | **70 / 13,300** | grapes 3 -> 14, green fruit 0 -> 15, Fall 30 mushrooms -> 41 poison mushrooms (counted from the phase reasons) |
| frames a day | 21,408 | 28,176 | wall 55 min -> 70 min for the year |
| bad days / rescues | 1 / 0 | 2 / 0 | |

- Forage is +8,350G of the bin gold. Corn is the rest of the Summer gain: 80 more shipped (about 9,600G) from the 103-tile nights that now water in full
  and from the earlier Spring cash. Spending also differs (a house upgrade and feather bought, one cow fewer), so the wallet delta is smaller than the sum.
- `y1_bench` printed 7,200 for forage because the journal's `shipping_deposit` kept only the last loop of a two-trip day; it now records the bin total
  from the start of the phase (`MountainGrapeShipTask.phase_shipping_before`), and `block_calendar_report track` counts trips from the phase reason.
- Fall poison mushrooms: 41 in 27 phases, because the task stopped starting loops at 13:00 and a loop is ~3 game hours, so most days ran two. The bail hour
  is now 14 (not yet benched).
- The third cow was not bought (`cow_target`): hay 180 at Fall D30 against 234, with 10 `CUT_FODDER` days against 20. The new run's mornings also carry wife gifts
  and wood (its wallet trace has a feather and a house upgrade the base lacks), so this is not isolated to forage. Cow chores end `incomplete cow care` a few dozen
  times in both runs.
- Watering still hits the 24k-frame phase watchdog 5-6 times a year in both runs (a pond-refill thrash at the south lip); unchanged.


## Notch blocks and fast press from the Day-3 pin (2026-11-02)

Same pin `LN_d2_perfect_poweron`, same tree except the layout, frozen copies, one draw each, Clean, 0 rescues. Reports and gzipped
logs: `recordings/block_calendar/runs/<tag>/` (gitignored); plan, tracks and map page: `recordings/block_calendar/`. Neither run has
B on path-centring steps (`_run_centring`, benched after they started).

| run | layout | Spring D30 wallet | potatoes shipped | corn blocks planted | end |
| --- | --- | ---: | ---: | --- | --- |
| `fastbase` | legacy nine plots, fast press, four bin stands, tick window | 10,790 | 223 (to Fall D1) | 9 bags S1-S7 | Fall D1 **31,660G**, 2 cows, 2 hens, corn 361, 3 watering watchdogs, 0 bad days |
| `blocks` | `HARVEST_RING_LAYOUT=blocks` (12 core + 2 late notch blocks) | **12,450** (forecast 11,740) | 270 by Summer D7 | 4 on S1, 6 on S2, 3 on S6 | Summer D7 10,300G after 13 corn bags |

For reference only (older tree, 48k day cap): `base` ended Spring at 11,260 and Fall D1 at 26,740G with the same 2 cows and 2 hens,
corn 289. The packed-ring runs ended Spring at 4,650-8,540.

What the block run showed:

- The frame model holds: across 270 ship-log records the measured pickup-to-ship cycle per block is 0.97-1.17x the planned frames
  (`block_track_blocks.json`; (19,28) is the outlier at 1.17, a yard walk around the well).
- The two late blocks were sown in mid-Spring but the stamina gate (`ring_establish_affordable`, one spa) refused them from D20 on
  (`no_work: stamina for hoe and standing-crop watering ((4, 43))`), so no late potatoes ripened in Summer. Twelve blocks is what one
  spa soak a day funds.
- Summer D1 shipped nothing: the D25 potatoes ripened on D2, a night late (a dry night), so their 6 blocks took corn on D2. Corn
  planted D1-D2 still gets six harvests.
- Watering is the open risk: `CROP_WATER exceeded 24000 frames` on Summer D1 and D2 nights at 12-13 blocks.

Summer estimate for `blocks` as planted (`block_calendar.forecast` from Summer D1, 12,450G, the run's planting days): ~77,000G at
Summer D30 from 571 corn, plus ~3,200G of D2-D5 potatoes, if every block is watered every night and nothing is bought. A missed watering
delays a block's harvests by a day each; the run already shows two watchdog nights.

## Packed ring lattice from the Day-3 pin (2026-11-02)

`LN_d2_perfect_poweron` to Fall D1 (`base`, `packed1`, `packed2`) or Summer D4 (`u_*`), frozen tree copies, one draw each. Layout and
switches: [plan.md](plan.md#packed-ring-lattice-east-pasture-uncapped-days-2026-11-02-handoff). Reports and gzipped logs in
`recordings/spring_y1/packed/` (gitignored). The first three ran with the old 48,000-frame day cap.

| run | change | rings | Spring-end wallet | Spring frames | bad days (max in a row) | Fall D1 wallet |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | current tree, legacy nine rings | 9 | 11,260 | 713k | 0 | 26,740 (1 cow, 1 hen) |
| packed1 | `HARVEST_RING_LAYOUT=packed`, **south pasture still on** (the rings reserve it: 475 tiles) | 26 | 11,550 | 971k | 5 | 46,870 (0 cows, 0 hens, `planted_grass` 9; base 103) |
| packed2 | + shared-stand watering tour, south pasture | 26 | ~3,200 | 1.01M | 3 | pending (260 at Summer D14) |

Spring D23-D27 of `packed1` are five bad days in a row: three to seven new rings planted a day, each followed by a full-field watering and
a spa soak, trip the 48k cap (32 cap trips against 5 for `base`). `packed2` spent all its gold on 24 corn bags on Summer D1-D3, then its
watering ran into the cap on D10-D14 and no crop ripened until D15. `packed1` and `packed2` were frozen before the east pasture existed: the packed rings reserve the whole south pasture, so no grass bag could land,
every `CUT_FODDER` failed `stored_grass=0->0` (439 log lines), the shop's planted-grass gate never opened and no cow or hen was bought. That is why
`packed1`'s 46,870G at Fall D1 is wallet only (`base`: 26,740G plus a cow, a hen and the Summer purchases) and cannot be read as a win. The `u_*` runs add the
east pasture and the 216k cap; their results are not in this table yet.

## Barn, dog and tool-fetch stalls: feeding the herd (2026-10-31)

Fall D1 to Year 2 Spring D1 from `PR_final_d3_Y1_Fall_D1` (2 cows, 2 hens, 24,130G), one draw
each on frozen tree copies; reports in `recordings/spring_y1/fallwinter/` (gitignored).
"Fed" counts animal-days with every adult's fed bit set at the pre-sleep read, which
undercounts a late `feed_check`.

| run | change | net worth | bad days | fed animal-days | `COW_CHORES` failures | cows at end | frames/day |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fv2 | tree before this work | 67,130 | 13 | 53/60 (88%) | 22 | 3 | 21,635 |
| cw1 | fodder dispenser also from tile (13,10) | 70,830 | 7 | 55/60 (92%) | 5 | 3 | 19,480 |
| cw2 | + BFS walk detour, `cow walk blocked`, barn re-entry | 71,330 | 3 | 59/60 (98%) | 0 | 3 | 18,150 |
| cw3 | + exit-prep detour (cow beside the milk bin) | 75,630 | 1 | 59/60 (98%) | 0 | 3 | 16,353 |

What stalled, each reproduced on a pin and fixed with a non-ROM test:

- **A cow body in the barn aisle.** The fodder, trough and milk-bin walks are fixed pixel
  lanes that ignore cows; one cow in the way held `COW_CHORES` for its 30,000-frame timeout,
  and because the barn clock does not tick indoors the farmer slept at 11:06 with the day
  undone. Seven days in fv2. A cow parked on the dispenser stays put for 24k+ frames, so
  waiting never helps. Fixes: the dispenser answers from (13,10) as well; a walk phase that
  stands still for 90 steps routes around the cows with the BFS navigator
  (`CowChoresTask._walk_unstick`); two detours that never arrive fail the task with
  `cow walk blocked`, and the day plan splices one `EXIT_BARN, NAV_TO_BARN, ENTER_BARN,
  COW_CHORES` (`splice_barn_reset`), because entering the barn puts every cow back in its
  stall (a 570-frame round trip).
- **The farm dog.** The full chain from `LN_d2_perfect_poweron` (`y1a`, tree before the dog and
  exit-prep fixes) ended at 62,750 net worth with only two of its three cows. Winter D13's
  required `ENSURE_ANIMAL_TOOLS` failed `pixel_stuck` at the west entrance (98,430), the day
  aborted, the herd went unfed (`feed_check` hit the same wall) and a sick cow died on D19.
  The dog sat in the corridor and the navigators missed it: its animation cycles sprite ids
  `0x01D9-0x01F7` on the farm and only `0x01D9` was named. `FARM_DOG_SPRITES` labels them, and
  a failed tool fetch is now skipped after its recovery attempt (`SKIP_AFTER_RECOVERY_PHASES`)
  so the brush and milker cannot cost the herd its feed. Winter D13 and D19 both fetch the
  tools and feed three cows on their pins.
- **Full chain, final tree (`y1b`)**: `LN_d2_perfect_poweron` to Year 2 Spring D1, 2,411,867
  frames (20,268 a day), **76,140 net worth** (wallet 59,140, 3 cows, 2 hens; `y1a` 62,750 with
  2 cows), Spring 9,260G, Summer 27,290G, Fall 36,740G, 511 crops shipped, 127 milk shipments,
  zero `COW_CHORES` failures, all three cows alive to the end. 7 bad days, none of them a
  lost herd: Summer D1 (48k day budget), D14 and D15 (the known spa-exit wedge on the
  mountain map), D17 (`NAV_TO_COOP` landed on 0x04), Fall D1 (a cow in the barn's
  top-left lane held the exit prep at (58,57); the re-entry's `EXIT_BARN` hit the same
  lane at (38,75) and cost one rescue), Fall D24 (`COOP_CHORES` started outside the coop,
  hens unfed) and Star Night. One draw each; Summer storm days differ per run.
- Dog check: the same y1b tree with `FARM_DOG_SPRITES` emptied reproduces `y1a`'s Spring
  exactly (9,830G, 198 crops), so the other changes do not touch Spring; with the dog
  labelled Spring ends at 9,260G and 177 crops (paths diverge from about D9). That is
  inside the spread other tree variants showed (9,480-10,720G), so the label stays: the dog
  is a physical blocker. Not isolated over a full year.
- Left: Winter D24 (Star Night) fails its first `EXIT_TO_FARM` on a house dialogue (text
  `0x0290`, 361 frames) and the day rolls; the forced `feed_check` still feeds the herd.
  The BFS cow tiles ignore the pixel spill of a cow into the next tile (centres under 16px
  collide), which is why a parked cow still needs the barn re-entry.

Repro: `run_to_day2 --state PR_fv2_Y1_Fall_D5 --days 1` (also `_D10`, `_D16`, `_D26`, `Winter_D12`,
`_D18`), `PR_cw2_Y1_Fall_D28` / `PR_cw2_Y1_Winter_D13` for the exit, `PR_y1a_Y1_Winter_D13` /
`_D19` for the dog (pins are local).

## Completion and seasonal sales specification

Run the complete farm from power-on, then generate the sales report:

```bash
HEADLESS=1 QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 \
  --power-on --until-ending --save-season-pins --save-end-state LN_completion_ending \
  --out recordings/y1_bench/completion.json
uv run python -m harvest.tools.y1_bench recordings/y1_bench/completion.json \
  --sales --out debug_completion/season_sales.md
uv run python -m harvest.tools.y1_bench recordings/y1_bench/completion.json \
  --out debug_completion/scorecard.json
```

These commands run from `snes/harvest`. Reports, logs and states stay game-local
and gitignored. The committed seasonal tables below preserve the useful results.

| Check | Required receipt |
| --- | --- |
| Basic game completion | Natural Y3 Summer D30 arrival; `success`, `terminal`, `ending.finished` and `ending.final.terminal` true; final scene index `0x25`; 900 frames with unchanged ending signals. |
| Completion with score | Read the ROM's `ranch_mastery_0x25`, including a score of zero. The model's oracle is a cross-check, not a substitute for the RAM score. |
| Continuous Clean power-on proof | Completed new-diary bootstrap; exactly zero initial loads, mid-run loads and RAM writes; empty assists; intervention class `Clean`. Missing audit fields fail this check. |
| Perfect | The separate target is 999. A lower score can complete the credits. The published-video rung and Perfect campaign remain separate: see [CONTEXT](../CONTEXT.md) and [ADR 0007](adr/0007-perfect-rancher.md), including its later load ruling. |

The runner journals the last completed workday after credits settle, so Summer
D29's shipments and overnight credit are retained. An ending-state pin does
not invent a preceding workday. The new `start_economy`, `end_economy` and
`sales_ledger` fields are read-only observations; they do not change controller
input, load states or write RAM.

| Sales field | Meaning and limits |
| --- | --- |
| Bin shipping G | Sum of positive `shipping_money` increases, observed every frame from the controllable start through the ending. Night payout resets do not subtract from gross sales. A pin's opening bin is excluded and reported separately. |
| Crop quantities / G | Positive lifetime shipment-counter deltas by season, multiplied by the prices below. A counter decrease is flagged and invalidates complete observation. |
| Forage G | Recorded `shipping_deposit` deltas, including partial trips. A trip that deposits goods but never leaves a receipt can contribute to unattributed shipping. |
| Milk + egg G (estimate) | Bin value minus crop-counter gold minus forage receipts on measured successful cow/coop chore days. These reports cannot identify milk tiers or count all eggs separately. |
| Unattributed bin G | Bin gold minus crops, forage receipts and estimated milk/egg gold. It retains missing or failed-phase receipts instead of assigning the remainder to livestock. |
| Hen sales G | Successful sale-event receipts times the native 500G resale price. This is direct wallet income, outside the bin. An unreceipted sale remains in other net cash. |
| Recorded gross sales G | Bin shipping plus receipted hen sales, before spending. Inventory valuation is excluded. |
| Paid sales G | Observed bin deposits less the closing unpaid bin (floored at zero), plus receipted hen sales. The final run has 50G still in the bin at credits, so paid sales are 50G below deposits plus hen receipts. |
| Wallet change | Season-end wallet minus prior season-end wallet; the final season uses the final report receipt. It includes purchases, other income and spending. It is not gross sales. |
| Other net cash G | Wallet change minus recorded gross sales plus receipted seed/animal purchases. House upgrades, tools, gifts, visitor purchases and missing receipts can remain here; this is not itemized spending or profit. |
| Coverage | `frame ledger` identifies measured shipping totals. `journal subtotal` identifies older, incomplete observations. Bin-snapshot counts show legacy coverage even when the frame ledger supplies the total. |

The calendar has 30 days per season and four seasons per year. Sales belong to
the season of deposit, even when the wallet receives the credit the next
morning. A completed report includes ten seasons, with only D1-D29 worked in
the last Summer; D30 is the ending. The bootstrap's Spring D1 is observed even
though the daily journal starts on D2. Festival/event jumps can leave fewer
journal rows than calendar days.
New Year updates the year and season in separate frames. Zero-sale transient
calendar rows outside the run's actual date range are omitted from the tables;
they remain in the raw ledger. Nonzero out-of-range sales invalidate complete
coverage. Closing unpaid bin gold is reported separately from paid income.

Prices come from the ROM-backed tables in
[`livestock_econ.py`](../harvest/planner/livestock_econ.py):

| Product | Sell G / unit | Seed bag G | Season |
| --- | ---: | ---: | --- |
| Turnip | 60 | 200 | Spring |
| Potato | 80 | 200 | Spring |
| Tomato | 100 | 300 | Summer |
| Corn | 120 | 300 | Summer |
| Egg | 50 | — | Year-round, with laying hens |
| Milk S / M / L | 150 / 250 / 350 | — | Year-round, with adult cows; tier follows happiness |
| Adult hen resale | 500 | — | Direct sale; purchase price is 1,000G |

## Current power-on completion and season sell totals (2026-10-30)

The current farm route completes one continuous Clean power-on run through Y3 Summer D30 credits, ending index `0x25`, with RAM score **195**. Final wallet: **26,730G**; **two cows and eleven chickens**; Eve is married (wife flag retained), first house upgrade complete, second upgrade unfinished. This proves basic credits completion with a score; Perfect 999 and the published-video rung remain open.

Run: `recordings/y1_bench/completion.json`, **5,022,801 frames**, **6,139.9 s wall** (1h 42m 20s). At 60 fps these frames represent 23h 15m of playback; no video was recorded. The audit reports **0 initial loads, 0 mid-run loads, 0 RAM writes and no assists**. The Summer D29→D30 journal entry is retained. The frame ledger covers the full controllable run; all 180 crop-counter increments reconcile to the final counters: 102 potatoes and 78 corn, with no counter decreases.

Reliability remains unfinished: **66 bad-day rows, 36 day-floor trips, nine rescues**, and 98/137 days with animals whose recorded snapshot shows every adult fed (71.5%). Snapshots are captured before any late barn retry and can undercount successful late feeding. Barn and coop exits, milk approaches and the Spring planting route still lose work. Late hay is tight: 15 before the last workday and 13 in the final receipt. The final day records only seven hens fed and no cows fed, so this is not a healthy-care or Perfect claim. Both cows survive to the ending with status `0x29` (adult, sick). The final feed retry fails at the coop exit, confirming the last-day feed shortfall.

The previous `y23_final` run had score 120, 15,680G, no cows, three chickens, 90 bad days and 33 rescues. The current result improves score by 75 and wallet by 11,050G while retaining the herd. Routes and timing differ, so this comparison does not isolate the effect of any one repair.

The integration is `HarvestMoon-Snes`, ROM SHA1 `a64a5634429a4f5341868a40c220d7be89fda70a`, stable-retro 0.9.8. Every route/ledger Python source still matches the launch hashes; only the independent scorecard formatter changed during verification. Raw report SHA256: `cd399beb373ee2a0f35d191a3ae2ff1837715d807d4570ee6bb847511eb4cf00`. Local receipts: `debug_completion/{source_hashes.json,scorecard.json,season_sales.md}` and `logs/completion/power_on.log`.

Clean power-on credits with score: True. Score: 195. Perfect 999: False.

All gold is G. Crop cells are quantity / gold. Gross is bin shipping plus receipted hen sales, before spending. Milk + eggs is a residual estimate on measured chore days; forage and hen sales come from phase receipts.

| Season | Turnip | Potato | Tomato | Corn | Crop G | Forage G | Milk + egg G (est.) | Unattributed bin G | Bin G | Hen sales G | Gross G | Paid G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y1 Spring | 0 / 0 | 4 / 320 | 0 / 0 | 0 / 0 | 320 | 900 | 0 | 150 | 1,370 | 0 | 1,370 | 1,370 |
| Y1 Summer | 0 / 0 | 4 / 320 | 0 / 0 | 19 / 2,280 | 2,600 | 0 | 0 | 0 | 2,600 | 0 | 2,600 | 2,600 |
| Y1 Fall | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Y1 Winter | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Y2 Spring | 0 / 0 | 80 / 6,400 | 0 / 0 | 0 / 0 | 6,400 | 0 | 0 | 0 | 6,400 | 0 | 6,400 | 6,400 |
| Y2 Summer | 0 / 0 | 1 / 80 | 0 / 0 | 59 / 7,080 | 7,160 | 0 | 1,100 | 0 | 8,260 | 0 | 8,260 | 8,260 |
| Y2 Fall | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 3,150 | 3,600 | 0 | 6,750 | 1,500 | 8,250 | 8,250 |
| Y2 Winter | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | 8,950 | 0 | 8,950 | 1,500 | 10,450 | 10,450 |
| Y3 Spring | 0 / 0 | 12 / 960 | 0 / 0 | 0 / 0 | 960 | 0 | 11,150 | 0 | 12,110 | 500 | 12,610 | 12,610 |
| Y3 Summer | 0 / 0 | 1 / 80 | 0 / 0 | 0 / 0 | 80 | 0 | 13,750 | 100 | 13,930 | 0 | 13,930 | 13,880 |

Total bin shipping: 60,370G; receipted hen sales: 3,500G; recorded gross sales: 63,870G.

Closing unpaid bin: 50G. Paid shipping from observed deposits: 60,320G; paid sales including receipted hens: 63,820G.

| Season | Source | Bin snapshots / journal rows | Wallet start | Wallet end | Wallet change | Purchase receipts G | Other net cash G |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Y1 Spring | frame ledger | 28/29 | 300 | 1,120 | +820 | 600 | +50 |
| Y1 Summer | frame ledger | 30/30 | 1,120 | 2,030 | +910 | 1,700 | +10 |
| Y1 Fall | frame ledger | 30/30 | 2,030 | 2,030 | +0 | 0 | +0 |
| Y1 Winter | frame ledger | 30/30 | 2,030 | 2,030 | +0 | 0 | +0 |
| Y2 Spring | frame ledger | 30/30 | 2,030 | 5,240 | +3,210 | 3,200 | +10 |
| Y2 Summer | frame ledger | 30/30 | 5,240 | 6,520 | +1,280 | 7,100 | +120 |
| Y2 Fall | frame ledger | 30/30 | 6,520 | 1,670 | -4,850 | 10,000 | -3,100 |
| Y2 Winter | frame ledger | 29/30 | 1,670 | 12,120 | +10,450 | 0 | +0 |
| Y3 Spring | frame ledger | 29/29 | 12,120 | 20,050 | +7,930 | 4,800 | +120 |
| Y3 Summer | frame ledger | 28/29 | 20,050 | 26,730 | +6,680 | 1,200 | -6,050 |

Other net cash = wallet change - recorded gross sales + purchase receipts. It includes unitemized spending and income, so it is not a profit figure.


## Review regression receipts (2026-10-31, pins only)

The reviewed completion bugs now have non-ROM regressions and audited ROM
replays. These pin checks establish task behavior, not campaign economy.
Every replay used zero RAM writes and zero mid-run loads.

| Pin / case | Receipt |
| --- | --- |
| `LN_cowfix_pin_fall_d21`, `TILL_FIELD` | Exactly 32 verified hoe changes; planted grass stays 34 and seed stock stays 2. Hoe work has its own limit and scans independently of seed bags. |
| `LN_y23_wood_y2s6`, native axe depletion then `CLEAR_BUSHES` | Weeds 511→508 while stamina remains 0. Completion is scoped to weeds, including when hammer and axe are carried (non-ROM regression). |
| `LN_cowfix_s15_morning`, complete sale cycle | Chickens 1→0, wallet 440→940G; the request baseline survives the return trip. |
| `LN_cowfix_pin_fall_d21`, first-Fall trade | Chickens 1→0, wallet 4450→4450G. The native trade is rejected as a paid-sale receipt. |
| `LN_y23_wood_y2s6`, sale task without a request | Chickens 3→3, wallet 6400→6400G; fails with `receipt missing` after 900 frames. Resetting a task cannot create a sale. |

The paid chicken branch is `$81ED3C`; `$81E969` is the cow branch. The
first-Fall chicken trade (`$81EECA`) removes the hen without `AddMoney`.

The fresh `review_fixes` power-on bench was stopped on **Y1 Fall D2** after
1,912,705 frames (2,084 seconds): `CLEAR_FIELD` left a held stone (`0x0D`),
and return-home drop/wander recovery cycled in the north yard around (265,70)
for more than 50,000 frames. The prefix has zero initial/mid-run loads, zero
RAM writes and no assists. It ends with 2,830G, two hens, one cow, 72 pasture
tiles and 87 hay; its 9,830G net worth is **partial**, not a completed-year
result. The first cow gate opened on Summer D27; it waited on season 28
mornings, money 15, shop closure 9, pasture 6 and weather 1.

Against the previous `run16_fixes_y1` report, the completed Spring wallet is
6,200 vs 1,120G and Summer is 2,580 vs 2,030G. The generated scorecard pair
(`review_fixes_scorecard.json` / `.txt`) also prints end-date deltas, which are
not a full-year comparison because this run stopped early. Next: diagnose
the north-yard held-stone return-home recovery before repeating the year.

## Complete-farm campaign and headed replays (2026-10-31)

The campaign now lets the required D2 clear finish instead of abandoning it
at the ordinary day floor. A blocked required clear terminates the run;
sleep cannot silently turn it into a completed day. Weeds across all four
quadrants are cleared before hauling posts/stones to ponds, and a met clear
quota must leave empty hands before it can report success.

New Year uses group `0x28` event 1's `$7F1F7A` conversation acknowledgements,
not a wait until 18:00. The native-predecessor ROM check completes the event
in **3,396 frames**, returning control shortly after **08:00**, with zero RAM writes,
zero mid-run loads and no day-floor trip. Navigation uses exact 8-pixel
controller-probed hops and verified talk stands. Rebuild its grid with:
`uv run python -m harvest.scripts.probe_walk_grid --state LN_newyear_entry --setup wait:240 --tilemap 0x3A --exact --out harvest/maps/new_year_walk.json`
(from `snes/harvest`).
The grid is geometry research; the campaign never reloads states to navigate.
On the identical `LN_newyear_entry` bytes, seed 1 improves from 35,580 to
3,358 frames (−32,222; 90.6% fewer). Receipt:
`recordings/y1_complete/new_year_policy.json`.

`run_to_day2 --headed --speed 4` opens the shared viewer on native Wayland;
`[` / `]` adjust speed, 8+ is unthrottled, and ESC writes the partial report.
`--save-day-pins --pin-prefix PR_complete_` retains settled mornings and planner
lineage, including the terminal morning. Each run needing retained pins uses
its own prefix. Progress checkpoints also track the current child phase.

The fresh power-on Year 1 run **failed on Spring D2** after 278,222 frames
(449.8 s), with zero loads, writes or assists. After weeds, posts and loose
stones were cleared, `ENSURE_HAMMER` reported success but `CLEAR_ROCKS`
entered/stayed in the shed (`0x26`, (54,166)), ending at its goal watchdog.
This is not a completed year or a clean-farm handoff. Log, report, progress,
launch arguments and source hashes are in `recordings/y1_complete/`.
The next replay and exact failure pins are in
[the handoff](tasks/rr-20w-run16-defects.md). No runners remain active.
The intended handoff remains `PR_complete_Y2_Spring_D1`, followed by a headed
Year 2 watch. Verify completion and actual farm clearance first. The
older `PR_Y2_Spring_D1` has 2,030G, no animals and the original uncleared
farm; it is useful only as a local failure predecessor.

## Years 2-3 repeatability sweep (2026-11-01)

Start `PR_y1b_Y2_Spring_D1` (end of `y1b`: 76,140 net worth, 3 cows, 2 hens). Frozen tree copies,
one draw each; reports in `recordings/spring_y1/` and `recordings/y23_sweep/` (gitignored). Tool:
`scripts/y23_sweep.sh`, verdict `harvest.tools.sweep_report` (CLEAN = ending reached, zero rescues,
Clean intervention). Perturbations: `HARVEST_PHASE_JITTER=<max>` idles 0..max neutral frames before
each phase (seeded), `HARVEST_FAULT_PHASES/RATE/SEED` fails a share of phases. A neutral wait in
the house changes nothing (the ROM is frame-identical; `--start-jitter` 97 matched the baseline
at every frame and was removed), so the jitter is per outdoor phase.

First-pass runs on the unfixed tree found six wedges, listed with causes, fixes and pins in
[plan.md](plan.md#years-2-3-repeatability-sweep-2026-11-01-in-progress). Three of them ended a
run (8 rescues in one day): the spa pond after a mid-soak day floor (Y2 Spring D19), the well
pit / Harvest Sprites' cave (Y2 Summer D4) and a rider at the house door (Y2 Spring D19 with
jitter). Two cost a day of work (coop and barn chores recovered outside the building, Summer D13
and Fall D8) and one a day in Year 1 (recovery budget 900 frames, Summer D14).

| run | start | perturbation | tree | result |
| --- | --- | --- | --- | --- |
| y23a | `PR_y1b_Y2_Spring_D1` | none | unfixed | Y2 Spring D19: 8 rescues (spa pond), killed |
| p60 / p60b | same | jitter 60 | unfixed | Y2 Summer D4: 8 rescues (well hole, Sprites' cave) |
| p150 / p150b | same | jitter 150 | unfixed | Y2 Spring D19: 8 rescues (rider at the door) |
| base2 | same | none | partly fixed | Y2 Fall D19: 8 rescues (tired farmer read as a cutscene on a festival eve) |
| t2_s12 | `PR_t1_s1_Y2_Spring_D29` | jitter 90 | pit fix, one hole | Y2 Summer D16: 8 rescues (second hole) |
| **f7 + f7b** | same, resumed at `PR_f7_Y2_Spring_D24` | faults `*:fail` 1%, 42 injected | unfixed then partly fixed | **CLEAN**: ending credits `0x25` at Y3 Summer D30, 0 rescues, 23 bad days, 156 days / 4.63M frames / 7,784 s in `f7b`, 322,330G, 4 cows, 2 hens, Clean (initial load 1, zero mid-run loads, zero RAM writes) |
| **y3b** | `PR_Y3_Summer_D1` | jitter 90 | first dismount filter | **CLEAN**: ending credits at Y3 Summer D30, 0 rescues, 1 bad day, 1.18M frames, 1,493 s |
| y3a | `PR_Y3_Spring_D1` | jitter 90 | first dismount filter | hung at Y3 Spring D20 17:00 for 1.6M frames: the first `dismount_horse` held A and starved the shipping box of its A edge |
| t4_s42 | `PR_y1b_Y2_Spring_D1` | jitter 90 | dismount fixed | Y2 Spring D10: 8 sleep rescues (an A during the 5pm scene's `input_lock` 0 set `game_state & 0x0800` for the day; fixed in the shipping predicate) |
| **t4z_s46** | `PR_Y3_Summer_D1` | jitter 90 | dismount fixed | **CLEAN**: ending credits, 0 rescues, 2 bad days, 0.997M frames |

`f7b` crossed Year 2 and Year 3 end to end with 39 injected phase failures, all recovered. It
started from a day pin written by `f7` (one initial load, no mid-run load), so a single
uninterrupted power-on-style run from `PR_y1b_Y2_Spring_D1` is still the open proof; the
`t4*` sweep runs (final tree) are the attempt, and `sweep_report` is their verdict.

Time hot spots (frames, `y1b` Year 1): CROP_WATER 702k (29.1%, 263 frames per watered tile; 76% is
navigation between water stands, 21% the watering action; costliest stands (4,35) 130k, (9,35) 114k,
(41,33) 75k, (13,28) 74k, (4,40) 68k); COW_CHORES 272k (11.3%, ~1,000 frames a cow, spread evenly
over nav, verify and milk); HOT_SPRING_STAMINA 194k (8.0%); HARVEST_ROUTE 177k (7.3%); SLEEP 159k;
RETURN_HOME 106k; EXIT_TO_FARM 84k (3.5%, 420 a call). Y2 Spring days run 40-50k frames and trip the
48k budget on D3, D9, D11, D12: replay of D3 = potato bags trip 8k (`NAV_FARM_EXIT` 1.7k +
`BUY_SEEDS` 6.4k), establish 9k (three `hoe_until_tilled cannot face` failures), watering 6.4k +
soak 5.3k + second pass 7.2k (interrupted by the floor).

## End-to-end power-on runs and the saddle A/B (2026-11-01)

`scripts/e2e_run.sh <tag>` runs power-on to the credits on a frozen tree copy, with one-hour video
parts (or `VIDEO=0`), and writes `recordings/e2e/<tag>/{report.json,run.log,FINAL.md}`;
`harvest.tools.final_score` prints cash, the ranch score by term, shipped-crop counters and the
cost of the run. Video does not change emulation (lean step only skips the screen copy): `base0`
(no video) and `base1` (video) failed at the same frame.

| run | tree | result |
| --- | --- | --- |
| base0 / base1 | working tree, before the fix | Y1 Spring D2 frame 52,254: `BUY_SEEDS` "town coords never settled after shop exit", the farmer stood in town and `D2_FARM_CLEAR` ended `stale_farm_map`. The town fade-in after the seed shop takes ~190 frames and the settle cap was 180 |
| v1 / v1n | + `TOWN_SETTLE_FRAMES` | `v1n` (lean) reached **Y2 Winter D24** (5.7M frames, 3.2 h) and ended there: 8 sleep rescues, the baby-day house trap below. `v1` (video) was stopped at Y2 Spring D7 |
| v2 | + pond takes unrecognised items, saddle opt-in | stopped at Y1 Winter once the trap was found (it would have hit the same day) |
| **v3n** | + the baby-day script (`planner/tasks/labor.py`) | **credits reached** from power-on at Y3 Summer D30 (7.97M frames, 3.3 h wall, lean): wallet **254,260G**, ranch score **357** (Ranch Master), 4 cows, 2 hens, 2 children, 1 rescue (Y2 Winter D19, `EXIT_BARN` timeout at (182,280) / (198,316), the barn exit lane stall), 0 RAM writes, 0 mid-run loads. Y2 Winter D24 passed with no rescue: the baby-day script works on the real timeline. Scorecard: `recordings/e2e/v3n/FINAL.md` |
| **v3** | same tree as `v3n`, with one-hour video parts | the recorded proof run, **credits reached**: 7,974,544 frames (17,069 s wall, ~470 fps), wallet 254,260G and score 357, frame for frame the same as `v3n` (video does not change emulation). 37 one-hour parts, all decode with `ffprobe` (768x672, 60 fps, 36.9 h of video, 18 GB; the report counts 7,974,904 frames with the 360 neutral ones at the end). Scorecard: `recordings/e2e/v3/FINAL.md` |

**Strategy legs (2026-11-01, `planner/run_strategy.py`, pins local).** `cash` and `score` replayed from the `v3` day pin
`PR_e2ev3_Y3_Summer_D20` (one draw each; the pin comes from a cash run, so its corn counter is 643, already
wrapped past 511, and 671 potatoes):

| leg | days | corn shipped | wallet | frames | notes |
| --- | ---: | --- | ---: | ---: | --- |
| `cash` | 2 | 643 to 695 (+52) | 221,750 to 230,790G | 85,641 | `HARVEST_ROUTE` on 24 and 39 ripe tiles |
| `score` | 2 | 643 to 643 | 221,750 to 224,550G | 41,187 | no `HARVEST_ROUTE` planned: every ripe tile is past its cap; Summer seed `tomato` on D20 (corn capped); same `COW_CHORES` failure as `cash` |
| `score`, corn counter poked to 500 (`LN_score_whatif_corn500`) | 1 | 500 to **511** | 221,750 to 224,470G | 30,134 | "Detected 11 ripe crop targets" (28 more ripe tiles left on the plant), exactly 11 shipped, then stop |

The capped days take about half the frames, because the harvest walks are the day's longest phase. From
`PR_e2ev3_Y3_Summer_D10` (3 days) the only ripe tile was the sealed west-ring tile (12,28): `cash` planned `NAV_CROP` and
`HARVEST_ROUTE` for it every day and both failed (unreachable); `score` planned neither. Pin replays drift (the D10 pair
diverged on a rescue), so these show the mechanism, not a size; the number comes from a full power-on `score` run.

**The baby-day trap (Y2 Winter D24, found by `v1n`).** The morning `kid1_age` reaches 60 the ROM plays
the birth scene (event group `0x1F`): the wife is in bed, a doctor stands in the middle of the doorway
and three villagers stand around the bed. The farmer is free but cannot leave (the doctor fills the
doorway), cannot reach the bed stand (a villager at (296,120)), and the house clock does not tick, so
waiting for the crowd to go never works; talking to the doctor or the villagers only repeats a line.
Run `v1n` stood there for 200k frames and then spent 8 rescues. What works, all measured on the pin
`PR_e2ev1n_Y2_Winter_D24` and written down in `labor.py`: talk to the wife from the east stand
(296,102) facing left and take the **second** row, "No, no, no" (the first, "Really?", sends the farmer
to work and the baby is born on his return; the second also gives the wife +50 hearts); idle about
3,800 frames until the "Waah-waah" cutscene puts the baby in the crib and the farmer at the door;
chat with the villager at (312,104) from tile (20,6) (he walks off east), which opens the east stand;
talk to the wife again: "Darling, please name him", the name grid (`name_entry_press`), and the ROM
ends the day itself (the 4 cows are not fed that night: -8 happiness each). The script costs about
6,400 frames and no game time. `tests/test_labor_morning.py` and `test_labor_morning_rom.py`.

Other stalls the sweeps found on the way (all with non-ROM tests that go red on the bug):

- `t5f_s54` (1% forced failures), Y2 Summer D18: 8 rescues, run over. The farmer came out of
  `CROP_WATER` holding item `0x57` (not debris, not a crop). `RETURN_HOME` tried the drop spot and
  the wander drop; every yard tile (`0xA8`) refuses a toss, and `PondTossTask` only took
  `POND_TOSS_HELD`. It now takes anything but a crop, egg, milk, fodder or a hen
  (`farm_toss.pond_takes`). The item was not identified; the first draw that holds it again will.
- `t4f_s44`, Y2 Summer D25: finished the ending (321,440G, ranch score 343) with one rescue.
  `COW_CHORES` ended `cow walk blocked` at (150,278), the barn re-entry splice's `EXIT_BARN` then
  timed out at (180,316) and (198,316), the two right-aisle lanes, and the day was lost to a rescue.
  Six jitter replays of that morning did not reproduce it; open in [plan.md](plan.md).

### Saddle shipping A/B (Y2 Summer D13 -> Fall D1, one draw each, same pin)

Mechanism and ROM facts: [F8_shipping.md](perfect_rancher/research/F8_shipping.md#saddlebag-the-adult-horse-ships-a-thrown-crop-2026-11-01-rom-probes).
Pin `PR_t4_s41_Y2_Summer_D13`, `JITTER=0`, `UNTIL='--until-season 2 --until-day 1'`, frozen tree copies,
`HARVEST_SADDLE` the only difference.

| leg | policy | end wallet | corn shipped | picked D13-D30 | frames | rescues |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sb0 | off (bin only) | **144,310** | **485** | **202** | 693k | 0 |
| sb1 | park the horse up front when far crops are 16+ tiles out | 135,400 | 402 | 120 | 596k | 0 |
| sd1 | park only after the near crops are done; ship at the home horse when nearer | 141,130 | 451 | 165 | 615k | 0 |

On the harvest pin `LN_saddle_whatif_harvest_y2d22` (12:11, 48 ripe crops, farmer far from the bin) the
parked horse shipped 14 corn (1,680G) in 3,300 frames against 6 (720G) at the bin. The A/B says that
does not carry to the real schedule: the day plan reaches `HARVEST_ROUTE` at about 12:00 (two wife-gift
loops ~4,000 frames, `HOUSE_WOOD` ~2,500, `ENSURE_ANIMAL_TOOLS` ~2,700 come first; barn and coop time
does not run the clock), the near rings are cheaper at the bin (~170 frames a crop) than any ring at
the horse (~235), so the up-front park spent the window and picked fewer crops, and the late park never
fired (the near crops are never done by 14:30). The home-horse option cost 4 x 600 frames of "horse out
of reach" walks. Saddle shipping is therefore **off by default** (`HARVEST_SADDLE=1` opts in). What
would make it pay is an earlier harvest: with the harvest at 07:00 the far rings fit the day and the
horse ships them at ~235 frames a crop instead of ~500-640.

Score note: the ranch money term caps at 78 points (about 99,840G), and these runs end with 140k to
320k, so more cash is worth nothing to the score. Shipped-crop counters are worth 1 point per 16
(31 at 496 to 511; a count past 511 wraps to 0), and the bot ships no tomatoes or turnips: that is the
cheapest unspent score (up to +62).

## Runs so far (power-on, Clean, 120 days, reports in `recordings/y1_bench/`, gitignored)

| run | net worth | wallet Sp/Su/Fa/Wi | animals at end | crops shipped | bad days | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline_wip` | 6,590 | 3,710 / 2,840 / 1,590 / 1,590 | 1 cow (no tools), 0 hens | 39 | 14 | pre-work tree |
| `v1_animals` | **20,270** | 3,710 / 5,520 / 13,520 / 16,270 | 4 hens, 0 cows | 50 | 10 | hen-first chain, brush/milker trip, hen yard delivery |
| `v2_pasture` | 13,810 | 3,710 / 1,110 / 8,460 / 10,810 | 3 hens, 0 cows | 27 | 26 | larger pasture target, Winter hay stock: **regressed** |
| `cow_ship_fix` | 8,840 | 1,550 / 690 / 6,390 / 7,840 | 1 hen, 0 cows | 79 | 14 | east-field ship walk fixed; the Summer D15 can stall left the corn unwatered |
| `cow_fix2` | 6,750 | 1,780 / 2,700 / 6,750 / 6,750 | none | 127 | 13 | + can route; 8 `PLANT_PASTURE` and 20 `CUT_FODDER` nav timeouts, hen gate 1 tile short |
| `cow_fix4` | 9,240 | 1,780 / 1,340 / 6,790 / 8,240 | 1 hen, 0 cows | 101 | 15 | + pasture lane; 25 planted tiles against the cow gate's 32 |
| `cow_fix6` | 9,710 | 1,780 / 410 / 1,060 / 2,710 | 2 hens, **1 cow (Fall D27)** | 97 | 17 | + corn-reserve cutoff, planting before cutting: first clean power-on cow |
| `s1_ring_cutoff` | 21,900 | 3,310 / 2,550 / 5,850 / 14,900 | 2 hens, 1 cow (Summer D27) | 205 | 12 | Spring sowing stops D24, rings replanted to corn on D3. A good draw: no Summer storm, no stuck field |
| `s1_ring_fix` | **25,730** | 2,970 / 2,930 / 10,180 / 18,730 | 2 hens, 1 cow (Summer D21), brush D22, milker D27 | 222 | 11 | + parked water tiles retried after 3 days, bin-stand exit. Best so far; see "Summer 1 tuning" |
| `pasture_ctl` | **25,730** | 2,970 / 2,930 / 10,180 / 18,730 | 2 hens, 1 cow (Summer D21) | 222 | 11 | the 2026-10-29 tree without the pasture target: reproduces `s1_ring_fix` exactly |
| `pasture_fix` | 22,980 | 2,970 / 1,530 / 9,230 / 16,980 | 6 hens, **0 cows** (bought Summer D20, gone by Winter D30) | 227 | 26 | + `target_tiles`: pasture 68 tiles, but the flock outran the Winter hay and the coop failed on 11 mornings. **Regression** |

v2 is worse than v1 on every line after Summer D13, so v1 is the number to beat. `cow_fix6` is the first power-on run to buy a cow (Fall D27, 5,310G to 310G), but its net worth is still below `v1_animals`: the rings put every Summer crop 40+ tiles from the bin, and the cow arrives too late to pay for the tools before Winter shuts the town trips. Brush, milker and milk are measured in pin runs below, not yet in a power-on year. `s1_ring_fix` (Summer 1 tuning, below) beats `v1_animals` by 5,460 and `cow_fix6` by 16,020, with a cow from Summer D21, the brush and milker owned, and 36 milk shipments.

### Why v2 regressed (found, fixed 2026-10-28)

On Summer D13 the hen was delivered, then `BUY_GRASS_SEEDS` + `PLANT_PASTURE` ran, then `ENSURE_WATERING_CAN` failed (`route failed: pixel_stuck pos=(336,489)`, `logs/y1_bench/v2_pasture.log` lines 7314-7323). Every later morning `CROP_WATER` failed with `tool 0x10 not selected` and `dry_crops=7 skipped=7` (222 log lines, 0 in v1) from D13 to D30. The summer corn died unwatered: 5 crops shipped in Summer against 28 in v1, and the 3,000G of grass bags bought did not buy a cow or any new hens. Lead: the bigger pasture (or a grass tile planted that day) now seals or confuses the route to the can pickup near (336,489) on tilemap 0x00. **Found and fixed 2026-10-28:** not the pasture. `field_to_shed` aims at tiles (21,31) and (25,31), which Summer regrows as fence; BFS retargeted to the standable tile north of it, but arrival was judged against the fence pixel (15px away, radius 12), so the farmer pressed into the fence until `pixel_stuck` failed the route at exactly (336,489). `MultiMapNavTask._at_standable_substitute` now counts a plain waypoint on an unwalkable tile as reached on its nearest standable tile. Reproduce from `PR_Y1_Summer_D1` (or `LN_poweron_summer_d1`) with `run_to_day2 --state <pin> --until-season 2 --until-day 20` and watch `ENSURE_WATERING_CAN` after the first hen and grass buy; v1 never failed it. Those pins are not bench-grade: a pin run proves the failure, not the economy.

### State at handoff

Late-game comparison (2026-10-29): the earlier `y23_a` tree completes a
continuous Clean power-on run through Y3 Summer D30 credits: 5,061,449 frames,
score 65, 4,270G, two cows, four chickens, 56 bad-day journal rows and seven
rescues. Audit: no initial or mid-run loads, RAM writes or assists. This tree
predates the later navigation fixes and does not marry Eve. It is a baseline,
not validation of the current tree. Report: `recordings/y1_bench/y23_a.json`.

Final code (2026-10-30), `recordings/y1_bench/y23_final.json`: fresh Clean
power-on through Y3 Summer D30 credits, score 120, 5,996,523 frames,
6,780.5 s wall, no loads/writes/assists. Eve is married and both houses are
complete; wallet 15,680G, zero cows, three chickens. It uses 33 rescues and
90 bad days, so reliability and livestock health remain open. Against
`y23_a`, wallet rises 11,410G and net worth rises only 410G (18,680G), with
two fewer cows. Year-1 season wallets: 3,640 / 4,240 / 1,940 / 8,490G.
`debug_y23/y23_final_scorecard.md` holds both scorecards and the delta. The
cow gate waited on season 148 mornings, shop closure 43, money 26, pasture
16 and reserve six over the whole run. Fully-fed share: 39%.

Main tree has every lane merged, including `lane_rings.patch` (data-driven `POCKET_PLANT_CENTERS`, stamina gate, per-tile water/harvest failure memory). The non-ROM suite passes after the rings merge. The rings merge is **not yet benchmarked** (only a Y1_D3 ROM lane: Summer D1 wallet 3,760 -> 6,520, potatoes 32 -> 90, bad days 7 -> 1). `harvest/planner/day_phase_registry.py` is 1,013 lines, over the 1,000 soft cap.

### Cow economy session (2026-10-28)

Each defect below was found on a real-ROM replay of a deterministic pin and
fixed with a non-ROM test. Pins are local (`custom_integrations/.../LN_cowfix_*`).
Re-create the Summer pins with `run_to_day2 --power-on --until-season 1
--until-day N --save-end-state <name>`; a pin replay drifts from the power-on
timeline after about a week, so the power-on run is the check.

| defect | symptom | cause | fix | check |
| --- | --- | --- | --- | --- |
| East-field ship stall | after the (34,36) pick the farmer wandered 20k frames, `HARVEST_ROUTE` hit the 24k watchdog | `(11,26)-(11,28)` are static no-go, a grown west ring plus a regrown weed on (12,30) seal the bin stand (11,30) from BFS; the greedy fallback only worked from beside the bin | `Pathfinder.allow_tiles` lets `HarvestTask` use the x=11 lane; an unreachable bin now tosses the crop and fails the route after `SHIP_NAV_TOTAL_LIMIT` | Summer D5 pin: 11 crops shipped (880G), was 1 |
| Can fetch stall | `ENSURE_WATERING_CAN` `pixel_stuck (336,489)`, `CROP_WATER` skipped every crop | `field_to_shed` waypoints on regrown fence (see above) | `MultiMapNavTask._at_standable_substitute` | Summer D15 pin: `tool 0x10 ready`, 27 crops watered, was 0 |
| Pasture route stall | `PLANT_PASTURE` and `CUT_FODDER` `multi_nav timeout` (8 + 20 in one year), the farmer circled the north yard | the y=31 gap had moved to x=11-12; the only way to it is the same x=11 lane, a 93-step detour otherwise | the pasture `MultiMapNavTask` allows the lane | Summer D27 pin: 18 tiles planted, hen bought D28 |
| Tool trip | 4 wasted town trips on Fall D25-D29 (`no shelf tool to buy`) | gate was brush price + reserve even with only the milker (1,800G) missing | `cow_tool_trip_affordable` prices the next missing tool; no reserve in Fall | pin run: brush bought Fall D21 (1,390 to 590G) |
| Grass reserve | 810G held on Summer D28 behind 1,200G of corn-seed bags that cannot harvest, pasture stuck at 25 tiles | `pasture_reserve_g` reserved ring seed to D30 | seed reserve ends at `SUMMER_LAST_CORN_BAG_DAY` (20) | `cow_fix6`: 34 tiles, cow bought |
| Cut before plant | Summer D28 and D29, the last planting days, cut fodder instead | `pasture_day_phases` cut when hay < 48 | planting goes first unless hay is below `hay_wanted` | `cow_fix6`: D28 `BUY_GRASS_SEEDS` + `PLANT_PASTURE` |

Opening the lane for **every** farm path was tried and rejected: 40
`CROP_ESTABLISH` failures and 41 day floors in one year, against 6 and 12. The
west pocket's hoe-stand navigation relies on it staying closed.

Measured on the real ROM:

- Buy and feed: `cow_fix6` bought the cow on Fall D27 in a Clean power-on run
  and fed it from D28. A `PR_Y1_Summer_D1` pin run bought it on Fall D1 and fed
  it 94% of animal-days to Winter D5.
- Brush: bought on Fall D21 in the pin run, daily brushing after.
- Milker and milk: `LN_cowfix_whatif_milker` (Spring D6 what-if, brush owned,
  2,200G poked in) bought tool `0x0e` on the first open day (2,200 to 400G), and
  the next morning's chores milked and shipped 150G. A power-on year has not yet
  reached the milker: Winter shuts town trips (`include_shop_run`, the Winter D1
  horse event strands the bot), so it must be bought by Fall D29.
- Winter feed: hay falls 4 a day for a cow and three hens. From Winter D5 (86 hay)
  it hit 0 on D27 and the animals went unfed for three days; in `cow_fix6` (cow
  and two hens, 61 hay on Winter D1) it hit 0 around D26, 86% fully fed. Fall cuts
  work (9 of 9); 34 tiles produce about what the herd eats, so no Winter reserve builds.

### Economy consolidation (2026-10-28)

Prices, shop gates, reserves and the pasture, hay and tool decisions moved
from `sale_menu`, `day_phase_catalog`, `cow_geometry` and `day_plan_status`
into `livestock_econ` (one price table; `buy_seeds` reads it). Each animal gate
now has a name (`livestock_block`), and the journal records them each morning.
Check: a power-on year on the refactored tree reproduced `cow_fix6`'s journal
exactly (119 days, 1,519,848 frames, all fields except the new `gates`).

That run's gates explain the Fall D27 cow: it waited on `season` 59 mornings,
`pasture` 20 (under 32 planted tiles), `money` 20, `shop_closed` 17, `weather`
1 and `reserve` 1. Summer lost the cow to pasture and cash in equal parts.
(That run's `tool_trip` field predates the cow-owned condition; ignore it.)

### Summer 1 tuning (2026-10-29)

**Where the money was.** `cow_fix6` shipped 25 crops in Summer. Corn is the Summer
crop (300G a bag, 120G a crop) and it is ripe 12 nights after planting (measured:
tile `0x3A` to `0x52`; tomato `0x26` to `0x38` is 9), then regrows every 3 nights, so a
ring planted on Summer D1-D3 gets six harvests and one planted on D8 or D13 gets
four or two. The four working rings (`POCKET_PLANT_CENTERS`; (19,28) sits on a
2x2 stump that needs the axe) were full of late-Spring potatoes until Summer D6-D10,
sown on Spring D24-D30, so corn went in on D8 (one ring), D13 (three) and D16 (one).
Cash was not the limit: 1,400-2,000G sat idle from D2 to D12.

**Changes** (all in code, tests in `test_crop_planner`, `test_two_bag_replant`,
`test_livestock_econ`, `test_seasonal_prep`, `test_ring_capacity`, `test_bin_lane_exit`):

| change | where | why |
| --- | --- | --- |
| Potato and turnip are not sown after Spring D24 | `crop_planner.SPRING_LAST_SOW_DAY` | the rings are empty on Summer D1; the D29-D30 bags (800G) and their 28 potatoes were worth less than 2-3 weeks of earlier corn |
| A ring of ripe Spring crops counts as about-to-be-empty in Summer too | `farm_pond.pocket_plant_targets(replant_ripe=True)` | the shop trip happens on the harvest morning, not the day after |
| The grass errand waits until D3 while rings lack corn bags | `livestock_econ.pasture_errand`, `RING_FIRST_PLANT_DAYS` | two town trips on one day: the second failed (`exit_walk timeout`) or landed past the 13:00 window |
| A water tile skipped twice is parked for 3 days, then retried | `crop_planter.WATER_PARK_DAYS` | a parked tile is never watered, so it never stopped being dry: two bad days parked a whole field for the Summer (see below) |
| A walk that starts on the bin stand or lane may leave by the lane | `multi_nav._lane_exit`, `farm_pond.SHIP_BIN_STAND_TILES` | with the west ring grown the lane is the only way out |
| `SUMMER_LAST_CORN_BAG_DAY` 20 to 18 | `livestock_econ` | a bag planted after D18 cannot ripen before Fall |

**The failure that made it variable.** Without the water-tile expiry, 3 of 4
samples lost most of the corn. `water_end` ring census (a probe printed every ring's
tile ids after each `CROP_WATER`) showed the tiles frozen one stage short of ripe
(even id `0x48`/`0x50`) and the journal `CROP_WATER no_work: water-only; no dry crop
tiles` every day to Fall: two bad days (a sealed bin lane, a stuck can fetch) skipped the pairs of
every tile twice, all were parked, and parked tiles are never watered.

Net worth at Fall D1 (wallet + 1,000 per chicken + 5,000 per cow), power-on, one
sample each; the cutoff day is varied only to move the RNG path:

| tree | cutoff | Fall D1 | Summer crops | cow | note |
| --- | --- | --- | --- | --- | --- |
| `cow_fix6` | none | 1,410 | 25 | Fall D27 | baseline |
| `s1_cutoff_c24` (cutoff + replant) | 24 | 9,550 | 131 | Su D27 | no Summer storm |
| `s1_cutoff_pasture_wait` | 24 | 4,300 | 39 | none | field parked from D21 |
| `s1_cutoff_c22` | 22 | 1,750 | 0 | none | field parked from D11, 185 sealed walks |
| `s1_cutoff_c26` | 26 | 2,830 | 83 | none | |
| `s1_fix_c24` (+ park expiry, stand exit) | 24 | 9,930 | 152 | Su D21 | brush D22, milker D27 |
| `s1_fix_c22` | 22 | 10,260 | 118 | Su D27 | |
| `s1_fix_c26` | 26 | 6,960 | 100 | Su D30 | |

Full year: `s1_ring_cutoff` 21,900 (the c24 draw above) and `s1_ring_fix` 25,730.

**Tried and rejected.**

- *Lane fallback*: when BFS finds no path, retry once with the lane open. It removed
  every "sealed" event but raised parked tiles to 94-137 a run and bad days to 7-8 in
  Summer; Fall D1 net worth 1,810 / 4,160 / 7,810 for cutoffs 24 / 22 / 26
  (`s1_lane_fallback_c*`), worse than without it in two of three. The lane is closed to
  walks that have a path for a reason (the hoe stands).
- *Tomato for the rings*: it ripens in 9 nights, but at a D1-D3 planting it pays the same
  per tile as corn (7 x 100G against 6 x 120G) and carries 20% less gold per crop
  walked to the bin. It wins only for a planting after about D8 (a ring replanted after
  a hurricane). Using it needs one source for the Summer seed: `policy.summer_crop`
  (the buy), `DEFAULT_SEASONAL_SEED` (the plan) and `resolve_seed_type_for_date` (the
  rank) all choose independently, and the plan overrode a tomato buy with corn
  (`ENSURE_CROP_SEEDS -> no stored corn seeds`).
- *Chopping the stump at (19,28)*: a bounded `CLEAR_FIELD` (`priority=["STUMP"]`) looped in
  "Approach plant pocket" at (201,469) on the bin lane and never reached it.
- *Correcting `CROP_SPECS`* (corn 12, tomato 9): the rancher ledger is pinned on the
  current figures (`test_rancher_ledger_result_is_pinned`), so the measured numbers are a
  comment only.

**Caveats.** One power-on run per tree and cutoff, so these are draws, not a
distribution; the old baseline is one draw too. Hurricanes are the open risk: each
wipes about a quarter of the standing tiles and a ring missing fewer than 3 is not
refilled. Summer storm days, from the journals: `cow_fix6` D12 and D25; the three
`s1_fix` draws D24, D9 and D2/D24/D26; `s1_ring_fix` D24; `s1_ring_cutoff` none. The
three-storm draw has the lowest tuned net worth (6,960). The harvest failure memory
(`harvest_task.drop_failed_harvest_tiles`) has the same permanent shape as the water
one and parked tile (12,28) in one run.

### Chicken economy audit (2026-10-29)

Question: is "hatch every egg, sell each chicken as soon as possible" the best Year-1 chicken
policy? Checked against the ROM and the `s1_ring_fix` journal, then a day-level model, then a power-on
A/B. **The A/B contradicted the model: the pasture fix lowered net worth by 2,750** (see "Measured" below).

ROM: a sale pays a flat 500G for an adult only; eggs and chicks never eat (only adults take the
night's feed); an egg placed on day t is an adult on morning t+10 and the incubator frees on the
hatch night, so one chick every 3 days at most. Hatching never meets the shop gate, but every
hatched chick counts in it (16 tiles per animal).

What `s1_ring_fix` did: one hatch all year (Summer D22), against about 30 possible. The cause was
the pasture, not the sale policy. `PLANT_PASTURE` reported success on every morning from Summer
D15 to D30 (about 3,300 frames each) with `planted_grass` flat at 34-35 and 2-5 grass bags (up to
2,500G) in stock: `pasture_day_phases` never passed `target_tiles`, so the planter used its
default of 32 and returned at once while `pasture_target_tiles` wanted 48-80. That capped
`pasture_feeds_more` (no hatches), the second cow's grass gate, and Winter hay (0 at Year 2 Day 1).
Cash was idle from Summer D20 (5k to 18k), so it was not the limit; the cow date follows corn
income (about D20) and hay, which a first chicken sale (D24 at the earliest) cannot move.

Model (calibrated on that run, final wallet within 4%, 80 weather seeds, liquid net worth = wallet
+ 500 per hen + 5,000 per cow):

| policy | pasture stuck at 35 | pasture sown to about 77 |
| --- | --- | --- |
| baseline (hatch gate shut, keep hens) | 25.4k | 33.6k |
| hatch all, sell every adult at once | 25.2k | 29.5k |
| same, cow bought first | 29.6k | 29.6k |
| hatch only what the hay feeds, keep | 26.8k | 34.8k |
| hatch all, keep to the hay limit, sell the excess, cow first | 29.6k | 35.3k |

Reading: selling is worth about +4k only while the pasture is stuck, and only with the cow bought
first (otherwise the chicks hold the grass gate shut and the cow slips to about D57). With the
pasture sown, keeping hens up to what the hay feeds is as good as flipping (a kept hen lays 50G a
day against a one-off 500G), so the sale pipeline's risk buys nothing. Rain and big-harvest gating
of the sale trip barely moved the result. The chicken sale tasks have no passing Clean power-on run:
the May 2026 logs under `logs/chicken_sale_debug*` are all failures.

Changed: `pasture_day_phases` passes `target_tiles=pasture_target_tiles(ram)`
(`test_pasture_planter_keeps_sowing_past_the_cow_gate` fails without it); the scorecard prints an
`at resale` net worth beside the 1,000-per-hen headline, which is unchanged so earlier rows still
compare.

**Measured (power-on A/B, same tree, only the target differs; the control reproduces
`s1_ring_fix` to the gold):**

| | `pasture_ctl` | `pasture_fix` | B-A |
| --- | --- | --- | --- |
| net worth (1,000 per hen) | 25,730 | 22,980 | -2,750 |
| at resale (500 per hen) | 24,730 | 19,980 | -4,750 |
| planted grass / peak hay | 34 / 67 | 68 / 126 | |
| hens, cows at the end | 2, 1 | 6, 0 | +4, -1 |
| bad days | 11 | 26 | +15 |
| fully fed animal-days | 79.5% | 64.9% | -14.6 |
| milk + egg gold (est.) | 12,650 | 13,350 | +700 |

The planter fix did what it should (68 tiles, hay 126 on Fall D28, six hens hatched), and the model
was wrong about what that buys:

- **The hatch gate is not Winter-safe.** `pasture_feeds_more` counts 9 tiles per animal for the
  current season only. Six hens and the cow ate 7 hay a day against 106 on Winter D1; hay was 0 from
  Winter D19, the animals went unfed for 11 mornings, and the cow was in the barn through Winter D28
  and gone by D30 (5,000 off the headline). The ROM kills a sick adult cow when its counter runs out
  (`docs/perfect_rancher/research/F3_herd.md`, "Death"); that is the likely cause, not observed
  directly. The model treated an unfed night as lost milk only, which hid this.
- **Coop chores do not scale with the flock.** 11 mornings failed `COOP_CHORES ... not in coop
  tilemap=0x00` (Fall D13, D15, D16, D27, Winter D1-D14) and three more failed on `NAV_TO_COOP` /
  `EXIT_COOP`; 12 of 37 Fall/Winter chore days collected and shipped no egg although 6 hens were fed.
  Egg-ship days fell from 59 to 38 while hens tripled, so milk + egg gold rose only 700.
- Bad days rose from 11 to 26, and Summer wallet fell 1,400 (bin gold -2,020 with more crops
  shipped), which this audit has not explained.

Reading for the next change: pasture size was not what capped Year 1. Do not re-enable a larger
pasture target on its own. It needs a hatch gate that covers hay to Year 2 Day 1 (stock plus the
rest of Fall's production against every animal's remaining days), and the coop entry and multi-hen
egg pickup need to hold at 4-6 hens first. One draw each, so treat the sign and size as indicative.
`pasture_ctl.json` and `pasture_fix.json` are under `recordings/y1_bench/` (gitignored).

### Next session, in order

1. A fifth ring. (19,28) is the closest 8 tiles to the bin and sat on its 2x2 stump
   all Year 1 (the monthly clutter regrowth puts it back over bare tiles). Needs a
   stump-chop route that does not stall in "Approach plant pocket"; worth about a
   quarter more corn, and it keeps the stump from regrowing in Spring.
2. Sealed yard walks. `s1_cutoff_c22` logged 185 `no_path sealed` events, 168 at
   (250,454), the pocket between the grown west ring, the well, the stump and the y=31
   fence: with the west ring planted the bin lane is the only way home from the south-east
   yard, and it is no-go in BFS. The broad fallback made it worse (see above), so the
   fix has to be narrower: for example `RETURN_HOME` ending its south route west of the
   ring, or leaving the ring's notch walkable.
3. Hurricanes and late replants. A ring missing fewer than 3 tiles is not refilled and a
   replant after about D8 is better done with tomato (9 nights); both need the single
   Summer-seed source described above. The `ENSURE_CROP_SEEDS` seed type and the
   `BUY_SEEDS` crop must agree first.
4. Harvest throughput. A 32-tile cycle comes in as 15-17 crops one day and 7-8 the
   next (the `HARVEST_ROUTE` watchdog); the last cycle must be picked on Summer D30
   or it dies on Fall D1. Pace the rings so the cycle does not straddle a day.
5. Winter hay. 34 planted tiles feed a cow and two hens but leave no reserve
   (`s1_ring_fix`: fully fed 79.5% of animal-days, 0 hay at Y2 D1). With cash from
   Summer, plant more grass or hold the second hatch.
6. Bin-lane debris. A weed or stone regrown on (11,27)-(11,30) seals the ship walk
   again; today it tosses the crop and fails the route. A bounded `clear_field`
   (`farm_bounds`, `priority=[WEED, STONE]`) before `HARVEST_ROUTE` would close it.
7. `day_phase_registry.py` is 1,014 lines, over the soft cap. The unscheduled
   Ann `ProposalTask` (about 290 lines, uncommitted, nothing in the day plan
   emits `PhaseKind.PROPOSE`) is the cut; the owner decides. `EXIT_BARN` has the
   same 5-frame settle bug the coop exit had (`day_phase_cow.py`, settle 90).

### Cow gate follow-up

The v2 journal had hay during Fall (69 on D18), but only 35 planted grass
with four chickens. The vendor needs 80 for those chickens plus a cow. Grass
planted in Fall does not increase the vendor's counter, so this must be
addressed before Fall. The planner now reserves the first cow's vendor grass
requirement when deciding whether to incubate; the delivery-day coop ships
eggs until the next live planning decision. This is a policy fix, not yet a
verified power-on cow purchase or an end-to-end hay result. A replay from
`PR_Y1_Summer_D1` with the policy bought a cow on Fall D1 (`0->1`,
5,340->340G) and fed it on subsequent days. This pin is not a clean
power-on benchmark. At Fall D20 it had one cow, three chickens, 39 planted
tiles, 107 hay and 1,240G; it stopped at the morning before the first
affordable brush trip. The tool trip and Winter stock remain to be verified.

The `PR_Y1_Summer_D1` replay with the merged crop rings bought and delivered
the hen on D13, but failed `CROP_ESTABLISH` at `nav_hoe_ring_1_left: nav
timeout` before reaching the old watering-can failure. It also had a D3
pasture navigation failure at `(134,407)`. This replay does not reproduce or
disprove the v2 can stall; see `logs/y1_bench/cow_debug_pin.log`.

The first power-on run with the hatch policy reached Summer D5 and then
repeatedly deferred adjacent unreachable harvest targets. A regression test
covers the cycle; the task skips a target after it has already been deferred
without a successful pick. The eastern-field shipping stall that followed is
fixed below.


### Component probes, 2026-11-02 (no new full run)

Medicine purchase/delivery/pickup and turnip purchase return to the farm with
ROM receipts. Two- and three-cow care pass four idle perturbations each; the
12-cow research gate remains blocked on feed/navigation/exit. Crop siting now
penalizes shipping distance for Spring crops too, and prioritizes closer
west-strip sites. Mirrored eight-crop side notches pass single-bag planting
and immediate watering: ring 1451+1079 frames; east notch 1454+1019; west
notch 1611+1021, same (9,35) plot from the natural Spring-D3 pin. This is not
a mature harvest or paired-layout benchmark. No new economy or ending claim.
Evidence and remaining gates: [plan.md](plan.md#isolated-reliability-work-toward-999-2026-11-02)
and [component receipts](perfect_rancher/research/care_components_20261102.json).

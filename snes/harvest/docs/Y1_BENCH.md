# Year 1 benchmark

Run power-on through Winter D30 from `snes/harvest` with its venv (`uv sync` once; PyYAML is a declared dependency, the root venv works too): `HEADLESS=1 .venv/bin/python -m harvest.scripts.run_to_day2 --power-on --until-year 2 --out recordings/y1_bench/<name>.json`.
Score it: `.venv/bin/python -m harvest.tools.y1_bench recordings/y1_bench/<name>.json [other.json] [--out card.md|card.json]`; two reports add a B-A delta.
Every economy number (prices, reserves, pasture and hay targets, the cow target) is in `harvest/planner/livestock_econ.py`. Journal rows carry `gates` (`ranch_gates`, the same morning read as `economy`), and the scorecard prints the first morning the cow gate opened and how many mornings each gate held it (`cow waited on`). Reports older than 2026-10-28 have no `gates`.
Headline is net worth = wallet + 1,000 x chickens + 5,000 x cows, both parts stated. Only a power-on, zero-write, zero-assist run is marked comparable.
Journal rows carry `economy` (morning read, same moment as `money`) and `eod` (pre-sleep read). `shipping_money` is read after the night payout and is always 0, so bin gold comes from `eod.bin_gold`.
Milk and egg gold is bin gold minus the lifetime crop counters' delta at ship prices minus forage `shipping_deposit`s, on days with cow or coop chores. Milk tier and egg count are unrecorded, so it is an estimate.
Reports older than the `economy` fields print `n/a` for brush, milker, bin gold and animal gold, and mark flock sizes `~` (from purchase receipts). `planted` is `n/a` while the establish phase reports "sequence complete". Chickens include chicks and incubating eggs.

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

v2 is worse than v1 on every line after Summer D13, so v1 is the number to beat. `cow_fix6` is the first power-on run to buy a cow (Fall D27, 5,310G to 310G), but its net worth is still below `v1_animals`: the rings put every Summer crop 40+ tiles from the bin, and the cow arrives too late to pay for the tools before Winter shuts the town trips. Brush, milker and milk are measured in pin runs below, not yet in a power-on year. `s1_ring_fix` (Summer 1 tuning, below) beats `v1_animals` by 5,460 and `cow_fix6` by 16,020, with a cow from Summer D21, the brush and milker owned, and 36 milk shipments.

### Why v2 regressed (found, fixed 2026-10-28)

On Summer D13 the hen was delivered, then `BUY_GRASS_SEEDS` + `PLANT_PASTURE` ran, then `ENSURE_WATERING_CAN` failed (`route failed: pixel_stuck pos=(336,489)`, `logs/y1_bench/v2_pasture.log` lines 7314-7323). Every later morning `CROP_WATER` failed with `tool 0x10 not selected` and `dry_crops=7 skipped=7` (222 log lines, 0 in v1) from D13 to D30. The summer corn died unwatered: 5 crops shipped in Summer against 28 in v1, and the 3,000G of grass bags bought did not buy a cow or any new hens. Lead: the bigger pasture (or a grass tile planted that day) now seals or confuses the route to the can pickup near (336,489) on tilemap 0x00. **Found and fixed 2026-10-28:** not the pasture. `field_to_shed` aims at tiles (21,31) and (25,31), which Summer regrows as fence; BFS retargeted to the standable tile north of it, but arrival was judged against the fence pixel (15px away, radius 12), so the farmer pressed into the fence until `pixel_stuck` failed the route at exactly (336,489). `MultiMapNavTask._at_standable_substitute` now counts a plain waypoint on an unwalkable tile as reached on its nearest standable tile. Reproduce from `PR_Y1_Summer_D1` (or `LN_poweron_summer_d1`) with `run_to_day2 --state <pin> --until-season 2 --until-day 20` and watch `ENSURE_WATERING_CAN` after the first hen and grass buy; v1 never failed it. Those pins are not bench-grade: a pin run proves the failure, not the economy.

### State at handoff

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
policy? Checked against the ROM and the `s1_ring_fix` journal. **The pasture fix below is not
yet benched** (non-ROM test only); the model numbers are a day-level model, not a ROM run.

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
compare. To check: a power-on year against `s1_ring_fix`, reading planted grass, hens hatched,
`cow waited on`, and hay at Year 2 Day 1.

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

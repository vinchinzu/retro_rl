# Harvest agent notes

Package `harvest` (disk `snes/harvest/`, nested import root). Repo rules:
[root AGENTS.md](../../AGENTS.md), [CODING_STANDARDS.md](../../CODING_STANDARDS.md).
Living residual:
[docs/tasks/rr-20w-run16-defects.md](docs/tasks/rr-20w-run16-defects.md).

## Run

Run from `snes/harvest` with its own venv (`uv sync` once). Every morning plan
reads `docs/perfect_rancher/milestones.yaml`, so PyYAML is a declared
dependency in `pyproject.toml`; the root venv works too.

```bash
PY=.venv/bin/python
HEADLESS=1 $PY -m harvest.scripts.run_to_day2 --power-on --until-year 2 \
  --out recordings/y1_bench/<name>.json            # Year-1 bench, ~25 min
$PY -m harvest.tools.y1_bench recordings/y1_bench/<a>.json [<b>.json]   # scorecard, B-A delta
HEADLESS=1 $PY -m harvest.scripts.run_to_day2 --power-on --until-season 1 \
  --until-day 15 --save-end-state LN_<lane>_summer_d15   # mint a faithful pin
HEADLESS=1 $PY -m harvest.scripts.run_to_day2 --state <pin> --until-season 2 --until-day 20
HEADLESS=1 $PY -m harvest.scripts.run_to_day2 --power-on --stop-after-d2-clear \
  --out recordings/power_on_d2_farm_clear.json      # opening farm clear
$PY -m harvest.scripts.interact_scan tape <name>    # or: search <word>
$PY -m harvest.scripts.event_dis group 5 --event 0  # ROM event scripts
QT_QPA_PLATFORM=offscreen uv run python -m pytest snes/harvest/tests/test_perfect_ending.py -m rom -q   # every 999-score term vs the ROM calculator (~3 min, local pin)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest snes/harvest/tests -q -m "not rom and not ml"  # from repo root
```

Full end to end, recorded and scored (power-on to the credits; copies the tree first, ~3 h without
video and ~5 h with it, about 25 GB of one-hour video parts):

```bash
scripts/e2e_run.sh <tag>                  # recordings/e2e/<tag>/{report.json,run.log,video_partNNN.mp4,FINAL.md}
VIDEO=0 scripts/e2e_run.sh <tag>          # same run, lean step, no video (frame-identical, ~1.6x faster)
HARVEST_STRATEGY=score scripts/e2e_run.sh <tag>   # optimize the ranch score, not cash (default cash; run_to_day2 --strategy)
$PY -m harvest.tools.final_score recordings/e2e/<tag>/report.json [<other>/report.json]  # cash, ranch score, terms, cost, B-A
```

Repeatability sweep (Years 2-3 to the ending, perturbed and seeded; copies the tree first):

```bash
JITTER=90 FAULTS='*:fail' RATE=0.01 scripts/y23_sweep.sh PR_y1b_Y2_Spring_D1 <tag> 1 2 3
$PY -m harvest.tools.sweep_report recordings/y23_sweep/<tag>_s*.json   # CLEAN = ending, zero rescues
```

`HARVEST_PHASE_JITTER=<max>` idles 0..max neutral frames before each phase (seed
`HARVEST_FAULT_SEED`): sprites, animals and ROM RNG move without a RAM write.
Neutral frames in the house do nothing (the ROM is frame-identical), so jitter is per
outdoor phase. `HARVEST_FAULT_PHASES/RATE` force phases to fail (see Clock and days).

Targets: `--until-year N` (1-based), `--until-ending`; `--until-season/--until-day`
stop on that day's morning. Video: `--video <stem>.mp4` writes one-hour parts
that stay playable if the run stops. Glance: `harvest.clock_glance`. Natural
entry is power-on.

Headed Year 2 watch (Wayland/Omarchy; root or game venv):
`uv run python -m harvest.scripts.run_to_day2 --headed --speed 4 --state PR_Y2_Spring_D1 --until-year 3 --save-day-pins --pin-prefix PR_watch_`.
`[` / `]` change speed; 8+ is unthrottled. Daily pins include planner lineage:
resume with `--state PR_watch_Y2_Spring_D2 --days 1` for a short replay.
Use a separate prefix for each run whose morning pins must be preserved.

## Change loop

1. **Reproduce** on a pin (`--state <pin>`). A pin replay drifts from the
   power-on timeline after about a week, so a pin proves a failure, never the
   economy. Done: the failing phase and frame are in the log.
2. **Fix** with a non-ROM test that goes red on the bug. Done: the non-ROM suite
   is green.
3. **Bench** a fresh power-on year and score it against the last report. A
   power-on run is deterministic for unchanged code: a refactor must reproduce
   the previous journal exactly, and a policy change shows up as the scorecard
   delta. Done: both scorecards compared, `cow waited on` read.

Results go in [docs/Y1_BENCH.md](docs/Y1_BENCH.md) (runs table, next steps).
`docs/STATUS.md` holds power-on results only.

## Year-1 economy

Every price, gate and policy knob is in `harvest/planner/livestock_econ.py`:
ROM facts, then Year-1 policy, then gates. Tune a knob there and bench it. The
journal stores `ranch_gates` each morning; the scorecard prints the first
morning the cow gate opened and every reason it stayed shut.

The calendar is the binding constraint:

- **Spring**: crops only. Potato and turnip cannot be sown after Spring. A
  watered potato ripens five nights after sowing, so `SPRING_LAST_SOW_DAY` (26)
  lets the last bags ripen by Summer D1-D2, when the ring takes its corn bag
  (Summer D1 is a Wednesday; the shop is shut Sat and Sun).
- **Summer**: corn is the money crop: 300G a bag, ripe 12 nights after planting,
  then every 3 (tomato: 9 and 3). A ring planted by D3 gets six harvests, one
  planted on D4 five, so the grass errand waits until D3 and the rings' corn bags
  are the day's only town trip (`RING_FIRST_PLANT_DAYS`). Corn bought after D18
  cannot harvest before Fall. Planted grass is the only kind that counts toward
  the animal shop's gate (16 planted tiles per animal after the buy, chicks
  included).
- **Layout switches** (process-wide, read at import): `HARVEST_RING_LAYOUT=legacy|packed|packed18|packed12|blocks` (default `legacy`; `packed` is 26 rings,
  four rows of six at pitch 4 sharing one lane, see `farm_pond.PACKED_RING_ORDER`; `blocks` is the owner's notch-block plan, below), `HARVEST_PASTURE_ZONE=south|east` (east when packed),
  `HARVEST_WATER_TOUR=0` (packed only), `HARVEST_DAY_FRAME_BUDGET` / `HARVEST_DAY_CLOCK_GRACE` (code defaults 0 / -1 = off; older text said 216,000 / 60,000; owner: no per-day
  cap, only stalls end a day). Tests built on legacy sites or the south pasture use `tests/layout_markers.py`; run the offline suite under both
  layouts. Hard-coded farm walks (`map_routes`, `farm_gate`) cross the south field: a ring tile on a waypoint is moved to the nearest standable
  tile by the nav, but keep lanes at x=6, y=33 and y=37.
- **Year-3 score switches** (all `year >= 2` only, so Years 1-2 keep their frames; `=0` switches one off for an A/B): `HARVEST_Y3_JOINT_BUY` (corn and tomato bought on one shop
  visit, `BuyShelfItemTask.then`), `HARVEST_EARLY_COVER` (ragged stands when no full stand is left, a sown ring's band released, 60-bag Summer loads), `HARVEST_WATER_RETRY` (a wedged walk
  to the can is retried twice). Not switches: the Eve gift unstick (`eve_gift.py`) and the house-exit sidestep (`ExitBuildingTask`). `HARVEST_Y3_SUMMER_EXTRA_SITES=N` stays 0 (crops
  up, development down). Evidence and open bugs: `docs/plan.md` "2026-11-17 evening", `docs/tasks/NEXT_SESSION_TODO.md`.
- **Notch blocks** (`HARVEST_RING_LAYOUT=blocks`; owner: say "block", not "ring"): 3x3 plots with the centre planted and one side
  cell bare, `xxx/xxo/xxx` and `xxx/oxx/xxx`, on the pitch-4 lattice in mirrored pairs whose notches face across the shared lane.
  `planner/block_calendar.py` ranks them by measured pickup-to-ship frames and sets each block's calendar
  (`farm_pond.BLOCK_PLAN_ROWS`, gated in `pocket_plant_targets` by `calendar_bag`): 12 `core` blocks sow their last potato on
  Spring D25/D26 (ripe Summer D1/D2, 6 watered nights) and take corn that morning; 2 `late` blocks sow to D29/D30 (ripe Summer D5/D6).
  A bag is sown standing on the hoed centre, entered through the notch. Report: `python -m harvest.tools.block_calendar_report
  plan|track <run>/report.json` (plan JSON, forecast, herd estimate, map page, run vs plan).
- **Forage** (`planner/forage_calendar.py`, one ROM table; `HARVEST_FORAGE_FIRST=0` puts the trip last again for an A/B): mountain tiles give
  a different item per season, grape Spring, green fruit Summer (tile (28,10), stand (454,187) facing up with a 16-frame push), poison mushroom
  Fall (200G, stand (678,633) facing right); a tile gives again the same day. With nothing ripe to ship the trip goes ahead of the crop work
  (the clock freezes at 18:xx and work goes on, only the 17:00 bin is a deadline; it may start up to 12:00, `trips_today` refuses a loop past 17:00);
  Spring/Summer take two loops, Fall as many as fit (bail hour 14, unbenched). Count trips from the phase reason, not `shipping_deposit`. Keep the
  40px north pull after the mountain landing on every inbound route or the BFS walks back off the south exit.
- **Spa refills**: `CROP_WATER` may splice up to four soaks a day (`day_phase_stamina.spa_refresh_allowed`); a watering is 2 stamina, so one bar
  waters ~46 tiles. The 1,000-line cap on `day_plan_orchestrator.py` is tested (`test_composer_seams`): new rules go in a sibling helper.
- **Rings** (`farm_pond.POCKET_PLANT_CENTERS`, data): the two yard rings, the
  three south-field rings and the four `WEST_STRIP_PLANT_CENTERS` south-west of
  the bin. The D2 full clear leaves the farm array (`$7EA4E6`) with zero debris
  through Summer D1, so the old (19,28) stump and y=31 fence do not come back.
  Nine rings are 72 tiles, about 150 stamina of watering a day: `CROP_WATER`
  arms what the bar pays for and ends `stamina_low unfunded=N`, the day plan
  splices one `HOT_SPRING_STAMINA` soak and reruns it (`TOOL_STAMINA_PHASES`),
  and `ring_establish_affordable` counts that soak. A ring takes a bag only with
  3+ open cells and a centre that a side cell reaches (`ring_center_reachable`):
  a corner-only opening walls the sowing stand in. A `CROP_ESTABLISH` that still
  fails after its recovery retry is skipped, so the day's watering still runs.
- **Flower Festival** (Spring D23) puts the farmer to bed, so `apply_eve_phases`
  waters first: `CROP_WATER` with `stop_hour` 12 (optional) arms only the tiles
  that finish before noon (`WATER_MINUTES_PER_TILE`), then the dance.
- **Fall**: last grass. Cut hay to cover the rest of Fall plus 31 Winter
  days (`hay_cut_target`). The grass errand is planned before noon; after noon
  only a cut for a herd short of hay runs (y23_final Y2 Summer: wood trips ate
  every morning and the hay hit 0 with 101 tiles grown).
- **Year 2+ Spring**: Winter's night check wipes the grass and decrements
  `planted_grass`, so Spring starts bare with only the hay left. With a herd,
  the pasture is sown again from Spring D1 (`pasture_season`; a bag counts in
  Spring and Summer only, ROM `season < 2`), and the Fall cut also stocks
  `SPRING_HAY_DAYS` per animal. The shop's fodder (600G) and chicken feed
  (300G, 12 uses each, used as tools on the trough) are not bought yet.
- **Winter**: the pasture is wiped on D1, animals eat stored hay only. Town
  trips run (tool buys, hen sales): the path is tilemap `0x0F`
  (`PATH_TILEMAP_IDS`), and the first path walk from Winter Y1 plays the horse
  scene, met alone by `MEET_HORSE` on the first Winter morning (its actors seal
  the road west for that visit; the name grid is `unowned_scene_press`). Pin
  `LN_cowfix_pin_winter_d5`: horse, milker bought, chores, and a sale the next
  day. Star Night (Winter D24) talks after 18:00. Year 2+ New Year (Spring D1)
  follows the native `$7F1F7A` talk flags, then the exit woman; it has no 18:00
  gate (`FestivalEscapeTask`, exact-hop `maps/new_year_walk.json`).

Animal facts the plan rests on:

- Hen first: 1,000G, eggs ship 50G, and an incubated egg is a free hen. One hay
  per animal per day; an unfed night sets status `0x10` and costs about three
  laying days. A `chicken_feed` buy sets the stock to 12, it does not add.
  Only adults eat: an egg placed on day t hatches on night t+2 (the incubator
  frees) and is an adult on morning t+10, so a chick costs no hay. A sale pays a
  flat 500G, adults only (`CHICKEN_SALE_PRICE`); the bench headline counts a hen
  at 1,000G and prints an `at resale` line. Hatched chicks count in the shop's
  16-tiles-per-animal gate, so buy the cow before a hatch fills the pipeline.
  A neglected cow dies: with the pasture sown to 68 tiles (`pasture_ctl` vs
  `pasture_fix` in Y1_BENCH) six hens and the cow outran the Winter hay, hay hit 0
  on Winter D19 and the cow was gone by D30.
- Flock rule (owner): keep `HENS_KEPT` (2) laying hens throughout the campaign
  with `FLOCK_CAP` (2) birds total counting chicks and the incubating egg
  (`hatch_allowed` requires room under that cap plus the pasture gate). Once at
  two birds, `coop_mode` stays `"ship"`. Surplus adults from older pins still
  follow the sale route after coop chores without being fed.
- A sale: carry the hen from the coop to the farm drop (60,480), ask at the
  animal-shop counter (text `$030B` sets `$7F1F5A` bit `$0008`), walk home. On
  farm re-entry the ROM clears the bit and starts the buyer's scene, which pays
  500G (`$81ED3C`) and takes the hen after its dialogue; nothing checks the hour.
  Retain count/wallet baselines from the first request across the return trip;
  success requires both the hen count drop and 500G per sale. The request bit
  also clears without payment; the first-Fall trade takes a hen without gold.
  Night reset clears the bit. The counter refuses everything in rain or snow
  (`$0196 & 0x000A`, `$81DE45`), so the planner skips the trip then. The trip is
  done by 10:00-11:00.
- Cow: 5,000G, no milk for 21 child nights. Talk +1 and brush +3 happiness a
  day; milk ships 150G below happiness `0x60`, 250G below `0xC0`, else 350G. Brush 800G and
  milker 1,800G are in the General Store, not the seed shop. Year 1 buys two
  cows in Summer/Fall; Year 2+ buys up to `COW_TARGET_LATER` (4), Spring too.
  The ending scores 3 a cow plus `min(25, happiness >> 3)`.
- A barn adult whose fed bit (`$0932`: `1 << slot`, pregnant `0x1000`) is clear
  at night loses 8 happiness and rolls sickness; a sick cow dies seven nights
  later. Every y23_final cow died after a day the chores never fed it. Chores
  feed first, and the planner runs one bounded barn retry (`feed_check`,
  `BarnFeedCheckTask`) before going home when a barn adult is still unfed.
  The barn clock does not tick indoors, so a chore pass costs frames, not hours.
  **Barn chores** (`tasks/cow_task.py`, map and walker in `tasks/cow_geometry.py`, ROM proof
  `tools/barn_fuzz.py`) are a plan read from RAM each step, not a route: fodder, cure, talk and brush, milk.
  Facts measured on the ROM, 2026-11-05 (old notes that a cow can be run through are wrong):
  - The walk map is what farmers actually stood on (`BARN_WALK_ROWS`). A map poked cell by cell lies: tile
    collision only stops a step *into* a solid, so a poked farmer moves freely inside the rails.
  - Walking into a cow stops for good at 13 px. Running stands ~40 frames, then **shoves her** (2 px per 3 frames)
    unless a wall is behind her. The farmer shoved cows along the hall and out of the door, so routes close
    the cells round every cow (`plan_route(clear=True)`) and only push when no other route exists.
  - A press reaches the point 16 px ahead of the farmer, +-12 px (`press_reaches`). Talk reads it at the press;
    a tool reads the contact left by the last **blocked step**, so a swing needs a real push into her from the
    stand (`_step_in`) and misses if she walks off in the ~65 frames before it lands. Wait until she stops.
  - **A cow on the door tile `$C2` leaves the barn** (`$84A595`): her home map becomes the season number, so in
    Spring it reads 0, and she stops getting chores. Tile `$E1` at farm (20,22) puts her back (`$84A540`).
    No farmer, 8 idle runs of 30,000 frames from a fresh entry: nobody left, the first cow came within 72 px of the
    door at 1,870 (median 14,500) and within 32 px at 13,490 (older: 28 runs of 40,000, first out at 15,550). With the
    farmer working it is faster: fuzz seed 78 had a cow walk from 57 px to the door tile in 350 frames, inside one job,
    6,000 frames after entry. A pass therefore walks out and back in (each entry resets the herd) when a cow is within
    `BARN_DOOR_DANGER_PX` (72) of the door, checked at every job boundary *and every walking frame with empty hands*
    (`WATCHED_JOBS`), or after `BARN_SHIFT_FRAMES` (14,000). The hall rows y 308..329 lead nowhere (the old `y >= 308`
    rule fired within 1,000-4,000 frames). Resets are not free to skip: jobs run ~10% slower per 10,000 frames the herd
    has scattered, and 9,000-14,000 was the flat minimum of the 16-pin timer sweeps. A bottle in hand blocks a shift, so a
    cow on the bin stand is waited out at `BIN_HOLD_PX`, not nudged (seed 20 lost a bottle to the reset).
    `COW_ROUNDUP` is a best-effort net.
  - Children and calves are never tested for a fed bit (`.babyonfarm`), so they get no hay. A cranky cow refuses the
    milker (status `& 0x70`). The tired animation replaces a brush swing at 0 stamina. Milk from 17:00 pays nothing.
  - `$0953` is one shared counter, not a per-cow step timer.
  - **12 cows play at half speed.** From 11 cows the ROM runs the barn one game tick per two frames (10: ~1.4 frames,
    9 or fewer: one), so a B-run is 1 px a frame and every animation lock doubles: 9 cows cost 11.2k frames a Y3 day,
    12 cost 24.1k. Locks after a press: fodder 55, trough 43, talk 105, brush 136, milk 154, ship 46 frames; an X tool
    swap is a 69-frame animation and X is ignored while walking. Floor of a full-care day: ~17,700 frames (12 forkful
    round trips 4,030 px, 12 bottle round trips to the far bin 5,740 px, a ~700 px care tour, ~6,500 locks, entry and exit),
    so halving it needs less care, not a better route. Profile a morning: `python -m harvest.tools.barn_profile run <pin> --out f.json`.
  - A cow bumped by the farmer walks away (the bump sets her flag `0x08`, `$84A617`). A tool press with no bump (a cow
    pinned to a wall, pressed from the side) still makes the bottle but never sets the milked flag: a failed press now tries
    another stand (`_tried`), a bottle in hand ends the job for a ship, and a struck-out cow waits 600 frames behind the others.
  - The fodder stand tolerance is 1 px: a run stopping at x=214 presses out of the dispenser's reach (every third pickup).
  - Talk and brush on a cow at happiness 255 change nothing the score or the milk price reads (`AddCowHappiness` clamps;
    the night pass only takes happiness down). `HARVEST_BARN_SKIP_CAPPED=1` skips them (the bench ledgers expect the flags).
  Fuzz it after any barn change: `.venv/bin/python -m harvest.tools.barn_fuzz run <tag> --profile all --trials 84`.
  The farm dog animates through sprite ids `0x01D9-0x01F7` (`FARM_DOG_SPRITES`); a
  sitting dog in the west corridor (98,430) failed the tool fetch and cost a cow.
  A required phase that is only a means (`ENSURE_ANIMAL_TOOLS`) must be skippable:
  feeding never waits on the brush or milker.
- Carpenter payment waits for the first cow (`house_payment_allowed`).
- Birth day cascade (Y2 Winter D17 in `perfect999b`): the labor morning is a lost day, so every barn adult goes unfed once and rolls sick 1/4
  (-30 player happiness, -12 cow each; the happiness term is `>> 5`, so three sick cows cost ~3 points). The cure is one dose a day
  (single shed slot, animal-shop trip, shop shut in snow/rain), a cow that dies costs -100 and its funeral scene eats the whole next day
  (church, tilemap 7, 14:00) so deaths chain. Keep every medicine trip free of wedges; replay `PR_..._Y2_Winter_D17..D24` after any
  change to the shop routes, cure planner or labor task.
- Summer crops are walking-bound: the farmer carries one crop to the bin stand
  (11,30), and the east rings are 40+ tiles away.
- A crop dropped in the bin at 17:00 or later is destroyed: `Dropedonsaleplace`
  (bank_81 `$8193BF`) skips the `shipping_money` add from hour 17. A ripe crop
  keeps on the plant, so `HarvestTask` stops picking when `ship_window_fits`
  says the walk misses 17:00. Before that fix 27 Summer corn (3,240G) went in
  late and the log called them "money settles 5pm".

## Traps

**Tools that replace tools**
- The sprinkler (0x15) takes the watering can's shelf slot: an owner has no can (like the gold hammer and the plain one).
  Asking for the can failed `tool 0x10 missing after shelf` and ended 41 of 57 days of the 531 run at 13:00, before the
  evening cover. `EnsureCarryToolTask` resolves can to sprinkler, and `SprinklerWaterTask` waters leftovers itself.

**Shipping speed (ROM, `harvest.scripts.ship_bench <pin> [--exit-house]`)**
- The clock counts 900 ticks an hour: the minute byte runs 0-14 (60 ticks each), one tick a frame outdoors, frozen at 18:00.
  `harvest_task.clock_tick`; 17:00 is tick 15,300. Never read `minute` as 0-59.
- The bin throw reads the tile two ahead into F2 (8-9, 29-30): stands (11,29)/(11,30) face left, (8,32)/(9,32) face up.
- Fast press (`HARVEST_FAST_PRESS=0` restores the centred press): the lift ignores A until the farmer is pushed against the crop;
  a one-frame turn after a run may not register (a throw facing the wrong way drops the crop: paid 0), so A waits for
  `player_direction`; a potato tile turns to soil on the A edge, 23 frames before the carry bit, so the verify waits out the lift.
  Each `HARVEST_ROUTE` journal row carries `ship_log` (`harvest.ship_log/1`: frames and ticks at select, pick, throw, ship).

**Navigation**
- Viewport BFS covers about 16 by 14 tiles. Keep hop targets within 7 tiles, or
  call `densify_waypoints`.
- Ellen (sprite 0x023C) stands at town (648,904) on weekday mornings, in front of the animal shop door: no route may put a tight
  waypoint on that pixel. `animal_shop_town_to_farm` stops at (632,904), swings south via (656,936) and joins x=696 at (696,920).
  A sprite whose 90-frame yield times out is parked: `_pad_parked` ring-pads it for the rest of the waypoint (any map).
- Cover trips read `[COVER] morning ... reason=busy morning (N ripe)`; N must be the tiles the harvest will pick, not every ripe tile. Capped
  corn/tomato stays ripe on the plant (`Left N ripe ... at the score_herd ship cap`, `Detected 0 ripe crop targets`) and, counted as work, blocked all
  of Y3 Summer D22-D29 in `fixD` (development 42, not 49). `WorldProbe.ripe_crop_count` filters through `run_strategy.within_ship_cap`.
- Year 3 Spring plants 24 sites (`farm_pond.SCORE_Y3_EXTRA_CENTERS`: 12 more on the pitch-5 grid) because potato/turnip cannot be sown after
  Spring D26 and Years 1-2 supply only ~242/215 of the 496 each. From Summer on `active_plot_centers(year, season)` drops the extras (the
  cover takes their 5x5 back); `reserved_cells` and the bag cap read the same `(year, season)`. Env switches are listed in `docs/plan.md`
  "Score push 2". Harvest throughput (~15-45 crops a day, 170 frames each) is the real ceiling: a shop day harvests ~1-10, so late sites
  only help if they ripen before the season ends.
- A cover walk (`COVER_FIELD`/`COVER_TILL` nav, `MultiMapNavTask(soft_timeout=True)`) that times out ends where the farmer stands and the
  planter takes the next stand. Before this a first stand the BFS route could not reach (north-east spur: `(48,15)` 0xA3 and the 0xA8 bank
  at `(48,16)` are walkable ids that push-block) failed the phase every evening (`y3r_s3` Summer D8-D19: 11 failures, development 47).
- `HARVEST_PHASE_JITTER_FROM_YEAR=2` limits the seeded phase jitter to Year 3, so a *power-on* run plays Years 1-2 exactly like the
  deterministic campaign and only Year 3 draws differ (`scripts/e2e_run.sh` passes the env through; same seed + code replays identically).
  Pin previews (`y23_sweep.sh`) drift from the power-on timeline; a failure seen only in a preview (shed door `(26,30)` "enter failed:
  directional transition timeout", house door `(9,23)` 0xA8 push-block in `y3r_s5`/`y3r_s2`) did not reproduce from its own D24 pin.
- The seed-shop walk (farm exit to counter door) is ~1h50 of game time and the door shuts at 17:00, so a trip that leaves after
  ~15:00 reaches a closed door and the log shows `town_to_shop_door failed: exit_walk timeout on tilemap 0x04`. It is the clock, not a
  route bug (y3e Summer D1 and D6, 15:12 and 15:13 starts; the 14:12 start on D7 bought). Year 3 therefore goes to the shop before it
  forages (`SHOP_FIRST_FROM_YEAR`/`SHOP_FIRST_LATEST_HOUR` in `day_phase_berry.py`); Years 1-2 keep the forage-first timeline.
  `ReturnHomeTask` walks out of the animal shop (tilemap 0x24, or the plaza south-west of the town map) with
  `animal_shop_to_town` + `animal_shop_town_to_farm`; `ExitToFarmTask` only exits farm buildings.
  The medicine buy's exit segment `town_animal_shop_to_path` has the same swing (it once started on her pixel with radius 4: Y2 Winter D19 lost
  the day, `test_animal_shop_exit_routes_keep_clear_of_ellens_stand` guards both routes).
- Centring before a sideways step (`Navigator.follow_path`) presses toward the tile centre even when a solid sits there: the
  farmer then stops 7 px short for good. `stasis` (which releases it at 30) is reset by every replan, and the 48-frame
  pixel-stuck replan always came first, so the mountain Harvest Festival walk (fish in hand, (566,593)) failed 4 replans. The
  opt-in `Navigator.center_stall_limit` (`MultiMapNavTask(center_stall_frames=CENTER_STALL_FRAMES)`, set by `FestivalEntryTask`
  only) counts centring frames that leave the farmer on one pixel across replans and yields to the path, on live-pad frames only
  (`_pad_is_live`: faded in, input free, walking or carrying). Do NOT turn it on for every walk: skipping centring after each
  replan broke the D2 opening plant, and the detector on all walks died on Y1 Spring D2 twice (dead-pad fade-in frames counted;
  then a required `nav_hoe_ring_3_down` timeout went terminal). Pin ROM tests do not cover the power-on house exit: compare a
  fresh power-on run's morning-pin frames with the bench. Carrying (`player_state` 3) is not in
  `PLAYER_STATES_FREE`, so a carrying farmer's wall is invisible for 240 frames; known, left alone.
- WEED `0x03` is not travel-walkable. Stand on a neighbour to clear debris;
  BFS onto debris or push tiles fails.
- Bin lane: `(11,26)-(11,28)` are in `FARM_NO_GO_TILES` but are open ground.
  The harvest walk and the pasture route allow them (`SHIP_BIN_LANE_TILES`);
  every other farm path keeps them closed, because the west pocket's hoe stands
  depend on it. A walk that *starts* on the lane or the bin stand
  (`SHIP_BIN_STAND_TILES`) may leave by it: with the west ring grown it is the
  only way out, and a sealed RETURN_HOME or can fetch cost whole days of
  watering. A ship walk that cannot reach the bin tosses the crop and fails
  after `SHIP_NAV_TOTAL_LIMIT`. A weed regrown on the lane seals it again.
- An uncleared farm showed fence on y=31 in Summer (after the D2 full clear the
  array stays debris-free through Summer D1). A static waypoint on an unwalkable tile counts
  as reached on its nearest standable tile (`MultiMapNavTask._at_standable_substitute`).
- Clutter on an uncleared farm regrows monthly. The hoe ignores weeds; lift and toss them. A tool at
  0 stamina does nothing. Weeds (`0x03`) disintegrate immediately upon lift (A)
  with 0 stamina and 0 held items. Small stones (`0x04`) become held item `0x0D`
  and require tossable terrain. When unarmed (`Tool.HAMMER` missing),
  `drop_unarmed_debris` prioritizes weeds before stones (`[WEED, STONE]`), and
  `grass_planter` clears only weeds (`_target_ids()` returning `frozenset({WEED})`)
  to prevent holding stones with seeds equipped. Idle afternoon daylight before
  18:00 runs `CLEAR_BUSHES_PHASE` (`include_bush_clear`), clearing 100+ weeds per
  day with zero stamina to boost farm development score toward 2,950 tiles (50 pts).
- The yard ground south of the house door (tile `0xA8`) refuses every toss, with
  or without the d-pad, and the drop spot `drop_spot_px` (136,480) sits on the
  bin ditch (`0xF2`). A stone carried home (the far-east escape's A presses lift
  one) sealed Summer D27 for 8 rescues. RETURN_HOME now carries debris
  (`POND_TOSS_HELD`) to the main pond lip (`PondTossTask`, pond first when it is
  nearer). Pin `LN_y23_wander_d27` is that wedge.
- South-field walks to the west gate (spa, shop) take the steps at
  (3-4,30-31), tile `0xE0` (`_FARM_SOUTH_WEST_TO_WEST_GATE`). The old x=13
  climb runs through the (13,28) ring and stopped 18 of 44 soaks once it grew.
- On a festival eve the scene's location is FESTIVAL on the town, path and
  mountain maps, so a non-walking `player_state` reads as a cutscene. The hot
  spring soak (0x21) is excluded by `SPA_POOL_TILE_BOX`; without it the spa
  task pressed dismiss until it timed out in the water.
- The fenced yard behind the animal shop opens only through the shop's back
  door (36,51). `in_animal_yard` (whole fence box) picks `animal_yard_to_farm`
  for the hen walk and for RETURN_HOME; a town route from inside pixel-sticks.
- Arriving on a map, the pad is dead until the fade-in ends (WRAM `$0025`
  reaches 15, `screen_faded_in`), ~70 frames after a coop exit lands at
  (456,360); the tilemap reads the farm ~160 frames earlier. A route started
  mid-fade cycles left/down/up and the up re-enters the coop: 27 of 33
  y23_final rescues. Building exits to the farm wait for it. The mountain landing
  (328,728) needs the north pull (`_MOUNTAIN_LANDING_NORTH_PULL`): 12 of 13
  Fall mushroom trips walked back onto the path without it. A shop trip bounced
  back to the path walks in again (`TOWN_BOUNCE_RETRIES`).
- Tile ids 0x72/0x75/0x76 are both "stale viewport" markers (`STALE_TILE_IDS`)
  and pasture grass stages. `farm_map_loaded` skips `PASTURE_BOUNDS`; a Year 2
  pasture otherwise reads as an unloaded farm and the clearer walks north-west
  for good (pin `LN_y23_wood_y2s6`).
- The house door is (8,20) at the top of an open walkway (8,22)-(8,26). Enter by
  pushing up from any walkway tile (`in_house_walkway`): from Year 2 a sprite
  (object 0xC7, probably the horse) can park on the stand (8,26).
- A crop tile whose water pair is skipped twice within `WATER_PARK_DAYS` (3) is
  parked for those days, then retried. Parked tiles are never watered, so a
  permanent park left a whole corn field dry for the rest of Summer after two
  bad days (a sealed bin lane); a stuck field shows as `no_work: no dry crop
  tiles` while the ring tiles sit at an even id one step before ripe.
- The farm's two holes (tile `0xC7` at (50,55) and `0xC8` at (37,48)) are the Harvest
  Sprites' cave, map `0x2A`. It is not BFS-walkable, but the far-east escape charge ("left 90,
  up 80" from (873,937)) ends under it and walks in. The cave has no exit the bot
  found (85 cells from the spawn, A pulses on the ladder do nothing) and the scene
  reader calls it a dialogue (stale text `0x031A`), so return_home mashed A/B for 8
  rescues (Y2 Summer D4). `hold_back_from_pit` (in `sanitize_action`, applied to
  every planner action) holds a farm step whose next 12px lands on either.
- An A pulse beside the parked horse mounts it (`player_state 0x11`, sprite `0x97`);
  the house door refuses a rider and the enter push timed out 3 x 11k frames (Y2
  Spring D19). Riding is harmless elsewhere (walking mounted works). `dismount_horse`
  presses A only in the door stand's zone (x 104..168, y 340..470), with input free and
  the farmer idle, as a pulse. Its first version held A anywhere on the farm: the walk
  froze (Y2 Fall D23, `pixel_stuck` at (259,376), unfed herd, three cows lost) and the 17:00
  shipping box never got its A edge (Y3 Spring D20, 1.6M frames). A filter that rewrites
  every action must never hold a button.
- Summer D1-D3 courting yields to the rings' corn bags (`social_calendar.ring_corn_trip_first`): when ripe crops are due, the
  day's one town trip is the shop (a ring planted by D3 gets six harvests; D4-D5 are the shop's closed weekend). A run that
  fell one watering night behind lost the whole Summer's crop income to a Nina court that ended at 16:01.
- Run strategy (`planner/run_strategy.py`, `HARVEST_STRATEGY` / `--strategy`): `cash` (default) overrides nothing, so
  its runs stay comparable with every earlier row. `score` ships each crop only to 496 (the ROM counter is nine bits and
  wraps past 511; 496 leaves 15 of slack for stray throws) and buys the Summer seed with the most room left; `score_herd` adds a bigger herd and flock (unproven). A goal-
  dependent number belongs in a strategy field read at one decision point, with a `None` meaning "module default".
  The report records `strategy`; `final_score` prints it, the points lost to wraps and the lever for each missing term.
  The score family (`score`, `score_herd`, `perfect`; `cover` is an alias of `score`) also builds the whole field (`planner/field_cover.py`):
  `HARVEST_FIELD_COVER=1|0` forces it for an A/B, `cash` never does.
- Saddle shipping (`tasks/horse.py`, opt in with `HARVEST_SADDLE=1`, off by default after a negative
  A/B): a crop thrown at the adult horse pays only while `$7F1F5C & 0x10` is up (stand N, W or S of
  it, face it, ~5 frames); a throw before that eats the crop for nothing. Centring on the stand tile
  never finishes beside the horse (its body is in the way), so `ON_STAND_FRAMES` on the tile is
  arrival, same for the ride's park tile. The horse wanders about a tile per 400 frames; a leg that
  cannot reach it in 600 steps ships at the bin. Details: `research/F8_shipping.md`.
- The morning `kid1_age`/`kid2_age` reaches 60 (Y2 Winter D24 in the power-on timeline) the house
  holds a birth scene: a doctor stands in the doorway, villagers seal the bed stand, and the house
  clock does not tick, so nothing leaves on its own (e2e `v1n`, 8 sleep rescues). `ExitToFarmTask`
  runs `planner/tasks/labor.py` first: wife (east stand, facing left) -> second menu row -> idle
  ~3,800 frames -> east villager from tile (20,6) -> wife again -> name grid; the ROM then ends the
  day. Never press A at the wife except in that order; she repeats "uugh" otherwise.
- A day floor that fires mid-soak leaves the farmer in the spa water (x >= 624 on map
  `0x10`); a plain walk shuffles on the `0xF7` edge. `return_home` jumps out west with
  Left+A first (`spa_pond_exit_actions`, shared with the post-soak settle).
- The spa bath is closed-loop (`BathCrossing` in `tasks/hot_spring.py`): walk flush to the 0xF7 edge, hold B+dir+A there until `player_action == 3`,
  ride the jump, choose the next side from x. A press before the farmer is flush becomes a tool swing (`player_action` 4/5) and eats the press; the old
  fixed-length script did that after a left jump overshot the west lip (605 instead of 617) and sat at 132/160 for 3,600 frames. Four misses end the soak
  (`bath stalled`); an unrestored soak still crosses out and walks home before it fails. Test: `HotSpringBathLoopTests` (pond model); ROM check: save the
  state at the lip and re-run the soak from several idle offsets (`PR_g2_s2_Y3_Spring_D12`; a replay of that pin day alone does not reproduce it).
- `RecoveryTask` gets 900 frames from a building or the farm and 6,000 from the
  mountain, path, town or a shop (`FAR_RECOVERY_TIMEOUT`): 900 ended one map short.
- `COOP_CHORES` that times out at the feed bin is recovered by an exit to the farm;
  the retry read "not in coop" and left the hens unfed. `splice_coop_reentry` walks
  back in once a day (`splice_barn_reentry` likewise for `COW_CHORES`). The timeout's
  cause: the left-aisle route (8,6) -> (4,5) -> (2,6) waited on a hen parked on the
  via-point (4,5) for 784 steps; `_navigate_to_left_top_goal` now skips an occupied
  via-point (never the goal).

**Clock and days**
- With the day floor off, `_phase_watchdog_frames` is the only thing that ends a looping phase: CROP/HARVEST get 24,000 frames,
  every other phase `PHASE_BACKSTOP_FRAMES` (120,000, or twice its own `timeout` param), and `COVER_FIELD`, `COVER_TILL`,
  `EVE_TALK_LOOP`, `HARVEST_HAPPINESS` and any `ends_day` phase none. Before the backstop a silent loop in `ENSURE_ANIMAL_TOOLS`
  held Y2 Spring D8 for 2M frames (`compose6`). The longest ordinary phase in a finished Y3 run is 31k frames (`COVER_FIELD`).
- `farm_map_loaded` / `stale_under_player` treat 0x72/0x75/0x76 as grass inside `_grass_bounds()`: the pasture, or
  `COVER_FIELD_BOUNDS` (2,3,62,60) under the whole-field cover (`0xFF` counts everywhere). The cover's young grass (389 tiles
  outside the pasture on Y2 Spring D8) made the loaded farm read as unloaded all day, and `EnsureCarryToolTask`'s
  "leave the shed door" walk then restarted forever.
- `CROP_WATER` `stop_hour` (festival mornings: 12) arms tiles from a 13-minute estimate AND ends the task at `stop_hour` +
  `STOP_HOUR_GRACE_HOURS` (1) as SUCCESS. Refill detours at the pond's south stand (32,34) blew the estimate on Y1 Spring D23
  (06:04 to 18:12) and the Flower Festival, seed and dance were lost; the loop appears only on the power-on timeline, never
  on a pin replay of the same day.
- `DayFloor.note_bad()` only marks the day bad; it must not disarm `check()`. It once set the
  `_tripped` flag the check reads, so one failed optional phase left the rest of the day with
  no frame budget and a 17:00 stall hung the planner for 1.6M frames (Y3 Spring D20).
- The 5pm ShippingScene begins at `input_lock` 0 and opens its own box (lock 2) about 180
  frames later. Pulse A only from lock 2: an A during lock 0 set `game_state & 0x0800` for the
  rest of the day, after which the bed ignores every press (8 sleep rescues, Y2 Spring D10).
  That bit is not only the dog pickup: the "stuck dog" release cannot clear it.
- Festival eve: the scene location reads FESTIVAL on town/path/mountain, so `player_state`
  decides cutscene vs walking. Free states: 0, 1 and 1 plus carry `0x02` / tired `0x08` / ride
  `0x10` / bath `0x20`; `0x04` (tool animation) and `0x40`/`0x80` are events
  (`scene._walking_state`). A tired farmer walks about 4x slower.
- Year 2+ New Year (Spring D1) is a ROM-forced bed day: the farmer is on the mountain
  top at 06:03, back in the house at 08:08, walks to the bed itself (text `0x0290`)
  and sleeps. Cow happiness and status are unchanged across it; the "EXIT_TO_FARM
  dialogue dismiss timeout" bad day is cosmetic (Star Night is the same scene).
- The clock stops indoors; a stuck `[RUN]` time means tilemap `0x15` or `0x26`.
- Tilemap `0x0F` is the Winter path and also the overnight fade (about 17
  frames). `scene.is_sleep_transition` tells them apart by the clock byte
  (`time_running`: path 0 or 1, fade 6). Reading `0x0F` alone as a sleep sent
  the farmer to bed at 09:00 on a Winter town trip.
- The hour caps at 18 and only the bed ends a day, so work after 18:00 costs no
  game time and a tighter day floor cuts real watering. `DayFloor`
  (`core/recovery.py`) ends bad days: 48k-frame budget, or 18:00 + 15k frames,
  then return_home, rescue walk, sleep. Fault hook:
  `HARVEST_FAULT_PHASES=crop:fail,harvest:stall,*:stall` (plus
  `HARVEST_FAULT_RATE`/`HARVEST_FAULT_SEED`, `HARVEST_DAY_FRAME_BUDGET`).
- `game_state` bit `0x0800` (dog pick-up animation) left set indoors kills the
  bed and every A for the day. Press A away from the dog; the sleep rescue
  releases it outdoors.
- Start D2 work from power-on or `Y1_Inside_House`; from
  `Y1_D2_Morning_After_D1` the grape return seals at the house fence (rr-oqri).

**Farm development and herd care (ranch score)**
- The 50-point development term needs 2,950 counted tiles: `min(100, counted * 10 // 295) >> 1`, so
  2,949 scores 49. Only fence (`0x05/0x06`) and ids `>= 0x1D` below `0xA0` count; hoed soil (`0x07/0x08`)
  does not, it only takes the seed. That is 99.4% of the 2,968-cell clear-pin field (99.2% of the 2,974 Fall
  mask), so every cell is hoed and then sown. `farm_final_layout.terrain_ledger` reads it (worked share, counted,
  swings left, bags left, unhoed cells inside the bag plan, full 3x3 blocks); the day journal's
  `development_census.ledger` and the `final_score` terrain table carry it. A bag is 9 tiles at best; the raster
  plan needs 364-391 bags on the real field (328 is the floor), 500G each.
- Whole-field cover (`planner/field_cover.py`, phases `COVER_FIELD` + `COVER_TILL`): Year 2-3 Spring and Summer only (a Fall bag writes `0x1D`, wiped on Winter D1). No schedule
  line and no daily quota (owner: hoe the whole thing in one evening, refill in the spa). The morning grass trip buys every bag the gold above `cash_floor_g` pays for
  (>= 40 bags unless it finishes the field, <= 255: **the seed stack is a u8 and the clerk charges for bags past it**), on a light morning (<= 30 ripe crops) or for a
  big load (>= 100 bags), so a handful of trips and next to no delayed picking (log: `[COVER] morning ...`; the weekday shifts one a year, Y2 Summer D17/D18 are Sat/Sun). The `evening_clear` step then sows **every bag in hand** (`GrassPlantTask(stands=...)`, bag_limit; a stack carried
  overnight broke `CROP_ESTABLISH`) and hoes the rest of the field (`till_only` stands), **east side first** (`east_first`), so a field the gold does not stretch over is left bare
  beside the bin; the bin lane stays reserved to the end. The planter reports `stamina_low`; the day plan soaks and reruns with **no refill cap** while the last pass made
  progress; `TaskSequence(keep_last_reason=True)` lets the reason through, and the cover planter has `max_failures=400` (each rerun restarts at the first stand). **The farm dog seals the walk to the spa**: it flees west ahead of the farmer and can be cornered on the one-tile west gate (pixel (22,425)), then barely moves (a tile every few
  thousand frames). `HotSpringStaminaTask` retreats to the yard and waits 3,000f after 18:00 (600 by day), up to 8 tries; a cover pass or empty-bar restart that did nothing is idle, 3 idle ends the evening.
  A blocked evening leaves its bags in hand overnight (open risk for the next `CROP_ESTABLISH`).
  Probe: `run_to_day2 --state <pin> --days N --strategy score` (`HARVEST_FIELD_COVER=0` is the control). A pin's `.lineage.json` holds the absolute path of
  the tree that minted it (`/tmp/e2e_s4/...`), so a replay fails with `FileNotFoundError`: copy the `.state` under a new name and rewrite `pin` and `path` in a new lineage file.
- The seed shelf is season-gated: Summer sells corn and grass but not turnip or potato (ROM probe, same pin and
  task as the Spring control). An out-of-season seed writes `0x1E`, which counts and only toggles with rain
  (`NightlyFarmTilesCheck`); a Spring-bought turnip bag held into Summer would be a 200G filler. Not planted yet.
- Cow medicine (`harvest/tasks/cow_medicine_ops.py`, planner `planner/cow_cure.py`): face a sick adult, use item 9;
  the cow script clears the sick bit and the death timer. The swing's animation outlives a 20-frame poll and the
  bit falls after the dose is gone, so verify waits for `player_action == 0` and never presses with an empty hand.
  A day plans `BUY_MEDICINE` (animal shop, own trip, before 15:00), `ENSURE_MEDICINE`, `COW_CURE`, then the ordinary
  chores refetch the brush. Herd research pins are cut by `harvest.scripts.mint_herd_whatif`; a pin whose cow
  slots were edited needs a barn exit and re-entry first, or stale cow objects block the aisle.

**Shops and buys**
- After the seed-shop buy the town fades in for ~190 frames (`$0025` below 15), and
  `town_coords_settled` waits for it. `BuyShelfItemTask` gave the settle 180 frames, so the power-on
  D2 buy failed ("town coords never settled after shop exit"), the farmer stood in town and
  `D2_FARM_CLEAR` blocked on `stale_farm_map` (e2e `base0`/`base1`, frame 52,254). The cap is now
  `TOWN_SETTLE_FRAMES` (480).
- A buy is proven by the shop tilemap plus a wallet or stock change. A shop
  menu, or a CrossMap return to the origin map, proves nothing.
- Prove a planted bag by its seed field falling (`grass_seeds`).
- The 5pm farm ShippingScene wants A pulses; a held A stalls it.
- General Store: tilemap `0x22`, town door tile (25,53), shut Sat, Sun and from
  17:00. Stand at x=217 facing right (milker y=344, brush y=358); the shed bit
  sets when the "delivered to your shed" box closes. Its `0x22`->`0x04` exit
  passes bogus coords (408,408) before (408,872): settle on y>=850.
- The animal shop sells one animal a day and refuses delivery in rain or snow.
  A bought hen waits in the yard behind it (town `0x04`, about (568,792)),
  reached only through the shop's back door (interior tile (5,3)), which shuts
  at 17:00 and seals a farmer inside (`HenDeliveryTask`: enter by 16:20, leave
  by 16:45, else retry next morning via `hen_pickup`). The coop drop needs one
  tile inside the door mat.
- Leaving the animal shop, the town door holds input for a few frames at
  (600,888). A walk started with no settle reads that as stuck and replans back
  into the shop, where recovery A presses bought 900G of feed in one pin run. Leave
  by `animal_shop_town_to_farm` with its default settle (east first).
- Arriving in town from the path lands at (744,424), already inside a
  (728,424) radius-18 waypoint, and the column below is wall: run west on y=424
  to x=696 before turning south (`_TOWN_EAST_GATE_TO_LOWER_ROAD`). A route with
  `initial_settle_frames: 0` also skips the settle after a path-to-town
  relocalization; both bounced four Fall sale trips back out to the path.
- A wallet of 30,000G or more on a clear Year 2+ 6:00 draws the farm-road
  peddler once (flag68 `0x0400`). Text `$014D`: `$019B` bit 0x80 sets when the
  text ends and the menu goes live (bit 0x10) ~12 frames later; an A in that
  gap buys for 20,000G. `peddler_decline_action` holds the pad, then steers to
  row 1 (Down toggles `$018F`). Pin `LN_y23_whatif_peddler_y2w1`.
- A hen or egg parked on the coop's crossing row or door column `(134,73)` blocks a single fixed exit
  lane: coop exits detour off `x=136` and shift crossing rows on stall. Pins `LN_y23_coop_hen_block`,
  `LN_y23_coop_fade_landing`. The barn exit walks the barn map instead (`BarnWalker`).

**Pins and lanes**
- Pins are `gzip.compress(em.get_state())` under `custom_integrations/`;
  `memory.assign` takes `0x7E0000 + WRAM`. Research pins are
  `LN_<lane>_whatif_*`. Pin results stay out of `docs/STATUS.md`.
- `scripts/pr999_lane.sh <lane> [run -- <cmd> | pytest]` makes a worktree at
  `../rr_harvest_pr999_<lane>`. Its `roms`, `.venv`, pins and untracked tapes
  are symlinks into the main tree, so writing an existing pin name writes
  through. After `setsid nohup`, `$!` is not the python pid.
- Scan existing tapes (`interact_scan`) before recording an interact tape;
  record only a walk no search can close.

**Power berries**
- Cave berry (`PowerBerryTask(source="cave")`): the special stump (10,38) is in a west pocket behind a boulder that
  `$7F1F64` bit 0x0002 removes (first Y1 Winter snowstorm, by D8). Before that the task is `BLOCKED`. The walk to the
  stump stand is `CaveWalkTask(target=(10,40), mid_tol=6, lane_align=2)`: a 1 px tolerance never pops a path tile and a
  corridor on a cliff lip drops the farmer a lane if it walks 6 px off the centre line.
- Statue berry: the berry's life is ~33 frames unless the statue's text is open; take it from (88,110) facing up, never
  from under the statue (A examines the statue). Green on the ROM (`test_statue_berry_is_hammered_picked_and_eaten`).
- **The planner schedules berries** (`planner/power_berry_calendar.py`, `RunStrategy.collect_power_berries`, `HARVEST_POWER_BERRIES`):
  one source a day from RAM flags, ahead of the tool trip but behind an animal buy. Ten sources, nine with receipts (F6, plan.md).
  The ROM sets a berry's permanent bit when it *spawns*, so an unpicked hoe or stump berry is lost: `collect_spawned_berry` runs
  from the day plan step, skips while a `PowerBerryTask` phase owns the berry (the Kappa's cutscene sprite looks stray), and drops a
  failed pickup with a 1,200-frame cooldown instead of failing the plan.
- Egg Festival: the prize is a coin flip (turtle shell 0x1000, or 0x2000 arms the house berry). The house berry lifts from its east
  side facing left, never from above (the table is between); barrel 7 needs the pen gate `0xA6`; festival trips end when the 15:00
  close cannot be met. Idle frames move the ROM's RNG only outdoors (`farm_idle` in the berry tests).
- Test both layers for a berry change: `test_power_berries_rom.py` (bare task) and `test_power_berry_planner_rom.py` (inside
  `DayPlanTask`; the hook bug above only showed there).
- `AuditedEnv` counts only `data.set_value`; a raw `data.memory.assign` poke is invisible to `ram_writes`. Declare it with
  `record_ram_write` and restart the attempt, as `test_power_berries_rom._declare_research_poke` does.
- A what-if poke writes bus addresses (`0x7E0000 + WRAM`); `to_storage(raw=True)` still clamps to the field's range, so wrap
  tests (hearts 1008) must encode bytes themselves (`tests/perfect_ending_lab.py`).

**Hearts (F11)**: every house entry at hour 18 costs a married wife (>= 200 hearts) 2 hearts, measured 376 to 374 against 376 to
376 at 17:05. See [research/F11_social_levers.md](docs/perfect_rancher/research/F11_social_levers.md) (venue table, section 4)
before touching `eve_gift`, `home_sleep`, `relationships`, `social_calendar` or the festival escape.
- Courting (`planner/social_calendar.py`, switch `HARVEST_SOCIAL_CHURCH=1|0`, on for the score family via `RunStrategy.court_girls`):
  clear Sundays run `ChurchCourtTask` (Maria and Ann +4 each per church re-entry, ~2,100 frames, ~6 clock ticks), Fall/Spring
  weekdays `TownCourtTask` for Ellen and Summer/Winter weekdays for Nina (+8 per town reload through the shop door beside her).
  Caps: 199 until the wedding (a girl at 200 while unmarried plays an event whose flag mutes her church question for a
  while), then 500 minus the Flower Festivals, New Years and Egg Festivals still ahead (15/10/10). Every stop
  checks hearts + gain <= cap: a church that stopped only at the cap took both girls 196 -> 200 unmarried (f4all) and
  played their 200-heart events.
- Interior BFS (`room_walk_action`) only takes 0x00/0x01/0xA0/0xC3: the church door spawn (8,28) is 0xFF and Maria's stand is
  on 0xA2, so stands walk to the tile behind and step in (`stand_walk_action`); seated actors' +-12 px boxes can seal a room
  (Star Night bar), where a fixed corridor path is used instead. A seated villager stops the farmer short of a 13 px stand:
  a blocked step within 20 px counts as arrival (the press reaches 16 px ahead +-12).
- Star Night: one A turns the first page, then Down/Up step `$018F`; the morning settle's A+B every frame picks index 0. With Eve
  the wife, `star_night_menu_press` (first thing in `MultiDayPlannerTask._step_inner`) picks index 2 and `StarNightSceneTask` talks
  six villagers then Eve: Eve +30 instead of -30, Ellen gets nothing.
- Harvest Festival (Fall D12): map `0x09` has no edge exit and a frozen clock. `FestivalEscapeTask` talks to the girls,
  moves the mayor out of the host's entrance, accepts the dance and chooses Eve. The berry trip is enabled;
  `HARVEST_HARVEST_FESTIVAL_BERRY=0` restores the old calendar for comparison. Reserve its question gains in social caps.
  Y2 Fall D12 (score family, before 15:00, wallet above `happiness_cash_floor` + 300G, floor >= 60,000G) also runs
  `HarvestHappinessTask`: the dumpling stall sells +3 player happiness for 300G, bought until 999 (345 -> 999 was 218 buys, 65,400G, 155k
  frames, native). A purchase press on the last villager's dialogue reopens it, so the buyer waits for the farmer. Test:
  `test_native_harvest_stall_reaches_999_and_finishes_the_festival` (`-m rom`).

**Eve and the house**: the bar loop, Flower Festival, feather proposal and
carpenter steps live in [docs/EVE_BAR.md](docs/EVE_BAR.md). Read it before
touching `eve_*`, `proposal`, `home` or the Saturday/Sunday phases.

- Walking wedges end fast (2026-11-14): `Navigator.stasis` resets on a tile change, so a wobble across a tile boundary never grows it. `NavTask` and `MultiMapNavTask` therefore watch a 24 px
  `MotionBox` (`core/task_progress.py`): 600 f still = one clean repath, the next 600 f still = FAILURE; `NavTask` also fails on no progress (1,200 f, 600 f within 64 px). A new walking task
  needs no wedge code of its own; a phase timeout above ~3,000 f on a pure walk is a smell. Look for `[NAV_WEDGE]` in run logs and `work_metrics.idle` / the scorecard "Idle frames" section.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/Y1_BENCH.md](docs/Y1_BENCH.md) · [docs/FARM_CLEAR_D2.md](docs/FARM_CLEAR_D2.md) ·
[docs/INTERACT.md](docs/INTERACT.md) · [docs/EVE_BAR.md](docs/EVE_BAR.md).
Skills: `harvest-route`, `harvest-interact`, `harvest-shop`.

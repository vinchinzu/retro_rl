# Harvest agent notes

Package `harvest` (disk `snes/harvest/`, nested import root). Repo rules:
[root AGENTS.md](../../AGENTS.md), [CODING_STANDARDS.md](../../CODING_STANDARDS.md).
Session loop: `.grok/skills/harvest-session/SKILL.md`. Tracker:
`bd ready -l harvest -l spine`. Living residual:
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
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest snes/harvest/tests -q -m "not rom and not ml"  # from repo root
```

Targets: `--until-year N` (1-based), `--until-ending`; `--until-season/--until-day`
stop on that day's morning. Video: `--video <stem>.mp4` writes one-hour parts
that stay playable if the run stops. Glance: `harvest.clock_glance`. Natural
entry is power-on.

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

- **Spring**: crops only. Potato and turnip cannot be sown after Spring, and
  `SPRING_LAST_SOW_DAY` (24) stops them there so the working rings are empty for
  corn on Summer D1 (a Wednesday; the shop is shut Sat and Sun).
- **Summer**: corn is the money crop: 300G a bag, ripe 12 nights after planting,
  then every 3 (tomato: 9 and 3). A ring planted by D3 gets six harvests, one
  planted on D4 five, so the grass errand waits until D3 and the rings' corn bags
  are the day's only town trip (`RING_FIRST_PLANT_DAYS`). Only four rings work:
  (19,28) sits on a 2x2 stump that needs the axe. Corn bought after D18 cannot
  harvest before Fall. Planted grass is the only kind that counts toward the
  animal shop's gate (16 planted tiles per animal after the buy, chicks included).
- **Fall**: last town trips. Cut hay to cover the rest of Fall plus 31 Winter
  days (`hay_cut_target`). The brush and milker must be bought by Fall D29.
- **Winter**: no town trips (the Winter D1 horse event strands the bot), the
  pasture is wiped on D1, animals eat stored hay only. Star Night (Winter D24)
  and Year 2+ New Year (Spring D1) are forced festivals: talk to every villager
  after 18:00 (`FestivalEscapeTask`).

Animal facts the plan rests on:

- Hen first: 1,000G, eggs ship 50G, and an incubated egg is a free hen. One hay
  per animal per day; an unfed night sets status `0x10` and costs about three
  laying days. A `chicken_feed` buy sets the stock to 12, it does not add.
  Only adults eat: an egg placed on day t hatches on night t+2 (the incubator
  frees) and is an adult on morning t+10, so a chick costs no hay. A sale pays a
  flat 500G, adults only (`CHICKEN_SALE_PRICE`); the bench headline counts a hen
  at 1,000G and prints an `at resale` line. Hatched chicks count in the shop's
  16-tiles-per-animal gate, so buy the cow before a hatch fills the pipeline.
- Cow: 5,000G, no milk for 21 child nights. Talk +1 and brush +3 happiness a
  day; milk ships 150G below happiness `0x60`, 250G below `0xC0`, else 350G. Brush 800G and
  milker 1,800G are in the General Store, not the seed shop.
- Carpenter payment waits for the first cow (`house_payment_allowed`).
- Summer crops are walking-bound: the farmer carries one crop to the bin stand
  (11,30), and the east rings are 40+ tiles away.

## Traps

**Navigation**
- Viewport BFS covers about 16 by 14 tiles. Keep hop targets within 7 tiles, or
  call `densify_waypoints`.
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
- Summer regrows fence on y=31. A static waypoint on an unwalkable tile counts
  as reached on its nearest standable tile (`MultiMapNavTask._at_standable_substitute`).
- Clutter regrows monthly. The hoe ignores weeds; lift and toss them. A tool at
  0 stamina does nothing.
- A crop tile whose water pair is skipped twice within `WATER_PARK_DAYS` (3) is
  parked for those days, then retried. Parked tiles are never watered, so a
  permanent park left a whole corn field dry for the rest of Summer after two
  bad days (a sealed bin lane); a stuck field shows as `no_work: no dry crop
  tiles` while the ring tiles sit at an even id one step before ripe.

**Clock and days**
- The clock stops indoors; a stuck `[RUN]` time means tilemap `0x15` or `0x26`.
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

**Shops and buys**
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
- A wallet of 30,000G or more draws the farm-road peddler (texts 333-335), whose
  A pulses spend 20,000G. Keep poked Year-1 wallets below that.

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

**Eve and the house**: the bar loop, Flower Festival, feather proposal and
carpenter steps live in [docs/EVE_BAR.md](docs/EVE_BAR.md). Read it before
touching `eve_*`, `proposal`, `home` or the Saturday/Sunday phases.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/Y1_BENCH.md](docs/Y1_BENCH.md) · [docs/FARM_CLEAR_D2.md](docs/FARM_CLEAR_D2.md) ·
[docs/INTERACT.md](docs/INTERACT.md) · [docs/EVE_BAR.md](docs/EVE_BAR.md).
Skills: `harvest-session`, `harvest-route`, `harvest-interact`, `harvest-shop`.

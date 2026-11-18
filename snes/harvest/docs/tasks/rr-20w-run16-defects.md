# Berries in the planner, Star Night and wife drift (2026-11-03)

Supersedes the statue, fish and "register the source dispatch" items below. Full write-up and evidence:
[plan.md](../plan.md#power-berry-calendar-star-night-and-wife-drift-2026-11-03-handoff).

- Done: the statue eat, the fish return, the hawker hen trade, the Harvest Festival fish and both Egg Festival prizes are green
  on the ROM (bare task and inside `DayPlanTask`); `planner/power_berry_calendar.py` schedules one source a day for the
  score-family strategies; opportunistic hoe and stump berries are picked up by the plan step hook (e2e `s4` lost three).
- Open, in order: (1) a power-on `score` run to Y1 Winter counting berries by season; (2) the snow flower receipt (tenth source);
  (3) Star Night place choice for the wife (F11, unverified option-to-index order); (4) wife gifts in every house level and from
  milk; (5) whole-field development (2,950 tiles); (6) the hen trade rides the existing hen-sale route, which wedges on 6 of 8
  Fall pins (`chicken pickup timeout`, `pixel_stuck` on the return walk) and is the next reliability item for that source.
- Both offline layouts green after adding `@south_pasture_only` to two `test_seasonal_prep.py` tests that assume the south pasture.

# Perfect-score parameter tests, cave berry (2026-11-02, later session)

Owner ask: finish the berry handoff and get closer to the perfect score with a test for every parameter
of the perfect ending, using RAM edits inside the tests. Nothing committed or pushed; everything sits on
top of the pre-existing uncommitted checkout. Verification below was run on this tree.

## Every score term, poked and judged by the ROM (new)

`tests/perfect_ending_lab.py` (method, case table, ROM driver) and `tests/test_perfect_ending.py`.
Base pin `latest_y3_ending_ranch_master_999` (local, gitignored; Y3 Summer D29 17:00, all terms at max).
Point values are written out literally from [MASTER_PLAN](../perfect_rancher/MASTER_PLAN.md) section 2;
the ROM's own calculator (`$83F26D`, ending index 0x20) is the referee and `harvest.core.ranch_master`
is checked against it, not the other way round. Every poke is a RAM write: research only, never Clean.

- **302 offline** checks (default run): the table sums to 999, all 152 cases agree with the oracle, each
  case moves only its own ledger row.
- **185 ROM** tests (172 s): one natural final night is played to index 0x1B and snapshotted; each case
  restores it, pokes, and reads `$7F1F54` at 0x25. All 21 terms, both window edges and the first value
  outside them (495/496/511/512, 1008 wraps to 496, 999 scores 30, shipped 9999 scores 16), the money
  low-word wrap, all 12 cow slots one by one (199 / absent / 0), composites (all low edges, all high
  edges, defects add, everything empty = 0).
- **Natural final night** (D29 17:00 to the first ending scene, ~2.7k frames each; scoring fields are
  final from there, tested). ROM facts the plan can rely on:
  - Each of the five wives (one marriage bit set) loses exactly **2 hearts** on the night and nobody else
    moves: entry window **498-511** (497 lands on 495 and costs a point). Eve works like the others.
  - The shipping pile is credited before the money term (9970 + 14 -> 9984 -> 78; + 13 -> 77).
  - Farm development: **2,950** counted tiles -> byte 100 -> 50 pts; 2,949 -> 99 -> 49; 2,890 -> 97 -> 48
    (`CalculateFarmDevelopment` `$82AA0C`, then `tiles * 10 / 295`, cap 100).
  - Twelve fed cows at exactly 200 keep 200 (300 pts); at 199 they score 288.
  - A full evening vector (windows, both house bits, two children, clock, turtle, Eve at 500, 12 cows at
    200, 12 hens, grass farm) scores **999** after the night.
  - The planner's own D30 forecast (`morning_score_ledger`) equals the ROM verdict in every night case.
  - Over-max terms (13 cows, 11 berries) are absorbed by the 999 cap; only the oracle ledger shows them.
- Mutation check: sabotaging `WIFE_DRIFT_DELTA` turns the wife night tests red.

Not claimed: this pins what the ROM rewards, not that the bot can reach the vector. The ten-berry
best-ending equality (`$83F097`) is not a score term and is not covered. Run:

```bash
QT_QPA_PLATFORM=offscreen uv run python -m pytest snes/harvest/tests/test_perfect_ending.py -m rom -q
```

## Cave berry: native recipe is green, and the gate is found

`PowerBerryTask(source="cave")` now runs end to end from `PR_spring_final_Y1_Spring_D3`: exit house, fetch
axe, farm to mountain landing, tile-BFS walk to the stump stand (10,40), six chops (353 frames, 12 stamina),
walk up into the hole (map 0x29 after 17 frames, landing beside the berry), pick and eat. **3,530 frames,
count 0 -> 1, max stamina +10, `$7F1F6A & 0x40` set, zero writes and loads inside the recipe.**

- **The pocket is gated.** The special stump (10,38), the goddess pond and the cave drop sit in a west pocket
  sealed by a 2x2 boulder at tile (23,39). Nothing opens it (74 hammer swings, every lip height tried).
  `ReplaceTilesMoutain` (`$81AC7F`) removes it once **`$7F1F64` bit 0x0002** is set, which
  `SetWeatherFlags` does when weather 4 (a Winter snowstorm) applies. `WeatherTomorrow` forces weather 4
  when the day reads 7 in Year 1 Winter, and the flag lands the following night: **open by the Year 1
  Winter D8 morning at the latest, earlier on a random 1-in-8 snowstorm** (the Winter D5 pin already had it).
  ROM tests: forecast 4 + a night sets the flag; a night into day 7 makes forecast 4 with the flag still
  clear. `PowerBerryTask` refuses the cave (`BLOCKED`, `can_start` False) until the flag is set.
- The cave test sets the flag with one declared RAM write (a stand-in for Winter). A natural two-night
  proof from the Winter D1 pin was not completed (the second sleep stalls outside the day planner).
- Old claims retired: the west door `(10,25)` only reaches the cave's upper rooms (the berry chamber is
  sealed from inside; the `0xD7` tile at (5,24) is a sign), and the old waypoint chain `(248,688)` ran into
  the x=17-18 wall. The way round is east along row 43, up the x=32 corridor, west along rows 39-41.
- Code: `CaveWalkTask` gained `mid_tol` and `lane_align` (off by default, so cave and statue walks are
  unchanged); `PowerBerryPickupTask` no longer treats a walker stopping a few pixels short as fatal and
  stops a walk once the berry is in reach.
- Research pins (local): `LN_score_mtn_landing` (mountain, axe), `LN_score_mtn_landing_open` (boulder flag
  set), `LN_score_cave_dropped` (in the chamber).

## Statue berry: pick works, eat does not (strict xfail)

The statue test was red when this session found it (the earlier "green" predates later edits). Findings:
the berry has a ~33-frame life unless the statue's own text (0x37C) is open; A from directly under the
statue examines it and the berry is lost; **from (88,110) facing up A lifts it** (held item 8 in 14 frames,
now `STATUE_PICKUP_STAND`). The eat then does not start: while carrying, A runs the put-down animation
(`player_action` 5, 9) and the berry is never eaten (1,700+ frames). In the cave the same A pulses eat it
(`player_action` 8, animation 0x3B) after a ~120-frame carry. `test_statue_berry_is_hammered_picked_and_eaten`
is `xfail(strict=True)` with this diagnosis; it fails loudly if someone fixes it. Not solved: why the eat
starts in the cave but not here (compare `$0915`/`$0901`/`game_state` after the pick, `bank_81.asm` `$81CA46`).

## Fish-return berry: tape evidence only

`tasks/mountain_fish_power_berry.json` (4,159 frames, brush in hand) on the mountain-spring map:
09:01 Left+A at tile (39,23) lifts item 7 (f2652); the farmer carries it to tile (42,25) and releases it
(f3062); at the same stand an A at f3668 lifts **item 8**, eaten at f3766. `interact_scan` only reports the
first held change. The tape's start pin (`latest_backup_mountain_fish_power_berry_*`) is absent, so there
is no live receipt and nothing is implemented. Do not build duplication routes on it yet.

## Audit gap

`AuditedEnv` counts only `data.set_value`. A raw `data.memory.assign` (the repo's research recipe) is
invisible to `ram_writes`, so a test that pokes that way must call `record_ram_write` itself
(`_declare_research_poke` does). Shared harness; not changed here.

## Verification this session

Harvest offline: **2,493 passed, 32 skipped** (legacy) and **2,491 passed, 34 skipped** (`blocks`).
Berry ROM file: 8 passed, 1 xfailed. Perfect-ending ROM: 185 passed. Docs test: see the final report.

## Next actions

1. Statue eat (above), then the explicit source dispatch and `PhaseKind` builders for field_1, field_2,
   statue, stump and cave (still not registered; automatic pickup is wired).
2. Fish-return berry: find or re-record a start pin, replay the tape, decode the release script.
3. Natural-night proof of the boulder flag from the Winter D1 pin through the Composer, not a bare
   `GoToSleepTask`; add "snowstorm seen" to the calendar so the cave phase schedules itself after it.
4. The frozen-tree power-on benchmark and the remaining economy work below are unchanged.

---

# Score optimization handoff — 2026-11-02 (owner requested stop)

This section supersedes the older priorities below. The owner now wants the
perfect ranch score: prioritize sprinkler, more grass and earlier cows, keep
chickens economical, allow different plot shapes and whole-field tilling,
work after 18:00 without day frame caps, and acquire/eat every power berry
with an individual test. All session runners are stopped. All work is uncommitted. Preserve the substantial
pre-existing changes throughout this checkout. No commit or push requested.

## Changes in this session

- Added `harvest/tasks/sprinkler.py`: selects reachable stands covering at
  least four dry crops, verifies every covered tile, uses eight stamina and
  no can water, falls back to the existing can skill for sparse leftovers.
  Reports `stamina_low watered=N unfunded=N` for the existing spa retry.
- Sprinkler purchase/delivery/pickup implemented through `BuyShelfItemTask`:
  tool `0x15`, 2,000G, shelf stand **(216,342) facing up**. Ownership is
  **row 3, `$7F1F02 & 0x10`**, unlike brush/milker row 2. `ToolShelf` now has
  `shed_addr`. Shed pickup is (96,168), the same stand as the can.
  `Tool.SPRINKLER`, the shared price table, planner shop gating and crop
  builder are wired. The normal water phase still fetches the can before
  the sprinkler fetch; removing that redundant fetch is future optimization.
- Year 2 blocks select `seven_east_access` / `seven_west_access`: bare centre
  and side notch, seven crops. Year 1 keeps its eight-crop shape. Geometry
  accepts `plant_center=False`; planting acceptance uses actual layout size;
  `farm_pond` counts seven cells for Year 2 target decisions. Extra Year 2
  blocks remain opt-in. New shape has component receipts, no season A/B.
- `DayFloor` defaults now disable the day work budget (0) and evening grace
  (-1); explicit environment overrides and phase stall/rescue checks remain.
- Late Summer (D19+, with cows) pasture target now budgets whole nine-day
  cuts at 80% yield against the anticipated herd through Winter and the
  14-day Spring bridge, capped at the existing 180 tiles. This is a policy
  model, **not a measured third-cow improvement**. Two hens remain the
  ordinary route cap; Perfect's separate 12-hen override is unchanged.
- Added `harvest/tasks/power_berry.py`: actual berry sprite `$0116`, pickup
  and eat receipt (`$7F1F36 +1`, max stamina +10), hard stop at exactly ten.
  Crop tool use, grass work and the day Composer pause for nearby berries.
  Grass's old transient `$0976` count handling was replaced. No RAM writes.
  Explicit field_1 / field_2 / stump / cave source tasks have ROM greens; the
  statue source picks but does not eat (strict xfail), see the top section.
  These source tasks are not yet registered as milestone recipes /
  `PhaseKind` builders; automatic pickup is wired, explicit source dispatch
  remains to be wired.

## Verification and artifacts

Everything is under `recordings/score_optimization/` (gitignored):

- `sprinkler.log`: purchase, return, shed pickup and watering **72 -> 0 dry
  tiles** passed on `PR_e2es4_Y2_Spring_D4`, zero writes / mid-run loads.
  Test: `test_care_shopping_rom.py -m rom -k sprinkler`.
- `notch.log`: both seven-crop orientations planted and watered successfully
  from `PR_spring_final_Y1_Spring_D3`; zero writes / mid-run loads.
- `berries.log`: field_1 and field_2 passed. The second test actually sleeps
  after acquiring the first, then digs/picks/eats the second on the next day.
- `statue.log`: an earlier statue pass, superseded: the test was red when the
  later session found it (top section). Approach `(4,6)` with radius 6, hammer
  up; the ground below the statue stops the farmer at about y=110.
- `mountain_berries.log`: stump source passed; its cave line is superseded
  (green, top section). Stump RNG is 1/16 per completed six-chop tree. The
  test has not been swept over multiple RNGs.
- `offline.log`: **2189 passed, 32 skipped, 94 deselected**, before the later
  hay-reserve and mountain/cave edits. `econ.log`: **16 passed** after the
  hay-reserve change. Neither is a final full-suite verification of the WIP.
- `cave_berry.log` (`multi_nav timeout` at frame 19,497) is superseded by the
  green cave recipe in the top section.
- A fresh power-on Year 1 run was interrupted at the owner's stop request.
  `year1.log` and `year1.json` contain the partial run. Shutdown wrote the
  report successfully: `reason="interrupted by signal 2"`, 3,508,520 frames,
  Year 1 Winter D8, **54,440G, three cows, two hens, 97 planted-grass tiles**.
  Maximum stamina was 130 in the live log.
  **No completed Year 1 scorecard or accepted new benchmark.** Modules were
  edited while it ran; repeat from a frozen copy for publishable comparisons.

Useful commands from repo root (all through `uv run`):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests/test_power_berries_rom.py -m rom -q
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests/test_care_shopping_rom.py snes/harvest/tests/test_crop_notch_rom.py -m rom -q
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests -m 'not rom and not ml' -q
HARVEST_RING_LAYOUT=blocks QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests -m 'not rom and not ml' -q
```

## Berry audit correction — do not trust the old designed-seven claim

`perfect_rancher/research/F6_collectibles.md` has a real script-bank mistake.
`event_dis group 23 --event 0` decodes `09 02 94 8e` at `$B48E44` as a child
at **$B48E94**, not the berry loop at **$B38E94**. The choice at `$B48E72`
writes tool `0x14` (gold axe) and +20 happiness. Do not register that
"goddess axe berry" as an acquisition recipe or use seven designed berries
as an established upper bound. Ten remains the ending's exact count check.

An existing tape supplies contrary evidence the audit missed:
`tasks/mountain_fish_power_berry.json` has held item **7 at f2652**, thrown
at **f3062**, then held item **8 at f3668**, cleared at **f3766**. It is a
fish-return berry. Its stored start pin
`latest_backup_mountain_fish_power_berry_20260427_150910` is absent locally,
so the tape trace is evidence to investigate, not a new live ROM receipt.
Scan it with `harvest.scripts.interact_scan` before implementing interaction.
Do not build duplication routes based on the faulty seven-source premise.

## Cave notes kept from the earlier session

Native cave stride is **32** (`$0181=2`); path stride is **16** (`$0181=1`), unlike the farm's 64; the local
floor walker accounts for that, do not change the shared tile reader. The special stump is **TL (10,38), raw
tile 0x11** with 0x12/0x13/0x14 companions (ordinary stumps are 0x09..0x0C; do not chop (37,5)). The berry is
at **(224,384)** in map `0x29`. Diagnostic pins `LN_score_field_spawn`, `LN_score_path`,
`LN_score_stump_entry` and `LN_score_cave_entry` (the upper cave rooms, not the chamber) are research only.
Throwaway probes are in `recordings/score_optimization/`; do not promote them to CLIs or tapes.

Re-run affected ROM tests and both offline layouts, then a frozen-tree power-on benchmark including the
hay-reserve policy. Keep the residual explicit: not all berries, 999, or the new economy are proved. No STATUS
promotion, no new recording, no commit or push.

---

# Clean Year 1 and headed Year 2 handoff — 2026-10-31

User requested handoff. All session runners have exited; nothing is watching or
playing. Work is uncommitted. Preserve the extensive pre-existing Harvest,
ALttP, SM and shared-harness changes. Do not commit or push unless requested.

## Owner intent and corrections

1. Earn a **fully completed Year 1 with a clean farm from power-on**, then use
   that actual save for the headed Year 2 run. The older `PR_Y2_Spring_D1`
   checkpoint has 2,030G, no animals and an uncleared farm; the owner rejected
   using it as the campaign handoff. Do not substitute an old or patched state.
2. Improve New Year day policy. This is implemented and ROM verified below.
3. Short tests should replay nearby pins, not repeat Day 1. The final campaign
   validation still starts from power-on after the local fixes pass.
4. The desktop is Arch / Omarchy / native Wayland. Headed playback worked.
5. The owner explicitly removed issue-board workflow from this task. Its
   commands and pointers were removed from root/Harvest agent notes and the
   plan header. Follow the local plan and this residual directly.
6. Preserve the earlier reliability work: two hens, better cow care, the four
   review fixes documented in `../Y1_BENCH.md`. Planting layout and Perfect 999
   are not the current goal. Do not edit STATUS from a pin.

## Day 2 perfect pass — current work

The owner now requires Day 2 to finish with every bush, post, stone, boulder
and stump removed, eight planted/watered potatoes, and planted grass. Use the
[acceptance checklist](../FARM_CLEAR_D2.md). Cash flow starts with the potatoes;
the free shed grass bag is sown without buying another bag.

The prior `recordings/y1_complete/report.json` remains a red Year-1 attempt:
D2 stalled in the shed at 278,222 frames. There is still no completed, accepted
Year-1 → Year-2 handoff. Preserve the old failure pins and report.

Confirmed causes and fixes:

- Shed departure aimed at `(25,28)`, on the roof behind the facade. The NavTask
  escape went west into the facade and could re-enter the shed. Leave south
  from `(26,30)`, then finish on open yard `(25,32)`.
- `ShedFetchItemTask` converted failed exits and failed step-offs to SUCCESS.
  Both now propagate failure. A transition already reaching its target before
  its first active tick is allowed to settle rather than fail its origin check.
- Bounded rock/stump clears kept running indoors. They now fail immediately;
  the pocket clearer retains its explicit off-farm approach route.
- After spa, stale-map recovery also used the fixed westward shed escape at
  the west farm gate, walking back onto the path. Both clearers now choose
  a yardward action from the live position.
- The corrected shed timing exposed a barn care route outside the right aisle
  at `(228,310)`. `recorded_interact_lane_action` now rejoins x=204 before
  climbing to a left-facing cow. The natural `PR_Y3_Summer_D1` Composer
  regression again feeds, talks, brushes, milks, and exits successfully.
- `d2_work` now requires nine `planted_grass` tiles in addition to eight wet
  potatoes, every debris count zero, and the pre-17:00 shipping receipt.

Short reproductions (no Day 1 replay):

- `test_d2_tools.py -m rom -k d2_hammer`: **3 passed**. Both saved shed
  positions leave a loaded, faded-in farm and stay loaded after neutral frames;
  the following real Composer rock phase reduces the boulder count.
- Native suffix from `y1_complete/progress_f255000`: **124,594 frames / 138.2 s**,
  SUCCESS; every debris count zero, grass **0 → 9**, grass seeds **1 → 0**,
  stamina 81, wallet unchanged at 150G. Zero RAM writes or mid-run loads.
  Receipts: `recordings/d2_perfect/{suffix.json,suffix.log}`;
  leave pin `LN_d2_perfect_suffix`. This pin proof does not certify the economy.
- Harvest offline suite: **1,946 passed**, 30 skipped, 39 deselected,
  63 subtests. `recordings/d2_perfect/non_rom.log`.

Fresh power-on acceptance **PASS**: `recordings/d2_perfect/power_on.json`,
**374,794 frames / 702.9 s**, terminal reason `d2 farm clear complete`.

- Weeds **506 → 0**, posts **80 → 0**, stones **185 → 0**, boulders **51 → 0**,
  stumps **38 → 0**; no small or damaged boulders remain.
- **8 planted / 8 watered potatoes**; grass **0 → 9**, grass seeds **1 → 0**.
- Shipping before 17:00 verified; two settled farm observations, stable input,
  empty hands. Ends Spring D2 18:14, wallet 110G, wood 228, stamina 33.
- Continuous **Clean**: initial loads 0, mid-run loads 0, RAM writes 0, assists
  empty; no day failures or day-floor trips.
- Accepted Day-2 checkpoint: `LN_d2_perfect_poweron`. This is not a completed
  Year-1 save or an accepted Year-2 handoff.
- Final selected ROM suite: **10 passed**, 185 deselected (shed, native boulder
  handoff, feather pickup, building exits, visitor entry, and daily cow care).
  `recordings/d2_perfect/rom.log`. Docs: **8 passed**.

All session runners have stopped. No commit or push. Next action: verify
D3–D8 watering and the first 640G potato harvest, then benchmark a complete
clean-farm Year 1 before the headed Year-2 run. Highest-score cash flow remains
an economy benchmark, not a claim from the D2 establishment receipt.


## Previous handoff implementation

- `run_to_day2.py`: shared `--headed` viewer, `--speed` (default 4), HUD,
  Wayland setup, graceful ESC/window-close report, pygame cleanup, live
  sidecar enabled for headed runs.
- `campaign_composer.py` and runner: `--save-day-pins`, `--pin-prefix PR_*`,
  settled morning lineage including the successful terminal morning. Prefixes
  preserve older campaign pins. Replay one day with `--state <PR_pin> --days 1`.
- `task_progress.py`: checkpoint keys include the child phase, so a campaign
  retains nearby phase checkpoints as well as mornings. The live outer
  `stall` counter is still coarse during repeated clear chunks; the D2 goal
  watchdog independently reads real farm progress.
- `multi_day_planner.py`: the required D2 clear owns its completion rather
  than being abandoned at the 48k / 18:00 day floor. Required clear failures
  now stop visibly instead of silently sleeping. Forced festival time is
  excluded from the ordinary work budget.
- `d2_work.py`: sweep weeds in all quadrants before hauling posts/stones.
  Hauling while other quadrants were still weed-blocked stranded carries.
- `farm_clear_quota.py`: quota SUCCESS now goes through `_finish_or_drop`;
  it cannot leave a held weed behind. The saved zero-weed carry failure
  clears its hands and succeeds in 300 native frames with no writes/loads.
- `home_recover.py:PondTossTask`: north-yard carries invoke the existing
  `CarryToPondStand` corridor immediately instead of first walking a single
  distant south-pond waypoint into the barns. Native north carry replay:
  602 frames, held 0x0F -> 0, no writes/loads.
- New Year policy and geometry, detailed next. Root/Harvest agent context
  was also cleaned of the unwanted tracker commands.

No new dispatcher, no RAM/calendar/resource editing, no commit, no push.
Speculative changes to `carry_toss.py` were reverted; that file has no session
change. Use the existing carry skill rather than reintroducing those probes.

## New Year — verified

`FestivalEscapeTask` now follows group 0x28 event 1's native `$7F1F7A`
conversation acknowledgements. It prays, talks to outstanding villagers, then
addresses the exit woman. **There is no 18:00 gate for New Year.** Star Night
keeps its own evening behavior.

Walking uses `harvest/maps/new_year_walk.json`, an exact 8-pixel controller-
probed grid, and precise talk stands. Approximate hop endpoints accumulated
position drift; `probe_walk_grid --exact` makes the grid reproducible. Queued
turn/release/A avoids walking away from the verified talk stand between pulses.
Progress snapshots expose talked slots/target/side. New Year exhausts its
bounded route with BLOCKED rather than random wandering.

Receipts in `recordings/y1_complete/new_year_policy.json`:

- Identical entry bytes (`LN_newyear_entry`), seed 1: **35,580 -> 3,358 frames**,
  delta -32,222 (90.6% fewer); finishes at 08:08.
- Full natural predecessor Composer regression: **3,396 festival frames**,
  returns during the 08:00 hour, no day floor, zero writes/mid-run loads.
- Regenerating the exact grid produced the same 315 cells.

Rebuild from the game directory:

```bash
HEADLESS=1 QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.probe_walk_grid \
  --state LN_newyear_entry --setup wait:240 --tilemap 0x3A --exact \
  --out harvest/maps/new_year_walk.json
```

The probing uses state restores for geometry research. Live campaign navigation
uses the resulting geometry and controller inputs, with no state restores.

## Earlier validation and artifacts

- Harvest non-ROM: **1,941 passed**, 30 skipped, 36 deselected, 59 subtests.
  `recordings/y1_complete/non_rom_checks.log`.
- Selected live-ROM calendar/home regressions: **6 passed** (47 deselected).
  `recordings/y1_complete/rom_checks.log`.
- Docs: **8 passed**. `git diff --check -- AGENTS.md snes/harvest` passed.
- Natural cleanup suffix: all 80 posts removed after the weed sweep, zero
  writes/loads. `recordings/y1_complete/fences_open_farm.log`. That scoped
  DayPlan then auto-returned/slept; its final interior zero counts do not prove
  boulder/stump clearance.
- Partial Year 1: `recordings/y1_complete/{report.json,run.log,progress.json,
  source_hashes.json,launch.json}`. This is a red report, not a completed year.
- Rejected headed run and local New Year traces: `recordings/year2_watch/`.
  The old headed run was closed normally; it did not crash.

Useful commands (repo root unless stated):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests -q -m 'not rom and not ml'
QT_QPA_PLATFORM=offscreen HEADLESS=1 uv run pytest \
  snes/harvest/tests/test_calendar_festival_escape.py \
  snes/harvest/tests/test_day_plan_home_return.py -q -m rom
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_docs.py -q
```

After the shed/rock exit is proven, start the next fresh campaign from
`snes/harvest` with the existing runner. Preserve the failure pin/report first:

```bash
HEADLESS=1 QT_QPA_PLATFORM=offscreen uv run python -u -m harvest.scripts.run_to_day2 \
  --power-on --until-year 2 --save-day-pins --pin-prefix PR_complete_ \
  --progress-every 10000 --progress-sidecar-every 1000 \
  --progress-sidecar recordings/y1_complete/progress.json \
  --checkpoint-dir custom_integrations/HarvestMoon-Snes/y1_complete \
  --save-end-state LN_complete_year1 --out recordings/y1_complete/report.json \
  > recordings/y1_complete/run.log 2>&1
```

Only after that year completes and actual farm clearance is verified, use
`PR_complete_Y2_Spring_D1` for the headed Year 2 run:

```bash
QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 \
  --headed --speed 4 --state PR_complete_Y2_Spring_D1 --until-year 3 \
  --save-day-pins --pin-prefix PR_watch_ --out recordings/year2_watch/report.json
```

`[` / `]` adjust speed; 8+ is turbo. Native Wayland succeeded on this desktop.
Hyprland uses Lua dispatchers here, not the old string dispatcher syntax. If
needed, float/resize the specific watcher with `hl.dsp.window.float(...)` and
`hl.dsp.window.resize(...)` through `hyprctl dispatch`, after reading the local
Omarchy skill. No desktop config was changed.

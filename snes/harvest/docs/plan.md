# Plan: Harvest Moon (SNES)

**Next session: read [tasks/NEXT_SESSION_TODO.md](tasks/NEXT_SESSION_TODO.md) first** (runs in flight, how to record a 999, the overplant/stagger design).

Future work. Verified results: [STATUS.md](STATUS.md). Campaign detail:
[MASTER_PLAN.md](perfect_rancher/MASTER_PLAN.md). Current priorities are below.

## 2026-11-17 evening: why r41..r47 missed 999, and the Year-3 switches (nothing committed)

`rdet` + `r41..r47` (lean, `score_herd`, no extras) scored 986-998 with Y1/Y2 identical in all eight (Summer corn+tomato 195, then 417), so every miss was Year 3: crops short in all
eight (Y3 Summer owes corn 249 + tomato 131 = 380, delivered 232-366), development in four (1-2 points), Eve hearts in one. Causes, all measured in `report.json`/`run.log`:

- **One crop per shop trip.** Summer D1 is a Friday (Sat/Sun shut): 4-6 corn bags on D1, tomatoes on D5-D6, so half the sites stood bare a week. 3 of 12 sites also hold capped, ripe
  Spring potato/turnip (`pocket_plant_targets` never replants a held ring), leaving 9-11 free rings on D1. Fix: `BuyShelfItemTask.then` buys the second crop on the same visit (the shelf is one
  row, 16 px apart; ROM-verified `bought corn_seeds 0->8 ... then tomato_seeds 0->3`), `run_strategy.joint_summer_bags` splits the free sites by remaining need / harvests a ring sown today
  still delivers (`HARVEST_Y3_JOINT_BUY`, default on, Year 3 Summer only).
- **A lost watering night delays every plant a day (~25 crops).** Cycles run 3-4 days, 5 harvests a Summer. 17 of 592 Y3 Summer `ENSURE_WATERING_CAN` calls failed, 16 right after the evening
  `CUT_FODDER`. Only the wedged walks are transient (3 of 42 retries worked), so `splice_water_retry` (`HARVEST_WATER_RETRY`, default on, Year 3) retries only "route failed" /
  "exit_to_farm failed". Open: "enter failed" at the shed door (door stand (424,486), `held=0x00`, `lock` 1, no sprite near) and "tool 0x15 missing after shelf" (pair [sickle, milker], sprinkler
  bit set in shed row 2, A ignored three times) repeat exactly; repros `PR_ab_D_r44_s1_Y3_Summer_D24` and `PR_ab_A_r47_s1_Y3_Summer_D3` with `run_to_day2 --state ... --days 1`.
- **Development sat flat D9-D21** ("no stands": full stands done, every crop site's 5x5 reserved to D20), then 405 tiles in nine mornings, and harvest days leave no clock for the shop
  ("shop start deadline 15:00", a wedged `farm_to_path`). Fix (`HARVEST_EARLY_COVER`, default on, Year 3): ragged stands the day no full stand is left, a sown ring's 5x5 band released
  (`field_cover.released_cells`; the 3x3 block stays), 60-bag Summer loads from D4 (`planter.release_cells`). Counted tiles at D29 went from 2,768-2,956 to 2,941-2,963.
- **Eve hearts**: `GIVE_WIFE` "gift walk stuck" for nine mornings (an NPC 13 px east of the farmer blocked the walker's last 2 px sidestep; Eve 492). `eve_gift.py`: vertical first
  after 40 still frames, 6 px of x slack. ROM: `PR_e2er47_20261117_Y3_Summer_D21` replays `eve 504->508`.
- **House exit after a gift** (found by the previews): the blind exit walk only presses down, so a child on the farmer's column (327,156 / NPC 328,169) timed it out; the barn feed
  check failed too, 12 cows unfed, cow happiness -25, ROM 949 (D r45) and 10 such days in ~100 runs, all right after `GIVE_WIFE`. `ExitBuildingTask._sidestep_if_blocked`: after 30 still
  frames step left then right for 24 frames (Year 3 only). ROM: `PR_ab_D_r45_s1_Y3_Summer_D18` exits.

Paired previews, ROM score minus the pin-only "Player happiness -1" (planner state is not restored from a pin), 8 pins each, JITTER=0 unless noted:

| variant | mean | at 999 | notes |
| --- | ---: | ---: | --- |
| base (the batch's tree) | 993.1 | 0 of 12 | crops short in all, development in 8 |
| joint buy | 994.8 | 1 of 8 | |
| early cover | 995.8 | 2 of 8 | |
| joint + early + water retry (A) | 996.0 | 3 of 8 | |
| A + 60-bag load + Eve unstick (D) | 989.8 | 1 of 8 | one house-exit disaster (949); 995.5 without it |
| D + house-exit sidestep (H) | 995.6 | 1 of 8 | storms and can failures remain |
| D, jitter 90, seeds 1-2 (E, n=16) | 995.3 | 1 of 16 | crops short in 15 of 16 |
| H + 4 extra Summer sites (F) | 996.4 | 4 of 8 | crops fixed, development -1..-3 in 3 |
| H + 8 extra Summer sites (G) | 997.1 | 0 of 7 | crops fixed, development short in 7 of 7 |

Reading: the three switches are worth about +3 points and remove the systematic misses, but 999 is still a 1-in-4 event on a pin. What is left is a handful of independent day-level
failures, each ~1 lost day: the can pickup (above), `COOP_CHORES`/`EXIT_COOP` timeouts (32 bad days in ~100 runs: `coop_nav_feed_bin timeout`, "directional transition timeout" inside the coop),
storms that wipe 130-200 counted tiles (D10-D11, D26-D27; the repair needs shop days that harvest days and weekends do not leave), and the 9-11 free rings. Extra Summer sites trade
crops for development one for one (F and G above); they stay off. Certify with `rdet` + 7 seeds, never from pins (see `tasks/NEXT_SESSION_TODO.md`).

## HANDOFF — 2026-11-14 (read first): why Year-3 scores vary 994-999, and how to make 999 repeatable

Owner's brief: the score should not depend on luck. Any term that can be *over*-done safely (crops: the cap leaves extras on the plant;
development: more cover is never wrong; hearts: aim mid-window) must be over-done, so the only variance left is the unavoidable final timing.
Better code, not more seeds, is the goal. Nothing is committed or pushed.

### Result so far (2026-11-14 ~17:00): fresh power-on, all Clean
`proof1_20261114` (deterministic timeline, the tree frozen at launch, before the cover `soft_timeout` and jitter-year edits; `git_head.txt` d669a6a1 + the dirty tree in
`git_dirty.txt`): **999/999**, 0 initial/mid-run loads, 0 RAM writes, no assists, 11 recovered day failures in the journal.
**The recording**: `recordings/e2e/rec1_20261114/` (same code, VIDEO=1, 768x672 @60, h264) repeated it exactly: **999/999 Clean**, same wallet (337,280G), 0 loads, 0 writes.
52 one-hour parts `video_part001..052.mp4` (51.9 h, 27 GB, every part ffprobe-clean, ending at the credits), power-on to the ending; scorecard `FINAL.md`, report `report.json`.

Eight seeded Year-3 power-on runs (`jit21..28`, current tree, all CLEAN): **999 x3** (`jit23`, `jit25`, `jit28`), **998 x2** (`jit22`, `jit27`), **997 x3** (`jit21`, `jit24`, `jit26`).
Every miss is a single term, and the repeated one is development:
| run | ROM | misses | log signs |
| --- | ---: | --- | --- |
| `jit21` | 997 | tomato 492, development 99 | final census 2,928 of 2,950 |
| `jit22` | 998 | development 99 | final census 2,946 |
| `jit24` | 997 | turnip 490, development 98 | 2 `COVER_FIELD` failures, 10 shed `enter failed`; census 2,905 |
| `jit26` | 997 | tomato 488, development 99 | 22 shed `enter failed`; census 2,926 |
| `jit27` | 998 | development 99 | final census 2,922 |
| `jit23`/`25`/`28` | 999 | none | final census 2,954-2,958 |
Hearts were 155 in all eight (Nina 496 in every one). So **development is the main variance now** (5 of 8 runs lose it), then Summer tomatoes (2), turnip (1).

Development tail: the last census sits 4-45 tiles under 2,950 in the five misses and over it by 4-8 in the three perfect runs. The last mornings only close ~3-8 tiles a day
(`[COVER] morning ... bags=15..6 behind=146..126`): ragged one- and two-cell stands, so a bag buys well under a tile. The field has 2,968 cells and 2,950 are needed, so there is
almost no slack by construction (18 cells), and `uncovered` still reads 115-170 on the last morning because those cells cannot count (reserved lane, stumps, boulders, rings kept
for crops). The fix is more early margin, not luck: start the dense 3x3 sowing earlier, release a capped crop's footprint when it hits its cap instead of at day 289, and make
the last week's evening load larger than the ragged-stand rate (the evening hoes ~110 tiles whatever the bag count).

### Is the 52 h busy or stalled? (2026-11-14 evening, `rec1_20261114`)
Owner's question: could the run have sat idle for hours? Answer from the report and a frame scan of all 52 video parts (256x224, a frame counts as frozen when at most 4 pixels change):
- **No frozen screen anywhere.** The longest frozen picture is 4.5 s (the credits hold 9.7 s); the longest black fade is 7.3 s (sleep). 0 runs over 10 s in 11.2M frames.
- Only 15.4 h of the 51.9 h had the game clock running (900 frames an hour outdoors). The other ~36 h is barn, house, shop and menu time, where the clock is frozen; Y3 Spring and
  Summer average 19 min a day (60 days = 19 h), mostly 12-cow chores (about 20,000 frames a day) and cover work.
- **Real waste, not idleness:** (1) 16 `NAV_CROP` timeouts of exactly 9,000 frames (144,000 f = 0.67 h). The eight Y1 ones (Spring D12-13, Summer D4-10) started at 06:00 and ran the
  clock to ~15:14: a lost field day each. The eight Y2/Y3 ones (Y2 Spring D16/22/23, Y3 Spring D18/21, Summer D4/5/8) started at 17:00-18:00 after the day's work, so they cost wall time
  (and about an hour of evening on Y3 Summer D5/D8). Cause, reproduced from pin `PR_e2erec1_20261114_Y3_Spring_D18`: after CUT_FODDER in the north-east pasture the farmer stands at about
  (770,216); `ENSURE_CROP_SEEDS` (`multi_nav` route to the shed, 4,000 f) and then `NAV_CROP` (9,000 f) both sit there with a path that never advances. The journal records `skipped`, no failure, no
  rescue. One clean repath (clear the blocked-tile set and the path) frees it.
  (2) COW_CHORES: 463 `retry` results = 2.12M f (9.8 h) against 171 passes at 0.50M f; 455 of the retries are the designed `barn shift over at Nf` re-entry (about 6,000 f a shift), 7 are
  `incomplete cow care ... milk`, 1 is a blocked walk. Not a stall, but the single largest block of wall time.
- Code check now in the ledger: `WorkFrameLedger.observe(..., action=)` counts frames with a button held and *dead* frames (no button, same map and pixel position), per segment, and lists every
  dead run of 120+ frames in `work_metrics.idle`; `FINAL.md` prints an "Idle frames" section (`final_score.render_idle`). Tests in `tests/test_task_progress.py`.
- Instrumented re-run of the exact recorded tree: `recordings/e2e/idle1_20261114/` (tree `/tmp/e2e_idle1_20261114` = frozen rec1 tree plus the passive counter, `HARVEST_STRATEGY=score_herd`, lean).
  Result (finished 2026-11-15, 4.4 h wall): frames 11,216,662 and wallet 337,280G equal `rec1`, 999/999 Clean, so the counter is passive and a power-on run replays exactly.
  Idle: 11.19M stepped frames, 7.75M with a button held, 3.45M with none, **3.33M dead** (no button, same map and pixel as the last frame; 30%), but only 1,733 dead runs of 120+ frames
  (673k frames, 6%). Longest dead run: 7,130 f (119 s, ENDING credits); next 3,643 f (61 s, Y2 Winter D17 EXIT_TO_FARM birth scene), then 3,000 f (HOT_SPRING_STAMINA's deliberate dog wait).
  No hour-long stall anywhere. Dead frames are animation locks and waits (CUT_FODDER 158k, COVER_FIELD 133k, COW_CHORES 125k in runs). Walking wedges are *not* dead frames (the farmer
  wobbles), so the wedge evidence is the NAV_CROP/nav timeouts above; this old-code run logs no `[NAV_WEDGE]` lines because it has no guard.

- **Fix landed (2026-11-14 night): no wedge can run past ~1,200 f.** Root cause: `Navigator.stasis` resets on a *tile* change, so a wobble across a tile boundary (x 762-774 at the north-east pasture,
  x 48 at the water stands) never grows it, and the path never re-plans; each walk then ran to its own timeout (NAV_CROP 9,000, shed route 4,000, water-can route 8,000).
  `core/task_progress.MotionBox` now watches a 24 px box in both walking engines (`NavTask`, `MultiMapNavTask`): after `NAV_WEDGE_FRAMES` = 600 f one clean repath (clear the blocked set and the path),
  a second stall without moving = FAILURE; each new stall earns its own repath. `NavTask` also fails on *no progress*: 8 px closer to the target at least every 1,200 f (600 f when within 64 px, the
  circling case of Y1 D12). `NAV_CROP` timeout 9,000 to 2,400 (a real one is 60-700 f). Every wedge prints `[NAV_WEDGE] ... pos target path blocked`. Replays from the rec1 pins:
  Y3 Spring D18 and Summer D5 (shed route freed after 600 f, was 4,000 + 9,000), Y1 Spring D12 and Summer D5 (NAV_CROP fails at 600 f, was 9,000). Tests `tests/test_nav_wedge.py`.
  Also ended by this: the 4 `COLLECT_POWER_BERRY_STUMP` nav timeouts (~10,000 f each), 4 `ENSURE_WATERING_CAN` multi_nav timeouts (8,000 f), `CROP_ESTABLISH` nav timeouts (~5,000 f).
  Not covered: `TILL_FIELD` grass-planter "too many target timeouts" (2 x 24,000 f, Y1 D4 and D24): its targets are NavTasks so they shrink, but check the next run's journal.
- Old videos deleted at the owner's request: `rec1_20261114` and `v3` mp4 parts (45 GB); reports, FINAL.md and run logs stay. Re-record with `VIDEO=1 scripts/e2e_run.sh <tag>` after the final code lands.
  2026-11-17 check: no mp4/mkv/webm is left under `snes/harvest` (the 30 GB in `recordings/e2e` is ~50 runs of `report.json` + `run.log`, ~620 MB each). The rec1 999's *tree is gone* (`/tmp/e2e_*` is wiped
  by a reboot and `git_dirty.txt` is only a file list), so that 999 cannot be replayed; `e2e_run.sh` now saves `frozen_tree.tgz` + `env.txt` in each run dir (unpack, same env/seed, `VIDEO=1`).
  The current tree (unchanged since 2026-11-15 11:00) scored 985-995 in `hdet`/`hj31..36` (no 999): crops short 2-7 and development short 1-10 in every run. Batch `kdet_20261117` + `k41..k47_20261117`
  (same tree, seeds 41-47, lean) started 2026-11-17 09:48; a 999 Clean among them is recorded by re-running that tag with `VIDEO=1` (replay equality VIDEO=0 vs 1 was shown by `rec1`/`proof1`).
  **That batch died in all eight runs at Y1 Winter D1 (wedged after 8 rescues).** Cause: `MultiMapNavTask._camp_escape` (the 2026-11-15 08:50 hsdet fix, never run in a full batch: `hdet`/`hj31..36` froze at 07:02)
  changes the fish-berry walk out of the camp on Y1 Spring D4 (log line 4651, settle (650,422) vs (620,422)), which re-rolls Year 1: 4 berries at Winter D1 instead of 6 (egg festival turtle shell, two
  upgrade berries missing), the summit fence tile (24,2) stays `0xD5` (shut; `0xFF` when open), `SummitGateTask` stands at (391,54) for 1,800 f, and the `return_mountain_to_farm` route then wedges at the
  push-block (26,7) `0xA0` from (401,121). Fix: `_camp_escape` is now off unless `HARVEST_CAMP_ESCAPE=1` (`multi_nav.CAMP_ESCAPE`), so Year 1 replays the 07:02 timeline again (`rdet_20261117` must equal
  `hdet_20261115`: 11,290,538 f, 990). Still open (found, not fixed): the snow-flower trip never checks the summit gate before it walks there, and a failed `SummitGateTask` leaves the farmer in the lane
  (pin `PR_chk_kdet_Y1_Winter_D1` reproduces it); `hsdet`'s camp dead-end wedge (Y1 Winter D8, `HARVEST_BARN_SKIP_CAPPED=1` only) is therefore open again.
  **Why Y3 Summer misses crops (hj31..36, hdet: corn 432-479 of 496 in all seven, tomato 460-496 in four; potato/turnip always 496):** supply, not harvest time. Corn is sown once, on Summer D1 (the only
  shop day before the weekend), in 5 bags: `seed_bags_for_quota` holds back `ceil(plots*room_tomato/sum)` = 5 of 10 empty sites for tomatoes. Five sites give 37 corn a cycle, six cycles (D13..D29) = 222-232,
  and Y1+Y2 supply 247, so the need is 249. Harvest days cap at 37 crops (8:00-16:00, ~200 f a crop), and a cycle's ripe plants wait on the stem, so the limit is how many ripe plants a cycle offers.
  Levers, both env-gated and default off (Y1/Y2 untouched): `HARVEST_Y3_SUMMER_EXTRA_SITES=N` (new, `farm_pond.y3_summer_extra_site_count`) keeps N of the twelve Year-3 extra sites through Summer (12 = the old
  `HARVEST_Y3_SUMMER_SITES=1`); the cost is development (their 5x5 ground stays reserved). Paired previews from the `hj31/34/35/36` Y3 Summer D1 pins (`recordings/y23_sweep/ab_<variant>_<run>_s1.FINAL.md`; JITTER=0, so only the variant differs) decided the first cut (ROM, hj31/34/35/36):
  base 995/993/991/997 (mean 994; corn 418-482, tomato 473-496); **E4 997/994/997/998 (mean 996.5; tomato 496 in all, corn 496 in two)**; E12 995/998/991/992 (mean 994; every crop 496, development gap 1-7).
  More sites buy the crops and lose development (their 5x5 stays reserved until `FINAL_FILL_INDEX` 289, Summer D20). `E6` 998/986/996/991 (mean 992.8; corn 433 and 462 in two) and `E8` 986/994/998/995 (mean 993.3; every crop 496, development gap 2-9) came back the same way.
  Means: base 994, E4 996.5, E6 992.8, E8 993.3, E12 994: one pin per cell, so the differences are inside the noise. **Conclusion: extra Summer sites trade crops for development one for one; the switch stays
  off by default.** The shared bottleneck is the development tail (cover closes 3-8 tiles a day on the last mornings), so the next lever is cover throughput / releasing a capped crop's reserved ground early
  (idea 3 of Finding 3), not more crop sites. Certify anything with a power-on batch (`HARVEST_Y3_SUMMER_EXTRA_SITES=N` plus the seeded Year-3 jitter env), never from pins alone. Test: `test_year_three_summer_keeps_only_the_asked_extra_sites`.
  2026-11-17 cleanup: the superseded run logs (`report.json`/`run.log` of runs through Nov 13, batch 1 `det1`/`jit31..38_20261114`, `hsdet`) were deleted, scorecards kept (e2e 30 GB to 13.5 GB). Running at that
  point: `rdet_20261117` + `r41..r47_20261117` (power-on, no extras, the control for the new switch; ~11:00 start, ~4-5 h) and `ab_E6/E8_*` previews.
- Barn rework landed (section "Barn frames" below: ~6% fewer barn frames, resets 3.7 -> 1.75 a day; the ROM floor is ~17,700 f, so more needs `HARVEST_BARN_SKIP_CAPPED=1`, -26%).
  **Batch 1 (nav fix only: `det1_20261114`, `jit31..38_20261114`) scored 938-946, not 999.** Same terms short in all nine (Y1-Y2 are shared): power berries -5, Eve hearts 8 of 31, children -16, crops -9..-14.
  Two causes, both fixed 2026-11-15:
  1. A bug in my guard: the `NavTask` no-progress rule failed the shed-shelf walk (14 px from its stand, wobbling over the tile line at x=64) that the old code ended by soft arrival at 80% of
     the timeout. `COLLECT_POWER_BERRY_SNOW_FLOWER` failed on 57 mornings. Now a stalled walk inside `_arrival_limit` returns SUCCESS ("soft arrived stalled"); pin replay `PR_e2edet1_20261114_Y1_Winter_D1`
     eats the berry again. Test `test_a_stalled_walk_inside_the_soft_radius_arrives_instead_of_failing`.
  2. The rec1 999 had a lucky Year 1. Removing its 9,000 f wedges shifted Y1 from Spring D4; in the new timeline the farmer stood 228 lumber on Summer D6 (rec1: 384, needs 250) with 11,590G, and
     `apply_house_phases` refuses HOUSE_WOOD under two cows, so no first house, no feather, and the wedding came a year late (Y2 Summer D3): Eve hearts 8, one child. The score family (`court_girls`) now gathers
     first-house lumber before the second cow (`home.py`, test `test_score_family_gathers_first_house_lumber_before_the_second_cow`; payment still waits for a first cow). Replay from `PR_e2edet1_20261114_Y1_Summer_D6`:
     lumber D10-D13, house ordered D16, feather D26, proposal due the next sunny Saturday. rec1 married Summer D18, so a Y1 Fall wedding is later: watch the second child.
  Lesson: any code that moves Y1 frames re-rolls this chain; a 999 needs a full power-on batch after every such change, never a pin.
  First fixed-tree batch (`fdet`, `fsdet`, `fj31..36_20261115`) died in all eight runs at **Y1 Fall D12 (Harvest Festival)**: the farmer fished from 08:11 without a catch
  (the old run landed its fish at 12:05; a reel is a 1-in-3 draw) and at the 14:00 give-up the rod was still cast. `player_action` 17 (cast, waiting) ignores every button
  but Y, and Y only recasts, so the exit walk never freed him: 8 rescues failed at (633,371), "wedged after 8 rescues", run terminal. Fixes (2026-11-15):
  `fishing_exit_pad` (wait through the cast line, one Y on the bite (action 18), never recast, walk once the rod rests at action 15), `camp_exit_pad` (the camp fence opens only
  west at x=585 below y=390; the generic mountain exit swung 50 px between y=342 and 393 for 11,001 f), both wrapped in `PowerBerryTask._leave_fishing` on every failure of a
  fishing source; and `MultiMapNavTask` now fails a leg that gets no closer to its waypoint for `NAV_WAYPOINT_NO_PROGRESS_FRAMES` (2,400 f), whatever the wobble width.
  Replay of `PR_e2efdet_20261115_Y1_Fall_D12`: no rescue, no bad day, EXIT_TO_FARM 1,134 f. A missed fish costs that year's festival berry only: Fall D12 repeats in Y2/Y3.
  Second fixed-tree batch (`gdet`, `gsdet`, `gj31..36_20261115`) reached the credits in all eight runs but scored 860-918: Y1-Y2 now land (Eve 31 hearts, both children, 10 berries) but Y3 lost
  development (-37), shipped crops (-22) and cow happiness (-20..-58, cows 8-12 at 16-22). Cause: a poorer economy (Y3 Summer D1 wallet 149k vs rec1 348k, 12th cow bought a season
  late so its happiness could not reach 200). The Y1 Summer collapse (harvested 90, det1 449) was a stone left in hand by a skipped `clear_plant_stands` (`lift_only`): every can use
  in `crop_skills` idled on "drop held" with an EMPTY pad, so `CROP_WATER` hit its 24,000 f watchdog 14 times (73 `water_until_wet timeout ... held=0x04`). Now `toss_held_first`
  (`home_recover.py`, used by `_build_crop`) sends held debris to the pond (`PondTossTask`) before watering; replay `PR_e2egdet_20261115_Y1_Summer_D11`: CROP_WATER succeeds in ~12.9k f.
  Not fixed: Y1 `TILL_FIELD` grass-planter runs of ~23k f (D4, D16; evening, clock frozen, wall time only) and 8k `HARVEST_ROUTE` skips; the barn shift now fires at 14k f.
  Third fixed-tree batch (all of the above), lean, started 2026-11-15: `hdet_20261115` (default barn), `hsdet_20261115` (`HARVEST_BARN_SKIP_CAPPED=1`), `hj31..36_20261115`.
  Pass = 999/999 Clean, no `[NAV_WEDGE]` that fails a phase.
  Batch 4 (`hdet`, `hsdet`, `hj31..36_20261115`, tree before this note): `hsdet` died at Y1 Winter D8: after the cave berry the `EXIT_TO_FARM` route (31 waypoints from (569,201)) drifted into the
  fishing camp's dead end and wedged at (614,374) through 8 rescues (rec1 left the same spot after a 6,528 f stall, then walked home in 964 f; my 600/2,400 f limits now cut that stall short).
  Fix: `MultiMapNavTask._camp_escape` (mountain, inside `CAMP_BOX`, own target outside it) walks `camp_exit_pad(through_gap=False)` out first (max 900 f). Replay of
  `PR_e2ehsdet_20261115_Y1_Winter_D8` now ends as a bad day (EXIT_TO_FARM fails at 4,480 f, recovery finishes) without a rescue. The slow cave exit itself (route BFS from (470,454) with 11
  temp-blocked tiles) is not fixed: ~7.5k f, same as rec1.

### Barn frames (2026-11-14, `harvest.tools.barn_profile`, barn agent)
Measured on the ROM from the `rec1` morning pins (16 pins, Y2 Fall to Y3 Summer; barn frames = NAV_TO_BARN + ENTER + COW_CHORES + EXIT; a pin proves a day, not the economy):
- **Before** 23,960 a day (COW_CHORES 20,560, 3.7 resets), **after** 22,560 (20,510, 1.75 resets): -5.8%. Y3 pins alone (8): 25,410 to 24,150, resets 4.0 to 1.9. Real day Y3 Summer D15 replayed with
  `run_to_day2`: 26,740 to 25,270, same end state (12 cows fed, happiness 255 = 300/300, wallet 323,610 both). `barn_fuzz` paired by seed (67 trials): mean difference +100 +- 310 frames, resets 4.1 to 1.8;
  84 trials 69/84 clean before and after (barn care 67/69 to 68/69: `herd` seed 61 fixed, `knock` seed 68 is a Navigator `pixel_stuck`; `outside` 14 of 15 fail both times), no escapes.
- **Why not half.** The ROM runs the barn one game tick per two frames from 11 cows up (<= 9 cows: one per frame), so a B-run is 1 px/frame and every action locks the farmer for a fixed time
  (fodder 55, trough 43, talk 105, brush 136, milk 154, ship 46 frames; an X swap is 69 and X is ignored while walking). A full-care 12-cow day is ~10,400 px of walking (12 forkful round trips from the
  one dispenser 4,030 px, 12 bottle round trips to the bin in the far corner 5,740 px, a care tour ~700 px) plus ~6,500 frames of locks: **a floor of ~17,700**. Nine cows play at 11,150 on the same Y3 pins.
- **What moved**: tool-in-hand pass order (one swap a day, was two), walk-map distance for the next cow, fodder stand tolerance 1 px (every third pickup was a 60-frame miss), no `NAV_TO_BARN` in a reset,
  a missed press tries another stand and a struck-out cow waits behind the others, a cow on the bin stand is waited out with the bottle (fuzz seed 20), `COW_CURE` never asks for a shift (seed 61), and
  resets fire on a cow within 72 px of the door, watched every walking frame (fuzz seed 78 lost a cow to a 14,000-frame timer with the first draft of that rule) instead of `y >= 308` plus a 6,000-frame timer.
  Cows scatter (jobs ~10% slower per 10,000 frames), so fewer resets only paid for themselves in the counter, not in frames: timers 3,000 to 22,000 were all within +-5%.
- **Levers that change care, not taken** (owner's call), Y3 pins: `HARVEST_BARN_SKIP_CAPPED=1` skips talk and brush on a cow at happiness 255 (ROM: `AddCowHappiness` clamps, the night pass only lowers it):
  24,150 to 17,800 (-26%); plus `milk=False` on days the wallet does not need milk (milk is ~10,500 frames of a day, 350G a bottle): 7,700; fewer than 10 cows halves everything.
  The ledgers (above, "talk/brush/milk flag verified per slot") read the flags, so the cap skip is off by default.
- Open: a cow pinned to a wall gives a bottle with no milked flag from her side (now another stand is tried); no second bin stand was found (stands over the corner did not pay); `BARN_DOOR_DANGER_PX` 72
  also fires on bottom-stall cows 88 px from the door (a `y >= 316` refinement would save ~0.4 resets a day but was not fuzzed).

### State when this was written
- Fresh power-on runs in flight (`recordings/e2e/<tag>/`, scorecard `FINAL.md` when each ends, ~3.5 h; the video run ~5 h):
  `rec1_20261114` (VIDEO=1, deterministic), `proof1_20261114` (same code, VIDEO=0), `jit21..jit28_20261114` (VIDEO=0, seeded Year-3 jitter:
  `HARVEST_PHASE_JITTER=90 HARVEST_PHASE_JITTER_FROM_YEAR=2 HARVEST_FAULT_SEED=<n> HARVEST_FAULT_RATE=0`; Years 1-2 identical to the deterministic run).
  `rec1`/`proof1` were frozen before the cover soft-timeout and jitter-year edits; `jit*` include them. Read the scores before changing anything.
- To record a winner: same env and seed with `VIDEO=1 scripts/e2e_run.sh <newtag>` (`HARVEST_STRATEGY=score_herd`). Replay equality between VIDEO=0 and
  VIDEO=1 for the *same* code is assumed from earlier runs, not re-proved here: check the new run's score before calling it the recording.
- Uncommitted code from this push: `farm_pond.py`, `run_strategy.py`, `social_calendar.py`, `day_plan_phases.py`, `field_cover.py`, `day_phase_registry.py`,
  `day_plan_orchestrator.py`, `planner/tasks/multi_nav.py` (`soft_timeout`), `core/recovery.py` (`jitter_from_year`), tests (`test_y3_spring_sowing.py` new,
  `test_run_strategy.py`, `test_floor.py`, `test_day_plan_common_nav.py`), plus the older uncommitted `retro_harness/video.py` segmented-video work.

### What the previews show (all `score_herd`, from `PR_e2efixD_20261114_Y3_Spring_D1`, Clean unless noted)
| run | ROM | gap | cause found |
| --- | ---: | --- | --- |
| `y3r_s6` | 999 | none | everything lined up |
| `y3r_s2` | 997 | potato 487 (-1), development 99 (-1) | Summer D15 shed door `(26,30)` "enter failed: directional transition timeout": the day's watering and cover were lost |
| `y3q_s1` | 996 | tomato 470 (-2), development 99 (-1) | tomato 470 although this run had the tomato-reserve rule (`y3m` had 470 too): not diagnosed, start here |
| `y3r_s4` | 996 | tomato 494 (-1), development 97 (-2) | not diagnosed (no cover failures in its log; cover `behind` 180 on the last mornings) |
| `y3r_s3` | 995 | potato 495 (-1), development 95 (-3) | `COVER_FIELD multi_nav timeout` on 11 evenings (Summer D8-D19): first stand in the north-east spur, route through the push-blocking `(48,15)`/`(48,16)` banks |
| `y3r_s5` | n/a | wedged Spring D24, 8 rescues | shed door enter timeout 16:10, then `return_home` blocked at `(9,23)` (0xA8) east of the house door; a replay from its own D24 pin did not reproduce it |
Earlier previews: `y3m` 995, `y3n` 996/994, `y3o` 995, `y3p` 997/994. Power-on: `fixD` 977, `fixF` 984. Hearts were 155/155 in all five finished batch-3 runs
(Nina ended exactly 496 in every one: zero margin).

### Which terms can be over-done, and which cannot
| term | overshoot behaviour | margin now | variance driver |
| --- | --- | --- | --- |
| crops (4 kinds, 31 each) | `SHIP_CAP=496`: extras stay on the plant, harmless; 511 is the wrap | cap = floor, so any *supply* shortfall is a lost point | **harvest minutes**, not seeds (below) |
| development (50) | more counted tiles never hurt (max 2,968) | 95-100 of 100 | evening cover throughput, nav failures, storms (a storm wiped 141 tiles in `y3h`) |
| hearts (155) | window 496-511; Eve drains -2 per hour-18 house entry; counter wraps past 511 | Nina 496, Ann/Maria/Ellen 500, Eve 500-504 | court days lost to closed doors, Eve egg/gift timing |
| everything else | already full | | |

### Finding 1 (measured): crop supply is not the limit, harvest time is, and it depends on distance to the bin
Y3 Spring `HARVEST_ROUTE` always runs ~8:00 to ~16:10 and carries **one crop at a time** to the bin stand `(11,30)`. In `y3r_s6` (and `y3r_s3`, same shape):
- days that harvested 30-45 crops had a mean Manhattan distance of **8-10 tiles** from targets to the bin (D13, D18, D25: the west pockets, first target `(20,27)`);
- days that harvested 13-23 crops had **19-27 tiles** (D8-D12, D14-D17, D19-D24, D26-D30: first targets `(28,39)`, `(32,36)`, `(25,34)`, `(43,28)`);
- on most of those days there are no pick or ship failures (`PICK OK` == `SHIP OK` == `Target`; D17 and D19 had two `fast pick missed` retries): the cost is walking, ~380 frames a crop far against ~165 near;
- every day ended with `ship window closed: 30-90 ripe left`, and D30 still left 30-51 ripe tiles that die with the season. `ship_window_fits` is already
  ETA-exact to 17:00, so the window cannot be stretched; shop days (`BUY_SEEDS` 2.2-2.9 h), court days (`SOCIAL_TOWN` 4-4.9 h) and, in `y3r_s6`, **`COW_ROUNDUP` (3.1 h) on D28, D29 and D30** start the harvest at 11:00-13:00 (D28-D30 shipped 9-10 crops each with 30-51 ripe left).
- `HarvestTask._choose_next_step` sweeps whole sites, nearest to *where the farmer stands*, not nearest to the bin, and `steps.sort` is only the initial order.
- The twelve Year-3 extra sites (`SCORE_Y3_EXTRA_CENTERS`) are 15-40 tiles from the bin. They raised the *supply* (potato 386 -> 487-496) but each of their crops costs
  about twice a west-pocket crop to ship, so "just plant more" hits a wall: a rough estimate is ~19 usable harvest days x ~7,200 frames, about the whole Spring need
  (254 potato + 281 turnip beyond Years 1-2); compute it properly from the day journals. Not yet verified per crop: the D30 leftovers may be corn/tomato kept by the cap rather than potato. Break it down first
  (count ripe-left by crop id in `HARVEST_ROUTE`'s final summary).
Ideas, cheapest first (each needs a preview A/B and then power-on seeds):
1. Harvest the cheapest crops first across the whole farm (sort groups by distance to the bin, not to the farmer) and stop at the cap: with over-planting, the far
   sites then only supply what the near ones cannot. Log crops/day and mean distance to prove it.
2. Site the one-shot Spring crops (potato/turnip) on the near-bin pockets and give the far sites to Summer crops only; or choose the twelve extras by distance, not the pitch-5 grid.
3. Move load out of Y3 Spring: Year 2 Spring (12 sites, no extras) ships 215-242 of each; extras there (or a second Y2 cycle) cut what Y3 must carry.
4. Fewer lost harvest days: buy potato and turnip bags in the first two shop windows (the bags are 7-12 a trip; in `y3r_s6` `BUY_SEEDS` ran on D13/D16/D20/D22, each a 2.9 h trip),
   and move `COW_ROUNDUP` off the last three Spring days (or make it conditional on ripe tiles left).

### Finding 2 (measured): navigation failures cost whole days and have no memory
- Cover walk: first stand unreachable by the BFS route (`reachable()` treats 0xA3/0xA1/0xA8 bank ids as walkable; `FARM_NO_GO_TILES` has no north-east spur wall).
  Fixed in this tree by `MultiMapNavTask(soft_timeout=True)` for cover (the planter takes the next stand), untested on a full run before the `jit*` batch. A better fix
  adds the spur's physical walls (`farm_pond.EAST_SPUR_*` notes: wall at y=14 x=46-50, open at x=51) to the cover BFS.
- Shed door `(26,30)`: `DirectionalTransitionTask` "enter failed" at pos x=422-425 on `y3r_s2` (Summer D15) and `y3r_s5` (Spring D24), both right after a long
  harvest, 16:09. One failed enter burns ~2 game hours (`enter_timeout=1500`), then `ENSURE_WATERING_CAN` and `CROP_WATER` fail and the day is lost. `ShedFetchItemTask`
  has no retry on `enter` failure. Cause unknown (a sprite, held item, or door column). Add diagnostics (sprites near `(26,29)`, `held`, `player_state`) and a back-off retry.
- House door `(9,23)` (0xA8 bank, push-blocks from the east): `return_home` hugged y=383 from the east and failed 8 times in `y3r_s5` with `hands_clear=True`.
- Dead stands (`grass_planter.dead_stands`) reset every day, so the same failure repeats nightly; make failure memory persist across days.

### Finding 3: development is a throughput budget with a late start
Cover `behind` was 791 tiles at the start of final fill in `y3r_s6` (100/100) against 1,711 in `y3r_s3` (95), because of finding 2. Ideas: release a crop's reserved
5x5 footprints as soon as that crop hits its cap instead of at the fixed `FINAL_FILL_INDEX=289` (Summer D20); a second hoe/sow pass per evening while stamina and
gold last; treat any evening that sowed less than its bags as a failure to log.

### Finding 4: what the sample can and cannot say
Pin previews drift from the power-on timeline (a failure seen in one did not reproduce from its own pin), so a preview batch ranks fixes but does not prove them.
The `jit*` runs are the honest distribution: fresh power-on, Clean, Year-3 timing seeded. Build a small variance report from `FINAL.md` + the day journal
(crop raw per kind, development count, `COVER_FIELD`/`enter failed` counts, crops/day and mean bin distance, ripe-left by crop at D30) so each batch states *why* it
missed; today that is done by hand with `grep` on the logs. Acceptance bar for "consistent": at least 7 of 8 seeded power-on runs at 999 Clean.

## Score push 2 — 2026-11-14 (984 power-on `fixF` → 994-997 previews; final proof run pending)

`fixF` (fresh power-on, fix 5 code) scored **984/999**, Clean: crops 110/124 (potato 386, turnip 390, tomato 493), development 49/50, hearts 155.
Every other term was full, so the crops were the whole gap. `fixD`'s Y3 Spring journal showed why (potato and turnip cannot be sown after Spring D26,
and Year 1-2 only supply 242 and 215 of the 496 each):

1. **Twelve sites were the limit, not stamina** (a day waters ~90 tiles, two bars). `farm_pond.SCORE_Y3_EXTRA_CENTERS` adds twelve Year-3 Spring sites
   on the pitch-5 grid (`HARVEST_Y3_EXTRA_SITES=N`, default all; 0 restores twelve). From Summer on they drop out of `active_plot_centers(year, season)`
   (`HARVEST_Y3_SUMMER_SITES=1` keeps them): twelve sites grow all the corn and tomatoes, and the extras' 5x5 reserved ground goes back to the cover.
   Test: `test_year_three_sows_eight_more_sites...`, ROM `test_every_year_three_extra_site_takes_a_full_bag_from_the_natural_morning`.
2. **One crop was bought per trip for half the empty sites** (D3: 7 turnip bags for 12 empty sites). `seed_bags_for_quota` now buys a bag for every
   empty site of the crop it picked in Year 3 Spring (`HARVEST_Y3_ALL_BAGS=0` restores the proportional split).
3. **Nothing was bought before a shut weekend.** `pocket_plant_targets(replant_ripe=True)` also counts a ring that ripens within the shop's
   closed days (`shop_closed_days_ahead`, `soon_ripe_tiles`; `HARVEST_Y3_SOON_RIPE=0` off).
4. **D2 court ate the seed window** (`fixD`: Ellen's 4,189 frames, cutoff 13:11, twelve bare sites a day). `spring_sowing_trip_first` yields the court on D1-D3
   while sites are bare (`HARVEST_Y3_COURT_YIELD=0` off). The morning plan is made in the house, so it reads the persistent farm array, not the live map.
5. **Quota accounting**: `spring_committed_totals` reserves three spare rings per crop in Year 3 (`HARVEST_Y3_SPRING_SLACK`, tiles): `y3k_a` stopped buying with
   497 turnips "committed" and shipped 494.
6. **Court after a harvest that ends at 17:00 walks into closed doors** (`y3k_b`: loops=0 talks=0 on D27-D29, Ellen 484). Year 3 Spring weekday courts go
   before the harvest (`court_before_harvest`, `HARVEST_Y3_COURT_FIRST=0` off).
7. **Summer tomatoes** (`y3m`: 470 of 496): the first trip bought corn for every empty site and left tomatoes no ring. Year 3 holds back the rings the other crop's
   share needs while nothing of it stands (`HARVEST_Y3_TOMATO_RESERVE=0` off).
8. **Final fill started late with more ripe crops** (`y3p_s2`: 46-96 ripe every morning D14-D22, cover only from D25, 2,881 of 2,950): final-fill mornings now
   tolerate up to `FINAL_FILL_LIGHT_RIPE_TILES` (100) ripe tiles (`HARVEST_FINAL_FILL_RIPE`).
9. **A girl under the 496 floor is courted in the last window** even when cap 500 leaves under one +8 visit of need (`y3m`: Nina 495).

Previews from `PR_e2efixD_20261114_Y3_Spring_D1`, `score_herd`, Clean, items 1-5 (+6 for `y3o/p`): `y3m` 995, `y3n_s2` 996, `y3n_s3` 994, `y3o` 995, `y3p_s2` 997, `y3p_s3` 994.
Their gaps were scattered (potato 457/472, tomato 470, development 48-49, one heart girl each), which is what items 6-9 target. Open: Eve ends at 492-494 in
some runs (gift needs an egg, nightly entry costs 2; `EVE_GIFT_TARGET` 506 is the wrap-safe maximum, `test_ranch_master_published_rules`). The ROM
flock test needed `HARVEST_FINAL_FLOCK_START=250` for its D11 pin (final flock now starts D25): stale pin, not a code bug.

## Score push — 2026-11-13/14 (942 → 975 → 977 power-on; fix 5 preview and final power-on `fixE` pending)

`recordings/e2e/perfect999b_20261112/` was the first fresh power-on to the credits with Clean intervention: **942/999**. The causes
and fixes since, each with a red-then-green non-ROM test (the full offline suite is green: 2,782 passed):

1. **Y2 Winter cascade (−25 points).** The first child's labor morning (Y2 Winter D17) is an unavoidable lost day: cows go unfed and ~3
   roll sick. `BUY_MEDICINE` on D19 then used `town_animal_shop_to_path`, whose first waypoint sat on Ellen's weekday pixel (648,904),
   so the day was lost again, four cows died (−100 happiness each, a funeral day each) and Y3 Spring spent D3–D12 re-buying cows.
   `map_routes.py` now swings south of Ellen; `test_buy_seeds.py` keeps every 0x04 waypoint ≥16 px from her. Replay from
   `PR_e2eperfect999b_20261112_Y2_Winter_D17`: no deaths. Preview `y3c`: 967.
2. **Fish gifts ate the Summer planting mornings.** The dated Eve fish correction ran on Summer D1–D5 and the seed trip was never
   planned (no corn). `wife_fish_gift_wanted` is now off for Summer D1–D5. Preview `y3d`: 971.
3. **Evening heart drain (`eve_gift.py`).** Every house entry at hour 18 costs Eve exactly −2 hearts and zero game time (the clock is
   frozen on the entry fade), so the excess above the 496–511 window is shed by repeated enter/exit after the day's work instead of
   spending fish mornings. `HARVEST_HEART_DRAIN=0` restores the fish branch. Preview `y3e`: **979** (Eve 504, no wrap).
4. **Seed-shop clock (Year 3 only).** `y3e` Summer D1 ran two forage loops to 14:00 and left for the shop at 15:13; the walk is ~1h50 and
   the door shuts at 17:00, so the trip "failed" with `town_to_shop_door ... exit_walk timeout` (the clock, not the route). Saturday and
   Sunday are shut and Monday went to grass seed, so corn went in on D7 and **tomatoes never** (470 of 496, −2 points). Year 3 now
   goes to the shop before it forages (`SHOP_FIRST_FROM_YEAR`, `SHOP_FIRST_LATEST_HOUR`), and the Summer pick starts a crop that has
   room but nothing standing once the other is in the ground (`_summer_quota_bags`, Year 3 only). Years 1–2 keep their timeline.

5. **Capped crops blocked the last-week cover trips (found in `fixD`).** `fixD` (fresh power-on, fixes 1–4) scored **977**, Clean: tomato and corn 496,
   hearts 155, but development only 42/50 and potato/turnip 386/385. Y3 Summer D22–D29 logged `[COVER] busy morning (46..80 ripe)` every morning while the
   harvest said `Left 80 ripe corn/tomato tile(s) on the plant at the score_herd ship cap` and `Detected 0 ripe crop targets`: `WorldProbe.ripe_crop_count`
   counted tiles the ship cap leaves on the plant, so no grass trip ran after the D21 load (24 bags, 195 tiles still to hoe) and 445 tiles stayed
   uncounted (previews that happened to have a light morning reached 2,917 of 2,950). It now counts only what `within_ship_cap` keeps
   (`test_a_busy_morning_counts_only_the_ripe_tiles_the_harvest_will_pick`). Busy mornings with real ripe tiles still keep the harvest.
   Preview `y3h` (from `PR_e2efixD_20261114_Y3_Summer_D1`, this fix only): **979**, development 45 (a storm wiped 141 tiles on D22→D23 and the last
   nine mornings sowed ragged one- and two-cell stands, so 24 bags counted only ~35 tiles a day, 342 short at the tally). Final-fill days now
   load `FINAL_FILL_TARGET_BAGS = 60` bags (`field_cover.py`; the evening hoes ~110 tiles whatever the bag count, so bags were the limit;
   `test_final_fill_days_load_more_bags_than_a_building_day`). Building days keep 24.
   Preview `y3i` (same pin, both fixes): **984**, development **50/50** (2,955 counted; the D21–D25 loads were 60, 51, then the held bags), hearts 155,
   corn 496, tomato 493. Scorecard: `recordings/y23_sweep/score_push_20261113/y3i.FINAL.md`. Remaining misses: potato 386 / turnip 390 (−14), tomato 493 (−1).

**Power-on proof, code before fix 4: `recordings/e2e/fixC_20261113/` = 975/999**, Clean (0 initial loads, 0 mid-run loads, 0 RAM writes, 0 rescues),
3.6 h wall. Misses: development 41/50 (Y3 Summer D29 2,472 counted tiles), tomato 473, potato 391, turnip 408. Replays with fix 4 from the same
Y3 states: `y3s` (Summer only) 984, `y3f` 984, `y3g` (cutoff 14:00 on every Year-3 day, not kept) 983, all Clean, tomato and corn 496.

Remaining misses in `y3e` (979): potato 405 and turnip 393 of 496 (Spring plot capacity, −13), development 45/50 (hurricanes and
competing days reset grass; 2,675 of 2,950 counted tiles), tomatoes 470 (fix 4). Next levers if the proof run is short of 980: more
Spring Y3 sowing days (shop-first helps) and cover scheduling.
ROM eval for the changed files (`test_eve_gift`, `test_buy_seeds`, `test_day_plan_crop_phases`, `test_run_strategy`, `test_crop_quota_policy`, `-m rom`):
7 passed, 1 failed. The failure, `test_score_flock_incubates_an_egg_from_a_natural_final_spring_pin` (local pin `PR_e2es4_Y3_Spring_D11`), is not from
this pass: the frozen pre-change tree `/tmp/e2e_perfect999b_20261112` gives the same `egg_mode == "ship"` (`hatch_allowed` from the uncommitted
`livestock_econ.py` work). Open item.
Evidence: previews `y3c/y3d/y3e/y3f` (pins `PR_y3[cdef]_*`, scorecards and logs under `recordings/y23_sweep/score_push_20261113/`),
power-on proofs `recordings/e2e/fixC_20261113/` (code before fix 4) and `recordings/e2e/fixD_20261114/` (final code).

## Active continuation — 2026-11-12

The handoff below is historical; work has resumed. No composed 999 is claimed.
`recordings/y23_sweep/carry999_20261112_s1.{log,json}` is the frozen-code,
zero-jitter continuation from the stable Y2 Spring D25 pin. Its daily pins use
`PR_carry999_20261112_s1_`. Check the process and newest pin before resuming.
The fresh power-on candidate is `recordings/e2e/perfect999b_20261112/`
(`FOREGROUND=1 VIDEO=0`, supervised terminal session). The first attempt,
`perfect999_20261112`, failed the D1 starter-tool walk and is retained as red
evidence. Both active runs use frozen source trees; the continuation predates
the D1 navigation scoping correction below.

Repairs and native receipts in `recordings/wedge_fixes_20261112/`:

- Renamed `LN_handoff_*` pins now restore their adjacent lineage, just like
  `PR_*`. Previously the stable copies silently skipped the planner history
  and core resync, changing the replay. The regression fails before the fix.
- A completed/failed day-ending phase no longer owns the rest of the day
  plan's date changes. The failed Egg Festival previously resumed ordinary
  work on D21, then waited for morning at 18:xx. The original lineage replay
  now reaches settled D21 in **40,866 frames**, zero rescues/writes/mid-run
  loads (`settle_after.json`). Morning settling also has its own 3,600-frame
  rescue bound, independent of the disabled per-day floor.
- Once the opening fences are gone, a real full BFS path to the pond wins
  over densify crumbs inside crops. The original Summer D14 lineage replay
  waters **71 tiles / 3 refills** and finishes the entire day in **24,817
  frames** (`refill_path_after.json`), versus the old watering watchdog.
  Selection also gives the unreliable (32,34) corner a two-step penalty,
  preferring adjacent (33,34) when the routes are comparable.
- The handoff's only hen is on farm map byte 1 at (510,969), not in the
  coop. This explains the absent eggs, stalled family affection and missing
  hen-trade berry. `HenDeliveryTask` now recovers farm hens; the day planner
  schedules that recovery. A native recovery/feed/sleep test confirms that
  she returns to the coop and lays. The sole egg first rebuilds the laying
  pair, then morning wife gifts resume. The two-day composer replay incubates
  the replacement on D26 (`hen_days_after.json`).
- The first fresh power-on attempt exposed the previous session's parked
  sprite padding on the D1 dog. Its ring included the house-approach target
  (137,375), so the starter-tool walk reported `no_path sealed`. Forced
  padding now applies to town villagers (the Ellen fix) while the farm uses
  ordinary body avoidance. The new power-on attempt clears the D1 handoff in
  **16,924 frames**, with both starter tools, and reaches its natural D2 pin.
  `scripts/e2e_run.sh` accepts `FOREGROUND=1` for supervised execution and saves
  an end-state on success or failure; its background mode was killed when the
  launching tool session ended in this environment.

Validation: **2,777** offline tests pass after the D1 scoping correction;
docs checks **8**. The original **13** wedge ROM tests pass; their combined
rerun with the calendar and supply-trip tests is in
`recordings/wedge_fixes_rom_complete.log`. The sparse-sprinkler ROM check now
performs the composer's native farm exit before inspecting crops: its old
morning fixture was indoors, so it read the house buffer and watered zero.

Remaining proof on both active runs: the second child, all four shipment
windows, the 2,950-tile final census, the final affection windows and the
native ending tally. The frozen power-on candidate is running alongside the
continuation.

Continuation receipts now reached: hen-trade berry **9 -> 10 on Y2 Fall D4**,
retained on D5; first child born on **Y2 Fall D13**; player happiness
**708 -> 999 on Y2 Fall D12** (97 dumplings, 86,620G left); turtle shell won
on **Y2 Fall D20**, followed by settled D21 and D22 mornings. Twelve cows
remain alive. These close the continuation's collectible/night-settle checks,
not the ending score or power-on proof.

The fresh `perfect999b_20261112` run has reached Winter after retaining the
five early berries, eating the Harvest Festival berry, winning the turtle on
Y1 Fall D20, acquiring the clock by Summer D14 and marrying Eve on Summer D18.
Its Year-1 pasture stopped at 27 counted planted tiles, below the 48-tile cow
gate; Fall cannot raise that counter. Verify Spring herd acquisition and
family timing rather than assuming the old Year-1 herd benchmark applies.

## HANDOFF — 2026-11-12 evening (read this first)

**Not done.** No composed 999 ending exists. Everything below was stopped on purpose (all emulator runs killed, nothing running).
Nothing is committed or pushed; the whole tree is the working-tree state on top of the long-dirty checkout.

**Best state reached** (`score_herd`, power-on to Y2 Spring D16 in `compose7`, then a pin continuation `cont8` with the final code):
Y2 Spring D26 morning: 9 cows, 1 hen, 7,450G, **9 of 10 berries / 190 max stamina**, Eve married (Y1 Summer D18; hearts 248),
shipped corn 175 / tomato 101 / turnip 232 / potato 228. Year 1 from power-on is deterministic and shared by `compose6`/`compose7`
(Y2 Spring D1 = frame 3,077,316). The older scout run (pre-fix code, `compose_20261112`) proved the Y2 Fall D12 happiness top-up on a
natural timeline (`happiness 509 -> 999`, 164 dumplings) and had 12 cows, 165,700G, development 1,104 of 2,950 at Y2 Fall D21.

**Saves** (`custom_integrations/HarvestMoon-Snes/`, git-ignored; lineage paths already rewritten off `/tmp`, so they survive a reboot):

| pin | what it is |
| --- | --- |
| `LN_handoff_20261112_Y2_Spring_D25` | newest; resume here (`cont8`, final code). A day plays Clean to Y2S0D26 |
| `LN_handoff_20261112_Y2_Spring_D16` | the morning of the animal-shop/Ellen wedge, now fixed (35,690 frames, no rescue) |
| `LN_handoff_20261112_Y2_Spring_D1` | Year 2 start of `compose7` (9 berries, married) |
| `LN_handoff_20261112_settle_morning_wedge_Y2_Fall_D20` | **reproduces the open `SETTLE_MORNING` hang** (old-code timeline, 8 berries) |
| `LN_handoff_20261112_refill_loop_Y1_Summer_D14` | reproduces the pond-refill loop (`CROP_WATER` 24,001-frame watchdog) |

Resume to the ending (continues the same timeline; copies the tree first, neutral jitter, no injected faults):

```bash
cd snes/harvest
JITTER=0 FAULTS= RATE=0 HARVEST_STRATEGY=score_herd scripts/y23_sweep.sh LN_handoff_20261112_Y2_Spring_D25 <tag> 1
# log recordings/y23_sweep/<tag>_s1.log, pins PR_<tag>_s1_*; one day only:
HEADLESS=1 PYTHONPATH=. .venv/bin/python -m harvest.scripts.run_to_day2 --state <pin> --days 1 --strategy score_herd --out /tmp/x.json
```

A pin continuation is not the natural-entry proof. The claim needs a fresh power-on run to the credits on the final code
(`VIDEO=0 HARVEST_STRATEGY=score_herd scripts/e2e_run.sh <tag>`, ~3 h); do it after the wedges below are gone.

**Fixed this session** (all with a red-then-green test; details in [Y1_BENCH.md](Y1_BENCH.md) and the Harvest `AGENTS.md` traps):
Harvest Festival walk (opt-in `Navigator.center_stall_limit`, set only by `FestivalEntryTask`); Spring D23 festival lost to a 17k-frame
watering (`CROP_WATER stop_hour` is now a runtime deadline, +1 h); Summer D1-D3 courting no longer takes the rings' corn trip
(`social_calendar.ring_corn_trip_first`); Y2 Spring D8 hang (`farm_map_loaded` / `stale_under_player` use the cover bounds, plus a
120,000-frame backstop for every phase that had no watchdog, `phase_backstop_frames`); Y2 Spring D16 animal-shop wedge (Ellen's
spot on `animal_shop_town_to_farm`, `ReturnHomeTask` shop/plaza branch, parked-sprite ring padding). Offline suite: 2,770 passed
(legacy) and 2,764 (`blocks`); docs tests 8; the calendar/social/wedge ROM files (16) and six festival ROM tests pass.
Known failing ROM test, also failing on the pre-edit runtime:
`test_power_berry_planner_rom.py::test_planner_visits_the_missing_stump_after_the_early_berries_are_done` (stump trip first-attempt `nav timeout`).

**Open, in priority order**
1. **`SETTLE_MORNING` hang after the Y2 Egg Festival** (Fall D20 pin above). The date rolls to D21 at frame 40,750, the farmer is on the
   farm map at 18:xx, `MULTI_DAY` waits "Settling morning scene" forever (`stall` 16M frames in the scout). It is a multi-day planner
   phase, outside the new per-phase backstop: add a stall guard there and find why the Egg Festival night leaves the farmer outdoors.
2. **Pond-refill loop** (4+ watering days lost a year). Two mechanisms, both in [Y1_BENCH.md](Y1_BENCH.md): Spring D23 (farmer one tile from
   the south stand (32,34), "no path", power-on only) and Summer D14 (reproduces from the pin above: the south-west "densify" hop to
   (12,42) thrashes on growing crops, then the charge macros wander). Stand success rates: (32,34) 46%, (33,34) 97%.
3. **Berry 10 and the turtle.** Hen trade needs two adult hens in Fall (these runs had one); Y2 Fall is its second chance. The Egg
   Festival coin flip decides berry or turtle each year; `compose6`/`compose7` got the berry in Y1, so the Y2 Fall D20 festival must
   give the turtle (+21).
4. The rest of the 999 terms are still unverified end to end: both children (hearts >= 650 for the second), the 496 shipment window per
   crop, development 2,950 counted tiles, Eve/hearts final windows, money and the final-night losses.

**Process traps hit today**: a monitor/`tail` watcher expires after 30 minutes, so confirm the run is alive (`ps`, newest `morning pin`),
not just that no alert fired (`compose4` was dead for 30 minutes unnoticed); never `pkill -f` a pattern that appears in your own command
line (it killed the shell twice); the day floor is off by design (owner), so every phase needs its own watchdog.

## Score policy repairs (2026-11-12)

The audit below describes the historical 851 ending and the clockless ten-berry
checkpoint. Policy changes now address the clock, crop, family, turtle and
development causes. These component receipts are not a new ending score.

- **Clock recovered:** the first quote is scheduled from Y1 Summer D4 without
  waiting for two cows, affection or full construction funding. The next
  morning's carpenter visit must set the native arm before Fall. A continuation
  from `PR_tenberries_guarded_20261111_Y1_Summer_D6` paid the first house on D7
  and owns the clock on **Summer D12**, with **5 berries / 150 stamina**.
  Resume at `PR_clock_gift_20261112_Y1_Summer_D12`. Both segments have no RAM
  writes, assists or mid-run loads; they ran on the shared live tree, not a
  frozen annual benchmark. Hashes and reload evidence:
  `recordings/clock_recovery_20261112/gift_verification.json`.
- **More crops and earlier shipments:** score strategies add three native-proved
  west-strip plots from Year 2 (12 plots / 96 crops versus 9 / 72). Summer
  purchases reserve room for both crops and count standing plants before
  choosing the next seed. Spring can buy the final partial-quota bag; shipping
  still stops each crop at 496. Ripe crops precede courting and pasture trips,
  including when the morning read is indoors. On the exact historical Y3
  Summer D17 pin, **23 tomatoes ship before church**, versus zero in the old
  journal; all 24 added turnips plant from a natural Y2 Spring D2 pin. Both
  native checks have zero writes, mid-run loads or assists. Evidence:
  `recordings/quotas/20261112/rom_verified.log`.
- **Second-child threshold and final Eve window:** score strategies use both
  eggs toward **654 hearts** until the second-child flag, then target 506 and
  reserve remaining festival and child-age-60 affection gains. A bounded fish
  gift can remove 8 excess hearts after that flag. A declared research fixture
  starts Eve at 646 and child 1 at age 90; native gifts reach 654, the night
  leaves 652, and the ROM sets child 2's flag and age 1. Setup edits are not a
  natural family timeline. The native fish catch/return/gift separately proves
  the -8 change without writes or mid-run loads. Receipts:
  `recordings/berries/family_threshold_20261112.log` and
  `recordings/berries/family_fish_20261112.log`.
- **Turtle after ten berries:** both the calendar and the acquisition task
  previously stopped at ten, suppressing the second Egg Festival hunt. They
  now allow the missing turtle prize. Three native prize-order cases pass,
  including a declared ten-berry starting fixture; six existing source/planner
  Egg Festival cases also pass. Prior prize/count edits remain research setup,
  not campaign receipts. Logs: `recordings/turtle_policy_20261112.log` and
  `recordings/turtle_planner_20261112.log`.
- **Development route unblocked:** the southern well's A1/A3/A8 rim was treated
  as walkable. The 851 run timed out there for weeks with paid grass left in
  stock. Its natural Y3 Spring D9 route fails before the coordinate fix and
  passes afterward. Native 24-bag purchase/sow checks add **191** durable tiles
  on Y2 Spring D3 and **208** on that Y3 farm, spending every bought bag with
  zero writes or mid-run loads. This does not yet prove the final 2,950-tile census.
  Receipts: `recordings/cover_policy_20261112/`; combined evidence:
  `recordings/score_policy_20261112/targeted_verification.json`.

- **Player happiness top-up (native receipt):** `HarvestHappinessTask`
  (`planner/tasks/festival_escape.py`, phase `HARVEST_HAPPINESS`) spends the
  Year 2 Fall D12 Harvest Festival on the dumpling stall: enter the festival
  (`FestivalEntryTask`), buy one 300G dumpling at a time (+3 each) until the
  player reads 999, and let the native night end the day. It is due only for the
  score family on `(year, season, day) == (1, 2, 12)` before 15:00 with 999 not
  yet reached and the wallet above `happiness_cash_floor + 300` (the dated
  `field_cover.cash_floor_g`, never below 60,000G), so it cannot starve the
  money term. From `PR_t5_s51_Y2_Fall_D12` (happiness 345, 163,960G) the native
  test takes happiness to **999** with **218 dumplings (65,400G)**, leaving
  98,560G, in 155,392 frames, on day 13 with zero RAM writes and zero mid-run
  loads (`test_native_harvest_stall_reaches_999_and_finishes_the_festival`,
  `-m rom`, 139 s; receipt `recordings/happiness_20261112/stall_receipt.json`,
  end state `LN_happiness_999_native_20261112`). An earlier run of this test
  caught a timing bug: the purchase press reopened the last villager's dialogue
  before the farmer was ready (`test_happiness_wait_does_not_reopen_the_last_girls_dialogue`).
  The ROM pays `happiness >> 5`, so 992 is enough; the task still aims at 999
  to absorb the later cow buys and losses. The ten-berry calendar keeps its
  own Harvest Festival berry phase ahead of the top-up
  (`test_missing_harvest_berry_keeps_its_own_phase_before_happiness`).

- **Frozen power-on Year 1 (finished, CLEAN):** `recordings/y1_bench/score_policy_final_20261112/` reaches Y2 Spring D1
  with 68,850G, 2 cows, 1 hen, 522 crops shipped, **7 of 10 berries**, 1 cosmetic bad day. Three berries were missing, and
  two of the causes were bugs found and fixed here (details in [Y1_BENCH.md](Y1_BENCH.md)): the **Harvest Festival** walk
  (navigator centring deadlock, opt-in stalled-centring detector for the festival-entry walk; pin replay 6 -> 7 berries) and the **Spring D23 Flower Festival**
  (the capped `CROP_WATER` ran to 18:12; `stop_hour` is now a runtime deadline, +1 hour). The third, the Egg Festival,
  paid the turtle shell, so the berry waits for the Year 2 second try. Open: the south-stand pond-refill loop that caused
  the Spring D23 overrun also hit Summer D14 (24,001-frame watchdog).

Next action: read `recordings/e2e/compose2_20261112/` (power-on to the ending, `score_herd`, current tree with both fixes:
clock, berries, children, crop quotas, development, happiness top-up). It is the composed result; its `FINAL.md` names
which of the 999 terms are still short. `compose_20261112` (same Year 1 without the two fixes) is a scout for Year 2-3
faults only.

## 999-point gap audit (2026-11-12)

The latest completed ROM ending remains **851/999**, from
`recordings/y23_sweep/ending12_20261111_s1.json`. Its oracle terms sum exactly
to the ROM's terminal score; the seven deficits sum to **148**:

| Term | Ending points | Missing | Required change |
| --- | ---: | ---: | --- |
| Shipped crops | 62/124 | 62 | Reach 496-511 shipped of each of the four crops |
| Turtle shell | 0/21 | 21 | Win the other Egg Festival prize by Y2 Fall D20 |
| Berries | 30/50 | 20 | Ten berries; now achieved in the separate Y1 Winter continuation |
| Field development | 31/50 | 19 | Raise 1,849 counted tiles to 2,950 |
| Children | 16/32 | 16 | Raise the wife's hearts to at least 650 for child 2's flag |
| Player happiness | 22/31 | 9 | Raise 732 to at least 992; target 999 before final losses |
| Eve's relationship | 30/31 | 1 | Finish at 496-511 after the final house-entry losses (ending: 494) |

Crop detail: tomatoes 299 (13 points short, 197 more shipments to 496),
corn 202 (19 points, 294 shipments), potatoes 251 (16 points, 245 shipments),
turnips 275 (14 points, 221 shipments). No crop counter wrapped. Money,
12 cows, 12 chickens counted by the ROM, all cow happiness, both house
upgrades, marriage and the clock already score their maximum in that ending.
Applying only the verified berry improvement to those historical terms would
give 871, leaving 128 points; that is arithmetic, not a new ending result.

**The current ten-berry branch has a separate clock failure.** At
`PR_tenberries_egg_20261111_Y1_Winter_D10`, clock-owned
`$7F1F6E & $4000` and clock-arm `$7F1F70 & $0004` are both clear. The first
house quote was accepted on Fall D6, after the ROM's Y1 Summer D30 arm
deadline. `bank_81.asm` at `$81F14C` only arms it in year 0, season < 2;
the first-completion gift at `$81F354` requires that arm. Thus this checkpoint
cannot reach 999 through the normal remaining route: the clock alone caps it
at **977**, even if every other term is filled. Recover it from a predecessor
before Summer D30, preserving the berry fixes; continuing through Year 2 will
not repair this deadline.

The current checkpoint is still Y1 Winter, with two cows and two hens, so it
must not inherit the old ending's full herd/house/relationship terms. Its
current RAM-model score is **352**, with development recomputed from the farm
array (59 counted tiles, 1 point); this is not a ROM ending score. Berries pay
50/50, the clock and turtle are both absent, the house pays 16/32, and no
child flag is set. Current shipments: corn 343, potatoes 226, tomatoes and
turnips 0. A live reload plus 120 neutral frames confirmed the critical flags.
Full arithmetic and state/report hashes:
`recordings/berries/ten_from_egg20_20261111/score_audit_20261112.json`.

The child deficit also has a policy cause: `tasks/eve_gift.py` stops routine
gifts at **502 hearts**, below the **650** required for the second child.
Family progression needs a temporary higher target and then a dated return
to the final 496-511 scoring window. Buying a larger house alone will not
close that gap. Second-child age 60 also supplies a separate +100 happiness,
so its timing matters to the happiness term as well as the 16 child points.

Next action: repair and prove the clock arm from a natural pre-Summer-D30
checkpoint before another full 999 attempt; then compose child-2 progression,
four crop quotas and the final development census with the preserved herd
and ten-berry route.

## Ten berries retained through Winter D10 (verified 2026-11-12)

The owner requested a restart from the beginning. The first fresh attempt,
`recordings/berries/ten_from_power_on_20261111/`, reproduced the broken D2
approach: `SOCIAL_TOWN` ran before forage/shop on the uncleared farm, failed,
and displaced those trips below the fences. The optional trip is now deferred
on Y1 Spring D1-D2. The natural D2 pin replay buys and plants eight potatoes
before any courting detour (`test_social_rom.py::test_opening_day_two_plants_before_any_courting_detour`,
16.45 seconds, no writes or mid-run loads). The calendar regression went red
before the fix and passes after it.

That first correction reached Y1 Spring D5 with one field berry retained
overnight, then exposed another native statue failure. The impact effect's
CC `$15` cleanup can delete a held berry that reuses its sprite slot before
the effect ends. The old fixture passed; the new natural D5 fixture failed.
The picker now reads the impact timer (`$7EBAD6`) and presses A with one tick
left, so the held sprite is allocated after cleanup and before the ground
berry expires. Both fixtures now pass (4.39 seconds total). A source recipe
that ends without an eat receipt explicitly continues into pickup/eat.
The stopped attempt and diagnosis are in
`recordings/berries/ten_from_power_on_fixed_20261111/`.
The full D5 composer replay reaches D6 with **2 berries / 120 maximum stamina**
(`ten_from_power_on_guarded_20261111/statue_day5_to_day6.json` under
`recordings/berries/`), one initial load and zero mid-run loads, writes or
assists. Six relevant source/planner ROM checks pass (102.99 seconds);
2,719 offline tests and all eight docs checks pass.

The power-on prefix in `recordings/berries/ten_from_power_on_guarded_20261111/`
retained four berries / 140 maximum stamina, bought the Spring D23 snow-flower
seed, and raised two adult hens. It stopped after the first Fall hen trade
spent its source without eating the reward. The natural Fall D1 checkpoint
predates that loss. The animal-shop return now clears the west gate horizontally
before descending, and the scheduled hawker pickup tracks its reward across the
scripted walk (192 pixels instead of the generic 64). The actual Fall D1 trade,
the older trade fixture, and an ordinary cow purchase all pass native ROM checks.

Stump collection also has a calendar fallback after the four early berries;
this run had no wood-gathering phase before Fall. Its actual Summer D23 pin
now passes the native scheduled-stump check.

The first continuation, `recordings/berries/ten_from_fall1_20261111/`, exposed
the herd morning leaving the hawker west of the fence; the isolated score plan
had approached from inside. The scheduled picker now closes dialogue and clears
the gate before chasing the reward. The full herd day retains **5 / 150** at
Fall D2 (29.40-second native regression; zero writes or mid-run loads).

The gate continuation in `recordings/berries/ten_from_fall1_gate_20261111/`
retained the stump and hawker berries (6 / 160) and the Harvest Festival berry
(7 / 170). On Fall D20 the multi-day festival fallback took control before
the egg hunt could start. The fallback now respects scheduled berry festivals,
and the egg phase uses the existing `ends_day` contract so its house pickup
survives the native overnight transition. The actual herd-day regression now
wins the berry prize and reaches Fall D21 at **8 / 180** (16.39 seconds,
zero writes or mid-run loads). An older cash pin drops the hawker reward outside
the fence, so its picker skips the gate detour; all four hawker regressions pass.

The completed continuation is `recordings/berries/ten_from_egg20_20261111/`,
starting at `PR_tenberries_gate_20261111_Y1_Fall_D20`, with daily prefix
`PR_tenberries_egg_20261111_`. Its morning ledger confirms **8 / 180** at
Fall D21 with both the prize arm and eat flags. `source.json` records its parent
checkpoint hash and frozen code. The successful lineage begins at power-on,
replays Fall D1 after the hawker repair, and replays Fall D20 after the festival
repair. This is not an uninterrupted power-on result and stays out of STATUS.md.

The snow flower is eaten on Winter D1 (`8->9`, 6,625 phase frames), with the
seed consumed and **9 / 190** retained on D2. The first snowstorm opens the
cave pocket by D8; that day's cave phase eats the last berry (`9->10`, 2,784
phase frames). Both D9 and D10 morning states retain **10 eaten / 200 maximum
stamina**. The continuation finishes at D10 after 475,962 frames, with zero
bad days, rescues, RAM writes, assists or mid-run loads; it has one initial
checkpoint load.

`collection.json` holds all 20 morning receipts. Independent verification
re-read every saved state's RAM, checked both parent checkpoint hashes and all
423 frozen runtime file hashes, and loaded the final D10 state into the ROM
for 120 neutral frames. Nine permanent source receipts plus the consumed
snow-flower seed agree with ten eaten. `verification.json` preserves the
state/report/manifest hashes and audits for all three used segments. Resume pin:
`PR_tenberries_egg_20261111_Y1_Winter_D10` (with planner lineage); the final
snapshot is also `PR_tenberries_egg_20261111_Winter10`.

The Winter D1 and D8 natural morning pins now have full herd-day ROM
regressions that require the expected count, source receipts and maximum
stamina after sleep. Both pass (134.08 seconds), and all eight docs checks
pass. Results are in `winter_overnight_verification.log` and
`docs_verification.log` beside the continuation report.

Next action: replay the repaired route continuously from power-on through
Winter D10 before promoting the ten-berry result into STATUS.md or composing
it with a new ending score.

## Power berry route repair (2026-11-11)

The 851-point checkpoint has six eaten berries but seven spent source/eat receipts
(`$7F1F64 & 0x1E00`, `$7F1F66 & 0x6000`, `$7F1F6A & 0x0040`). At least one
spawn was lost before this checkpoint. Later code cannot respawn it. Collect all
ten from an earlier lineage, before the lost spawn and the Spring D23 seed seller;
keep the twelve-cow Winter hay repair described below.

Repairs in this pass:

- Hen pickup abandons a stand blocked by a hen instead of centering there until
  timeout, and tries another stand after an unregistered pickup. Ground berry
  pickup closes the final gap and pushes into the sprite before pressing A:
  proximity alone does not refresh the ROM's contact flag.
- The snow flower uses the live reward position and summit floor. The calendar
  buys the 2,000G seed on Spring D23 even when Eve no longer needs the dance,
  reads the permanent purchase receipt (`$7F1F6E & 0x0800`), and schedules its
  Winter planting while the shelf or backpack still owns the seed. Empty Winter
  farms no longer skip that trip. The summit now has an explicit native return.
- The seed seller's day excludes other berry trips. Bounded festival watering
  does not splice a spa detour after its noon cutoff. Harvest/Egg Festival trips
  defer optional feather and wife-gift errands, while animal care stays first.
  The fish deadline allows the measured camp-to-gate walk through 13:xx; the
  festival entrance still enforces the ROM's 15:00 close.
- Egg Hunt waits for the mayor's next question to close and its new colour to
  load, including a repeated colour. It requires a new prize bit to call a win.
  A second win that awards the turtle no longer chases last year's eaten house
  berry. Both orders of the two prizes have native component regressions.
- Shelf checks use the WRAM accessor, including the emulator's 0x4000 offset.
  This fixes seed ownership and prevents repeated plain-hammer fetches after
  the gold hammer replaces it. The gold-statue pickup remains experimental
  (`HARVEST_GOLD_STATUE=1`); collect the statue with the plain hammer early.

Native receipts are under `recordings/berries/*20261111.log`; regressions are in
`tests/test_berry_recovery.py`, `test_power_berry_calendar.py`, and the existing
source/planner ROM tests. These prove components and checkpoint composition,
not a continuous ten-berry campaign or a new ending score. Fishing still has a
native random bite/catch window; an unlucky dated attempt can miss closing.

Final validation on the repaired code: **26 berry ROM checks pass** (all source,
planner and recovery cases, 977.04 seconds); **2,717 offline tests pass** in the
legacy layout and **2,713** in blocks (32/36 skipped); **8 docs checks pass**.
The second-prize cases declare edited starting prize flags as research fixtures;
their hunt inputs are native. The snow purchase/plant/return/night receipts use
their natural checkpoints. Compile checks and `git diff --check` also pass.

The required fresh power-on benchmark was attempted with frozen code and stopped
on Y1 Spring D3 after the Day 2 `CLEAR_PLOT` approach at `(201,526)` failed:
68,646 frames / 501.5 seconds, zero loads/writes/assists/rescues. It is not an
annual comparison. Report, both scorecards and the old completed-year comparison
are recorded in [Y1_BENCH.md](Y1_BENCH.md#berry-calendar-power-on-prefix-2026-11-11).

Next action: run the repaired calendar from the early Spring lineage through
both Fall D20 hunts and the Winter snow flower, then require count 10 and maximum
stamina 200 before publishing an all-berries result. Eve's final-day coop-exit
failure in the 851 run also remains a separate replay target.

## Twelve cows and the Harvest Festival berry (2026-11-11)

Owner priority: power berries, twelve happy cows, and relationship composition. This section supersedes the eight-cow
target and disabled Harvest Festival trip in the older handoffs. The checkpoint ending now scores 851; a fresh
continuous power-on result is still required.

- `score_herd` now targets **12 cows** with a **420-tile pasture ceiling**, keeps two hens until the final flock window,
  and retains the Year 3 cash floor and Fall hay gate. `cash` and ordinary `score` keep their cow targets.
- Natural acquisition: the frozen continuation `recordings/y23_sweep/hw12b_20261111_s1.{json,log}` starts at
  `PR_f6final_s1_Y2_Spring_D2` and reaches **12 initialized cow slots by Y2 Spring D19**, 270 planted grass tiles,
  120 hay and 19,520G. Three older cows have happiness 255; the others are still growing toward 200. Pins use
  `PR_hw12b_20261111_s1_`. All twelve reach happiness 200 by Fall D9 and 255 by Fall D27; the subsequent Winter
  starvation failure is recorded below. These are mid-campaign receipts, not 336 points retained at the ending.
- Native twelve-cow day: `recordings/y1_bench/native_twelve_day_20261111.json`, D19 to D20, **23,415 frames**, all twelve
  retained, hay 120 to 115, 15 crops shipped, Eve 494 to 496, no failed days or rescues. Audit: one initial checkpoint
  load, zero mid-run loads, RAM writes or assists. This proves one mixed adult/young herd day, not fifty days of care.
- Twelve-adult component checks: `recordings/barn_fuzz/score12_20261111/`, **8/8** pass all feed/talk/brush/milk/exit
  receipts, worst 34,595 frames and no escapes. These scenarios edit their starting herd as declared research;
  they do not establish natural twelve-adult arrival or campaign survival.
- Checkpoint portability: season/day resume now reads the state beside its local lineage file. The original absolute
  path referred to a removed temporary checkout and prevented this continuation from starting. The lineage file on
  disk is preserved; the native D19 replay above also exercises the corrected resume.

The **Harvest Festival exit is repaired**, and score strategies may attempt its fish offering again. From the natural
post-berry pin `LN_berries_20261103_harvest_festival_Y1_Fall_D12_after`, `FestivalEscapeTask` leaves in **6,740 frames**,
and the next morning settles by frame **7,437**, retaining the berry count and maximum stamina. Eve receives **+30**
for the dance, in addition to her first question; each other girl gains 8. The controller reads the native conversation
bits, clears the mayor from the narrow entrance, talks diagonally to the dance host, then chooses Eve. The host stand
needs exact alignment; a two-pixel tolerance misses her. `HARVEST_HARVEST_FESTIVAL_BERRY=0` disables the trip for comparison.
The social cap now reserves **8 points for at most one future Harvest Festival visit** (the berry receipt retires it),
with final-year top-ups releasing an unused margin. This prevents the restored berry route from silently consuming
the affection window's slack.

Native receipts: `recordings/y1_bench/harvest_exit_20261111_after.log` and `harvest_festival_day_20261111.json`. The full
composer reaches Fall D13 in 7,450 frames, zero rescues/writes/mid-run loads. It still labels the festival's automatic
overnight as one bad day; the journal classification needs a separate repair. The frozen herd run predates this berry
and social-margin change, so its ending will validate herd growth, not their combined campaign effect.

Validation: **2,701 offline tests pass** (32 skipped), the four existing native courting receipts pass, and the new
festival exit/next-morning ROM regression passes. The berry-planner check has four passes and two existing failures:
the hawker route times out picking up the hen, and the fishing draw reaches the Harvest Festival deadline too late.
The optional three-morning bought-flower/family test also remains red: on the second morning `GIVE_WIFE` reports
`wife not home` and Eve loses the nightly 2. Logs are `berry_planner_20261111.log` and
`relationship_chain_20261111_before.log` under `recordings/y1_bench/`. Do not treat isolated conversation receipts as
proof of that daily family chain.

### Winter herd failure and repair (2026-11-11 continuation)

The `hw12b_20261111_s1` run was stopped at **Y3 Spring D7**, 4,790,861 frames, after the herd fell from twelve to three.
Its report was preserved (one initial checkpoint load; zero writes, mid-run loads or assists). Read-only morning-pin
census: `recordings/y1_bench/twelve_cow_milestones_20261111.json`. All twelve had happiness 255 on Fall D27, but Winter
started with **115 hay**. Hay reached 3 on Winter D9, sickness began D10, and deaths followed. Do not promote the
twelve-cow strategy from acquisition and component-care checks alone.

Cause: **1,028 mature cover-grass tiles were ignored** while the small pasture regrew. The pasture errand and cutter
only considered `PASTURE_BOUNDS`, afternoon cuts stopped at a few days' hay rather than the Fall reserve, and cuts
did not receive stamina refills. After expanding the bounds, the planter's 30-tile approach limit still skipped distant
grass; preserving the cutter's `stamina_low` reason and allowing full-field cutting walks were also necessary.

The repaired errand cuts the cover field for score strategies, uses the Fall stock target after chores, stops at its
hay goal, and refills only while making hay progress (the existing idle-retry guard remains). The natural Fall D28
component replay banks **157 to 658 hay** through native swings and spa visits. Commands and audit output:
`recordings/y1_bench/winter_hay_20261111_{before,after}.log`; regression in `tests/test_livestock_econ.py`. The component
isolates cutting and omits the day's chores; the full planner must feed before that work and retain happiness overnight.

The corrected full daily-planner validation is **`winter12c_20261111_s1`**, starting from the natural
`PR_hw12b_20261111_s1_Y2_Fall_D28` and targeting Y3 Spring D2. Logs/report: `recordings/y23_sweep/`, morning pins
`PR_winter12c_20261111_s1_*`. The earlier `winter12`/`winter12b` attempts were stopped during setup/diagnosis; they
are not winter-survival receipts. Latest offline gate: **2,702 pass**, 32 skipped.

**Winter validation passed:** Y3 Spring D2, **all twelve cows alive, happiness 255 and status 0x09 in every slot**,
224 hay remaining, 310,300G. 1,116,390 frames / 1,357.8 seconds wall: **822 fps, 13.7 times real time** (5.17 emulated
hours in 22.63 minutes). Audit: one initial checkpoint load, zero mid-run loads/writes/assists and zero rescues.
Two automatic festival/birth overnights are still classified as bad days. The cut-only regression banks 157 to 658
hay in 99,044 frames without advancing the date. Barn chores account for 687,339 frames (61.6%); optimize their
route/retries before attributing runtime to the planner file sizes.

**Ending validation passed: ROM score 851**, credits index `0x25`, Y3 Summer D30. `ending12_20261111_s1` starts from
`PR_winter12c_20261111_s1_Y3_Spring_D2`, uses frozen code, `score_herd`, zero jitter and no injected faults. It takes
**3,708,687 frames / 4,370.1 seconds wall (72.8 minutes, 849 fps / 14.1 times real time)**. All twelve cows finish
with happiness **255**, status **0x09** in every slot; cow count and happiness pay the full **336/336 points**.
Twelve chickens pay 36/36. Wallet 494,490G, hay 60. Audit: one initial checkpoint load, zero mid-run loads, writes,
assists or rescues. Two required-phase failures remain: Spring D4 coop navigation and D6 cure map lock.

This is **82 points above the previous 769 checkpoint ending**, with a different start and route; it is not an
isolated A/B or a fresh continuous power-on proof. `STATUS.md` is unchanged. Final report/log and morning pins use
the existing sweep locations; scorecard: `recordings/y23_sweep/ending12_20261111_FINAL.md`.

| Remaining term | Points | Gap |
| --- | ---: | ---: |
| Power berries (6/10) | 30/50 | 20 |
| Five relationships | 154/155 | 1 |
| Shipped crops | 62/124 | 62 |
| Player happiness | 22/31 | 9 |
| Children | 16/32 | 16 |
| Clock/turtle shell | 22/43 | 21 |
| Field development | 31/50 | 19 |

Maria 498, Ann 500, Nina 496 and Ellen 496 finish in the scoring window. Eve finishes **494**, losing the final
point to the last-night window; her D29 morning was 496. Do not call the complete relationship term landed yet.
The earlier berry exit repair is present, but this late checkpoint cannot recover its missed Fall dates or the
snow-flower seed. Hawker pickup, fishing deadlines, snow flower and reliable Egg Festival prizes remain open.

Next: reproduce Eve's final D29 gift from `PR_ending12_20261111_s1_Y3_Summer_D29`, then work the missing berries
from an earlier seasonal checkpoint while preserving the now-verified twelve-cow winter budget. A fresh Year 1
run is still needed to measure all changes together.

Current handoff: [clean Year 1 / headed Year 2](tasks/rr-20w-run16-defects.md).
New Year is ROM verified in 3,396 frames without its old evening wait.
The D2 shed/rock handoff and the spa-return stale-map escape are fixed; nearby
ROM replays clear every debris type and plant the free grass bag. The revised
[Day 2 checklist](FARM_CLEAR_D2.md) also requires eight wet potatoes and shipping
before 17:00. Fresh power-on acceptance passed: zero debris, eight wet potatoes, nine grass
tiles, zero loads/writes/assists (374,794 frames; `recordings/d2_perfect/power_on.json`).
The post-D2 continuation now reaches Summer D1 with 127 native potato
shipments and 7,470G in 562,850 frames. The first eight potatoes ship on D8
and credit 640G overnight. It has zero rescues or day-floor trips, but one
required watering failure on D30. Detailed gates and frame rates are in
[Y1_BENCH.md](Y1_BENCH.md#spring-frame-measurements). Nine rings with spa
watering now end Spring at 9,830G and Fall D1 at 36,130 net worth from the
same pin (below). No completed Year 1 save has been accepted for the headed
Year 2 run.


## Courting calendar from the Year 1 Fall D1 pin (2026-11-08)

Owner ask: start at Year 1 Fall D1 from a pin, court the girls strategically, build a Sunday church policy that chats and
answers questions, add other long-run score levers; goal a full run above 900; do not run the whole game. The four girls who
are not the wife were 8-68 hearts (0-4 points of 124) at the end of every run so far. ROM decode and receipts:
[F11 section 4](perfect_rancher/research/F11_social_levers.md). Start pin `PR_fall1base_Y1_Fall_D1` (a copy of the power-on
`PR_e2et1_Y1_Fall_D1`: 9,200G, 2 cows, 2 hens, clock owned, Eve 250, unmarried). Replays are one draw each, frozen tree copies
(`scripts/y23_sweep.sh` with `UNTIL`), reports `recordings/y23_sweep/f*_s1.{json,log}`, day pins `PR_<tag>_s1_*`.

Built (all behind `HARVEST_SOCIAL_CHURCH`, on for the score family through `RunStrategy.court_girls`; `cash` unchanged):

- `planner/social_calendar.py`: one courting stop a day from RAM. Clear Sundays: `ChurchCourtTask` (Maria and Ann +4 each per
  church entry, one entry per +4 the neediest lacks, at most 30; fewer than 3 wait for the final year). Fall and Spring weekdays: `TownCourtTask` for Ellen, Summer and
  Winter weekdays for Nina (+8 per town reload through the shop door beside her; 8 loops in Fall/Winter, 2 in crop seasons and
  only for a girl 80+ short, except the last year). Caps: 199 until the wedding (the 200-heart event), then 500 minus the
  Flower Festivals (15), New Years (10) and Egg Festivals (10) still ahead, so the scheduled one-shots land the girl in 496-511, not past 511.
- Star Night: with Eve the wife the planner picks her scene (`festival_escape.star_night_menu_press`, `StarNightSceneTask`):
  Eve +30 instead of -30 each Winter D24, Ellen no longer gets the default +30.
- `farm_to_church` was broken (pixel-stuck at (278,128) for 18k frames); now the D1 Maria road. `church_to_farm` reversed.
- The Harvest Festival berry is off by default (`HARVEST_HARVEST_FESTIVAL_BERRY=1`): when the fish bites the day wedges on
  festival map 0x09 (no edge exit, clock 07:07). It ended the pre-change control at Fall D12 after 8 rescues.

ROM receipts (zero RAM writes and loads; `tests/test_social_rom.py`, 3 passed):

| what | pin | result |
| --- | --- | --- |
| Sunday church, 20 entries | `PR_f1ctl_s1_Y1_Fall_D3` | Maria and Ann 8 -> 88 each, 42,499 frames, clock 07:11 -> 07:13 |
| church inside the planner, 30 entries | same Sunday (`f2on`) | +120 each, 61,588 frames, home on the farm |
| weekday Ellen, 6 loops | `PR_f1ctl_s1_Y1_Fall_D4` | 8 -> 56, 12,517 frames, house 06:03 to farm 11:08 |
| Star Night, Eve's scene | `PR_e2ec1_Y1_Winter_D24` | Eve 430 -> 460 -> 458 next morning (default path: 398, Ellen +30) |

Fall D1 to Winter D26 Year 1, same pin, one draw each (`f2off` control `HARVEST_SOCIAL_CHURCH=0`, `f2on` church + Star Night
only, `f3all` church + weekday town + Star Night; morning pins `PR_<tag>_s1_*`):

| arm | Maria / Ann | Nina / Ellen | Eve | cash | RAM-model score | rescues / bad days | frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f2off` | 8 / 8 | 8 / 38 | 438 | 50,650G | 296 | 0 / 2 | 933,489 |
| `f2on` | 472 / 472 | 8 / 18 | 386 | 50,350G | 342 | 0 / 2 | 1,085,900 |
| `f3all` | 472 / 472 | 464 / 464 | 486 | 48,800G | 407 | 0 / 2 | 1,346,585 |
| `f4all` (caps 450, festival-eve skip, walk home after a berry) | 452 / 452 | 416 / 448 | 486 | 48,350G | 400 | 0 / 3 | 1,345,866 |
| `f6final` (final caps, stop before the cap), Winter D26 | 440 / 440 | 440 / 440 | 440 | 49,850G | 396 | 0 / 3 | — |
| `f6final`, Year 2 Spring D2 (New Year done) | 450 / 440 | 440 / 440 | 464 | 55,250G | 404 | 0 / 3 | 1,385,831 |

- The church took Maria and Ann 8 -> 128 -> 248 -> 368 -> 472 over the four Fall Sundays (30 entries, ~61k frames each).
  Ellen hit her cap in seven Fall weekdays (+64 a visit), Nina in Winter (+40-64 a visit, one visit failed after one talk and
  went home cleanly). All four end Year 1 at 29 points each (464-472), from 0.
- Star Night inside the planner: Eve +28 the next morning in both social arms; the control lost 32 and Ellen took +30.
- `f2on` lost a hen on Fall D13 (the hawker hen trade handed the hen over, then pixel-stuck at the west gate (24,448): no berry)
  and its Eve gained ~2 a day on one egg instead of ~6 on two. That is the trade's bug, not the courting.
- Cost: -1,850G by Winter D26 (-3.7%) and +44% frames, no rescues. Unplanned gains seen: the Egg Festival roam gave Ellen +10.
- `f4all`'s weather moved the proposal from Fall D9 to D16, and on the unmarried D10 Sunday the church stopped only once
  hearts reached the 199 cap: Maria and Ann 196 -> 200. Their 200-heart events played (`$7F1F6A` 0x1000 on D14, 0x4000 on D17)
  and Ann paid nothing on the D17 Sunday; the flags then moved to 0x2000 / 0x8000 and both paid again (an event state machine,
  not a permanent loss). Fixed: every stop checks hearts + gain <= cap (`test_church_stops_before_a_question_would_pass...`).
- `f6final` is the final tree except the 3-entry church minimum (one 4,677-frame visit for +4 each happened). Unmarried until
  Fall D16-17 in this timeline too; the D10 church stopped at 196 (17 entries). New Year paid Maria +10 (the margin's
  reason); Ann, Nina and Ellen were not paid there. Every girl sits at her Year 1 cap (27-28 points each); the Flower
  Festivals, the Year 2 Egg Festival, New Year and the Year 3 top-ups are what take them to 496-511.
- These arms ran with the first caps (470, no New Year or Egg Festival margin). At 472 a girl can wrap past 511 from the
  festivals still ahead, so do not branch Year 2 from their pins; the final caps leave room for all of them (`f6final`).

### Years 2-3 to the ending (2026-11-08)

`HARVEST_STRATEGY=score y23_sweep.sh PR_f6final_s1_Y2_Spring_D2 g<n> <seed>` with `--until-ending`, frozen tree, zero RAM writes,
one initial load, no assists (`recordings/y23_sweep/g{1,2}_s*.{json,log}`). Both draws ended at Y3 Summer D30 (credits index 0x25).

| draw | jitter | ROM ranch score | rescues | bad days | cash | cows / hens | frames | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g1_s1` | 0 | **663** | 0 | 9 | 138,790G | 4 / 10 | 6,415,411 | 2h12m |
| `g2_s2` | 90 | **662** | 0 | 13 | 186,030G | 4 / 11 | 6,134,376 | 2h07m |

The best power-on e2e before the courting work (`recordings/e2e/v3n`) ended at 357. Terms (`g2`; `g1` within 2 points on every term):
money 78/78, chickens 36/36, house 32/32, marriage 32/32, **hearts 154/155** (Maria, Ann, Nina, Ellen 496-500, Eve 490), cows 12/36,
power berries 30/50, shipped crops 88/124, happiness 21/31, children 16/32, special items 22/43, pasture 41/50, cow happiness 100/300.

- Star Night picked Eve's scene (`[STAR_NIGHT] choosing index 2 (eve)`) on Y2 Winter D24 and the baby day passed: no wedge.
- The Year 3 top-ups landed every girl in 496-511 in both draws; no counter wrapped past 511. Eve is 2-6 short (494 in `g1`, 490 in `g2`): her
  hearts come from the nightly -2 / gift / egg loop, not the courting stops.
- Bad days (`g1` 9, `g2` 13): zero rescues, but each lost a required phase for the day. They are real defects, not noise:
  - **Spa soak timeout strands the farmer (4 days, same pixel in both draws).** Y3 Spring D12 and Y3 Summer D9, D10 (`g2`) and D11
    (`g1`): `CROP_WATER` runs out of stamina after ~52 tiles, splices `HOT_SPRING_STAMINA`, the soak ends `soak timeout 132/160` after
    3,600 frames, the farmer is still in the pond at (661,198), the required `CROP_WATER` fails its map lock (`have=0x10 need=0x00`)
    and the day goes home with the rest of the field unwatered. These are the watering days the 36-38 missing crop points cost.
    **Fixed 2026-11-08 (`BathCrossing`, `tasks/hot_spring.py`).** Cause, found by saving the ROM at the lip and re-running the soak from 30
    clock/idle offsets (1 stalled; replaying the pin day under 7 jitter seeds never did, because the barn timing differs): the bath was an
    open-loop button script (4+14+20 frames per half-cycle). A left jump from the east shelf can land 12 px past the west lip (605, not
    617); the script then walked back and pressed A *before* the farmer stood flush against the water, the press became a tool swing
    (`player_action` 4/5), and every later cycle repeated it (the log shows 30 cycles with no jump). Pressing A from a flush start always
    jumped. The loop is now closed: walk toward the water until x stops changing, hold B+dir+A only there until `player_action == 3`,
    hold dir until the jump ends, settle, and choose the next direction from x (west of 624 goes right, east of 640 goes left). Four
    presses in a row without a jump end the soak as `bath stalled` (not the 3,600-frame timeout). A soak that ends unrestored now
    crosses back, walks home and only then reports failure, so a failed optional spa phase no longer leaves the farmer on the mountain.
    Measured on the repro pin: 42 clock/idle offsets all fill 0 to 160 (the old script: 1 of 30 stalled), in 762-860 frames against 1,188.
    Offline: `HotSpringBathLoopTests` (a pond model with the overshoot and the swing-eats-the-press rule; the old code gets 2 jumps and 80
    swings from it). Not yet seen in a full run: `hw4`/`hw6`/`hw8` below carry it.
  - **`COOP_CHORES` fails** (`not in coop tilemap=0x00`, or `coop timeout`/`coop_nav_feed_bin timeout`): `g1` Y2 Summer D3, D19, Y3
    Summer D26-27; `g2` Y2 Spring D27, Y2 Summer D5, D19, Y3 Summer D22, D25, D26, D29. Mostly Summer, both years.
  - **ROM advances the day mid-plan** (`overnight`) on Y2 Fall D20, Y2 Winter D24 (Star Night, the baby day) and Y3 Spring D1 in both
    draws: the plan loses the rest of that day. Expected on the dated scenes; the Y3 Spring D1 and Y2 Fall D20 ones are unexplained.
- Caveat: the Year 1 half is the pin's power-on timeline (`PR_e2et1`), not a fresh power-on of this tree. A fresh `e2e_run.sh` with
  the score strategy is the final proof of "power-on to credits clean"; the 3 h lean run is cheap on this machine.

## Revised plan to 900 (2026-11-08)

Gap to 999 is 336 points. Cash is not the constraint: 139k-186k at the ending against the 99,840G that maxes the money term, so
40-86k G is free to spend. Ranked by points per unit of risk:

| lever | gap | what it takes | stance |
| --- | ---: | --- | --- |
| Herd + cow happiness | 224 | each cow is 3 + 25 points; 4 cows give 112, 12 give 336. Owner: test 3/5/8 cows with medicine, not 12 | DONE for 6 and 8: `hw6` 708, `hw8` 769 (results below). Next arms are 10 and 12 cows if the owner wants them |
| Spa soak timeout on heavy watering days | part of the 36 below | closed-loop bath, see "Years 2-3 to the ending" | fixed 2026-11-08; confirm in `hw4` (bad days, `soak ended unrestored` count) |
| Shipped crops | 36 | corn 418, tomato 289, potato 371, turnip 343 against the 496 floor; stop at 496, do not pass 511 | spend the surplus on seed and earlier harvest |
| Special items | 21 | turtle shell (Egg Festival turtle test is 3 of 8) | fix the festival turtle ROM test |
| Power berries | 20 | 6 of 10 collected; spent spawn bits cannot recover | calendar in `power_berry` memory/F6 |
| Children | 16 | second child needs the wife bit by Y2 Fall D9 and hearts >= 650 (F4) | decode before building |
| Happiness | 10 | dumpling/rice-ball seller (+1/200G, +3/300G), seller not located | surplus gold converts here |
| Pasture | 5-9 | 2,950 counted tiles | whole-field cover, A/B in `FARM_CLEAR_D2` |
| Eve hearts | 1-2 | 490 -> 496 | gift cap vs nightly -2 |

Reaching 900 needs +237 from 662. Twelve cows at full happiness alone is +224, so 900 requires either the full herd or an eight-cow
herd with nearly every small term. Eight cows is +112 and the small terms add up to about +110, which lands near 885: treat 900 as
the eight-cow-plus-everything case and measure before promising it.


**Why the herd stayed at 3-4 cows: a plan-order bug, fixed 2026-11-08.** `HOUSE_WOOD` (a mountain chop that ends at (704,726)) was inserted right after the
day's first `EXIT_TO_FARM`; on days with the wife-gift chain the next phase was `NAV_TO_ANIMAL_SHOP`, which failed `expected tilemap 0x00, got 0x10` and
deferred everything behind it: the cow buy, `BUY_MEDICINE`, and on Y2 Spring D2 the barn chores. `g1` and `g2` deferred the cow-buy chain 8 times in two
years and bought one cow; the first herd arms (`hs6`, `hs8`, `hc4`, aborted at Y2 Spring D15-D20) deferred it 4 times in 15 days and bought none, and in `hc4`
the D2 night was unfed, two cows rolled sick (D2 happiness 255 -> 235/216), `BUY_MEDICINE` then failed the same way, and both died on D9. Fix:
`apply_house_phases` adds an `EXIT_TO_FARM` after an early `HOUSE_WOOD` unless the next phase walks from anywhere (`_FROM_ANYWHERE`). Test:
`test_lumber_walks_back_to_the_farm_before_a_shop_or_barn_phase` (red without the fix). Earlier herd failures that blamed hay and barn stalls may have
been this too; the old draws were not re-read for it.

Herd A/B relaunched 2026-11-08 with both fixes (all from `PR_f6final_s1_Y2_Spring_D2`, jitter 0, seed 1, `--until-ending`, frozen trees `/tmp/y23_sweep_<tag>`):

| tag | strategy | cows asked for in Year 2+ | what it isolates |
| --- | --- | ---: | --- |
| `hw4` | `score` | 4 (default) | the two fixes alone against `g1` (663) |
| `hw6` | `score_herd` | 6 (`HARVEST_COW_TARGET_LATER=6`) | +2 cows, 187-tile pasture |
| `hw8` | `score_herd` | 8 | +4 cows, 234-tile pasture (cap 240) |

**Results (2026-11-08, one draw each, all CLEAN, 0 rescues, ended Y3 Summer D30):**

| run | cows / hens | cash | ROM score | cow happiness | crops | player happiness | hearts | dev | spa failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g1` (before both fixes) | 4 / 10 | 138,790G | 663 | 100 | 86 | 21 | 153 | 45 | 4 |
| `hw4` (`score`, fixes) | 4 / 10 | 129,680G | 643 | 100 | 70 | 17 | 154 | 44 | 1 |
| `hw6` (`score_herd`, 6) | 6 / 11 | 235,300G | **708** | 150 | 75 | 20 | 155 | 44 | 0 |
| `hw8` (`score_herd`, 8) | 8 / 11 | 312,410G | **769** | 200 | 81 | 22 | 155 | 41 | 0 |

- Each extra cow is +28 as modelled (3 + 25): every cow ended at happiness 255, none sick, none dead. Gap to 999 is now 230 (`hw8`). Reaching 900 needs the four cows
  still unbought (+112 -> ~881 from `hw8`) plus ~20 from the small terms; `hw8` also shows what is still short: crops 43, berries 20, special items 21, children 16.
- The herd also pays for itself: 8 cows end with 312k cash against 130k (the `score_herd` floor of 110k in Year 3 never bound), so 10-12 cows are affordable.
- Crops are the unexplained loss: 86 / 70 / 75 / 81 across the four runs with no trend in cow count, so it reads as single-draw variance (about +-8) plus whatever the
  extra D2 cow does to early spending; not isolated. `hw4` (the pure control on the fixed tree) is 20 below `g1`, mostly tomatoes (215 vs 299), corn (367 vs 433) and turnips (253 vs 359).
  Needs a second draw of each (`JITTER=90`, seed 2) before anyone attributes it to the fixes.
- Spa: the old logs show 4 stalled soaks per run, the new runs 1 (`hw4`, Y3 Summer D14: four flush presses at x=617 with stamina 0 never jumped; not reproduced from the
  pin, a morning-pin replay skips the soak entirely). Every missed press now logs `[SPA] press N never jumped: pos= act= player_state= held_item= input_lock= near=[...]`.
- Barn: `COW_CHORES unfinished` (shift limit, re-entry) fired about once a day at 8 cows and recovered every time; hay stayed above 40 and the cut kept up with 12 hens.

`score_herd` is required for the arms: only it sizes the pasture and the Fall hay gate for the planned herd (`grows_herd()` reads the strategy field, not
the env override), and it adds the 110,000G wallet floor on Year 3 buys. Read: `.venv/bin/python -m harvest.tools.final_score
recordings/y23_sweep/hw4_s1.json recordings/y23_sweep/hw6_s1.json recordings/y23_sweep/hw8_s1.json`; `scripts/pin_status.py hw8` mid-run. Read cow count
by date, hay, sick/death lines and rescues, not only the ending score. `hs6`, `hs8`, `hc4` logs stay in `recordings/y23_sweep/` as the aborted prefix.

Remaining, in order:

1. DONE 2026-11-08: Years 2-3 from `PR_f6final_s1_Y2_Spring_D2` finished CLEAN twice (table under "Courting calendar"). Next is the
   herd A/B in "Revised plan to 900" below. No girl-overshoot repair exists yet (a weed -2..-4 or poison mushroom -8 gift);
   the festivals never overshot in two draws, so it stays unbuilt until one does.
2. Nina's Sunday mountain and Ellen's Saturday question are decoded, not wired (no open door near Ellen on Saturday).
3. The hawker hen trade loses the hen when the farm return pixel-sticks at the west gate (24,448) (`f2on` Fall D13): hen gone,
   no berry. Pre-existing; worth a fix (3 points a hen, -30 happiness).
4. A Harvest Festival exit (map 0x09) would return a +5 berry and the festival's +30 girl scene (group 0x26).
5. Player happiness: the dumpling/rice-ball seller (actions 0x51/0x52, `$81F5B0`/`$81F6D7`: +1 for 200G, +3 for 300G, no flag in
   the handler) could spend Year 3's surplus gold on the 31-point term; who and where the seller is is not decoded.
6. Red on the tree as found, not from this pass (each re-run on the frozen pre-change copy): `test_power_berry_planner_rom.py`
   hen trade (pickup timeout), Harvest Festival fish (too late) and Egg Festival turtle (3 of 8; the other 5 pass), and
   `test_relationships.py::test_three_native_mornings_grow_nina_without_abandoning_wife_gifts` (Eve 490 -> 488, the gift cap
   against the nightly -2). One Winter D2 Nina visit found no Nina sprite in town and went home (1,714 frames).
7. Bigger levers for 900+ stay as listed below: 12 cows at 200 happiness (+200), whole-field cover (+12), crops (+40), berries.

## Ranch Master rule coverage (2026-11-05)

Owner asked that every published Ranch Master rule be embedded, tested and in this plan. Sources, read 2026-11-05: the Harvest Moon
Wiki page `Ranch_Master_Score_(HM)` (the owner pasted its text; an automated fetch answered HTTP 402) and the RetroAchievements
guide chart "Breaking Down Ranch Master". Both agree with the ROM decode in `MASTER_PLAN` §2 line for line; the wiki's "estimated,
unverified" rating tiers equal the ROM's `RATING_TEXT_BANDS` exactly. Tests: `tests/test_ranch_master_published_rules.py` (52, offline:
each rule against the oracle, plus the policy caps) on top of `test_perfect_ending.py` (every term on the ROM calculator, `-m rom`).

Rules only the wiki page states, and where each stands:

- **Money wraps above 655,350G** (a 655,400G wallet pays 0; the next band opens at ~755,200G). Tested on the oracle; no run comes near
  it (cash runs end at ~255k).
- **A pregnancy at the end counts as the child** (16 points, "even for a single day"). The second child needs the wife at 650+ hearts
  at some point, then a window repair back to 496–511 (a bad gift is −5 to −8, each house entry after she is in bed −2). Not worked in
  Year 3: Eve stops at `EVE_GIFT_TARGET` 502. Open lever, +16 points, high risk to Eve's own 31.
- **Cows count while unborn; chicks count, eggs (incubator or ground) do not.** The flock gate counts incubating eggs
  (`flock_birds`), the score does not, so an egg still in the incubator on the last day is worth nothing.
- **Immature cows are not scored for affection**; a cow bought by Spring D10 is adult after 21 nights, well before the end.
- **Stamina must be exactly 200** (the Happy Family Ending needs it too); the calendar stops at ten berries (tested).
- **Clock:** only before the first Fall of Year 1; **charm (turtle shell):** only through Year 2 Fall D20. Both are fixed at the Year 3
  start (the pin owns the clock, not the shell).
- **Crops sold to the peddler never count**; the bot declines him (`scene.peddler_decline_action`).
- **Development** pays per 2% (1% = 30–58 tiles, 2% = 59–88, 3% = 89–117); 2,950–2,968 tiles pay 50. Tilled soil does not count; a
  seed (even ungerminated), a crop or an undamaged fence does. Tested at every boundary.
- **Rating tiers:** 0–199, 200–399, 400–599, 600–799, 800–999. The 400 line is where the end text changes from "Try harder." to
  "Come on, you are so close!"; 800 is "the perfect rancher".

| Chart rule | Oracle / ROM test | Policy that must obey it (Year 3) |
| --- | --- | --- |
| Money 78 at 99,840G | `money_raw` cases | `field_cover.cash_floor_g` rises to 109,840 on the last morning; `late_cash_floor_g` 110,000 on `score_herd`; cover and cows spend only above it |
| Cows 36 (12), chickens 36 (12) | `num_cows`, `num_chickens` cases | `cow_target_later` 8 (`score_herd`), `final_flock_target` 12; nothing above 12 (tested) |
| Power berries 50 (10) | `max_stamina` cases | `power_berry_calendar`, target 10; Year 3 can still reach statue, cave (now any season), fish |
| Crops 31 each, 496–511, over 511 resets to 0 | nine-bit window cases | `ship_cap` 496 on every score strategy; `spring_committed_totals` reserves standing crops |
| Bachelorette 31 each, 496–511 | `*_hearts` cases | only the wife is worked; weeds (−2) and poison mushrooms (−8) are the chart's way down from an overshoot and are **not implemented** (no girl is near 511 in Year 3) |
| Cow affection 25 each (300) | `cow<slot>.happiness` cases | fed daily, brushed; a new cow needs ~50 care days, so the last useful buy is Spring D10 |
| House 16 each, children 16 each, marriage 32 | flag cases | all owned at the Year 3 start |
| Clock 22 | `incubator_flags & 0x4000` | owned at the Year 3 start (chart: "upgrade your house before end of Summer") |
| Turtle shell 21 | `family_event_flags & 0x1000` | Egg Festival of Year 1 or 2 only; Year 3 has no Fall, so it cannot be earned there (tested) |
| Development 50 ("cover the whole field") | `ranch_development` cases, 2,950-tile boundary | `field_cover` (Spring and Summer of Years 2–3) |
| Happiness 31 ("999") | `happiness` cases | ROM pays `h >> 5`, so 992 is enough; chart lists the sources below |
| Total 999; 994 for an all-girls-999 run | `test_all_five_girls_at_999_affection_make_the_994_run` | the 496–511 window gets 999 |

Happiness sources the chart lists (player happiness, 14 of 31 points at the end of `s4`): turtle shell +80, marriage +50, wife
anniversary gift +30 each, first child +50, second child +100, cow born +70, **buy a cow +20**, sell a cow −50, cow dies −100,
wife leaves −50 / returns +100. Measured on the Year 3 runs: each cow bought moved happiness by +20 (443 to 533 in `y3b2` over four buys and the +10 cave
berry), so a bought cow pays 3 (count) + 25 (cow happiness) + 0.6 (player happiness). The F7 ROM decode says the +20 fires
on the first cow payment only; the runs say every purchase, so F7 needs a re-read before it is quoted again.

## Year 3 score push from the `s4` start (2026-11-05)

Owner: best fully tested ending is `s4` at 404; start from Year 3 Spring D1 and change policy to beat it. Start pin
`PR_y3base_Y3_Spring_D1` (a copy of `PR_e2es4_Y3_Spring_D1`); branches come from the control's Spring D3 morning
(`PR_y3ctl_s1_Y3_Spring_D3`, after the long first cover evening). Launch:
`HARVEST_STRATEGY=<score|score_herd> JITTER=0 FAULTS= RATE=0 scripts/y23_sweep.sh <pin> <tag> 1` (reports in `recordings/y23_sweep/`).
These are pin replays (one draw each), not a power-on certification.

| run | what it is | ROM score | notes |
| --- | --- | ---: | --- |
| `s4` | full power-on, old tree | **404** | 4 cows, 2 hens, no berry, 3 development |
| `y3b1` | `score` + cave berry any season + flock from D1 | **472** | 12 hens (+30), development 41 (+38), cave berry (+5); hay hit 0 on Summer D23, cows sick, cow points 89 not 100, player happiness 9 |
| `y3ctl` | `score`, no changes | **488** (1 rescue) | no crop sown all Spring (below); 12 hens (+30), development 42, cow points 100, no berry |
| `y3b3` | `score` + shop-first + cave + flock from D1 | **483** Clean | crops 75 (+8: 204 turnips, no Summer corn or tomato shipped), development only 26 (-16 vs control), 164k gold left unspent |
| `y3herd`, `y3b2` | `score_herd` (8 cows), b2 adds cave + flock from D1 | **327**, **360** | herd collapsed: 8 cows to 1 and 2 (hay near 0 by Summer D5, unfed nights, sickness, deaths), player happiness 0 (a death is -100), 3 rescues in b2. b2 had read 507 on Summer D5 |
| `y3b4` | `score_herd` (8 cows) + shop-first + cave + flock from D1 | **552**, 6 rescues (not Clean) | 8 cows and 12 hens alive, cow points 166 (25,20,25,25,20,19,16,16), crops 76, development 19, wallet 181k. Rescues are barn-exit stalls (`directional transition timeout`, barn tilemap `0x27`, pos (1-2,19)). The only herd run that survived: its siblings collapsed |
| `y3c2` | `score`, every fix, shop-first **on** | **452** Clean | a cow lost (3 left), cow points 75, development 18 (-24), 185k gold unspent; shop-first is worse |
| `y3c4` | `score`, every fix, shop-first **off** (now the default) | **491** Clean | 4 cows at 25 each (100), 11 hens + chick (36), cave berry 5, development 38, crops 69, wallet 114,470G |
| `y3k5` | `score`, 5 cows (`HARVEST_COW_TARGET_LATER=5`), shop-first off | running, **~524 est.** on Summer D23 | 5 cows at 25/24, 11 hens, corn 284 shipped, development 36; read `recordings/y23_sweep/y3k5_s1.json` |
| `y3k6` | `score`, 6 cows | incomplete | wedged at the house door on Summer D2 (`return_home timeout phase=enter_house`, 8 rescues): the known dog/horse door wedge, not cows |

Defects found and fixed this pass (each is in the tree with a test):

1. **Cover starved the crop seeds.** The cover's grass trip (~3 game hours) then the grape trip pushed `BUY_SEEDS` to a shut 17:11 door
   (`exit_walk timeout`), so no potato or turnip was sown in Year 3 Spring. `HARVEST_SHOP_FIRST=1` puts the seed trip first and it works
   (72 turnip tiles by D8) but it is **off by default**: it cost more cover than it sowed (483 and 452 against 488 and 491 without it).
   A seed trip that does not displace the cover trip is the open repair (`test_shop_first_is_opt_in_and_buys_crop_seeds_before_forage_in_later_years`).
   **Update 2026-11-14:** Year 3 now drops the forage on a seed-trip day (not just reorders it) and caps the start at 14:00; replays from the same pins went
   979 → 984 and 983 (see the Score push section), so that part is on by default for Year 3 only. `s4` had the mirror image: capped ripe potatoes filled every
   ring at Summer D1 and nothing could be sown.
2. **Dead statue trip.** The pin owns the gold hammer, so the plain-hammer trip failed at the shelf every day (14 times in the control).
   With `HARVEST_GOLD_STATUE=1` the berry spawns (`$7F1F64 & 0x200` flips) but the pickup recipe ends "sequence complete" without eating:
   the source is spent for nothing. `statue_hammer_available` skips the trip; the gold pickup is the open repair (+5).
3. **Cave berry only in Winter.** Any season once the boulder flag is set (+5, +10 stamina).
4. **Hay.** Twelve hens plus the herd eat ~16 a day and the cut only reaches the pasture zone, so hay hit 0 on Summer D23 (`y3b1`). Hens hatch
   one per three days and a chick eats nothing for eight, so `final_flock_start_index` is now 264 (twelve at ~293).
   Open: let the cut reach the cover grass (evening, clock frozen) for the 8-cow herd.
5. Measured: each cow bought is +20 player happiness (443 to 533 over four buys and the cave berry), so the herd pays 3 + 0.6 + up to 25.

## Barn rewrite and the 12-cow push (2026-11-05, later session)

Owner ask: make the barn never fail, randomize it from every angle, 12 cows, max score. Measured with
`harvest/tools/barn_fuzz.py` (real morning pins, herd and day rewritten from a seed, the planner's own cow section on the ROM,
receipts read from RAM; research what-ifs, a pass proves a component, not a route).

| Barn code | Trials | Clean | Note |
| --- | ---: | ---: | --- |
| Old lanes (`cow_care`, `cow_nav_ops`, ...) | 40 | 6 | 0 of 28 twelve-cow trials; most never finished in 60,000 frames |
| Rewrite, first pass | 28 | 25 | farmer walked out during the fade-in |
| + fade wait, route round cows, shift limit | 84 | 84 | all profiles |
| + morning profile (tired, wrong tools, half-fed) | 48 | 48 | |

`CowChoresTask` went from seven files and 2,800 lines of fixed pixel lanes to one plan read from RAM (`cow_task.py`,
`cow_geometry.py`). Rules and ROM facts are in `AGENTS.md`. **Two findings explain old collapses:** cows leave the barn by the door
and silently stop getting care (`y3herd`, 327: slot 1 outside for 14 days), and the farmer's own shoving caused it.
**A second bug outside the barn:** the sprinkler replaces the can, so 41 of the 57 days of the 531 run ended at 13:00.

### Barn verification follow-up (2026-11-06)

The follow-up covers the actual farm approach, care, payment and exit. Farm horse animation frames
`0x01F8..0x0211` now enter the navigator's animal blockers (seed 1165). The barn walker walks near a
cow and retreats before a run can shove her; the final door approach also walks. A cow reaching the
lower hall ends the current shift before she can wander through the door (seeds 1066 and 1110).

Milk now ships from the ROM-paid near stand `(56,348)`, avoiding the lower-left corner's cow trap
(1167). A tool swing already being verified finishes even if its walking budget expires: resetting
between the milk flag and the late bottle lost 250G in seed 1105. Shipping requires an actual bin
credit, and feeding requires the trough's fed bit.

`tools/barn_fuzz.py` checks retained cow slots, available hay including prefilled/shared pregnancy
troughs, milk prices when the milk appears, actual shipping credit, and zero runtime RAM writes or
reloads. Idle/knock frames count against the trial budget; a knock cannot cross the doorway during
its hold. Batches retain configuration and source hashes, persist crashes, reject stale receipts
and duplicate seeds, and report incomplete or failed batches with a nonzero exit status.

Replay the five regression seeds with
`uv run python -m harvest.tools.barn_fuzz run <tag> --profile all --seeds 1066 1105 1110 1165 1167`.
Verified on the final controller (`04acfeb43a5d2abe` source hash prefix):

| Gate | Result | Evidence |
| --- | --- | --- |
| Same 200 seeds as `final1` (1000–1199) | **165/165 barn care**, **7/35 outside roundup**; overall **172/200** | `recordings/barn_fuzz/barn_complete/REPORT.txt`; no new escapes |
| Independent knock seeds 2000–2031 | **32/32**, no escapes | `recordings/barn_fuzz/barn_knock_complete/` |
| Former failures 1066, 1105, 1110, 1165, 1167 | **5/5** | `recordings/barn_fuzz/barn_animation_regression/` |
| ROM pytest: fuzz regressions, bin stands, bounded idle, saved barn pins, cure/sleep herd ladder, natural Y3 morning | **41 passed** | `test_barn_fuzz.py`, `test_barn_care_rom.py`, `test_herd_ladder_rom.py`, `test_day_plan_home_sequences.py -k barn_daily_care` |
| Harvest non-ROM suite | **2642 passed / 32 skipped** legacy; **2638 passed / 36 skipped** blocks | `barn_complete/pytest_legacy.log`, `pytest_blocks.log` |

The clean 200-seed trials had median 25,881 frames and worst 46,289. The all-profile report intentionally
exits nonzero for the 28 roundup failures. No failed/partial batch was substituted for these results.
The `outside` profile remains a strict best-effort roundup test and is reported separately from
barn care. Remaining work: farm routing and controlled turns for outside cows; do not excuse those
failures or treat a what-if barn batch as a power-on campaign result.

The fresh `score` power-on benchmark, `recordings/y1_bench/barn_complete.json`, was stopped at
Y1 Spring D4, 639,020 frames / 1,080.1 seconds: return-home repeatedly could not clear held item
`0x08` near the pond and recorded four `return_home/rescue` failures. No cow or hen had been bought.
The audit still has zero RAM writes and zero initial/mid-run loads, but this is an incomplete prefix,
not a completed-year or barn-economy result. Both scorecards and their comparison are retained;
the prior completed year ends on a different date, so its wallet delta is not an annual comparison.
Remaining campaign gate: fix the Day-4 clear-hands/return-home loop and repeat the full year;
details and commands are in `docs/Y1_BENCH.md`.

**Handoff (end of session 2026-11-05).** Default policy for the next full power-on: `score` with the fixes below (c4: **491 Clean**), and
5 cows if `y3k5` ends Clean above 491. Do not make `score_herd` the default: 327, 360, 552 on three draws.

1. Read `y3k5` when it lands: `.venv/bin/python -m harvest.tools.final_score recordings/y23_sweep/y3k5_s1.json`; mid-run
   `.venv/bin/python scripts/pin_status.py y3k5` (new). If it ends Clean above 491, make `HARVEST_COW_TARGET_LATER` 5 the `score` default
   (`RunStrategy.cow_target_later=5`) and add a test.
2. Why did herd runs differ (552 alive vs 327/360 collapsed)? Compare hay, sick bits and deaths day by day on the pins
   `PR_y3b4_s1_*`, `PR_y3herd_s1_*`, `PR_y3b2_s1_*` (`pin_status.py --pin ...`). Hay hit 0 by Summer D5 in `y3herd`. Then give the hay cut the
   cover grass (evening, clock frozen, `hay_goal` stop) and fix the barn-exit stall that cost b4 six rescues.
3. Repeatability: run the c4 configuration with `JITTER=90 FAULTS='*:fail' RATE=0.01` and two more seeds before trusting 491.
4. ROM-level checks not re-run this session (need the local pins): `test_power_berries_rom.py`, `test_power_berry_planner_rom.py`
   (cave in Spring/Summer, statue skip), `test_perfect_ending.py -m rom`. Offline suite: 2,612 passed.
5. Files touched (all uncommitted; several are untracked in git already, so commit them together): `planner/power_berry_calendar.py`
   (cave any season, `statue_hammer_available`), `planner/run_strategy.py` (flock start 264, `HARVEST_COW_TARGET_LATER`),
   `planner/livestock_econ.py` (`HARVEST_FINAL_FLOCK_START`), `planner/day_plan_phases.py` (opt-in `HARVEST_SHOP_FIRST`),
   `planner/tasks/multi_nav.py` and `tasks/power_berry.py` (diagnostic prints), `scripts/pin_status.py`, tests
   `test_ranch_master_published_rules.py` (new), `test_power_berry_calendar.py`, `test_run_strategy.py`, `test_day_plan_crop_phases.py`.
6. Local state: pins `PR_y3base_*` (start), `PR_y3ctl_s1_*`, `PR_y3b*`, `PR_y3c*`, `PR_y3k*` under `custom_integrations/`; logs and reports
   `recordings/y23_sweep/y3*_s1.{log,json}`; frozen tree copies `/tmp/y23_sweep_y3*`. `y3k5` may still be running (it writes its own report).

Open levers after this pass, in rough order of points: hay cut over the cover (keeps cows at 25, ~+16); corn/tomato split of the nine
Summer rings (one crop type is bought per trip, so tomatoes get none: ~+10); gold-hammer statue pickup (+5); fish berry (+5, runs behind
the statue slot); second child needs Eve at 650 then a window repair (+16, risky).

## Final-year ranch-score replay (2026-11-04)

Current owner scope: start at **Winter D30 Year 2** and improve the remaining playable year, measuring the ending score.
This supersedes the earlier short-replay-only scope below. Keep policy changes applicable to earlier campaign dates;
do not replay Year 1 for this comparison or promote STATUS from a checkpoint.

Frozen starting pin: `PR_finalyear_20261104_Y2_Winter_D30`, copied byte-for-byte from
`PR_e2es4_Y2_Winter_D30`, SHA-256 `babd56b97de1ed2d2261151d14ef3003cf65811a0aeff89787145f16f75d5657`.
The copy preserves planner lineage and uses a local path. Reports, source snapshots, hashes and daily pins are under
`recordings/final_year_20261104/`; pin prefixes `PR_fycontrol_` and `PR_fycandidate_` preserve independent timelines.
Start RAM-model score **368**, 187,840G, three cows, two hens, zero eaten berries, 119 counted field tiles,
Eve 384, first child age 70, second child age 0. `start_score.json` records all 21 score terms and raw windows.

Changes under ROM evaluation:

- **Cave return:** the original score policy ate the cave berry on Winter D30, then exhausted eight rescues in map
  `0x29`. `ExitToFarmTask` now follows the cave's native 32-wide floor around the east tunnel and exits to the mountain,
  then builds its normal farm return. Native Year-1 cave-to-house regression passes; this repair is common to both arms.
- **Eve in the second house:** gifting was restricted to `0x16`; the upgraded house is `0x17`. Select the current house
  in a copied phase recipe, walk the coop return through short farm waypoints, and navigate the larger room's live grid.
  Native two-egg chain: **+8 hearts**, empty hands and farm return. `HARVEST_WIFE_GIFTS=0` disables the chain for an A/B.
  First-house gifting remains ROM tested. The nightly -2 and final 496–511 heart window still apply.
- **Ending flock:** `score` and `score_herd` keep two birds until Y3 Spring D11, then hatch toward twelve. Fields
  `RunStrategy.final_flock_target` and `final_flock_start_index` own the target/date; `None` retains the ordinary flock.
  The existing pasture/feed gate still applies. Native Spring-D11 planner recipe incubates a real egg with no writes.
  `cash` keeps two throughout. `score_herd` also uses its existing eight-cow/240-tile target; compare its real care costs.
- **Measurement:** `harvest.tools.final_score` now prints every recorded score window and its first-to-last change,
  including partial runs. The first recorded morning is explicitly after sleep, not the loaded pin's baseline.

Initial validation: default offline suite **2,555 passed**, blocks **2,551 passed**; wife ROM **2 passed**;
return/labor ROM **11 passed**; flock ROM **1 passed**. Full ending comparison is running; no final gain is claimed yet.

Remaining measurements: turnip harvest/credit in the same Spring, all final score terms, herd survival/feed, flock growth,
cover spending versus wallet points, and rescues. The control also exposes a late statue trip requesting ordinary hammer
`0x03` although the native shelf supplies gold hammer `0x13`. A gold-tool fetch alone is not an eaten-berry receipt;
the existing statue pickup/eating failure below remains relevant. Missed Fall prizes and spent berry spawn flags need
earlier back-application; the Year-2 checkpoint cannot recover those dates. Second-child score needs wife hearts 650
and first-child age 90, followed by a separate heart-window repair; keeping Eve near 500 alone does not earn it.

## Spread score work across the campaign (2026-11-04)

Owner direction: ease off the final-days 200-bag grass push, invest in useful planting earlier when the cash term can
still be funded, find cheaper score gains, and test short continuations from D3, Summer D1 or Year 2. No full run this pass.
The cover purchase rules in this section supersede the November 3 bulk-load rules below.

- `field_cover.cover_errand` starts from Year 2 as before, but buys **8–24 cover bags** on a light morning (at most
  30 ripe crops). Eight bags need only 4,000G of surplus instead of waiting for the old 20,000G minimum. Held stock
  counts toward the 24-bag target; a failed evening does not cause another stack purchase. The herd's own seed reserve
  remains separate. The final four mornings have **no busy-morning exception**, and a rich wallet cannot force a bulk trip.
- The cash floor is retained: `max(10,000, 99,840 + 10,000 - 500 * days_left)` plus the caller's animal/crop reserves.
  This is an income assumption, not proof of the final wallet. All paid stock may still be worked that evening, with
  spa refills; the purchase target is not a daily sowing quota. Existing whole-field tilling and east-first geometry remain.
- Cheap crop points: **Spring shop, selected seed and planter now agree on turnips**. Previously the plan could say
  turnip while `_berry_run_phases` bought potatoes, and turnip requests inherited the potato stock field. The shared
  Spring capacity gate also checked potatoes alone. Both crops now reserve their own standing tiles against their caps;
  the persistent farm array supplies this reservation indoors. A capped potato plot cannot be replanted until it is
  actually harvestable, while other empty plots may take turnips. Stop at 496 shipped as before.
- **Year 1 Spring keeps potatoes for income**, using any already-bought seed first. Spring balancing starts in Year 2,
  when the first farm investment has had time to pay. Summer balancing is unchanged.

Bounded native receipts, `recordings/score_spread_20261104/` (local, no RAM writes or mid-run loads):

| start / scope | result |
| --- | --- |
| `PR_e2es4_Y2_Spring_D3`, pasture buy + evening cover through `DayPlanTask` | **24 bags, 12,000G, all 24 sown, +202 durable tiles**, 79,715 frames, still Y2 Spring D3. The test waits natively until evening and uses the normal peddler/input guards. It omits intervening crop/herd chores and the separate whole-field hoe pass; this is a component receipt. |
| `PR_spread_20261104_Y3_Spring_D3`, six-day comparison, cover off in both | Both reach D9, no bad days or rescues. Before: 204,640 frames, 189,490G, buys nine potato bags and leaves two unused. After: 217,826 frames, 189,940G, **buys and plants nine turnip bags**, no unused seed. Neither has harvested a Spring crop yet; the +450G is not a measured turnip return or a final-score gain. |
| D3 cash-first policy, then its natural D4 pin | D3 buys potatoes and advances normally; the D4 statue phase eats one berry and gains **10 maximum stamina**. The alternate early-turnip D4 pin below fails this handoff. |
| `LN_spread_cashfirst_Y1_Spring_D4`, five complete farm days after the D3 predecessor | Reaches **D9**, **two berries by D5** (statue, fish; 10 berry-score points, maximum stamina 120), 510G, five potatoes shipped, 152,545 frames. Zero rescues and one bad watering day (D7); this is early progress, not a strong Spring economy receipt. |

Regression coverage: `test_field_cover.py` covers modest purchases, held stock, protected busy mornings including D26-D29,
cash reserves and the native Y2 receipt. `test_run_strategy.py` covers Spring shop agreement, both crop commitments,
unharvestable capped plots and Year 1 investment. The full offline suite and existing berry ROM checks are logged alongside
the reports: 2,552 offline passes on the default layout, 2,548 on `blocks`, plus a later added indoor-reservation regression
(30 strategy tests pass); docs checks 8 pass, existing berry ROM checks 21 pass and the new cover ROM check passes.
These are checkpoint/component checks, not a new power-on score or a Perfect claim.

Remaining work, in order:

1. Continue a later Spring pin far enough to **harvest and credit turnips**, checking counters remain inside 496–511.
   The six-day D3→D9 comparison only establishes planting; its first turnip sowing is D6. Preserve daily pins for the next
   short comparison instead of restarting a full campaign.
2. The alternate early-turnip route exposes a **statue pickup/eating handoff failure** on
   `LN_spread_20261104_Y1_Spring_D4`: item 8 is lifted, the eating dialog never appears, and a later A throws it. The
   source's spawn bit is already spent; watering then stalls on the stale held item. Deterministic seven-second reproduction
   through `_plan_and_run` in `test_power_berry_planner_rom.py`; logs `statue_before.log`, `statue_compare_trace.log`.
   Waiting, turning before eating and small walking offsets did not repair it. The proposed facing change was removed.
   Do not call this alternate route berry-safe; the cash-first predecessor has a separate successful native receipt.
3. Test the smaller grass errands across several actual farm days, including animal feed and crop income. One component
   evening proves planting, not the calendar's total cash or 2,950 development tiles. The old dead-stand/spa-gate residual
   below still applies; avoid treating an unfinished field as an ending-week emergency.
4. The cash-first D4→D9 segment still loses watering on D7 (required phase watchdog after repeated can-use timeouts,
   `early_cashfirst.log`). It finishes without rescue, but only five potatoes have shipped. Mint a D7 morning from the
   D4 predecessor before using this short segment as an economy baseline; `LN_spread_early_end` is the D9 continuation pin.

## Earlier whole-field experiments (2026-11-03; purchase policy superseded above)

Owner ask: hoe the entire field in one evening and refill in the spa as often as it takes; no straight-line schedule, no per-day
time caps; the wallet pays for grass and nothing else; grass on the east side, far from the shipping lanes. Landed, uncommitted,
on top of the dirty tree; `cash` is untouched. The first version (a straight line to 2,950 tiles, 33 tiles a day, two-errand shop
batches) is gone: it added ~2.9M frames to a run and two shop trips cost 17 potatoes of delayed picking.

- **No line.** Sowing windows are Spring and Summer of Years 2 and 3 (index 120-298): a Fall or Winter bag writes `0x1D`, which Winter D1
  wipes; Year 1 gold compounds in crops. Inside them the whole field is wanted from the first morning, and the only limit is gold above
  `cash_floor_g` = 99,840 (the money term) + 10,000 margin - 500G x days left, never under 10,000G, plus the herd reserve (`pasture_reserve_g`).
- **The buy** (`planner/field_cover.cover_errand`, joined to the grass shop trip by `pasture_day_phases`): one trip for every bag the gold pays for, at least
  `MIN_TRIP_BAGS` (40) unless it finishes the field, at most `STACK_CAP` 255. **The seed stack is a u8 and the clerk still charges past it**
  (ROM probe `PR_e2es4_Y2_Spring_D4`, 250,000G poked: 130 bags came in 74,398f, 300 bags cost 150,000G and 255 arrived, 22,500G gone).
  The trip (~2,500f, ~2.8 game hours) goes on a **light morning** (at most `LIGHT_RIPE_TILES` 30 ripe crops waiting, `WorldProbe.ripe_crop_count`) or for a **big load**
  (`BIG_LOAD_BAGS` 100, 50,000G): in Summer the blocks ripen on staggered days, so nearly every morning has ripe crops (ROM probe, Y2 Summer D16-D20: all busy), and a
  strict "nothing ripe" gate would never buy. A handful of trips in the whole run, ~0.7k G of delayed picking each, where the first version made forty. Bags already in hand are sown whatever the
  morning, and Year 3 Summer D26-D29 (`LAST_CHANCE_INDEX`) may always go. The shop is shut Sat and Sun, **and the weekday shifts by one each year** (120 days = 17 weeks + 1: Y2 Summer D17/D18 are Sat/Sun).
  The log line is `[COVER] morning shop=... buy=... reason=...`; the buy timeout is `4,000 + 650f x bags` (the shop clock is frozen).
- **The work** (`multi_day_planner` `evening_clear`, after the 5pm shipping wait, clock frozen at 18:xx): `COVER_FIELD` sows every bag in hand, then
  `COVER_TILL` hoes every stand that still holds bare ground (`till_only`, the whole rest of the field), both walking the planner's stand list
  **east side first** (`east_first`: three-column bands from the east edge, alternating north and south). So gold that does not stretch over
  the field leaves the west bare, next to the bin; the bin lane and its stands stay reserved to the end (`reserved_cells`), crop footprints until Year 3 Summer D20.
  The planner is handed every stand and stops at `bag_limit`, so one it cannot reach only moves it on instead of leaving a stack in the carry pair overnight
  (a stack carried overnight broke the next `CROP_ESTABLISH`, `0x02 not in carry pair`). Hoed soil never reverts, so later bags need only the sowing.
- **Soaks.** `planner` reports `stamina_low`; the day plan splices a hot-spring soak and reruns. `COVER_FIELD`/`COVER_TILL` have **no refill cap**: another soak is
  allowed while the last pass hoed or sowed something (`tilled=N planted=M` in the reason). The whole field is ~6,200 stamina, about sixty soaks of ~6.7k frames.
  `DayFloor` has no frame budget or clock grace by default, and only crop and harvest phases have a watchdog, so nothing else caps the evening.
  `TaskSequence(keep_last_reason=True)` is what lets `stamina_low` through; the cover planter runs with `max_failures=400` because every rerun restarts at the first stand.
  **The farm dog seals the walk to the spa.** It flees west ahead of the farmer along the corridor (row y=26) and can end up cornered ON the gate tile, pixel (22,425),
  which is the only way out of the farm; the exit is one tile tall, so nothing passes (`exit_walk timeout` / `pixel_stuck` / `no_path sealed`). It then barely moves: a tile every few thousand
  frames (`dog_watch` over a day: ~8px in 8,000f), so retries at once only hammer the same spot. ROM samples: 2 of ~9 soaks in one evening, one permanent in another (hundreds of failed
  walks), 1 of 12 morning starts blocked and freed after ~15,000f of waiting. `HotSpringStaminaTask` now **retreats to the yard (232,392) and waits** `GATE_WAIT_NIGHT_FRAMES` (3,000, frozen clock after 18:00;
  600 by day) before walking again, up to `GATE_RETRIES` (8), logging `[SPA] Walk to the spa blocked ... dog=...`. A cover pass or an empty-bar restart that did nothing counts toward `COVER_IDLE_PASSES` (3),
  so a gate that never frees ends the evening instead of looping (the first fix counted only passes that ran, and a ROM probe looped on `empty_before_start` for 800 failed walks).
  **Open: a stack left in hand by a blocked evening.** If the gate never frees, the bags stay in the carry pair overnight and the next morning's `CROP_ESTABLISH` is at risk (see above); no task puts a stack back on the shelf.
- **Switch.** On for `score`, `score_herd`, `perfect` (and `cover`, now an alias of `score`); `HARVEST_FIELD_COVER=1|0` forces it for an A/B.
- **Pasture.** `pasture_target_tiles` sizes for the animals a herd-growing strategy plans to own from Year 2 on, and `pasture_max_tiles` lifts the 180
  cap (`score_herd` 240, `perfect` 420). `block_calendar.herd_options` (offline): 4 cows + 2 hens 141 tiles, 6 cows 187, 8 cows 234; the cash path's 4 cows fit under 180, so `cash` is unchanged.

Evidence (offline): 2,547 passed on the default layout, 2,543 on `blocks`. Evidence (ROM, Clean except the poked gold, `score`): the local what-if pin `FC_rich_Y2_Summer_D19`
(`PR_e2es4_Y2_Summer_D19` + 190,000G poked; `FC_probe_*`/`FC_rich_*` pins are gitignored copies with rewritten lineage):

| step | result |
| --- | --- |
| morning, busy (21+ ripe crops), big load | `[COVER] buy=254`: **255 bags for 127,500G in one visit** (190,000G -> 62,500G). Picking that day matched the no-cover control (5 shipped, 21 ripe left in both) |
| evening 1, first build (older planter) | 18 soaks, 54 bags sown, 325 tiles hoed, then `COVER_FIELD: too many target timeouts` after ~400k frames; 942k frames |
| evening 1, dead-stand planter | **20 soaks, 94 bags sown, counted tiles 196 -> 1,008**, 30 stands dead and skipped, one spa walk blocked by the dog and freed after waiting; **436k frames**, clock frozen at 18:xx |
| evening 1, + restart at the first stand with work left | **29 soaks, 143 bags sown, counted tiles 196 -> 1,433 (+1,237)**; 40 stands dead, 64 skipped as `bag not ready`, 2 blocked spa walks freed by waiting; the pass then ended `Complete` with ~110 bags left; ~370k frames |
| next evenings | the leftover bags (159-200) were carried through the next day with no planting trouble (that day had none to do) and resumed the next evening; the field is built over a few evenings, not one |

So one evening is limited by what the planter can reach, not by gold or stamina: ~50 hoed tiles (about 5 stands) per soak, and stands it cannot reach (`dead` on a timeout, or `bag not ready` when the hoe pass could not take a square) are left for the next evening. **The restart bug** (walking to the first stand of the list after every soak, so every far stand looked unreachable) cost half the first evening's work; `restart_route_px` fixes it. The 2,950 count is **not reached or verified**; what is open is why ~100 stands stay unworkable (the `[GRASS] ... is dead for today` and `skip chunk ... bag not ready` lines list them).

Open: a full power-on `score` run for the final count (nothing replays a Summer hurricane yet, and the D29 -> D30 night cannot be repaired);
`score_herd`'s bigger pasture has only offline tests; the turnip filler (200G a bag, Spring-bought; not tried); estimate: s4's 252k end wallet less the money
term's 110k is ~142k spendable, ~285 bags against the field's 330-372, so the last tiles (the west, by the bin) are the first to give. A planting-day morning
still makes a second shop trip for the grass (the crop-seed buy and the grass buy are two visits: `BuyShelfItemTask` buys one item per visit).

## Power berry calendar, Star Night and wife drift (2026-11-03, handoff)

Owner ask: improve the ranch master policy, plant more grass, get all the power berries, and mine the docs and walkthroughs
for end-of-year bugs and hacks that raise the social (hearts) terms. Nothing committed or pushed; all of it sits on top of
the existing uncommitted checkout. Evidence below was run on this tree.

**Power berries are now scheduled by the planner** (`planner/power_berry_calendar.py`, strategy flag
`RunStrategy.collect_power_berries`, on for `score`, `score_herd`, `cover`, `perfect`; off for `cash`;
`HARVEST_POWER_BERRIES=1|0` forces it for an A/B). Before this nothing in the day plan emitted a `power_berry` phase: the ROM
tests drove `PowerBerryTask` alone. Worse, e2e `s4` ended with `$7F1F64` bits 0x0800, 0x1000 and 0x0400 all set and
`power_berry_count` 0: three berries (both hoe berries and the stump) **spawned during the run and were never picked up**, and
the ROM sets the permanent bit at spawn, so each was lost for good (30 points).

| source | gate (RAM only) | planner ROM test |
| --- | --- | --- |
| statue | `$7F1F64 & 0x0200` clear, clear weather, from Y1 Spring D4 | green |
| fish | `$7F1F66 & 0x2000` clear, clear weather, not Winter | green |
| cave | Winter, `$7F1F64 & 0x0002`, `$7F1F6A & 0x0040` clear | green |
| chicken_trade | Fall, `$7F1F66 & 0x4000` clear, 2+ adult hens, shop open and dry | green on `PR_y1new_Y1_Fall_D13`; only 2 of 8 Fall pins finished, the rest hit the existing hen-sale route (`chicken pickup timeout`, `pixel_stuck` on the return walk) |
| harvest_festival | Fall D12, `$7F1F6C & 0x8000` clear | green on one RNG draw; 1 of 8 made the 13:00 deadline |
| egg_festival | Fall D20, `$7F1F6C & 0x3000` not both | turtle and berry draws green; 8 of 10 planner draws won a prize |

One source a day, decided from RAM alone; a berry trip yields to an animal purchase but **not** to the tool trip (on the old
cash-run Fall pins `BUY_COW_TOOLS` was planned every weekday and would have starved the hen trade). Each receipt flag was
checked to survive the night reset. Field and stump berries stay opportunistic: `collect_spawned_berry` now runs from the day
plan's step (skipped while a `PowerBerryTask` phase owns the berry) and drops a failed pickup instead of failing the plan.

Bugs found and fixed while moving the berries from bare tasks to the planner (each has a test):

- The plan's own berry hook grabbed the Kappa's cutscene sprite during the fish trip, failed with `no power berry ground sprite`
  and, because the hook's result was returned as the plan's, failed the whole day (Y1 Spring D5).
- Egg Festival: barrel slot 7 sits in a fenced pen whose gate is tile `0xA6`, missing from the hunt's walkable set: 4 of 12 RNG
  draws ended "no cave floor path"; now 11 of 12 win (the last was a start past 15:00).
- Egg Festival house berry: the prize scene seats the farmer in the east annex while the room is rebuilt (stride byte 4); the
  pickup now waits for the house map to settle, lifts the berry **from its east side facing left** (from above the table lies
  between and A does nothing), and a pickup whose walk ends on its first step no longer reports the whole pickup as done.
- A turtle-shell win is a success (+21 ranch points), not BLOCKED. The prize is a coin flip; Y1 and Y2 Fall D20 cover both bits.
- Festival trips now end at once when the 15:00 close cannot be met (Y1 Fall D12 replay: fishing ran to 13:00, the farmer reached
  town at 16:00 and waited until bedtime).

Measured: a 9-day Spring D4→D13 replay from `PR_spring_final_Y1_Spring_D4`, `score` against `cash`, zero bad days and rescues in
both: **186,517 frames against 264,888 (-30%), wallet 990G against 1,510G**. Two berries by D5 (statue D4, fish D5) cost about
520G at D13 and the run used far fewer frames, plausibly from fewer spa soaks at 120 max stamina (not isolated). Berry ROM
files: 21 tests green (`test_power_berries_rom.py`, `test_power_berry_planner_rom.py`); offline 2,514 passed (legacy) and green
under `blocks` after two missing `@south_pasture_only` markers in `test_seasonal_prep.py`.

Not done: the **snow flower** (peddler seed 2,000G on Spring D23, planted on the summit in Winter; pins `LN_berries_snow_*`
and `snow_flower_recipe` exist, no green receipt, not scheduled). Nine of the ten sources now have a green ROM receipt (two hoe
berries, stump, statue, cave, fish, hen trade, harvest festival, egg house berry); the hoe and stump berries are RNG draws that
the pickup hook collects when they spawn, and the egg berry needs a winning hunt and the right coin face, so none of the nine is
guaranteed in a given run. The tenth needs the snow flower. A full power-on `score` run is the real proof and has not been run
on this tree.

**Social levers found in the ROM** (`perfect_rancher/research/F11_social_levers.md`; the web walkthroughs have nothing on
hearts mechanics or glitches):

1. **Star Night (Winter D24) is a choice, and the bot makes the wrong one.** The forced festival opens a five-option menu
   (text `0x02B6`, "Where should I go?": Public square, Mountain top, Church, Spa, Bar). Each option loads one scene that gives
   **one girl +30 hearts** and takes **-30 from the married wife unless she is that girl** (group `0x0F` events 1-5; choice index
   0 is Ellen, 1 Maria, 2 Eve, 3 Nina and 4 Ann, the last two inferred). The bot dismisses with the default cursor, which is index
   0, so every Star Night is +30 Ellen and -30 for the wife. `s4` shows it: Eve 410 → 376 and Ellen 38 → 68 across Y2 Winter D24.
   Choosing the wife's scene is +30 instead of -30, a 60-heart (~4-point) swing a year. **Not implemented**: which on-screen option
   is which index is unverified (the transit maps put index 1 at the church), so the cursor path needs a Winter D24 pin first.
2. **Every house entry at hour 18 costs a married wife (hearts ≥ 200) 2 hearts**, measured: same `s4` pin, entry at 17:05 leaves
   Eve at 376, entry at 18:05 makes 374. The bed refuses A before 18:00 and the clock is frozen indoors, so a farmer who sleeps
   must enter at 18:xx: the cost is fixed per married night (~250 nights, ~500 hearts) and the wife gift chain has to out-earn it.
   It does only in the remodeled house with an egg laid (`eve_gift.py`); Eve slid ~96 hearts over `s4`'s last 110 days.
3. Final-night window: wife entry hearts 498-511 (F0). Verified earlier; still the target.

**Grass**: see the whole-field cover section above (built later the same day).

Next, in order: (1) a power-on `score` run to Y1 Winter with `track`-style berry receipts (count by season) to prove the calendar
in a natural run; (2) Star Night place choice for the wife, verified on `PR_e2ec1_Y2_Winter_D24`; (3) snow flower receipt;
(4) the wife gift chain in every house level and for milk (items 0x10-0x17 are +4 to +5 for Eve); (5) whole-field development is built (section above): prove it in the power-on run.

## Latest stop / score optimization handoff (2026-11-02)

The owner now prioritizes Perfect ranch score, sprinkler, grass/cow growth,
all power berries with individual tests, and overnight work without day caps.
Current changes, exact ROM receipts, the red cave test, interrupted benchmark,
and next action are in the [current handoff](tasks/rr-20w-run16-defects.md).
Sprinkler acquisition and a 72-tile water pass, both seven-crop plot shapes,
and four berry source components have passing ROM tests. All berries and the
new economy remain unproved. The old F6 seven-source claim has a script-bank
error: the goddess axe event grants the gold axe; an existing fish-return tape
contains held item 8 and needs a native acquisition route. Read the handoff
before trusting the old collectible model or building duplication routes.
Later the same day: every term of the 999 score now has a ROM-verified RAM-edit test (302 offline, 185 ROM,
[test_perfect_ending.py](../tests/test_perfect_ending.py)), including the final night's wife drift (-2),
shipping credit and the 2,950-tile development edge. The cave berry has a green native recipe and a found
gate (Year 1 Winter's first snowstorm); the statue berry picks but does not eat; the fish-return berry is
tape evidence only. Details and next actions: the handoff's top section.


## Forage calendar, spa refills, Year 2 levers (2026-11-02, handoff)

Owner ask: keep picking berries and the summer fruit when there is time (the calendar page showed none), check the economy is
optimized, then re-run Year 1 and rebuild Year 2 planting (more cash, the horse). Landed, uncommitted, layout still behind
`HARVEST_RING_LAYOUT=blocks`; the forage and spa changes apply to every layout:

| change | where | evidence |
| --- | --- | --- |
| Forage by season from one ROM table: grapes Spring (150G), green fruit Summer (200G, tile (28,10)), poison mushroom Fall (200G, tile (43,39)); a tile gives again the same day | `planner/forage_calendar.py`, `tasks/mountain_berry.FORAGE_STANDS`, `maps/map_routes` (`mountain_entry_to_green_fruit`, `..._poison_mushroom`, `*_to_shipping_bin`) | ROM: green fruit 2/2 shipped on `LN_cowfix_poweron_summer_d5` (home 10:30 and 15:00); poison mushroom 2/2 on `PR_y1a_Y1_Fall_D10` (4,545 frames, home 12:00) |
| Forage first: with nothing ripe to ship the trip goes ahead of crop work (after the shop on a buy day); a harvest day keeps its morning for the bin. Berry run now on in Spring and Summer of every year; Fall plans up to four poison-mushroom trips | `forage_calendar.forage_first_today`, `day_plan_phases` (`forage_after_shop`, Fall `MUSHROOM_GATHER`), `day_phase_types` | A/B from the Day-3 pin, blocks: Spring D30 14,130G vs 13,060G forage-last; Summer D9/D12 replays ship 2 fruit then water everything |
| Up to four spa refills a day for `CROP_WATER` while each pass watered something (was one per phase name, two a day); the sowing gate counts three | `day_phase_stamina.spa_refresh_allowed`, `day_phase_registry._build_crop` | replay of `PR_s2base_Y1_Summer_D9`: 103 tiles watered in 3 passes, 0 unfunded (before: 20-50 tiles dry most nights) |
| Grape inbound takes the 40px north pull after the landing | `map_routes._MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED` | Spring D3 of `y1new` failed three trips on a pixel-stuck landing; D3-D5 replays ship |

Why forage was missing: the berry phases sat after all crop work, and watering fills the day, so Spring D3-D30 ran two grape trips. The clock is
not the limit (days run 28-65k frames: it freezes at 18:xx and work goes on); only the 17:00 bin is, so a morning trip costs crop work only on
a day with something ripe. Summer harvests nearly every day once corn cycles, so Summer fruit is free on D2-D12 only.
Sprinkler probe (RAM poke, not Clean; the shop buy is not built): one use at a block notch watered 5 crops for 8 stamina and 88 frames with no
can water; the tool is Y, the square is the 3x3 around the farmer. See `memory` notes `harvest-forage-calendar-facts`, `harvest-sprinkler-probe-facts`.

Year 2 forecast (offline `block_calendar.forecast`, Y2 Spring D1, 60,000G, every block already hoed; estimates, not ROM):

| watering cap (tiles a day) | blocks | Spring D30 | Summer D30 | notes |
| --- | ---: | ---: | ---: | --- |
| 112 (old, ~1 spa soak) | 16 | 88,360 | 169,840 | 683 crops shipped |
| 140 (4 soaks, landed) | 16 / 20 | 89,960 / 89,900 | 180,520 / 182,880 | bin clock binds: ripe left on the plant grows with blocks |
| 200 (sprinkler plots) | 20 / 24 | 92,760 / 96,040 | 185,480 / 187,960 | ships ~820: the 9,900-tick window is full |
| 140 + horse parked in the field (cycle 112 frames) | 20 | 95,600 | 194,900 | |
| 200 + horse | 20 / 24 | 100,280 / 107,020 | 215,960 / 242,140 | 954 / 1,146 shipped |

Year 2 first look, from `PR_y1new_Y2_Spring_D1` (66,550G, blocks, 14 blocks as control `y2c`):

- Six more lattice blocks are in as `farm_pond.Y2_BLOCK_PLAN_ROWS` (Year 1 never sows them; `active_plot_centers(year)`), **off by default**:
  `HARVEST_Y2_BLOCKS=1`. Through Spring they only break even: 92,720G against 93,730G at D27 (the extra bags are bought up front and potatoes are bin-clock bound), with
  more watering watchdogs (6 against 2 by D20) and `CROP_ESTABLISH` timing out on the far block (24,35). Corn is where they should pay (forecast +13k at Summer D30), so
  the A/B has to run to Summer D20 or later; it was stopped at Summer D8.
- Year 2 mornings (wife gift, `HOUSE_WOOD`, the herd) start the outdoor plan at 09:00-12:00, so forage-first at hour <= 9 fired twice in Spring (34 berry windows hit the
  14:00 cutoff). `FORAGE_FIRST_LATEST_HOUR` is 12 now (`trips_today` still refuses a loop past 17:00): 6 trips and +640G at D19 (`y2f`, 82,150 against 81,510).
- Year 2 days also fail `NAV_TO_COOP` / `NAV_TO_ANIMAL_SHOP` ("expected tilemap 0x00, got 0x10": the farmer is still on the mountain after `HOUSE_WOOD`) in both variants.

So Year 2 order of work: (1) run `HARVEST_Y2_BLOCKS=1` against the control to Summer D20, fix the far-block sowing and the pond-refill watchdog first,
(2) re-A/B the horse as a field-side bin (the old negative A/B was morning errands, see `harvest-e2e-and-saddle`; the model above says it is worth more than six blocks),
(3) buy the sprinkler (2,000G, tool shop shelf `$51`) and a centre-bare 7-crop plot it can serve, which needs the owner's call on plot shape.

Result: full Year 1 from the Day-3 pin ended at **66,550G** against 53,540G ([Y1_BENCH.md](Y1_BENCH.md#full-year-1-from-the-day-3-pin-all-four-changes-2026-11-02)),
591 crops shipped against 485, net worth 78,550 against 70,540 with one cow fewer.
Open: Fall bail hour 14 is unbenched, the third cow and hay (10 `CUT_FODDER` days against 20), the pond-refill watchdog (5-6 a year), `harvest_task.py` still ~1,400 lines.

## Notch blocks, block calendar, fast press (2026-11-02, handoff)

Owner direction: replan the Spring Y1 plots; the potatoes nearest the bin ripen on Summer D1-D2 so corn goes in close with every
harvest cycle; potatoes still growing into Summer go on the far plots; one-tile lanes; fast pickup-to-ship; a map and a data schema
to check frames against; then grass and cows from Summer cash (estimate). Mid-task: **no rings** - 3x3 blocks `xxx/xxo/xxx` and
`xxx/oxx/xxx`, mirrored pairs whose notches face across the lane.

Landed (uncommitted; layout behind `HARVEST_RING_LAYOUT=blocks`, the press and window are default-on for every layout):

| change | where | evidence |
| --- | --- | --- |
| ROM clock is 900 ticks an hour (minute byte 0-14, 60 ticks each, one a frame outdoors). The ship window is now in frames (`clock_tick`, `SHIP_CLOSE_TICK` 15,300); the old check read the byte as 0-59 | `tasks/harvest_task.py` | `UpdateTime` bank_82; one more crop inside the window on both far-field pins |
| Four bin stands: (11,29)/(11,30) facing left, (8,32)/(9,32) facing up (the throw reads two tiles ahead into F2 (8-9,29-30)); targets rank by the nearest | `harvest_task.SHIP_FALLBACKS`, `bin_distance` | each stand paid 120G a corn on `LN_saddle_whatif_harvest_y2d22` |
| Fast press: push into the crop until a frame moves nothing (the lift ignores A one pixel short), wait for `player_direction` before A (a missed one-frame turn threw a potato on the ground), wait out the 23-frame lift (a potato tile turns to soil on the A edge); a missed pick retries once centred | `harvest_task._fast_arrival`, `_step_press`, `_run_centring` | Spring D8 pin: 8/8 potatoes, 640G, 1,010 frames vs 1,181 centred (-14.5%); plus B on path-centring steps 4+ px off centre: 967 (-18%); far corn 3,447 vs 3,687 (-6.5%). The ROM runs below predate `_run_centring` |
| `ship_log` on every `HARVEST_ROUTE` journal row (`harvest.ship_log/1`: target, crop, stand, ship stand, frames and clock ticks at select/pick/throw/ship, paid) | `harvest_task`, `day_plan_orchestrator` | |
| Block calendar planner: BFS pickup-to-ship frames per crop over lanes and notches, roles, Spring last-sow and Summer bag days, day-by-day forecast (bags, cash, ripe, shipped within the 06:00-17:00 clock, watering cap), herd estimate | `planner/block_calendar.py`, `farm_pond.BLOCK_PLAN_ROWS`, `calendar_bag` | `tests/test_block_calendar.py` |
| Report: plan JSON + map page; `track` matches a run's ship log to the planned slots | `tools/block_calendar_report.py` (`plan`, `track`) | D2 plot measured 113.6 frames a crop vs 114 planned |
| Notch blocks are sown from the centre, entered through the notch (an object on an edge cell stopped the straight walk in: `PR_blocks_Y1_Spring_D9`, two nav timeouts) | `skills.farm_pocket_plant_skill` | D9 replay: 3/3 establishes, 4,404 frames vs 12,847 |

Forecast (one plan, `block_plan.json`; Day-3 pin, 260G, 112 watered tiles a day): Spring D30 11,740G, Summer D30 89,940G with 12 core +
2 late (570 corn, 96 tomato, 286 potato). 14 core + 0 late forecasts 91,420G: at this watering cap a late block costs ~750G, because it
holds water capacity all Summer. Watering, not walking, caps the plan: 144 tiles a day forecasts ~103k. Herd estimate (not ROM tested):
cash covers the grass first (Summer D1) and then a cow a day from Summer D13, so the table's best is the largest herd tried (8 cows,
234 grass tiles, 30 bags); care at 5+ cows (barn-aisle stalls, sickness) is the limit, not gold.

ROM runs (Day-3 pin, one draw each, Clean, 0 rescues; [Y1_BENCH.md](Y1_BENCH.md#notch-blocks-and-fast-press-from-the-day-3-pin-2026-11-02)):
`blocks` ended Spring D30 at **12,450G** (forecast 11,740; legacy plots on the same tree 10,790; packed rings 4,650-8,540), shipped 270
potatoes by Summer D7 and planted 10 corn blocks on D1-D2 and 3 on D6; measured pickup-to-ship frames are 0.97-1.17x the plan per block.
`fastbase` (legacy plots + the press, stands and window) reached Fall D1 at 31,660G with 2 cows and 2 hens (old-tree base: 26,740G, same
herd; not a clean A/B). Summer estimate for `blocks` as planted: ~80k at Summer D30 if watering holds.

Next, in order:
1. Watering at 12-13 blocks: `CROP_WATER exceeded 24000 frames` on Summer D1 and D2 nights. Read those two phases first; every missed
   night costs each dry block a harvest day. Then run `blocks` from the Day-3 pin to Fall D1 (`track` it) for the real Summer number.
2. The late blocks never got sown after D20 (one spa soak funds 12 blocks): either allow two soaks for a first hoe, or keep 12 core +
   0 late (the forecast prefers that at this watering cap anyway).
3. Grass and cows: the herd table in `block_plan.json` (estimate) says the cash covers grass on Summer D1 and a cow a day from D13; the
   limit is care at 5+ cows (see the herd ladder), not gold.
4. `harvest_task.py` is ~1,400 lines (soft max 1,000): fold the press helpers into the Composer or trim before more work there.

Not done: late blocks plant the day's bought Summer seed (corn under `cash`), not tomato; the forecast's per-block tomato needs the
shop to buy two crops. The 14+0 split has no switch yet (edit `BLOCK_PLAN_ROWS` or regenerate with `plan --core 14 --late 0`).

## Packed ring lattice, east pasture, uncapped days (2026-11-02, handoff)

Owner direction: start over from the Day-3 pin; pack Spring rings tight around the shipping bin (his manual play: 24+
eight-tile plots, rows of six); move grass east of the buildings and invest in it as the goal needs (cash, 100% score);
keep both the max-cash and max-score runs; **no per-day time cap** (a day may run an hour if the work is real; only a
stall, glitch or clip ends it); judge the layout by Summer D2-D3 (rings planted, corn bags, distance), not by Fall D1.

Landed (uncommitted, all switchable, default unchanged except the day budget):

| change | where | switch |
| --- | --- | --- |
| Packed lattice: 4 rows of 6 rings, pitch 4, one shared lane, plus the two pocket rings = 26 rings (208 crops). Lanes x=2,6,...,26 and y=33,37,...,49; keeps the lanes the hard-coded spa/shop walks use ((6,37), (6,33)). `PACKED_RING_ORDER` is closest-first by walking distance from the bin stand (11,30), every prefix reachable. Mean walk to a crop's lane stand: 9.7 tiles at 9 rings, 13.4 at 18, 15.6 at 24 (legacy nine: 18.0) | `maps/farm_pond.py` | `HARVEST_RING_LAYOUT=legacy\|packed\|packed18\|packed12` (default `legacy`) |
| Shared-stand watering tour: fewest stands that cover every dry tile (a lane stand serves a crop on each side), nearest-neighbour + 2-opt over real lane distances, tiles of one stand together. Offline, 12 rings: walk 427 -> 150 tiles | `tasks/crop_geometry.plan_water_route`, `crop_planter._water_step_groups` | packed layouts only; `HARVEST_WATER_TOUR=0` turns it off |
| South-field walks use the y=37 lane in the packed layout | `maps/farm_gate.py` | follows the layout |
| East pasture x31-60, y17-30, nav along the y=24 row from the house door, anchor (33,24) | `tasks/grass_planter.PASTURE_ZONES` | `HARVEST_PASTURE_ZONE=south\|east` (default `east` when the layout is packed). ROM probe from `PR_e2ec1_Y1_Summer_D1`: 2 bags, 16 tiles hoed, 8 planted at (33,24), 6,003 frames |
| `cover` strategy: `score` plus idle gold above 120,000G buys grass, 6 bags an errand, toward 2,950 tiles; `cash` and `score` byte-identical | `planner/run_strategy.py`, `livestock_econ.pasture_cover_target`, `day_phase_catalog.pasture_day_phases` | `HARVEST_STRATEGY=cover` |
| Day budget 48,000 -> **216,000** frames, clock grace 15,000 -> 60,000 | `core/recovery.py` | `HARVEST_DAY_FRAME_BUDGET`, `HARVEST_DAY_CLOCK_GRACE` (0 / -1 disable) |

Offline: 2,159 pass (legacy) and 2,155 pass (packed); the tests that build on legacy sites or the south pasture skip under the
other setting (`tests/layout_markers.py`). New: `test_ring_layout.py`, `test_water_route.py`, cover tests in `test_livestock_econ.py`.

Evidence (Day-3 pin `LN_d2_perfect_poweron`, frozen tree copies, one draw each, **48k day cap in force**, so these three are
the capped control; reports in `recordings/spring_y1/packed/`):

| run | rings | Spring-end wallet | Spring frames | bad days (max in a row) | cap trips | Summer wallet D1 / D14 / D24 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `base` (legacy nine, tree as left) | 9 | 11,260 | 713k | 0 | 5 | 10,040 / 3,360 / 13,390 (buys hen, cow, house 1 in D7-D14) |
| `packed1` (26 rings, no tour, south pasture) | 26 | 11,550 | 971k | 5 | 32 | 7,270 / 13,910 / 33,470; Fall D1 46,870 with no cow or hen |
| `packed2` (26 rings + tour) | 26 | ~3,200 | 1.01M | 3 | 28 | 470 / 260 / - : income stalled D6-D14 |

`base` ended Fall D1 at 26,740G (1 cow, 1 hen). Read: the 48k cap is what hurts the packed runs. Days that plant 4-7 rings, water 100+
tiles and soak at the spa run out of frames; `packed2` watered too little on D10-D14 and shipped nothing for eight days. `packed1`
is ahead of `base` in Summer wallet (46,870G at Fall D1 against 26,740G) but bought nothing: both packed runs predate the east pasture, the rings reserve the
south pasture, no grass bag could land, every `CUT_FODDER` failed and the shop's planted-grass gate never opened (`planted_grass` 9 against 103). So the wallet gap
is not a win until the herd is counted. The tour is not yet separated from the cap.

**Open comparison (started, not read):** the same pin to **Summer D4** with the new 216k budget, four variants:
`u_packed` (26 + tour), `u_packed18`, `u_packed12`, `u_packed_notour` (26, `HARVEST_WATER_TOUR=0`). Archived on exit to
`recordings/spring_y1/packed/u_*/{report.json,run.log.gz}`. Compare for each: corn bags bought on Summer D1-D3 and rings planted
(`BUY_SEEDS -> bought corn_seeds`), Spring-end wallet and frames, `bad_days`, rescues, cap trips (`grep -c "budget"`), and the Summer D1
wallet. Estimate Summer cash from the layout: a corn ring planted by D3 gets six harvests of eight crops at 120G = 5,760G gross; the
walk to the bin is ~2 x mean stand distance per crop (170-235 frames a crop at the near rings, `Y1_BENCH` saddle A/B).

Decision rule: flip `RING_LAYOUT_DEFAULT` to `packed` (or `packed18`/`packed12`) only when its Summer D4 position (rings planted,
wallet, bad days 0-1, no rescue) beats `base` there, and `u_packed` beats `u_packed_notour` before keeping the tour. Then run to Fall D1
to confirm, then the continuous power-on bench (a pin continuation certifies nothing).

Next, in order:
1. Read the four `u_*` runs (above); pick N rings and tour on/off.
2. Hard-coded farm routes still touch lattice tiles: `map_routes.ROUTES["farm_south_to_west_gate"]` waypoint (8,35) is a ring centre (the nav substitutes the nearest standable tile; replace it with the lane (6,35)); `_FARM_SOUTH_FIELD_TO_WEST_GATE` still climbs x=13. Prefer a lane-BFS hop builder over more waypoints.
3. Whole-field cover: built 2026-11-03 (`planner/field_cover.py`, `COVER_FIELD`); the old `cover` preset that only grew `PASTURE_BOUNDS` is gone (alias of `score`). Grass is walkable, so no stand order can wall the bin lane.
4. Watering stamina is the real cost: 26 rings = 208 tiles = ~416 stamina a day, ~4 spa soaks at ~6.7k frames each. The sprinkler (2,000G, unlocks Summer D20 in Year 1) waters a 3x3 for 8 stamina against 16 with the can; a ring with an open centre notch (`seven_*_access`) fits it. Not built.
5. The `(20,28)` crop of the (19,28) ring has no open neighbour (wall at x=21): one dead tile, legacy too.

## Isolated reliability work toward 999 (2026-11-02)

Owner direction: build and test the pieces from midgame saves; defer another
power-on year or ending run. Existing run scores are unchanged evidence.

Work backward from `tests/test_ranch_master.py` (the
`test_windowed_terms_sum_to_999` fixture) and ADR 0007. The terminal gate is
`ranch_mastery == 999` at ending index `0x25`; arithmetic on a fabricated RAM
array proves the score formula only.

| Term | Points | Required component evidence before a campaign |
|---|---:|---|
| Herd and cow happiness | 36 + 300 | 12 surviving cows, each happiness >=200; every required feed bit and talk/brush/milk flag verified per slot over consecutive nights and perturbed timing |
| Chickens | 36 | 12 birds, supported feeding and egg chores; expand only after care capacity passes |
| Five bachelorettes | 155 | Each hearts counter in 496..511; isolated talk/gift routes for all five, stop at the window and measure wife drift |
| Four crops | 124 | Each shipped counter in 496..511, stop at 496 with shipment slack; purchase/plant/water/harvest/ship each crop |
| Money | 78 | Ending wallet low word >=9984 (raw money units), preserve the score window |
| Berries | 50 | Exactly ten eaten; permitted native duplication still needs a route |
| Farmer happiness | 31 | 992..999 at the ending |
| House / marriage / children | 32 / 32 / 32 | Both upgrades, marriage, two children from natural arrival |
| Specials / farm development | 43 / 50 | Both item flags and the developed-field threshold |

Implemented component work:

- Medicine works end to end on the ROM (2026-11-02). `BuyShelfItemTask(tool_ids=(9,))` buys it at the **animal
  shop** (`0x24`, 1000G, shelf stand (58,134), shed row-2 bit 1); `planner/cow_cure.cure_day_phases` plans
  `BUY_MEDICINE` (only with 1,000G, the shop open, before 15:00, and not on a day another town trip owns unless
  the death timer is at 2 or less), `ENSURE_MEDICINE` (shed to carry pair), `COW_CURE`, then the ordinary chores
  refetch the brush. `CowChoresTask(medicine=True)` runs a select/nav/verify cure phase
  (`tasks/cow_medicine_ops.py`) that reuses the brush stands; the receipt is the sick bit falling on the target
  slot. ROM facts that cost a fix each: the tool animation outlives a 20-frame poll, and the cow script clears the
  sick bit after the dose has left the carry pair (8-cow barn), so verify waits for `player_action == 0` and
  never presses with an empty hand; a dose on a healthy cow or the wrong cow is spent for nothing.
- Herd-size ladder at 3, 5 and 8 cows: `tests/test_herd_ladder_rom.py` on pins cut by
  `python -m harvest.scripts.mint_herd_whatif` (recipe in its docstring). Gates: a sick cow with a bottle on the
  shelf, with no bottle (buy 1,000G then cure), and a healthy morning each end with the whole herd fed
  (`fed_cows_flags == (1 << N) - 1`), nothing sick, every cow alive after the night, zero RAM writes and loads;
  the cure also runs under 0/31/97/257 idle frames (12 runs, all land). 21 ROM tests, 11 minutes. Frames from the
  start of the morning to the end of the barn chores, one draw each (`resets` = incomplete passes that re-entered):

  | cows | healthy | bottle on shelf | buy 1,000G and cure |
  | ---: | ---: | ---: | ---: |
  | 3 | 3,902 (0 resets) | 8,934 (1) | 14,616 (1) |
  | 5 | 23,189 (4) | 18,953 (1) | 32,150 (3) |
  | 8 | 22,032 (2) | 26,026 (3) | 27,880 (2) |

  Care is not the failure: every herd is fed, healthy and alive after the night. Time is: from five cows the
  talk/brush walks stall and the barn reset runs two to four times, so the chores alone take 20-32k of the
  48k-frame day floor. The alarm in the test is 36k.
- Turnip shelf (200,342) costs 200G. The ROM test proves a turnip stock
  increase and farm return. Score strategies can now choose turnips.
- A seed receipt cannot be claimed from an unrelated debit before its first
  shelf interaction. Snapshot the wallet/stock at the shelf.
- Cow task completion now fails on outstanding requested talk/brush/milk,
  rather than returning success with false summary booleans. Incomplete care,
  blocked walking, and three failed feed placements request re-entry; the day
  plan permits at most three such resets, preserving ROM daily flags.
- Two-cow stalled-barn and three-cow nonadjacent-slot morning replays pass
  delays 0/31/97/257, with no writes or mid-run loads. These are bounded
  component results, not full-herd or consecutive-night evidence.
- The existing twelve-cow research save
  `LN_repair-herd_whatif_all12_finalnight` exposes repeated unsuccessful
  feeding and navigation stalls. It is a what-if save, not natural arrival.
  `tests/test_barn_care_rom.py` retains this as a strict expected-failure
  gate. After the feed retry fix, milk shipping and the left-side barn exit
  still wedge; an unexpected pass fails until the gate is reviewed.

Crop layout direction (owner screenshot, 2026-11-02):

- All crops should minimize harvest-to-bin travel; the old deliberate
  preference for distant Spring crops is removed. Establish nearby west-strip
  plots before remote eastern plots. Existing planted saves retain their crops.
- Experimental eight-crop layouts `eight_east_notch` (`xxx/xxo/xxx`) and
  `eight_west_notch` (`xxx/oxx/xxx`) plant the center and leave the side
  opening unhoed. `CropWaterTask(layout_name=...)` exposes isolated planting.
  These are not the seven-crop sprinkler shapes or a center-hole ring.
- ROM single-bag A/B at (9,35), natural Y1 Spring D3: ring planting /
  watering 1451 / 1079 frames; east notch 1454 / 1019; west notch
  1611 / 1021. All eight planted and immediately watered; side openings
  stayed empty. East saves 57 frames combined, west costs 102 on this one
  plot. This does not measure mirrored pairs or mature harvest paths.
  Evidence: `docs/perfect_rancher/research/care_components_20261102.json`.
- Pasture planning now rejects crop footprints plus perimeter working stands
  and the shipping-bin lane, even when supplied expanded grass bounds.
- Do not switch the entire campaign to these layouts until ROM probes prove
  all eight planted, all eight watered at blocking growth stages, all eight
  shipped, and lower frame cost for mirrored pairs. Reserve a continuous
  route to the bin, house and refill stands, and exclude grass from both crop
  footprints and working/haul lanes. Grass relocation and dense packing remain
  to be measured; changing a bounds constant alone is not a route proof.

Farm development, measured (owner asked how much terrain must be worked; 2026-11-02):

- The ROM scores `min(100, counted * 10 // 295) >> 1` (F2). Hoed soil does not count; only fence and ids
  `>= 0x1D` below `0xA0` do. 2,950 counted tiles is 50 points and 2,949 is 49, so the bar is **99.4% of the
  2,968-cell clear-pin field** (99.2% of the 2,974 Fall mask): at most 18 (24) cells may stay bare. "Worked" is
  not enough; every cell is hoed and then sown. The score steps are coarse below it: 2,891 to 2,949 counted all
  read 49.
- `farm_final_layout.terrain_ledger` now reads it from any farm array: pool, class counts (grass, fall seed,
  crop, fence, tilled, untilled, dried, weed, stone, heavy debris, blocked), worked share, counted, points, tiles
  to go, hoe swings left, bags left (floor `ceil(to_go / 9)` and a raster-greedy plan), cells in that plan not yet
  hoed (the gate before any bag can land), disjoint full 3x3 grass blocks, stamina left, and the
  `planted_grass` counter. It rides `development_census.ledger` in every morning journal row, prints as a
  `[MULTI_DAY] terrain ...` line, and `final_score` renders a terrain table per season start. The grass planter
  logs each spent bag (`bag_log`: stand and tiles; nine is a full ring).
- Real checkpoints (`s4`, the best score run, `PR_e2es4_*`): Y2 Spring D4 has 120 grass tiles, 211 of 2,974 worked
  (7.1%), 3/50 points; **Y3 Spring D1 has 116 counted, 258 worked (8.7%), 1/50 points, 2,713 hoe swings left, a
  364-bag plan (182,000G) against a floor of 315 bags**. On the clear pin itself the plan is 391 bags
  (195,500G) against 327. The 328-bag figure is a floor that only a perfect tiling reaches.
- Cost against the budget: `s4` ended with 252,430G and the money term holds at 99,840G, so about 152k is
  spendable, roughly 305 bags. The raster plan does not fit; a better packer (the plan can overlap and waste
  squares) or a cheaper filler is what makes this term affordable.
- Cheaper filler, evidence so far: an out-of-season seed writes `0x1E` (F2 `$EB`), which counts, is walkable
  (`0x1E-0x21` are outside `CROP_SOLID_TILES`) and only toggles `0x1E`/`0x1F` with rain
  (`NightlyFarmTilesCheck`). Turnip and potato are 200G a bag against 500G for grass. ROM probe, same Summer pin
  and `BuyShelfItemTask`: corn (300G) and grass (500G) sell, **turnip and potato do not** (28 A presses, stock
  unchanged), while turnip sells in Spring (200G, control). So the filler is Spring-bought turnip or potato held
  into Summer. Not tried: stock size limit, the Summer plant writing `0x1E`, and hurricane exposure (crops and
  tilled soil lose a tile 1 in 4 on a hurricane night, grass 1 in 16; a night entering D30 has no replanting
  morning, so either cover is at risk at the tally).
- Hoeing: stamina is 2.00 a tile on the ROM (36 for 18 tiles in a till-only bench from `PR_e2es4_Y3_Spring_D10`);
  frames per tile are only bounded (about 104 including the walk from the door, 18 tiles; the marginal rate is
  unmeasured, the composer assumes 30). 2,942 swings is about 5,900 stamina, so over the 59 days of Y3 Spring
  and Summer it is about 50 tiles a day plus 6 bags a day. The planner now does it: `COVER_FIELD` (see the 2026-11-03 section) works stands anywhere on
  the farm, paced to a deadline line; `TILL_FIELD` and `PLANT_PASTURE` still work inside `PASTURE_BOUNDS`.

Focused commands from the repo root (headless SDL/Qt):

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run pytest snes/harvest/tests/test_barn_care_rom.py snes/harvest/tests/test_care_shopping_rom.py snes/harvest/tests/test_crop_notch_rom.py -m rom -q
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests -m 'not rom and not ml' -q
```

Next action: the 5-8 cow first-pass chores cost 20k+ frames and need the planner's barn reset (talk and brush
`pixel_nav_stall` from a farmer stuck at tile (7,7) in the north-west aisle, then a cow parked beside the
stand); that is where the 48k day floor goes, not the cure. Fix the aisle stalls at 5 and 8 cows (tighten
`CHORES_FRAME_BUDGET` in the ladder as they close), then wire a `cover` run strategy (below) and prove a
complete mirrored-notch watering/shipping cycle. Do not start another full run yet.

## Spring and Summer Year 1 from the Day-3 pin

Nine rings with spa watering (2026-10-31, [Y1_BENCH.md](Y1_BENCH.md#day-3-pin-nine-rings-and-spa-watering-2026-10-31)):
from `LN_d2_perfect_poweron` to Fall D1 the current tree ends Spring at 9,830G
(base 7,470) and Fall D1 at 36,130 net worth (base 7,980 with three Summer
storms; the stormless base replay 17,950), with no failed soak or rescue. One draw each.
Compare from the same pin with frozen tree copies (`harvest-rom-run-env`
memory) and always quote the Summer storm days.

1. Two runs wedged outside this session's code: the far-east RETURN_HOME
   escape into map `0x2A` after an evening `CLEAR_FIELD` at (777,937)
   (`v5b64` Summer D16), and the west exit refusing a night soak walk
   (`LN_spa_whatif_exit_walk_FAIL`, Summer D2, weather byte 1). Bound the
   evening clear to the home half of the farm before any longer day cap.
2. Hen delivery from the yard still fails after the wedge fix: both `to_farm`
   walks step back into the shop at the town door (600,874), and the next
   morning's chase targets a bird outside the yard (104,598). Start from
   `PR_v2c24_d3_Y1_Summer_D6`/`D7` (pins local) or
   `LN_hen_whatif_yard_strand_FAIL`.
3. Watering on the bigger field: v2 passed the all-wet gate on 12 of 95
   watering phases (112 skipped pairs). The stands that fail are around
   (9,35), (13,28), (41,33) and (44,29); try bounded alternate stands first.
4. The 48k day cap tripped twice in v5 (Spring D11, Summer D3). Both 64k
   draws wedged (the spa bug, then the far-east escape); keep 48k until item 1
   is fixed.
5. Ship window: five west-strip crops still went in after 17:00 in v2, so the
   overhead is now 12 minutes (unbenched). 64 ripe-crop deferrals show the
   harvest walk, not the field, is now the Summer game-time limit.
6. Bound evening `TILL_FIELD` when stamina runs out (about 20-25k frames on
   four Spring evenings in the timer run, with repeated till failures).
7. Then run the continuous power-on Year 1 bench. A Day-3 continuation does not
   certify that run, and no completed Year 1 save is accepted for the headed
   Year 2 run.

## Fall and Winter Year 1 from the Fall-D1 pin (2026-10-31, in progress)

All runs start at `PR_final_d3_Y1_Fall_D1` (2 cows, 2 hens, 24,130G, 59 hay) and stop at
Y2 Spring D1. One draw each, frozen tree copies; reports are in the session scratchpad
(fbase, v1.. are gone; `fv2`, `cw1`, `cw2` are in `recordings/spring_y1/fallwinter/`, gitignored). Net
worth = wallet + 5,000 a cow + 1,000 a hen.

| run | change | end wallet | cows | mushrooms ok/fail | net worth |
| --- | --- | ---: | ---: | ---: | ---: |
| fbase | tree as left | 48,430 | 2 | 2/7 | 60,430 |
| v1 | + parked-tile fix, Fall lumber last, third cow (hay-gated) | 25,630 | 3 | 20/4 | 42,630 |
| v1a | v1 without the third cow | 39,080 | 2 | 23/5 | 51,080 |
| v2 | v1 + wedding-day chores | 50,130 | 3 | 15/7 | 67,130 |
| v2a | v1a + wedding-day chores | 47,180 | **1** | 23/4 | 54,180 |
| cw1 | v2 + dispenser north stand (cow parked at the stand) | 53,830 | 3 | 23/5 | 70,830 |
| cw2 | cw1 + BFS walk detour, fail fast, barn re-entry | 54,330 | 3 | 21/6 | 71,330 |
| cw3 | cw2 + exit-prep detour (cow beside the milk bin) | 58,630 | 3 | n/a | 75,630 |

cw3 is the first Fall-Winter run with every animal-day fed by the cow chores
themselves: 59/60 fully fed (v2 53/60), zero `COW_CHORES` failures (v2 22), 1 bad day (v2
13; Star Night), no rescue, 16,353 frames a day (v2 21,635). The one day short of fed is
the festival day, which has no barn chores in its plan.

The full Day-3-to-Year-2 chain on the final tree (`y1b`) finishes Year 1 at 76,140 net worth
with three cows (`y1a` below: 62,750 with two), 7 bad days and no `COW_CHORES` failure; the
bad days are listed in [Y1_BENCH.md](Y1_BENCH.md#barn-dog-and-tool-fetch-stalls-feeding-the-herd-2026-10-31).
Next: the barn's top-left lane (exit prep and `EXIT_BARN` both stop at a cow near (2,4)), Fall
D24 `COOP_CHORES` starting outside the coop, then the continuous power-on bench.

The earlier chain (`y1a`, tree before the exit-prep and dog fixes) finished
at 62,750 net worth with 3 bad days and **lost a cow**: Winter D13's required
`ENSURE_ANIMAL_TOOLS` died on `pixel_stuck` at the farm's west entrance (98,430), the day
aborted, the herd went unfed (`feed_check` hit the same wall) and a sick cow died on D19.
The wall was the farm dog: it sat in the corridor and the nav pushed into it, because the
catalog only named sprite `0x01D9` as the dog and its animation cycles `0x01D9-0x01F7`
(`FARM_DOG_SPRITES`, farm map only; `0x01E1` is a dog frame there). Both Winter D13 and D19
now fetch the tools and feed the herd on their pins, and a failed tool fetch is skipped
after its recovery attempt (`SKIP_AFTER_RECOVERY_PHASES`) so it cannot cost the feed again.

v2 (+6,700 over fbase, 13 bad days) kept all three cows; v2a still ended with one cow
(two lost despite the wedding fix; it ran 4 bad days), so unfed or sick nights other than
D9 still kill cows. Single draws: the third cow is not proven, and cow survival decided
every row. Read the v2a and v1 cow losses (`Completed sleep` happiness drops of -20)
before trusting any number here.

Findings (measured):

- **Mushrooms**: the baseline shipped 2 in 30 days. After D11 chores began at 12:04 (a
  07:10-10:03 lumber trip plus a tool swap came first), and the planner only adds the
  mushroom phase before noon. A loop is ~2,500-3,300 frames (4-5 game hours), so one loop
  a day is the cap (150G). A ripe tile the harvest had parked, (12,28) in the sealed west
  ring, also kept `NAV_CROP`/`HARVEST_ROUTE` in the Fall plan. v1/v1a reach 20-23 of ~28.
- **Cows**: the wedding plan (Fall D9) was `EXIT_TO_FARM, FETCH_FEATHER, PROPOSE_EVE`, so
  both cows spent that night unfed. An unfed adult takes -8 happiness and a sickness roll;
  a sick cow dies 7 nights later, and the death scene consumes that day, so the other cow
  goes unfed too. v1 and v1a each lost a cow this way (the baseline's roll was lucky).
  The replacement is a child with no milk, which is why both are below the baseline.
- Forage tiles (mountain): mushrooms `0x33` at (1,4), (46,13), (30,24), (14,36), (41,40);
  grapes `0x30` at (2,4), (45,5), (23,9), (20,26). All respawn daily, but one item is
  carried at a time, so a second loop does not fit the day.

Code changes this session (uncommitted, non-ROM tests green):
`WorldProbe.has_harvestable_crops` ignores parked tiles from Fall;
`apply_house_phases` puts Fall `HOUSE_WOOD` last; `_proposal_day_chores` runs the animal
chores before the proposal; `cow_target`/`fall_cow_hay_ok` allow a third cow in Fall when
stock plus 80% of regrowth covers Winter and `SPRING_HAY_DAYS` (`COW_TARGET_FALL`).

Open, in order:

1. Read v2/v2a against fbase. Keep the third cow only if v2 beats v2a; a draw is not proof.
2. Cow-chore stalls (fixed 2026-10-31, cw1/cw2): a cow body in the barn aisle held a
   pixel-lane walk until the 30,000-frame timeout and the farmer slept at 11:06 with the
   day's work undone (the late `feed_check` fed the herd, so the eod snapshot undercounts).
   Seven stalls in v2: the fodder dispenser (cow parked on or beside the stand, Fall D5,
   D10, D11), the lane mouth to the troughs (D16), the right aisle (Winter D12, D26) and
   the milk-bin walk (D26, D18). Fixes: the dispenser also answers from tile (13,10)
   (`FODDER_NORTH_STAND_PX`, ROM probe); a walk phase that stands still 90 steps routes
   around the cows with the BFS navigator (`_walk_unstick`); two detours that never
   arrive fail with `cow walk blocked` and the day plan re-enters the barn once
   (`splice_barn_reset`): entering puts every cow back in its stall (10,13)-(10,17)
   after a 570-frame round trip, and a parked cow does not move in 24k idle frames.
   Open: the left lane to the exit (cow at (1,20) while the farmer is at (2,19), cw2 Fall
   D28 and Winter D13) still times out `EXIT_BARN` and costs a rescue; the BFS cow
   tiles ignore the pixel spill of a cow into the next tile.
3. Sick cows: detection and the ROM constants landed (`animal_status.cow_is_sick`, `livestock_econ`),
   but nothing buys or applies medicine (1,000G in the General Store, `ToolUsedCowMedicine`).
   Buying and applying it would turn a 5,000G death into a 1,000G cost.
4. Eggs go to Eve's gift chain daily (about 100G a day not shipped); that is the wife-hearts
   plan, left alone.
5. Then the continuous power-on Year 1 bench, as in the Spring and Summer list above.

## Handoff (2026-11-01 17:40): `v3n` reached the credits; `v3` (video) is finishing

- `recordings/e2e/v3n` (lean) **reached the credits** from power-on: 254,260G, ranch score 357, 1 rescue (Y2 Winter
  D19 barn exit lane). Y2 Winter D24 passed clean, so `labor.py` works on the real timeline. Numbers are in
  [Y1_BENCH.md](Y1_BENCH.md#end-to-end-power-on-runs-and-the-saddle-ab-2026-11-01); the scorecard is `FINAL.md` there.
- `recordings/e2e/v3` (same tree, video, ~490 fps) was at Y3 Spring D22 (6.47M of ~7.97M frames) at 17:36, ETA ~18:30.
  It writes `FINAL.md` itself when it ends. Then: `ffprobe` the `video_partNNN.mp4` parts (~25 GB), and fill in the
  `v3` row of the e2e table (it should match `v3n` frame for frame, since video does not change emulation).
- Uncommitted: everything, including `scripts/e2e_run.sh`, `harvest/tools/final_score.py`,
  `harvest/tasks/horse.py`, `harvest/planner/tasks/labor.py` and their tests (offline 2,065 pass, saddle/labor/shop
  ROM tests pass). Nothing is pushed.

`v3` finished identical to `v3n` (7,974,544 frames, 254,260G, 357) and its 37 video parts all decode.

Open, in order: (1) a full power-on `HARVEST_STRATEGY=score` run against `v3n`'s 357 (the strategies section below);
(2) the two recoverable stalls in the sweep section (barn exit lane, hit again in `v3n`; house entry); (3) the other levers.

## Strategies: cash and ranch score (2026-11-01)

Cash stopped paying at 99,840G (78 points) and the runs end with 140k-320k, so the two goals now differ and the
planner takes the goal as a switch: `planner/run_strategy.py`, `HARVEST_STRATEGY=<name>` or `run_to_day2 --strategy`.
The default `cash` overrides nothing (every earlier row stays comparable); the report records `strategy`; `final_score`
prints it, the points lost to wrapped counters and the lever for each missing term.

| strategy | what it changes | status |
| --- | --- | --- |
| `cash` | nothing: 2-4 cows, 2 hens, ship every ripe crop, Summer seed forced to corn | the tuned path, `v3n`/`v3` = 254,260G / 357 |
| `score` | ship each crop only to 496 (the window floor; the nine-bit counter wraps past 511); ripe tiles past the cap stay on the plant and the day plan stops scheduling a harvest for them; a capped one-shot ring is not re-bagged; the seed is the buyable crop with the most room left, and the Summer shop and planter follow it (potato, corn, tomato only); Spring sows nothing once potato is capped | **404, Clean** in the full power-on `s4` (cash 357); `s5` adds the sowing guard |
| `score_herd` | `score` plus `cow_target_later` 8, `cow_target_fall` 4, the Fall hay gate for a bigger herd, and a 110,000G wallet floor on Year 3 buys; the flock stays at 2 (first cut had 6 hens, which cost the 5th cow, below) | **do not use yet**: `h2` 324, `h5` 395, `h6` 330 (cash 357, `score` 404 and 380); every herd run lost cows to sickness |
| `perfect` | `score_herd` at the ROM caps (12 cows, 12 hens) plus the `--perfect-rancher` milestone dispatch | **do not use for score yet**: full power-on `p2` ended Clean but at 164 (cash 357), see "Full power-on results" |

ROM check (2026-11-01, [Y1_BENCH.md](Y1_BENCH.md#end-to-end-power-on-runs-and-the-saddle-ab-2026-11-01)): from the `v3`
Y3 Summer D20 pin, `score` ships no corn past the cap and the two days take 41k frames against 86k for `cash`; with the
counter poked to 500 it shipped exactly 11 and stopped at 511 (with the first cut's cap of 511; the cap is now 496).

### Full power-on results (2026-11-01, one draw each, `VIDEO=0`, reports in `recordings/e2e/<tag>/`, pins `PR_e2e<tag>_*`)

| run | strategy | result | rescues | notes |
| --- | --- | --- | ---: | --- |
| `c1` | cash | 254,260G, **357**, 4 cows 2 hens, 7,974,544 frames | 1 | frame-for-frame identical to `v3n`: the strategy refactor changes nothing on the default path |
| `s2` | score (first cut) | 219,570G, **357**, corn 432, tomato 0, potato **512 (wrapped, 0 points)** | 7 | two bugs, both fixed below; plus the door stall |
| `h2` | score_herd (first cut, 6 hens) | 236,810G, **324**, only **2 cows**, 6 hens, no clock or turtle (special items 0), potato 511, corn 331 | 1 | the herd stalled at 4 cows from Y2 Fall (2 died and were not all replaced); the shop's `grass` gate blocked cow 5 with $110k-$200k idle; same two `s2` bugs |
| `p2` | perfect | 83,270G, **164**, 0 cows, 12 hens, not married, no house upgrade, no children, **Clean** (0 rescues) | 0 | the milestone dispatch defers most intents (`native_skill_missing`, `dependency_unverified`) and replaces the plan that does the marriage, house and cows; only the hens (36) and a partial money term came out |
| `s3` | score, seed wiring + 496 cap fixed, door fix covering horses only | **INCOMPLETE**: wedged at the house door on Y2 Summer D22 with 8 rescues, 100,940G; it was $10k ahead of `c1` at Y2 Summer D1 | 8 | the blocker was the farm dog at (126,358), sprite `0x1E3`; the horse-only sidestep missed it. The fixed tree replays that day from `PR_e2es3_Y2_Summer_D22` with 0 rescues (70k frames, a few door retries) |
| `s4` | score, all fixes above | **CLEAN** (0 rescues, 0 RAM writes, 0 loads): 252,430G, **404** (+47 on cash), 4 cows 2 hens, 7,279,669 frames (cash 7,974,544) | 0 | crops 67 points (corn 251, tomato 337, potato 497) against cash's 27; cow happiness 100 (97), happiness 14 (9). Year 3 Summer shipped nothing: the cap left ~55 ripe potato tiles standing, the rings stayed full and no corn was bought. Fixed next (`one_shot_sowing_ok`) |
| `h5` | score_herd (flock 2), dog and horse door fix | **CLEAN**: 92,100G, **395** (below `s4`'s 404), 8 cows, 2 hens, crops 70 | 0 | cows +12 but cow happiness 90 for 8 cows (`s4`: 100 for 4) and money 71 (-7): the herd's hay ran out in Y2 Winter (slots 0-2 fell 255 -> 8 around Y2 Fall D25), and late buys pulled the wallet under 99,840G. Both gated now, see "What `h5` taught" |
| `h6` | score_herd with the hay gate and cash floor | **CLEAN**: 114,760G (money 78: the floor held), **330**, 5 cows, crops 77 | 0 | worse than `h5`: cow happiness **23**. The herd went 3 cows (Y2 Summer D1) -> 1 (Y2 Fall D1) -> 5 happiness 0-4 (Y2 Winter D30, two sick) -> 1 cow in Y3 Spring, then 5 re-bought in Y3 Summer. Hay was 355 at Winter D3, so hay was not the cause |
| `s5` | score with the sowing guard | **CLEAN**: 242,950G, **380**, crops **82** (corn 496, tomato 337, potato 488), 7,729,355 frames | 0 | the guard worked (Y3 Summer D1 bought 9 corn bags, nothing left standing; crops 27 -> 67 -> 82 across `c1`/`s4`/`s5`), but two cows fell sick and died in Y3 Spring (D15 slot 3, D19 slot 0) and were re-bought, so cow happiness read 68 against `s4`'s 100 and happiness 7 against 14: a different draw after the guard changed the Spring routine on D7 |

What `s2` taught (each now has a test):

1. **The seed pick was only half wired.** The day plan said `seed=tomato` on 42 days, but the shop still bought corn
   (`DayPlannerPolicy.summer_crop`), so `CROP_ESTABLISH` found no tomato seed and planted nothing: 107 completed
   establishes against `c1`'s 128, 955 harvest picks against 1,497, corn 432 against 811 and tomatoes 0. The plan now
   sets `summer_crop` to the pick and stops sowing when every sellable crop is capped (`multi_day_planner`). ROM leg
   from `PR_e2ec1_Y1_Summer_D13`: `score` bought tomato seed, planted a ring and shipped tomatoes, with corn 221 against
   `cash`'s 227 and the wallet within 300G.
2. **A cap of 511 has no slack.** Potatoes hit 511 on Spring D21, then one more was shipped on Summer D24: the 55
   ripe tiles the cap leaves standing are picked up by other tasks, and a held crop is thrown at the bin. 512 wraps to 0
   and costs the 31 points. The cap is now 496, the window floor, with 15 shipments of slack.
3. **A dog or horse beside the doorway wedges the house door** (the open "house entry" stall). Object `0x1FB` (the horse family,
   `0x1F8-0x203`) stood at (126,358) while the farmer, at (136,371) and exhausted (`player_state 0x09`), pushed up: no
   progress for 1,500+ frames in a ROM replay (`PR_e2es2_Y3_Summer_D27`, reproduced from the morning pin in 42k
   frames), seven rescues in the run. The same state entered in 45 frames from x=141. `home_approach.door_sidestep`
   now aligns the push 5 px away from a dog or horse in the corridor (tolerance 1); with no animal there nothing changes.
   The first cut matched horses only and `s3` died of it (8 rescues, Y2 Summer D22): the same spot (126,358) held by a dog
   frame (`0x1E3`). The wedge hit both score runs and neither cash run, so it is draw-dependent, not strategy-dependent.
   Pins `LN_door_s2_d27_stall_FAIL` and `LN_door_s3_d22_stall_FAIL` (local) are the replay states.

What `s4` taught (404, Clean): the cap has a second cost. A Spring potato ring that cannot ship before the cap stays
ripe on the plant and keeps its ring occupied, so Year 3 Summer bought and planted no corn (c1 bought 9 bags that
morning): corn stayed at 251 and 16 points plus a month of income were lost. The sowing guard
(`run_strategy.one_shot_sowing_ok`, read in `farm_pond.pocket_plant_targets`) stops Spring sowing once potatoes shipped
plus potatoes standing leave less than one ring (8) under the cap. ROM leg from `PR_e2es4_Y3_Spring_D1` (34 days, one draw):
Summer D1 bought 9 corn bags and planted them (as `c1` does), nothing was left on the plant, potatoes ended at 488 (30
points: the guard stops up to 7 short of the cap to keep a ring free, a trade of one point for the Summer corn).

What `s5` taught: the crop term is now structural (27 -> 82 against cash), and what moves the total between score runs is cow
care. Across `c1`, `s4`, `s5` the other terms are identical (money 78, house 32, marriage 32, children 16, specials 22),
but cow happiness reads 97, 100 and 68 on the same 4 cows: sickness is random after an unfed night, a sick cow dies in 7
nights, and the replacement starts at happiness 0. Nothing buys or applies cow medicine (1,000G, `plan.md` item 3 above),
and `COW_CHORES` fails on a blocked walk 7-10 times a run in every strategy. That is now the biggest single lever
(cow happiness has 200 points open, and every unfed night risks a quarter of 25 points), ahead of more cows.

What `h6` taught (330, Clean): the Fall hay gate and the cash floor did their jobs (money back to 78, hay 355 at Winter D3), and
the herd still collapsed. Both herd runs were already down to **1 cow at Y2 Fall D1** (from 3 at Summer D1), where
`score` kept 3; the cow-sickness cascade then repeated in Winter (two sick at happiness 0) and left 1 cow by Y3 Spring.
Larger herds are fed and cared for by the same chore pass (`COW_CHORES` fails on a blocked walk 7-10 times a run), and a
cow that misses one night is a quarter chance of sickness and a 5,000G death 7 nights later. So `score_herd` is a
cow-care problem first: cow medicine (1,000G, `ToolUsedCowMedicine`), chores that do not skip a night on a blocked walk,
and the Y2 Summer pasture (118 planted tiles at D1) all come before buying more cows. Ranking from the full runs:
`score` 404 and 380 (mean 392), `score_herd` 395 and 330 (and 324), `cash` 357. Use `score`.

Why the order of work follows cost per point, from the `v3n` scorecard (357, 642 short). Money is full, so every gold
above the floor is spendable: 254,260G - 99,840G = 154k.

| lever | points | cost | status |
| --- | ---: | --- | --- |
| Stop crops at the 496 floor (`score`) | potato 9 to 31 measured in `s4`; crop terms 27 to 67 with the tomatoes | ~1,800G of cash and 9% fewer frames | built; `s2` ended at 512 with a cap of 511 (below), so the cap is 496 |
| Tomatoes by Summer balance (`score`) | +21 measured in `s4` (337 shipped); up to +31 | tomato 100G vs corn 120G | built and measured; one crop per Summer, picked by the room left |
| Turnips | up to +31 | a Spring split | blocked: `buy_seeds.SEED_STAND_X` has no turnip shelf stand (potato 182, tomato 168, corn 152 on y=342); measure it on the ROM |
| Cows 4 to 8 (`score_herd`) | +28 each (3 + 25 happiness at >= 200), +112 | 5,000G + hay each, ~180G a point | experimental; last feedable buy is Y3 Spring D10 (50 care days) |
| Hens 2 to 6 | +3 each | 1,000G, ~330G a point, and 16 pasture tiles each | **harmful**: hens share the shop's 16-tiles-per-animal gate (pasture tops out at 180), so 6 hens cost the herd its 5th cow (`h2`); grow cows first |
| Pasture and development | +47 | 328 grass bags at 500G, ~3.5k G a point | the sink for what is left of the surplus; needs 2,950 counted tiles |
| Harvest earlier | none directly; it is how 4 x 496 crops fit (v3n shipped 1,482, the four windows need 1,984) | move wife-gift loops, `HOUSE_WOOD`, tool trip after the bin closes | open, see "Next levers" |
| Hearts (132), berries (50), children (16), specials (21), happiness (22) | up to +241 | effort, not gold | no native route (F4, F6, F7, F9) |

The cap and the herd together project to about 357 + 35..51 + 112 + 12 = 516..532 ("good" band) without touching the
effort-bound terms. 999 needs all of them (`perfect_rancher/MASTER_PLAN.md`); nothing here claims it.

Adding a lever: put the goal-dependent number in a `RunStrategy` field (`None` = module default), read it at the one
decision point, and add a red/green test in `tests/test_run_strategy.py`. Run a strategy against `cash` from the same pin
or power-on, and quote both, as the A/B tables do.

## Next levers (2026-11-01, after the saddle A/B)

Measured on the real schedule, not guessed ([Y1_BENCH.md](Y1_BENCH.md#end-to-end-power-on-runs-and-the-saddle-ab-2026-11-01)):

1. **Harvest earlier.** Y2 Summer `HARVEST_ROUTE` starts about 12:00 and ships 6-20 of 40-60 ripe
   crops before 17:00 ("ship window closed: N ripe left"). Before it: two wife-gift loops (~4,000
   frames, ~4.6 game hours), `HOUSE_WOOD` (~2,500) and `ENSURE_ANIMAL_TOOLS` (~2,700). Barn and coop
   time does not run the clock, and work after 18:00 is free, so the gifts, wood and tool trip are the
   ones to move after the bin closes (or after the harvest). Then saddle shipping (`HARVEST_SADDLE=1`,
   `harvest/tasks/horse.py`) fits the day and is worth re-running as an A/B.
2. **Stop shipping at the window (found in `v3n`; built as the `score` strategy above, cap 496).** The ROM counter is nine bits and
   wraps. `v3n` shipped 811 corn and 671 potatoes, so both wrapped (299 and 159 left, 18 and 9 points) instead of 31
   each: +35 points, since cash is already past the 99,840G cap. Tomatoes now follow from the seed balance; turnips
   still need their shop stand.
3. **Score, not cash.** Money maxes its 78 points at ~99,840G and the runs end with 140k-320k.
   Shipped-crop counters give a point per 16 (31 max each; tomatoes and turnips are 0 shipped, +62
   available), cows 3 each (4 of 12), hens 3 each (2 of 12), 50 for ten power berries (0), 43 for
   special items (0), 47 for pasture development (3 of 50 used).
4. The Year 2 Spring day budget trips on D3/D9/D11/D12; `CROP_WATER` is 29% of Year 1 frames.

## Years 2-3 repeatability sweep (2026-11-01, in progress)

Goal (owner): a consistent clean finish of Years 2 and 3 under random perturbation, then the
hot spots. Start pin `PR_y1b_Y2_Spring_D1` (the end of the Year 1 `y1b` run, 76,140 net worth,
3 cows). A single power-on run is deterministic, so one clean finish proves one timeline; the
sweep tool shifts it: `scripts/y23_sweep.sh <pin> <tag> <seeds...>` (seeded
`HARVEST_PHASE_JITTER` neutral frames before each phase, optional `HARVEST_FAULT_*` forced
failures) and `harvest.tools.sweep_report` (CLEAN = ending reached, zero rescues). Results so
far are in [Y1_BENCH.md](Y1_BENCH.md#years-2-3-repeatability-sweep-2026-11-01).

Each wedge below ended a run (8 rescues in one day) or lost a day of work; each reproduced on a
local pin, got a non-ROM test that goes red on the bug and a ROM replay that clears it:

| wedge | cause | fix | pin (local) |
| --- | --- | --- | --- |
| Y2 Spring D19: 8 rescues, run over | the 48k day floor fired mid-spa-soak; the farmer stood in the pond (x >= 624) and `return_home`'s mountain route shuffled on the `0xF7` edge | `spa_pond_exit_actions` (Left+A) first, shared with the post-soak settle | `PR_y23a_Y2_Spring_D19` |
| Summer D14/D15 Y1 and any Y2+ spa day: day lost | `RecoveryTask` had 900 frames; the mountain -> path -> farm walk is ~3,000 | 6,000 from mountain/path/town/shop (`FAR_RECOVERY_TIMEOUT`) | `PR_y1b_Y1_Summer_D14` |
| Y2 Spring D13, D22, D3, Y1 Fall D24: hens unfed, rest of day lost | `coop_nav_feed_bin timeout`: the left-aisle route waits on a hen parked on the via-point (4,5) (stasis 784); recovery then exits the coop and the retry reads "not in coop" (a second attempt hit the same hen) | `_navigate_to_left_top_goal` skips an occupied via-point; `splice_coop_reentry` walks back in once a day | `PR_y23a_Y2_Spring_D13`, `LN_y23_coop_feedbin_fail` |
| Y2 Spring D28-D29: a hen on the farm, a day and a rescue lost in the shop yard | an A pulse at the feed bin lifts a hen (`held_item_block` on every egg); EXIT_COOP carries her out; `stray_hen_px` (map byte != coop) then plans `HEN_DELIVERY` to the shop yard, where the to-farm walk never exits (20k frames) | `CoopChoresTask` puts a lifted hen down (walks to open floor first); `stray_yard_hen_px` limits the errand to town map bytes | `PR_t1_s1_Y2_Spring_D29`, `LN_y23_coop_holding_hen` |
| Y2 Summer D13, Fall D8: rest of day lost | talk/brush stalls, recovery exits the barn, retry fails the barn map lock | `splice_barn_reentry`, same shape | `PR_base2_Y2_Summer_D13` |
| Y2 Summer D4 and D16: 8 rescues, run over (plan item 1 below) | two farm holes lead to the Harvest Sprites' cave, map `0x2A`: (50,55) tile `0xC7` (the far-east escape charge "left 90, up 80" from (873,937)) and (37,48) tile `0xC8` (a walk west along y=857 then up at x=598); no exit found, scene reader says dialogue (stale text `0x031A`/`0x032A`) | `hold_back_from_pit` in `sanitize_action` holds a step whose next 12px is either tile | `PR_p60b_Y2_Summer_D4`, `LN_y23_m2a_pre`, `LN_y23_m2b_approach` |
| Y2 Fall D19 (festival eve): 8 rescues, run over | on a festival eve the location reads FESTIVAL on town/path/mountain and any `player_state` outside {0,1} was a cutscene: a tired farmer (state `0x09`, empty hands, walks ~4x slower) was mashed A/B for 241f x 8 | `_walking_state`: bit `0x01` plus carry/tired/ride/bath modifiers is free; `0x05` (tool animation) and `0x40`/`0x80` stay cutscenes | `PR_base2_Y2_Fall_D19`, `LN_y23_festeve_carry` |
| Y2 Spring D30: a rescue | a calf parked in the barn doorway overlaps the exit lane from x~142; the push down at x=139 stood still for 1,800 frames (the stall counter also reset on the 347<->348 jitter) | `_barn_exit_action` sidesteps left (128, then 144, 120) after 45 still frames; a push into a body counts as still within 1px | `LN_y23_barn_exit_fail` |
| Y3 Spring D20 (`y3a`): planner hung 1.6M frames at 17:00 | `DayFloor.note_bad()` (any recorded phase failure) set `_tripped`, which disarmed `check()` for the rest of the day, so a stall after an optional `CLEAR_FIELD` failure was never cut | `_bad` is separate from `_tripped`; the budget still trips on a bad day | `PR_y3a_Y3_Spring_D20` (stall cause below) |
| Y2 Spring D19 (jitter): 8 rescues, run over | an A pulse beside the parked horse mounts it; the door refuses a rider; enter timed out 3 x 11k frames | `dismount_horse`: an A pulse in the door-stand zone only, input free, farmer idle | `LN_y23_door_p150` |
| Y2 Spring D10 (`t4_s42`): 8 sleep rescues, run over | the 5pm shipping scene starts at `input_lock` 0 and opens its own box (lock 2) about 180 frames later, but the planner pulsed A during lock 0 (pending flag / stale text `0x031A`): that A set `game_state & 0x0800` for the rest of the day, the bed ignored every press, and the "dog release" task cannot clear it (the sprites near the farmer were the horse family `0x1F8-0x203`, not the dog) | `shipping_scene_needs_dismiss` presses only once input is taken (lock != 0); the scene completes identically and `game_state` stays `0x4401` | `PR_t4_s42_Y2_Spring_D10` (does not reproduce on its own), `LN_y23_dogrise_d10`, `LN_y23_dogflag_d10` |
| Y3 Spring D20 (`y3a`) and Y2 Fall D23 (`n150`): hang and unfed herd, both caused by the first `dismount_horse` | it held A on every frame while mounted, anywhere: the walk froze at (259,376) (herd unfed, three cows died) and the 17:00 shipping box never saw an A edge (clock at 17:00 for 1.6M frames) | the zone/idle/pulse rule above; the 5pm test fails against the old filter | `LN_y3a_frozen17` |

| Y1 Spring D2 from power-on (e2e `base0`/`base1`): `D2_FARM_CLEAR` blocked `stale_farm_map`, run over at frame 52,254 | the seed-shop exit lands on the door apron and the town fades in for ~190 frames; `town_coords_settled` waits for the fade (`screen_faded_in`) but the settle allowed 180 frames, so `BUY_SEEDS` failed and the farmer clear-planned from town | the post-buy settle gets `TOWN_SETTLE_FRAMES` (480) | `PR_e2ebase0_Y1_Spring_D2` |
| Y2 Summer D18 (`t5f_s54`, forced failures): 8 rescues, run over | the farmer left `CROP_WATER` holding item `0x57`; the yard (`0xA8`) refuses every toss and the pond fallback only took debris | `farm_toss.pond_takes`: the pond takes anything but a crop, egg, milk, fodder or hen | none (the day end state was not saved) |

| Y2 Winter D24 from power-on (e2e `v1n`): 8 sleep rescues, run over at 5.7M frames | `kid1_age` 60 plays the birth scene: a doctor fills the doorway, villagers seal the bed stand, the house clock does not tick, so the farmer can neither leave nor sleep and waiting never clears it | `LaborMorningTask` (called from `ExitToFarmTask`): wife talk (second row), idle for the birth, chat with the east villager, naming talk, back to the wake spot; the ROM then ends the day | `PR_e2ev1n_Y2_Winter_D24`, `LN_labor_whatif_reentry` |

Not wedges: Y2 Spring D1 (New Year) is a ROM-forced bed day (cow happiness and status are
unchanged across it); the "EXIT_TO_FARM dialogue dismiss timeout" on it is cosmetic.
`--start-jitter` (neutral frames in the house) was tried and removed: the ROM is frame-identical.

Open, in order:

1. Finish the sweep. `t4f_s44` (forced failures, tree before the shipping-scene and pond fixes)
   reached the credits from `PR_y1b_Y2_Spring_D1` with one rescue; the power-on proof on the final
   tree is `recordings/e2e/v2` (see [Y1_BENCH.md](Y1_BENCH.md#end-to-end-power-on-runs-and-the-saddle-ab-2026-11-01)).
   Open stall from `t4f_s44` Y2 Summer D25: after `COW_CHORES blocked by a cow` the re-entry splice's
   `EXIT_BARN` timed out at (180,316) and (198,316), both right-aisle lanes (a cow in the lower
   right aisle); a third lane or a wait-for-the-cow step is untried and six jitter replays of the
   morning did not reproduce it. A second draw hit the same shape on the left lane: `t6_s62` Y2 Summer
   D19, `COW_CHORES` fine (milk 3, fed 4), `EXIT_BARN` timed out at (38,316), tile (2,19), one rescue. Both
   stop 10 pixels above the lower aisle (y=326); a cow standing in the lower aisle is the suspect.
   The house door shows the other recoverable shape: `t6_s61` Y3 Summer D15 18:09, six rescues in one
   day (`return_home timeout ... phase=enter_house`, four 11k-frame retries each, farmer stuck at
   (136,371), tile (8,23), hands clear), then it got in. The farm dog loiters on the house front after
   18:00 (probe: sprite 0x1D9..0x1F7 at (120..154,374)), but a dog poked onto the walkway did not block
   the push, so the blocker is unproven; the retries repeat the identical push and never vary the
   approach. Neither reproduced from the morning pin with six jitter seeds each.
2. The Harvest Sprites' cave (map `0x2A`) has no exit the bot knows: 85 reachable cells from the
   spawn (184,360), no hop leaves the map, A pulses on the ladder do nothing. The guard prevents
   entry; a cave exit would make any other entry survivable.
3. Evening `CLEAR_BUSHES`/`CLEAR_FIELD` walk to the far corners after 18:00 (weeds count only for
   the 50-point development score) and are where the wedges above start. Bounding them to the
   home half of the farm is the cheap safety margin if one more far-corner wedge shows up.
4. Time hot spots (Y1: CROP_WATER 29% of frames at 263 f/tile, 76% of it navigation between
   stands; COW_CHORES 11%; spa soaks 8%; harvest 7%). Y2 Spring days trip the 48k budget on D3,
   D9, D11, D12 (4 cows + nine rings: watering 9-11k a pass, soak 5k, second pass 7k, seed trip
   4-8k, tools 1-2k). A/B ring order, can refills and the seed-trip cadence only after the
   sweep is clean.
5. `n150` Y2 Fall D23 and `y3a` Y3 Spring D20 were both the first `dismount_horse` holding A while the
   farmer rode the horse near (255-259,362-376); both are fixed and covered by ROM tests. Rerun a full
   sweep to confirm no further hang (the sweep's tree has the fix).
6. The Summer D14-type mountain pocket: two NPCs on the spa road send the BFS north into a
   weed-filled pocket (tile `0x35` counts as walkable) before the recovery walks home. No
   longer fatal, costs ~2k frames.

## Years 2-3: finish the game with the livestock farm (2026-10-29)

Goal order (owner): finish power-on to the Y3 Summer D30 ending without a
wedge, then tune for cash. A run ends only when one day burns
`MAX_RESCUES_PER_DAY` (8) rescues; the hour stops at 18:00 and only the bed
ends a day, so any wedge that keeps the farmer out of the house is fatal.

The current code completes a fresh power-on run (2026-10-30): Y3 Summer
D30 credits complete, ending index `0x25`, RAM score **195**, 5,022,801
frames, 6,139.9 s wall. Eve remains married; the first house upgrade is
complete and the second unfinished. End wallet: **26,730G**, **two cows and
eleven chickens**. Audit: zero initial loads, mid-run loads, RAM writes or
assists. All route and ledger source hashes match the launch snapshot; only
the independent scorecard formatter changed during verification. Report:
`recordings/y1_bench/completion.json`; receipts:
`debug_completion/{source_hashes.json,scorecard.json,season_sales.md}`.

It finishes with **66 bad-day journal rows, 36 floor trips and nine rescues**.
On 98/137 days with animals the recorded snapshot shows all adults fed
(71.5%); these snapshots precede late feed retries and can undercount them.
Both cows survive with status `0x29` (adult, sick). The last workday's barn
retry fails at the coop exit, leaving both cows and four hens unfed. This is
not a healthy-care or Perfect result. Last-day hay falls from 15 to 13. Reproduce barn/coop
exits and milk approaches before tuning cash, and repair the weak first
Spring (four potatoes shipped). The current task does not promote STATUS.

The earlier `y23_final` baseline reached score 120 in 5,996,523 frames,
6,780.5 s wall, with 15,680G, zero cows, three chickens, both houses complete,
90 bad days and 33 rescues. Relative to it the current score rises by 75,
wallet by 11,050G, and the herd is retained. Different timing and routes mean
this is not an isolated A/B of any single repair.

[Completion specs and all ten season sell totals](Y1_BENCH.md) are now
recorded from a read-only frame ledger. Total bin deposits are **60,370G**,
receipted hen sales **3,500G**, and recorded gross sales **63,870G**. The
closing bin still contains 50G, so paid sales are **63,820G**. Crop counters
reconcile to 102 potatoes and 78 corn. Milk/egg attribution is estimated;
unattributed deposits stay explicit. New Year calendar transients with zero
sales are omitted, and the final Summer D29→D30 journal entry is retained.

Changes and remaining losses (fixes have non-ROM tests and pin replays):

| wedge or loss | cause | fix | pin |
| --- | --- | --- | --- |
| Summer D27 Y1 and Spring D10 Y2: 8 rescues holding a stone | the far-east escape's A lifts a stone; the yard (`0xA8`) and the drop spot refuse every toss | `PondTossTask` carries debris to the pond lip (pond first when nearer) | `LN_y23_wander_d27` |
| Y1 Fall D2, `review_fixes` power-on bench: 50k+ frames in return-home recovery | `CLEAR_FIELD` left held stone `0x0D`; drop/wander retries failed in the north yard around (265,70) | still open: diagnose the north-yard carry escape before repeating the full-year bench | power-on log and progress sidecar in `recordings/y1_bench/` |
| Y2 Spring D8: 4 rescues at the door | a sprite (object `0xC7`, likely the horse) parked on the stand (8,26); return home walked back to it | push up from any tile of the walkway (8,22)-(8,26) (`in_house_walkway`) | `LN_y23_door_y2d8` |
| Y2 Summer D6: 8 rescues, run over | 0x72/0x75/0x76 are both "stale viewport" ids and pasture grass stages; 57 grass tiles read as an unloaded farm, so the house-wood stump clearer held north-west into the barn | `farm_map_loaded` skips `PASTURE_BOUNDS` | `LN_y23_wood_y2s6` |
| Summer D17/D24: 2 rescues each, `no_path sealed` | grown west ring + Summer fence seal the strip east of the ring | `nav_house_front` may use the x=11 bin lane | (log only) |
| shop trip bounced off the town gate | a villager push-faced the east-gate run back onto the path | `TOWN_BOUNCE_RETRIES` walks back in | `LN_y23_summer_d27` |
| 12 of 13 Fall mushroom trips | a pixel-stuck replan at the mountain landing BFS'd off the south exit | north pull after landing, shared with the spa route | |
| Fall D13: 18k frames of sale wait | a late baseline misses earlier payment; the first-Fall trade can also take a hen without gold | retain pre-request count/wallet baselines through farm re-entry; require both the hen count drop and 500G per sale | |
| Y2 Spring: animals unfed until Summer | Winter wipes the pasture and `planted_grass`; grass was Summer-only | sow from Spring D1 when a herd exists; Fall cut stocks `SPRING_HAY_DAYS` | |
| Y2+ peddler takes 20,000G (still open) | text `$014D` "Buy it" is row 0 and every dismiss pulses A | `peddler_decline_action` has unit coverage but the live continuation still buys: Y2 Winter D18, 30,620→10,620G | `y23_flower_to_ending` log |
| Y2 Winter: wood trip strands the farmer and every farm phase fails | `ExitToFarmTask` sends neutral during the mountain exit's `player_state=0x81` transition, which needs Down held; Winter path `0x0F` was also absent from its known returns | keep the bounded outdoor route driving map transitions and recognize the Winter path | `LN_y23_winter_source`, `LN_y23_winter_wood_failure` |
| Y2 Spring D23: eight rescues on the Flower Festival map | Eve is already at the courting target, so the day plan omits attendance; an incidental town exit enters the festival and the generic escape cannot start the dance | `FestivalEscapeTask` delegates map `0x08` to the existing `FlowerFestivalTask` | `LN_y23_flower_failure` |
| Y3 Summer D9: dog flag keeps every bed attempt dead | `DogReleaseTask` replaces `ExitToFarmTask` on the first raw farm id, before the remodeled-house warp settles; the farmer is boxed at (136,344) and never reaches the dog | finish the exit child before starting dog navigation | `LN_y23_dog_failure` |
| Y3 Spring pasture softlock holding stone | grass planter lifted stones without a hammer; in-place toss failed and farmer carried stone into town and bed | grass planter clears weeds only (`frozenset({WEED})`); unarmed debris clearing sorts weeds first (`drop_unarmed_debris`) | `PR_Y3_Spring_D2` |
| Farm development tile deficit (<2,950 tiles) | weeds (`0x03`) do not count for development score; idle afternoon daylight was wasted going to bed at 14:00 | `CLEAR_BUSHES_PHASE` runs during afternoon daylight before 18:00, clearing 100+ weeds/day with zero stamina | `PR_Y3_Spring_D2` |

The continuation after the festival fix married Eve on Y2 Fall D8, then
reproduced the dog-release wedge on Y3 Summer D9. Preserved report:
`recordings/y1_bench/y23_before_dog_fix.json`. On its exact failure state,
the fixed release clears `game_state & 0x0800` in 1,046 frames; the normal
sleep task reaches D10 in another 1,092. Non-ROM and audited ROM regressions
pass. The previous power-on run was deliberately interrupted at Y2 Summer
after this new fix, and its snapshot remains in `debug_y23/before_dog_fix/`.
The current `debug_y23/final_run/` includes all fixes and its power-on
validation is complete, as recorded above.

That last continuation now completes the credits on the current code:
`recordings/y1_bench/y23_dog_to_ending.json`, 434,133 frames, score 123,
8,850G, Eve retained, both house upgrades, two cows, two hens, zero rescues.
Audit: one initial load, zero mid-run loads, writes or assists. It includes
the second-house 10,000G/500-wood debit and successful nights in map `0x17`.
This is a checkpoint receipt; the separate full power-on proof is above.

The first validation run stopped at that new Flower Festival wedge; preserved
as `recordings/y1_bench/y23_before_flower_fix.json` and
`debug_y23/before_flower_fix/`. On its exact failure save, generic escape is
still on D23 after 25,000 frames; the scripted fallback reaches D24 in 10,586.
The whole-planner replay reaches D25 in 21,179 frames
(`recordings/y1_bench/y23_flower_check.json`). Non-ROM and ROM regressions
pass. The subsequent continuation exposed the dog-release wedge; the final
power-on run includes both repairs.

Winter replay (2026-10-29): the natural Y2 Winter D1 checkpoint from `y2r2`
does yield six lumber. The old return freezes at (328,730) and fails after
656 frames; the fixed return reaches the farm in 730. A three-day continuation
(`recordings/y1_bench/y23_winter_check.json`, 36,928 frames) reaches D4 with
wood 32→50, the cow retained and fed every day, and zero rescues. D2 has a
separate coop-navigation timeout. Keep Winter gathering enabled: the reproduced
failure was the return, not a seasonal absence of the stump.

The same fixed tree continues from that checked Winter D4 to Y3 Summer D30
credits complete: `recordings/y1_bench/y23_winter_to_ending.json`, 1,438,627
frames, score 119, 25,340G, Eve's wife bit retained, one rescue and 22 bad-day
journal rows. Audit: one initial checkpoint load, zero mid-run loads, writes
or assists. The remaining cow dies during this continuation despite the
restored feed chores; livestock health is still an open loss. This is a
checkpoint continuation, not the required fresh power-on validation.

The east approach's barn-row change has a geometry regression check and a
faithful ROM replay. Replaying the earlier `y23_c` tree from power-on captures
Summer D21 immediately after the far-east escape at (678,550), grown west
corn ring (`LN_y23_east_natural`). On that identical state, the old y=440
slide fails after 3,226 frames, boxed at (250,454), `no_path sealed`; the
current y=376 approach reaches the house in 1,384 frames. The dedicated
`test_grown_corn_east_return_clears_the_well_and_reaches_home` checks that
natural predecessor with no writes or mid-run loads. The earlier
`LN_y23_east_s1d21` checkpoint remains insufficient: it starts north of the
fence and bypasses the changed branch.

Resume a later year without replaying Year 1: a `--save-season-pins` run
writes `PR_Y{n}_{Season}_D1` with the planner state, and
`run_to_day2 --state PR_Y2_Spring_D1 --until-ending` continues from it
with checkpoint planner fields restored. Replays can drift after several
days; the full power-on run remains the economy and completion proof. Main
tree aliases from the final run are `LN_y23_final_y2_winter_d1` and
`LN_y23_final_y3_spring_d1`; other season pins remain in the frozen snapshot.

Open, in order:

1. Daily reliability and cow health: the current run retains both cows, but
   needs nine rescues, has 66 bad days and feeds every adult on only 71.5%
   of the recorded days with animals. Both ending cows carry the sickness
   flag. Reproduce barn exits, milk approaches, coop exits and the
   final wife-gift chain from this run's natural checkpoints before tuning
   cash. Its last workday leaves both cows and four hens unfed.
2. Feed when the hay runs out. The animal shop sells cow feed (text `$0341`,
   12 uses) and chicken feed (`$0342`, 300G) from shelves inside the shop
   (tilemap `0x24`); the cow purchase already meets the cow-feed prompt and
   declines it (`CowPurchaseTask._advance_to_name`). Buying a bag and using it
   as a tool on a trough (`ToolUsedCowFeed` takes trough tiles `$E2`-`$EE`)
   would keep the Spring Y2 gap and a short Winter from starving the herd.
3. Rescue-costing traps that still recover: the west-ring centre box
   (`no_path sealed`, the rescue walk crosses tilled `0x07`), a villager
   parked on the lower town road (`Yield to sprite` then 4 pixel-stuck
   replans), coop entry that drops straight back to the farm (Y2 Spring D8,
   `COOP_CHORES BLOCKED: not in coop`).
4. Cash: Year 2+ now targets four cows, allows Spring animal purchases and
   retains the ending flock from Y3 Spring D22. The current run still ends
   with only two cows and eleven birds. Score full runs with
   `harvest.tools.y1_bench --sales`; its tables distinguish deposits, paid
   sales, purchase receipts and wallet change. The old continuation's
   20,000G peddler loss predates the pending-menu guard; the current full run
   never reaches the 30,000G trigger, so that guard still needs its separate
   native-menu regression whenever changed.

## Repair pass, 2026-09-30

Merged fixes now feed actual cow slots, retain unfinished milestone intents,
select native tasks before the advisor, and require immutable RAM receipts.
Use `run_to_day2 --perfect-rancher` to enable campaign dispatch; ordinary
farm runs retain their rule order. Unsupported acquisitions defer explicitly.
A cow receipt requires a newly initialized slot as well as the count change.

Merged native evidence:

- Twelve-cow feed/night fixture: 7,507 frames, feed flags 0x0FFF,
  hay 500→488, all twelve happiness values remain 200 after the Spring D16→17 sleep.
  The alleged missing slot-0 stall was a stale filled tile in an edited
  research pin. Native barn re-entry restores the trough. No replacement
  calf is required for slot 0; acquisition and fifty care days remain open.
- Dispatch fixture: legacy and milestone chicken purchases each take
  4,028 frames, count 0→1, wallet 4,090→3,090G, verified native receipt.
- Structure comparison: baseline, seeded changes and folded reporter each
  take 17,538 frames on the same D3 farm day, with identical end states.
  Runner/report/composer/multi-day sources meet the 1,000-line bar.

These are fixture proofs with zero live writes, mid-run loads or assists.
They do not establish natural campaign arrival or a 999 result.

## Next blockers

| Priority | Beads | Required evidence |
|----------|-------|-------------------|
| P0 | Exact-ten berries `rr-ux6x.16.7` | Execute native 7→10, guard 11 and preserve count/stamina after final sleep. The cabin quote/arm/clock gift now have a native first-house receipt below. |
| P0 | F7 `rr-ux6x.7.8`, F5 `rr-ux6x.7.6`, solver `rr-ux6x.8.2` | Validate one shared dated budget. Conditional YAML proposes marriage at index 128 plus 105 food repairs costing 31,500G; buy-three/breed-nine is an alternative. Repeated food, herd naming/feed and the complete Perfect cash schedule remain unproved. Basic first-house lumber and payment are received below. |
| P0 | Contract `rr-ux6x.4` | Resolve the handoff's parked questions versus ADR 0007's recorded duplication/load/fallback rulings. Preserve the recorded ADR; no new owner ruling is inferred. |
| P0 | Structure behavior `rr-ux6x.18.1` | Full ending requirement remains 2,175,350 frames. The 3,476,992 run changed policies/features, including daily clearing; the matched one-day proof passes. Do not close the full-run gate from that short comparison. |
| P1 | Dispatch `rr-ux6x.8.3`, arbiter `.8.4`, herd `.11.3/.11.5/.11.8`, social `.14.6/.14.7` | Current dispatch proof covers one purchase. Prove all due native domains and a real 12-cow/12-chicken day, measured care/medicine, and social windows. Model checks remain coverage only. |

Wife is Eve. Marriage can land in Fall or Winter. A cow comes first:
pasture stays on the calendar, and the first cow buys at 5,300G instead of
waiting behind the 6,000G house-and-feather fund. The carpenter payment,
including the 10,000G second house, waits until that cow exists. A Fall or
Winter first-house order does not arm the clock.

The first-house/marriage route now has native receipts
([EVE_BAR.md](EVE_BAR.md)): power-on festival decline preserves the wallet;
axe work banks 252 wood; corn funds the 5,000G/250-wood order on Summer D28;
four nights complete the house on Fall D2; the natural Sunday seller sells
the feather on Fall D3. The isolated wedding continuation finishes on Fall
D10 with Eve's wife bit set and retains it through two more nights. These
are Clean continuations from real power-on predecessors, with zero RAM writes
or mid-run loads. That route spent the summer wallet before any cow. The
Fall D18 resume (`LN_house2_wood_natural`) still has 0 cows, 680G, and
`wife_pregnancy` 9. The full campaign and Perfect result remain open.

The cabin's introduction needs a native exit/re-entry before the request
sign. The Summer order arms the clock, and the completion scene gives it;
`$7F1F6E & 0x4000` is set on the natural wedding checkpoint. First-house
materials, request, payment, construction, feather and wedding each have a
ROM check in `tests/test_house_upgrade.py`. House upgrade 2 is quoted
(500 wood, 10,000G) and one mountain chop from the married pin is in
[EVE_BAR.md](EVE_BAR.md); the order and the 450/650 pregnancy floors are
still open ([F4](perfect_rancher/research/F4_family.md)). Exact-ten berries and the
complete dated Perfect budget still need their own campaign receipts.

The [F10 verdict](perfect_rancher/research/F10_verdict.md) remains conditional:
seven designed berries give a table ceiling of 984 if every other term is
maxed. Neither 984 nor 999 has a verified campaign route. The calendar's
15 sightings cover five girls on three sampled days; native interactions
remain to be composed.

## Gradual relationships, 2026-11-06

All five **native bought-flower** deliveries are isolated and ROM-tested in
`tests/test_relationships.py`: Maria +4, Ann +2, Nina +6, Ellen +5, Eve +8.
The purchased flower is held item `0x19` and costs 300G; wild flower `0x05`
has a different ladder. See [F9](perfect_rancher/research/F9_hearts.md).

The optional `HARVEST_SOCIAL=off|talk|gifts` shopping hook now composes a
surplus-funded Nina flower and a small Maria conversation detour with an
existing seed trip; nearby Ellen conversations fit lower-town errands.
It records actual heart deltas, reserves farm/final cash, stops at 496,
shares visits for the day, and leaves Eve's wife routine responsible for her.
All five heart counters are now in daily economy snapshots.

Remaining, in order:

1. Finish the fresh power-on Year-1 `off`/`gifts` A/B and compare farm income,
   animal purchase dates, shipping deadlines and measured affection gains.
   Keep the hook explicitly opt-in until the economy result justifies it.
2. Prove gradual daily composition on a native multi-day checkpoint without
   starving chores/crops, then extend the measured schedule to Summer and
   weekends. The Spring-only hook cannot yet promise daily coverage of all
   five or 155 ending relationship points.
3. Carry inexpensive forage/eggs to Ann and Ellen along existing trips. Do
   not buy a 300G flower for Ann's +2 when a grape gives +5/+6. Reconcile
   spare eggs with Eve's pregnancy/child targets before diverting them.
4. Budget enough visits to reach 496–511 for every non-wife; finish the
   wife's second-child 650+ stage and safe descent separately. Verify the
   final natural predecessor, including festivals and the last-night -2.

## Constraints

Wife hearts must be 498–511 before hour-18 entry subtracts 2; Maria's
25 poison-mushroom offers measure 660→510, but the real wife approach is
still needed. Wife bit by Y2 Fall D9 suffices for children present at D29;
index 128 is the conditional earlier target for child 2's age-60 bonus.
Development needs 2,950 of 2,968 cover candidates after the final night.
Girls and crop counts need 496–511, happiness ≥992, money
99,840–655,350G after overnight credit, twelve cows ≥200 and twelve chickens.
Missed Fall D20 prizes need recovery; berry count above ten fails the best
ending check. Advance year checkpoints only from their real predecessors.

## Existing farm spine

Keep short-window work in
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md): summer-entry
Gate B (`rr-5in`), weak spring yield (`rr-bu62.25`), cow talk timeout
(`rr-bu62.30`), crop stand weeds (`rr-bu62.8`), name-entry warps
(`rr-bu62.18`/`.26`), dog release (`rr-bu62.21`/`.23`), and the money
visitor (`rr-bu62.14`). Purchase success must include a new initialized
cow slot after naming, not only wallet/count changes. The seasonal purchase
recipe remains [livestock_purchase_plan.md](livestock_purchase_plan.md).

The earlier 2,175,350-frame working-farm credits proof remains in STATUS.
The audit changes planning and evidence claims; it does not promote a
Perfect result or replace the verified product baseline.

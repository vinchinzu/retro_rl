# rr-npv.3 review — Clean L7 bait / Digdogger

Lane: `rr-npv.3` (exclusive `level7/**`). Fixture-live only.
Reviewed uncommitted diffs vs HEAD: `level7/{digdogger,hops,hungry,spine}.py`,
`tests/test_level7_{digdogger,hops,hungry}.py`, `docs/LEVEL7_ROUTE.md`,
`docs/tasks/rr-npv.3-residual.md`. No probe_level7 scripts remain to change.

## Summary

The Food-poke contract holds. `l7_hops(survival=False)` still builds
`NaturalBaitPurchaseController` (no `poke_food`). `continue_level7_spine`
forces `survival=False` when `run.allow_pokes is False`, and skips
`SPINE_L7_RUPEE_RETOPUP`. Hungry fail-closes on Food 0 with leftover
`hungry_goriya_requires_food`. Digdogger is a `HopController`: `STAND_SETTLE`
and the scripted 12×B hop are gone; dest is RAM (`0x38` / shrunk `0x18` /
play `0x0C`). No STATUS.md edit; LEVEL7_ROUTE and the residual both say
do not STATUS. Exclusive-tree writes for this sitting stay in `level7/**`
plus the listed tests/docs.

Acceptance TF `0x40` is not landed. Residual is honest: `Level7Entrance`
(Food 0, bombs 0) greens `level7_entry_first_door` 251f and first-reds
`level7_room69_west_bomb` timeout 22000f, leftover play `0x69` `(34,165)`
mode 8 health `0x20`, `food_poked=false`, deaths (mode 17) 0, Hungry
unreached. That is fixture-live, not a STATUS claim.

One correctness regression in the Digdogger BLOW rewrite: a single empty
object-slot frame now EXITs north (old code waited 240f in `BLOW_WAIT`
before `large_gone_after_blow`). Unit tests (92 passed in
`test_level7_{digdogger,hungry,hops,overworld}`; 1 ROM skipped) do not
cover that path. The two `test_level7_overworld.py` reds are
`farm_wait` vs expected DOWN/`off_east` and come from dirty
`overworld/path.py` (spine lock `rr-ps7.3`), not this sitting.

## Issues

### Issue 1 -- Severity: bug
- File: nes/zelda_i/level7/digdogger.py:342
- Description: BLOW treats one frame of “no type `0x38`” as a kill and
  walks north. Old `BLOW_WAIT` only took `large_gone_after_blow` after
  240 idle frames, which covered the whistle split (slots empty, then
  minis `0x18` spawn). New code:

  ```
  if not _large(snap):
      self.shrunk = True
      self.killed = True
      self._set_phase(DigdoggerPhase.EXIT, "large_gone_after_blow")
      return self._exit_north(snap)
  ```

  EXIT (`digdogger.py:358`) never re-checks `_shrunk_live` / `_large`, so
  minis that appear after the empty frame are ignored. `arrived()` can
  then green dest `0x0C` with `self.shrunk` set without ever seeing type
  `0x18`. `_sword` already has `EMPTY_SWORD_FRAMES` for this gap; BLOW
  skips it. No unit test plants empty slots during BLOW. Digdogger was
  not ROM-run this sitting (first red is room 0x69).
- Suggestion: If `_large` is gone and `_shrunk_live` is empty, stay in
  BLOW until `BLOW_WAIT_FRAMES` or enter SWORD so the 30f empty-slot
  confirm can run. Do not set `killed` / EXIT on the first empty frame.
  Add a fake-RAM test: plant `0x38`, start BLOW, clear the slot one
  frame, plant `0x18` hp>0, assert phase is SWORD not EXIT.
- Status: resolved

### Issue 2 -- Severity: suggestion
- File: nes/zelda_i/level7/digdogger.py:354
- Description: BLOW now holds B every play-mode frame for up to 240f
  (`blow_presses += 1` + `nes_action("B")`). The old hop was 12 taps
  then RAM-wait. Recorder use is edge-triggered; a held B is one blow
  if the dungeon stays in mode 5. Retry still exists (`BLOW_ATTEMPTS`),
  but each attempt is one edge plus 240f of held B, not 12 taps. Not
  ROM-checked on 0x1C.
- Suggestion: Press B on the rising edge (or a short tap), then idle
  while watching `_shrunk_live` / `_large` until `BLOW_WAIT_FRAMES`.
  That is still leftover-relative dest, without `idle(n)` as a required
  settle hop after walking to the stand.
- Status: resolved

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/level7/digdogger.py:197
- Description: `_bind_stand` only copies leftover Y when already within
  `ARRIVE_TOL` of 141; otherwise stand stays hardcoded `(120, 141)`.
  Off-row leftover `(120, 125)` is asserted to keep that stand
  (`test_level7_digdogger.py:119`). Walk-from-leftover is leftover-
  relative; the stand bind is not. Dest-is-RAM is the real gold-standard
  piece.
- Suggestion: Either drop `_bind_stand` (walk is enough) or bind from
  leftover door-row Y without the “already on 141” gate, and test a
  non-141 leftover if that is the intended pose.
- Status: open

### Issue 4 -- Severity: suggestion
- File: nes/zelda_i/tests/test_level7_hungry.py:136
- Description: Clean “never writes `ADDR_FOOD`” is only partly
  behavioral. `level7_entry_chapter_stages(survival=False)` is stepped
  and does not poke; `continue_level7_spine` is only
  `inspect.getsource` for `"allow_pokes"` / `"survival = False"`
  (`test_level7_hungry.py:138-140`).
  `test_continue_level7_spine_uses_the_survival_bait_fixture`
  (`test_level7_hops.py:401`) still only asserts `"survival=True" in
  src`, which matches the default kwarg even if the `allow_pokes`
  branch were deleted.
- Suggestion: Call `continue_level7_spine` with a `SpineRun`-shaped
  stub (`allow_pokes=False`) and a patched `attach_hops` / `l7_hops`,
  and assert the bait stage is `NaturalBaitPurchaseController` and
  `rupee_retopup` is empty. Keep the `survival=False` factory test.
- Status: open

### Issue 5 -- Severity: suggestion
- File: nes/zelda_i/level7/path.py:913
- Description: Residual first-red is `level7_room69_west_bomb` timeout
  after kill-clear, not `no_bombs`. `Room69WestBombController.step`
  fights live goriyas (`path.py:913-920`) before
  `BombWallController` can `_fail("no_bombs")`. Pin bombs=0 / 3 hearts
  therefore burns 22000f and dies into mode 8. Hungry
  `hungry_goriya_requires_food` is downstream and unreached, as the
  residual says. Pre-existing hop; leftover is still dishonest.
- Suggestion: Fail-closed on `snap.bombs <= 0` (and optionally Food 0
  before Hungry) before the goriya loop, or cap combat when hearts are
  empty. Do not change STATUS.
- Status: open

### Issue 6 -- Severity: nit
- File: nes/zelda_i/level7/hops.py:458
- Description: `level7_bait_shop_chapter_stages` has no bait-purchase
  stage (post / warp / shop approach only). The new docstring says
  “Clean `survival=False` keeps `NaturalBaitPurchaseController`”, which
  is a different chapter. Same WHAT restated at `hops.py:9` and
  `hops.py:579`.
- Suggestion: Keep “No Food write” on the shop chapter. Move the
  Natural vs Survival bait sentence to `level7_entry_chapter_stages` /
  `l7_hops` only.
- Status: open

### Issue 7 -- Severity: nit
- File: nes/zelda_i/level7/digdogger.py:94
- Description: `DigdoggerPhase.FAILED` is never set. `mark_fail` (death,
  dest_without_kill_edge, whistle_did_not_shrink) leaves `phase` at
  WALK/BLOW/SELECT, so `report()["phase"]` after a fail is the last
  live phase. Old `_fail` set FAILED.
- Suggestion: Set FAILED in a thin wrapper around `mark_fail`, or drop
  the unused enum member.
- Status: open

### Issue 8 -- Severity: nit
- File: nes/zelda_i/level7/hungry.py:155
- Description: Pause-select failure calls `_fail(self._select.fail_reason)`
  without `snap`, so `leftover` stays `None`. Food-0 / death / budget
  paths pass `snap` and fill leftover. Asymmetric for the leftover
  this sitting added.
- Suggestion: Pass `snap` on the pause-select fail path too.
- Status: open

### Issue 9 -- Severity: nit
- File: nes/zelda_i/tests/test_level7_hops.py:1
- Description: File was already 1024 LOC; this sitting adds 6 lines
  (Hungry leftover + `writes==0`). Soft max ~1000. `level7/path.py`
  is 1301 and was not grown. `digdogger.py` 677→681, under the cap.
- Suggestion: Do not keep stacking onto `test_level7_hops.py`; new
  Clean assertions belong in `test_level7_hungry.py` (already done for
  the poke test).
- Status: open

## Checks that passed (not issues)

- ADDR_FOOD is not written on Clean: `NaturalBaitPurchaseController.step`
  (`entry.py:127`) only reads; `SurvivalBaitPurchaseController` is
  survival-opt-in (`hops.py:433-437`); Hungry only reads `ADDR_FOOD`
  (`hungry.py:175`).
- `idle(n)` is not a Digdogger hop: `STAND_SETTLE` / `stand_settle` are
  gone. Remaining idles are scroll / empty-slot confirm, not an 8f
  stand hop. (Pond drain in `pond.py:102-116` still has 8f settle +
  12×B; out of this sitting’s Digdogger claim.)
- Exclusive tree: this sitting’s diffs are `level7/**` + listed tests +
  LEVEL7_ROUTE + residual. Working tree also has sibling dirt
  (`overworld/path.py`, `dungeon/{hop_controller,door_hop}.py`,
  `scripts/{run_survival_spine,level8_clear_lab}.py`); those are
  `rr-ps7.3` / `rr-npv.7` / `rr-npv.4`, not this worker.
- STATUS: `docs/STATUS.md` untouched. LEVEL7_ROUTE new block and
  residual both say do not STATUS. `route_eligible` stays false on
  Hungry and Digdogger reports.
- Verified tests: `test_level7_digdogger.py`, `test_level7_hungry.py`,
  `test_level7_hops.py` green. `test_level7_overworld.py` 2 reds
  (`test_post_l6_south_leftover_goes_down_not_into_cave`,
  `test_post_l6_0x13_east_mouth_left_not_down`) are `farm_wait` from
  `overworld/path.py`, not L7 bait/Digdogger.

## Verdict

Accept the Food-poke off / Hungry leftover / HopController shape as
fixture-live. Do not STATUS. Block on Issue 1 before treating forced
Digdogger as leftover-relative dest: empty-slot EXIT is a shrink-window
race the old 240f wait existed to close. Issues 2–5 are follow-ups
(B hold, stand bind, spine test, west-bomb leftover). Residual’s first
red and “do not retry this pin without bombs+Food” stand.

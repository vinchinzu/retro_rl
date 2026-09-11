# rr-20w — full-spring D3→D30: run16 baseline + fix pass

Session 2026-09-11. Evidence: `logs/spring_d3_30/run16_full_spring.log`
(`run_to_day2 --state Y1_D3_Morning --end-of-spring`, HEAD as of `181e846b`).
Baseline launched *before* this session's edits, so it prices the previous
session's nav work on its own.

**Do not STATUS.** Do not start from `Y1_D2_Morning_After_D1`.

## Fixed this session

Each entry names the run13 defect it closes, the mechanism, and the test.

### 1. Silent watering drop (run13 §4) — `crop_water_ops.py`

`_reorder_remaining_water_steps` rebuilt the remaining-step list from
`_best_water_variant`, and `continue`d past any target whose stands were
unreachable *at the instant the reorder ran*. The target was gone for the
rest of the day, with no log and no counter — run13 D3 declared `steps=8`
and reported `WATER DONE: 7/7 watered`, because the list had shrunk under
it. (12,28) then stayed at `0x58` while its siblings matured to `0x60/0x61`
and was correctly not harvested.

Unreachable-now steps are now **deferred to the tail** with their original
stand/face, counted in `_water_steps_deferred`, and logged. WATER DONE
prints `deferrals=N` so a shrunken plan can never again read as complete.

Test: `tests/test_crop_water_reorder.py` (5) — including that a deferred
target is retried once it becomes reachable.

**ROM A/B, same `Y1_D3_Morning` pin, D3 second plot:**

| run | code | plot 2 | day total |
|-----|------|--------|-----------|
| run16 | baseline | `WATER DONE: 7/7 watered` | `watered=15` |
| run17 | fixed | `WATER DONE: 8/8 watered deferrals=6` | `watered=16` |

run17 logs `WATER defer 1 tile(s) ... [(12, 28)]` six times, from six
different poses — the exact tile run13 §4 named — and waters it on the
seventh. Under the baseline that same tile is gone from the plan after the
first reorder, which is why run16's own harvest reads
`harvested=7 ... harvested=1` (the west ring short one cell, picked alone
later) while its second ring reads `harvested=8`.

### 2. `(11,28)` was both no-go and a staging stand (run13 §4, secondary)

It sat in `FARM_NO_GO_TILES` (shipping-bin ditch) *and*
`FARM_POND_ACCESS_STAGING_TILES`. `find_path` never returned it, so this was
dead weight rather than a live bug, but it read like a usable stand — which
is how the (12,28) diagnosis went sideways. Removed from staging.

### 3. `select_carry_0x07` (run13 §2) — tool-lock phase gate

Not a carry-swap bug. On run13 D10 `BUY_SEEDS_WINDOW` blew its cutoff, no bag
was bought, and `CROP_ESTABLISH` then **hoed the entire ring** before
`select_carry_0x07` discovered the bag was never in the carry pair. An
afternoon spent to reach a fact that was settled at 06:00.

`DayPlanTask._phase_tool_lock` is the tool analogue of the existing
`_phase_map_mismatch`: a phase whose required **seed bag** is not in the two
carry slots is advanced as **`no_work`**, not failed, so the day moves on to
phases that can still earn.

Two exemptions, both found by watching the gate misbehave rather than by
reading it:

- `ENSURE_*` kinds (`ACQUIRE_TOOL_KINDS`) — their `required_tools` is a
  postcondition, so a missing tag is the reason to run. The first version of
  this gate skipped `ENSURE_CROP_SEEDS` itself; a side run caught it live
  (`Phase ENSURE_CROP_SEEDS tool lock: no_work:missing_tool:seed`).
- **Tools, as opposed to the seed bag** (`_UNFETCHABLE_TOOL_TAGS`). A missing
  watering can already routes to `EnsureCarryToolTask` via
  `_make_recovery_task` — it goes and gets the can. Gating it would have
  replaced a fetch with a skip, which is strictly worse; run16 hits exactly
  that path once (`CROP_WATER FAILURE: watering can not in carry pair`). A
  bag is different: it needs a shop trip behind its own window, so a missing
  one mid-day is settled for the day.

Also: `CROP_ESTABLISH` with no ring needing seed now reports
`no_work:no pocket ring needs planting`, so the journal stops counting it as
crop work.

Test: `tests/test_day_plan_tool_lock.py` (6).

### 4. `CLEAR_FIELD` burns whole days indoors (run13 §6) — root cause found

run13 §6 called this "something in CLEAR_FIELD's idle/failure exit path
re-enters the house" and marked it unaudited. The run12 D8 log shows the
actual sequence, and the speculation was close but the mechanism is simpler:

```
[DAY_PLAN] EXIT_TO_FARM -> tilemap=0x00          <- genuinely on the farm
[CLEARER] Target: WEED at (4, 14) (SICKLE)       <- far NW target
[CLEARER] Debug @ 300f  pos=(86,166) ... targets=0
[CLEARER] Debug @ 600f  pos=(38,126) ... targets=0
[CLEARER] Debug @ 3300f pos=(38,126) ... targets=0   <- frozen, 3000f
[RUN] f=66000 date=S0D8 06:00 ... [RUN] f=68000 date=S0D8 06:00  <- clock frozen
[DAY_PLAN] Phase NAV_CROP map lock: map_mismatch:have=0x15:need=0x00
```

The clearer walked in through the farmhouse door chasing that NW weed. **The
SNES clock does not advance indoors**, which is why 3300 frames pass at a
literal 06:00 — the freeze is the tell, not a hang. It then scanned
`targets=0` for its whole 3500 f budget, and every later farm phase map-locked
on `0x15`. Two near-zero-productivity days (D8, D9).

Two fixes, both live:

- **`FarmClearTask` bails on frame 1 when off-farm** (unbounded whole-farm
  mode only; the pocket-approach mode has its own ExitToFarm recovery, and a
  pending shed startup fetch is still allowed). This closes the pre-existing
  red `test_unbounded_clear_off_farm_does_not_succeed`, which had been
  asserting exactly this behaviour.
- **`DayPlanTask._try_map_lock_exit`**: a farm phase that map-locks while the
  farmer is in the house splices one `EXIT_TO_FARM` ahead of itself and
  retries, once per phase per day, instead of forfeiting every farm phase
  behind it. The farmhouse is one EXIT_TO_FARM away; spend that rather than
  lose the day.

Not fixed: **why** the clearer's path went through the door. That needs a
live tile dump at the stall and an offline `find_path` replay — the bail
stops the loss either way.

### 5. One shop trip per harvest day, not one per ring (optimizer top lever)

`rr-20w-spring-optimizer.md` ranks the two runtime caps —
`max_bags_per_day = 1` and `grape_max_per_day = 1` — as worth **+38 %**
(~+3 200 G/spring), ahead of new ring sites. The grape half landed in
`edeef59a`. This is the bag half.

The shop round trip is ~2 480 f whether it carries one bag or two, and a
harvest day empties **both** pocket rings at once (run13 §4:
`harvested=7` west and `harvested=8` second, same day). One bag replanted
one ring; the other idled until the next day's trip.

- `BuySeedsTask.bags` (default 1, unchanged): the clerk A-pulse loop is
  already "keep pressing until bought", so N bags is the same loop with a
  later stop condition. `_bags_target()` clamps to the wallet read at
  `reset()`, so it can never overspend; `_bags_done()` reads whichever of
  stock/wallet has posted (they disagree mid-dialogue).
- `farm_pond.pocket_plant_targets(ram)` — the plural of
  `pocket_plant_target`, which now delegates to it (one source of truth).
  Answers "how many bags does the farm want today".
- `_shop_bag_count` sets `bags = waiting rings − bags already in stock`,
  capped by `max_bags` (default 2).
- `DayPlanTask._splice_second_establish` splices one more `CROP_ESTABLISH`
  after a successful one while a ring still wants seed and a bag is in the
  pocket. Bounded to `len(POCKET_PLANT_CENTERS) - 1` extras per day.

Tests: `tests/test_two_bag_replant.py` (12).

**ROM status: the 2-bag purchase is NOT yet ROM-proven** — see Non-claims.
`buy_seeds_probe --bags N` exists to prove it.

### 6. Success triggered the cow-purchase day, and it killed the run

**run16 is the first run ever to get rich enough to hit this.** At D26, with
~$5 000 in the wallet, the planner replaced the whole income plan with a
12-phase cow day:

```
[MULTI_DAY] Plan 0:26 phases=EXIT_TO_FARM, NAV_TO_ANIMAL_SHOP, BUY_COW_VENDOR,
  EXIT_ANIMAL_SHOP, RETURN_FARM_AFTER_COW_PURCHASE, NAME_COW,
  ENSURE_ANIMAL_TOOLS, NAV_TO_BARN, ENTER_BARN, COW_CHORES, EXIT_BARN,
  DYNAMIC_OUTDOOR_PLAN
[DAY_PLAN] Phase NAV_TO_ANIMAL_SHOP FAILURE: expected tilemap 0x0C, got 0x00
```

None of those phases declared a `failure_policy`, so every one of them was
**required** — a failure aborts the day before `DYNAMIC_OUTDOOR_PLAN` is ever
reached. D26 and D27 both earned nothing, and D28 died in `return_home`
(`exit_to_farm ... pixel_stuck pos=(598,248)`). The run ended two days short
of Summer with the crop engine still working perfectly.

This is a "success is punished" bug: the better the economy gets, the sooner
it fires. `livestock_econ.py` is an explicit stub whose inputs are
deliberately `None`, so this path was never meant to be load-bearing.

Fix: `OPTIONAL_COW_PURCHASE_PHASES` — the purchase leg **and** the barn tail
(skipping only the purchase would leave required chores for a cow that was
never bought, and the day would die one phase later instead). Buying a cow is
discretionary; the farm work behind it is not. A failure now defers the whole
route and falls through to the day's actual income work.

Note this also changed what `test_day_plan_aborts_required_missing_task`
demonstrates — it used `ENTER_BARN` as its example of a required phase. It
now uses `CROP_WATER`, so the assertion is unchanged and still meaningful.

## Still open

### NAV_CROP is now the top money leak (was run13 §3's "CROP_WATER variance")

run13 §3 correctly identified the *mechanism* — a fixed-order phase list with
no time-budget arbitration, so whatever runs slow eats everything behind it —
but attributed the slowness to CROP_WATER. With the grape and watering
defects closed, run16 isolates the actual consumer: **NAV_CROP**.

run16 D22, frames 264000-274000, is the clean reproduction:

```
[RUN] f=264000 date=S0D22 07:00 $4350 ... EXIT_TO_FARM -> tilemap=0x00
[DAY_PLAN] Starting phase 3/6: NAV_CROP (nav)
[NAVIGATOR] Push-facing block tile=(8, 25) / (7, 26)        <- house-door band
...
[NAVIGATOR] Push-facing block tile=(28, 28) / (28, 27) / (28, 26)  <- far east
[RUN] f=274000 date=S0D22 17:12 $4350 ...
[DAY_PLAN] Phase NAV_CROP FAILURE: nav timeout
```

**Eight in-game hours (09:03 → 17:12) in one NAV_CROP, ending in a timeout.**
Watering then started at 17:12 and `BERRY_RUN_WINDOW` failed its cutoff at
18:04. Money is flat at $4350 across D20-D22 for this reason, not because the
crop cycle stopped.

Calibrating the cost model against run16 + run13
(`spring_plan --calibrate`) shows the shape is **bimodal, not a general
slowdown** — which changes what the fix should be:

| phase | n | min | median | max |
|-------|---|-----|--------|-----|
| NAV_CROP | 47 | 43 | **154** | **6611** |
| CROP_WATER | 30 | 364 | 2194 | 13625 |

A typical NAV_CROP costs 154 f. Its worst case costs 6 611 f — about half a
day, against a measured day of 12 814 f (n=37). So the target is the tail
(replan / timeout / route selection on the bad draw), not nav speed in
general. Same shape for CROP_WATER.

Two details worth keeping:

- The push-block path is **not** itself a loop: `note_push_facing` →
  `block_push_facing` adds the tile to `temp_blocked` and clears `self.path`,
  so each block does force a replan. The cost is elsewhere — long stretches
  with no `[NAVIGATOR]` output at all between the two block clusters.
- The two clusters are on **opposite sides of the farm**: `(8,25)/(7,26)`
  (the farmhouse-door neighbours run13 §4b already flagged) and `(28,26-28)`.
  A nav aimed at the west pocket `(13,28)` has no business at `x=28`, so the
  suspicion is route selection, not local walk policy. Unverified.

**Push-block hotspots are stable across runs.** Counting
`Push-facing block tile=` over three independent D3→D30 logs gives the same
ranked list, which makes these cheap to fix *and* cheap to regression-test —
the property run13 §1 valued in the grape pins:

| tile | run16 | run17 | run13 | rough area |
|------|-------|-------|-------|------------|
| (21,45) | 52 | 11 | 47 | south of the y=31 fence row |
| (20,44) | 42 | 8 | 34 | south of the y=31 fence row |
| (9,0)   | 40 | 8 | 32 | far north edge |
| (14,7)  | 37 | 6 | 21 | far north |
| (8,22)  | 31 | 6 | 30 | farmhouse-door band |
| (1,29)  | 23 | 5 | 21 | west edge |
| (5,23)  | 22 | 5 | 19 | west of the house |

(run17 counts are lower only because it was still mid-run when this was
taken; the *ranking* is what matches.)

A second cluster — `(27,26) (28,26) (28,27) (28,28) (28,29) (27,29)` — recurs
verbatim in both run16 D22 and run17 D6, east of the well body that
`FARM_NO_GO_TILES` already covers at x=15-17.

Whether any of these *cause* the NAV_CROP tail is **unverified** — they are
co-located in the logs, nothing more. Do not add them to `FARM_NO_GO_TILES`
on that basis alone: the house-door band in particular is on the route home,
and sealing it would trade one failure for a worse one.

Next step is the method already proven on this lane: dump the live tile grid
at the stall and replay `find_path` offline, which separates bad tiles from a
stuck walk policy. That needs a pin at the stall — note that minting one by
running the campaign to a mid-spring day is itself unreliable right now
(a `--until-day 8` mint wedged in NAV_CROP on D6 this session, same class).
Related bead: `rr-20w.2.4`.

### Phase-ordering starvation (run13 §3)

The arbitration gap itself is unchanged and deliberately not patched here.
Reordering was considered and rejected: berries are ~300 G/day of same-day
cash, while a skipped watering day stalls growth on every tile and compounds,
so the two are close enough in value that reordering on a guess is not
obviously positive. §3's own warning applies — which phase gets starved keeps
moving as other bugs are fixed, so the fix is arbitration, not a new fixed
order.
- **Only two ring sites exist** (`POCKET_PLANT_CENTERS` = (13,28), (19,28)),
  so bag count saturates at 2. A third nav-executable ring needs proven
  hoe/water/harvest stands — still the next money lever after this.
- Second-grape pick miss at `GRAPE_STAND_PX`.
- Why the clearer routes through the farmhouse door (§4 above).

## run16 baseline result (HEAD, no fixes from this session)

| | run13 (prev best) | run16 |
|---|---|---|
| reached | D22 | **D28** |
| money | $3 990 | **$5 420** |
| days completed | 19 | 25 |
| terminal | `return_home timeout ... phase=exit_to_farm` | `exit_to_farm ... pixel_stuck pos=(598,248)` |
| journal phase failures | several | **none** (all absorbed as deferrals) |

349 405 frames / 1 014 s wall. The D22→D28 and +36 % money improvement is the
*previous* session's nav-route work paying off, measured here for the first
time — none of this session's fixes are in run16. The three deterministic
`soft_solid pin` failures run13 §1 catalogued do not occur at all.

## Non-claims

- No STATUS promotion.
- Did not start from `Y1_D2_Morning_After_D1`.
- **The 2-bag purchase has not been executed against the ROM.** The default
  stays `bags=1` everywhere it is not computed from live RAM, and the unit
  tests cover the clamp arithmetic, not the shop dialogue. Until
  `buy_seeds_probe --bags 2` is green from a ≥400 G pin, treat `max_bags=2`
  as unproven.
- Zelda lane untouched.

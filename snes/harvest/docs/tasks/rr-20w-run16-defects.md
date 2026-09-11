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

## Still open

- **Phase-ordering starvation (run13 §3).** Unchanged: a fixed-order phase
  list with no time-budget arbitration, so whatever runs slow consumes the
  budget of everything after it. §3's own analysis stands; nothing here
  addresses it, and which phase gets starved will keep moving.
- **Only two ring sites exist** (`POCKET_PLANT_CENTERS` = (13,28), (19,28)),
  so bag count saturates at 2. A third nav-executable ring needs proven
  hoe/water/harvest stands — still the next money lever after this.
- Second-grape pick miss at `GRAPE_STAND_PX`.
- Why the clearer routes through the farmhouse door (§4 above).

## Non-claims

- No STATUS promotion.
- Did not start from `Y1_D2_Morning_After_D1`.
- **The 2-bag purchase has not been executed against the ROM.** The default
  stays `bags=1` everywhere it is not computed from live RAM, and the unit
  tests cover the clamp arithmetic, not the shop dialogue. Until
  `buy_seeds_probe --bags 2` is green from a ≥400 G pin, treat `max_bags=2`
  as unproven.
- Zelda lane untouched.

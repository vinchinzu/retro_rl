# rr-20w — run13 full-spring: defect log

Session 2026-09-10. Evidence: `logs/spring_d3_30/run13_full_spring.log`
(`run_to_day2 --state Y1_D3_Morning --end-of-spring`, D3→D30, in flight at
time of writing). First run that has ever executed past D13. Cross-checked
against `run12_short.log` (D3→D13) and `run11_grapefix.log`.

**Do not STATUS.** Nothing here is a fix; this is a defect inventory.

## 1. Two fully deterministic nav pins (highest confidence, likely cheapest fix)

```
6x  MOUNTAIN_BERRY FAILURE: return_to_bin: soft_solid pin held=0x03 pos=(133,101) stasis=40
3x  MOUNTAIN_BERRY FAILURE: pick: farm_to_path: soft_solid pin held=0x00 pos=(312,377) stasis=41
```

Byte-identical every occurrence — same position, same held tile, same stasis
count. These are reproducible geometry traps, not flaky nav, which makes
them cheap to fix *and* cheap to regression-test.

Note both legs are hit: `pick:` (outbound, farm→path) and `return_to_bin:`
(return). A third site, `~(505,633)` on the carpenter terrace, was patched in
commit `2204e37b`. **Three known sites for one failure class suggests the
`2204e37b` fix addressed a site rather than the mechanism** — worth asking
whether `soft_solid`/`stasis` has a shared root cause (shared-waypoint
`run_direction`/`force_run` flags, or the stall-guard threshold) before
patching site four.

## 2. `7d07444d` (carry-swap) did not fix the bug

```
2x  CROP_ESTABLISH FAILURE: select_carry_0x07: 0x07 not in carry pair
```

`docs/tasks/rr-20w-spring-optimizer.md` records `7d07444d` as the fix for
`SwapCarrySlotsTask` timing out after BUY_SEEDS, with the honest caveat "Not
yet in a full run." It is now in a full run and the downstream symptom is
unchanged — this is the same `select_carry_0x07` establish loss seen on
run11 D3, reappearing around D10-D11. **Treat `7d07444d` as unproven at
best, ineffective at worst.**

Related (separate audit): buying N seed bags does *not* multiply
`SwapCarrySlotsTask` invocations — the swap gets the seed *type* into the
carry pair once regardless of bag count. So this defect is orthogonal to the
bag-count work, not a blocker for it.

## 3. Phase cutoffs are starvation by phase ordering (not a deadline bug)

```
2x  BERRY_RUN_WINDOW FAILURE: cutoff reached at 15:03
1x  BERRY_RUN_WINDOW FAILURE: cutoff reached at 16:12
1x  BUY_SEEDS_WINDOW FAILURE: cutoff reached at 16:09
```

Two readings by this doc's author were wrong and are corrected here.
First: "a fixed deadline missed by three minutes" — no. Second, the
correction to that: "`16:12` proves it is non-deterministic" — **also no.**

What is actually true:

- The deadline *is* fixed (`day_phase_types.py:317`,
  `berry_exit_cutoff_hour = 14`) and `DeadlineCheckTask.step()`
  (`inventory_time.py:31-40`) is a **one-shot read**, not a poll. The
  printed `15:03` is just whatever the clock said when the phase was
  reached — it measures how late the plan already was, nothing else.
- The harness *is* deterministic. `15:03` recurs because run12 D7 and run13
  D6 run an identical phase list from identical state; same code + same RAM
  + same inputs → frame-perfect same outcome. `16:12` differs because that
  is a **harvest day** with a 14-phase list (HARVEST_ROUTE shipping 8 items
  one at a time, then CROP_ESTABLISH, then CROP_WATER) where
  `BERRY_RUN_WINDOW` is phase 13 of 14.
- **The real defect is CROP_WATER variance.** Watering 2 plots (15 tiles +
  1 refill) takes ~2 h on a clean day and up to ~9 h when nav soft-arrives
  once: run13 D5 finished by 08:03 (`NAV_CROP -> arrived`, no retries);
  D6 hit `NAV_CROP -> soft arrived dist=30 stasis=90` plus one water-tile
  RETRY and ate the whole deadline. One nav stall consumes every downstream
  phase's budget.
- **Fixing the grape count moved the starvation onto a new victim.** With
  the (uncommitted) fix live, run13 D10's legitimate 2nd grape loop ran long,
  bailed correctly at `past return deadline` past 16:00, and the *next*
  phase — `BUY_SEEDS_WINDOW`, latest hour 16 — immediately failed at 16:09.
  That day bought no seed at all.

Root cause is the gap already named in `rr-20w-idle-day.md`: **a fixed-order
phase list with no time-budget arbitration.** Whatever runs slow fully
consumes the budget of everything scheduled after it — no reordering, no
slack reservation, no priority for the highest-G/hour phases. Which phase
gets starved will keep moving as other bugs are fixed.

## 4. The "missing tile" is a silent watering drop, not geometry

Earlier revisions of this doc (and `SPRING_ECONOMY.md` §11.4) blamed
`RING_TILES = 8`. **That was wrong on every count** — both rings are a full
3x3-minus-centre 8 (`crop_geometry.py:194-205`, `PLOT_RING_SIZE = 8` at
`crop_skills.py:56`), and establish only succeeds once
`count_ring_planted(...) >= 8` verifies against live RAM. Per-harvest tile
lists from run13:

```
harvested=7  (12,27) (12,29) (13,29) (14,29) (13,27) (14,28) (14,27)   west, centre (13,28)
harvested=8  (18,27) (18,29) (18,28) (19,29) (20,29) (19,27) (20,28) (20,27)   second, centre (19,28)
harvested=1  (12,28)                                     <-- picked alone, later
```

Exactly one tile — **(12,28)**, west-middle — falls out, and only on the
west ring. **Root cause: `crop_water_ops.py:169-190`,
`_reorder_remaining_water_steps`:**

```python
best = self._best_water_variant(ram, target, current_tile)
if best is None:
    continue                      # dropped: no log, no skip counter
...
self._water_steps = prefix + reordered   # list replaced WITHOUT the dropped target
```

If a stand is momentarily unreachable *at the instant the reorder runs*, the
target vanishes from the day's watering permanently — silently. Observed on
run13 D3: `steps=8` declared, `WATER DONE: 7/7 watered`. The tile is then
never watered, stays immature (3x3 RAM dumps show the west cell at `0x58`
while its siblings are `0x60`/`0x61` mature), and is correctly not harvested.
**So `harvested=7 skipped=0 unreachable=0` was accurate reporting of an
inaccurate ring state — the harvest task was never the bug.**

LIKELY trigger for why only (12,28): its preferred stand is `(11,28)`, which
is listed in `FARM_NO_GO_TILES` (`farm_pond.py:63`) — *and also* in
`FARM_POND_ACCESS_STAGING_TILES` (`farm_pond.py:99`). **That contradictory
double-membership is a defect in its own right.**

This is a **timing** loss, not a gold loss: mature crops never decay, so the
cost is fewer completed replant cycles over the spring, which compounds. The
earlier "~5 % optimistic per cycle" figure conflated the two and is withdrawn.

**Generalise the fix:** the silent drop is a footgun for *any* crop tile, not
just this one. A `dropped_targets` counter and a log line are warranted
regardless of the (11,28) question.

### 4b. A second, unrelated bug can zero the whole ring

```
HARVEST_ROUTE FAILURE: incomplete harvest: harvested=0 shipped=0 skipped=7 unreachable=0
[HARVEST] SKIP target=(12,29) (every stand unreachable: [((12,30),'up'), ((11,29),'right')])
[NAVIGATOR] Push-facing block tile=(8,25) / (7,26) / (8,27)
```

`(8,26)` is the farmhouse door (`transitions.py:57`,
`HOUSE_ENTER_STAND_TILE`); the three blocked tiles are its immediate
neighbours. On that morning `EXIT_TO_FARM` logged `tilemap=0x00` *without*
the usual `mid-warp settle y=344`, so the farmer's post-exit pose differed —
and from there the pathfinder could not route around the house at all. The
same day's `CROP_WATER` then failed to path to the **entire second ring**
(`SKIP water tile 1/8 (no path) target=(18,27)`), confirming a global
pathfinding blockage rather than per-tile geometry. Also a deferral, not a
loss (`Deferred HARVEST_ROUTE until tomorrow`) — unless it recurs on the
retry day. Why the door area blocks pathing on that particular pose is
SPECULATIVE; that it does, and cascades to a whole-ring skip, is CONFIRMED.

## 5. The grape fix is ALREADY WRITTEN AND UNCOMMITTED — commit it

`CostModel.grape_max_per_day` was set to 1 this session from run12, where
4 of 6 berry days reported `1/2 shipped; stopped early (shop window)`. That
calibration is wrong, and the reason is not sampling noise — **run12 and
run13 were produced by different code.**

`harvest/planner/day_phase_berry.py` has an uncommitted rewrite in the
working tree. The committed version contains **zero** occurrences of
`shop_bail_hour`, so `_build_mountain_berry`
(`day_phase_registry.py:336`) fell through to
`spec.params.get("shop_bail_hour", 10)` → `MountainGrapeShipTask`'s class
default of **10** on every day. The 2nd grape loop was gated at 10:00 and
could never start. The working-tree version adds
`GrapeDaySpec`/`grape_day_spec()` and threads `shop_bail_hour=spec.bail_hour`
through properly.

| run | code | 2/2 shipped | 1/2 shipped | phase failed outright |
|-----|------|-------------|-------------|------------------------|
| run12 | committed (bug) | 2 | 4 (`shop window`) | 0 |
| run13 | working tree (fixed) | 5 | 1 (`past return deadline`) | 4 |

run13 shows **not a single `(shop window)` bail**. The one remaining bail is
`past return deadline`, a legitimate `hard_return_hour` abort.

**This is the single highest-value action available and it is already done:
commit `day_phase_berry.py`.** It is worth ~+3 800 G/spring on the model and
has now proven itself in a live D3→D30 run.

Two consequences:

- The model should carry `grape_max_per_day = 2`, **plus a reliability term
  it currently cannot express**: the phase still fails outright on ~40 % of
  days (defect 1). "2 when it works, often broken" is not a cap.
- Any figure in `SPRING_ECONOMY.md` §11.2 or this session's task-doc entry
  that prices "fixing the grape cap" as future work is stale — that fix is
  in hand. Recalibrate from the completed run13.

## 6. Also seen

- `CROP_ESTABLISH FAILURE: nav_hoe_ring_3_down: nav timeout` (x1) — the WIP
  `crop_skills.py` ring nav (rr-20w.3.2), still unproven.
- `MOUNTAIN_BERRY FAILURE: pick: farm_to_path: multi_nav timeout` — a
  *fourth* distinct grape-leg failure mode, separate from the two pins.
- **`select_carry_0x07` is not caused by the berry phases** — neither
  `mountain_berry.py` nor `mountain_grape_ship.py` touches carry slots at
  all (they use the held-forage register, a different RAM concept). But it
  fired on run13 D10 — *the same day* `BUY_SEEDS_WINDOW` blew its cutoff and
  no seed bag was bought. Worth checking whether `ENSURE_CROP_SEEDS` /
  `CROP_ESTABLISH` handle "no bag purchased today" gracefully, or assume a
  carry-pair state that only exists after a successful BUY_SEEDS.
  SPECULATIVE, from temporal adjacency only.
- **`CLEAR_FIELD` burns whole days as a no-op.** run12 D8/D9: ~3300 frames
  with `targets=0 cleared=0 failed=0` and the position frozen, then
  `clear_budget cleared=0 remaining=0 lift_only`; every following phase then
  fails `map_mismatch:have=0x15:need=0x00` — the farmer is back *indoors*
  despite `EXIT_TO_FARM` having reported `tilemap=0x00` that same day. Two
  near-zero-productivity days. The map-lock check reads tilemap fresh per
  phase, so this is not a stale read; something in CLEAR_FIELD's idle/failure
  exit path re-enters the house. `harvest/tasks/farm_clearer.py` unaudited.

## Non-claims

Run was still in flight when this was written; counts are as-of ~D16 and
will grow. No fix attempted. No STATUS. Frame-cost recalibration deliberately
deferred until the run completes — mid-run numbers would be another
too-little-data calibration of exactly the kind §5 is about.

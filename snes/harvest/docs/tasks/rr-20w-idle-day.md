# rr-20w — the day is ~50-90% idle (the unpriced lever)

Session 2026-09-10. Evidence: `logs/spring_d3_30/run11_grapefix.log` +
its trailing `day_journal` JSON (D3→D11 from `Y1_D3_Morning`, killed
mid-D11 on the harvest stall).

## Measured

Per-day frame cost is flat — the day is a fixed clock, not a work budget:

| Day | frames | last productive phase ends | idle until 17:00 |
|-----|--------|---------------------------|------------------|
| D3  | 12 287 | NAV_CROP (establish lost to carry-swap) | — |
| D4  | 13 026 | CROP_ESTABLISH | ~4 h |
| D5  | 13 049 | MOUNTAIN_BERRY 13:12 | 3.8 h |
| D6  | 12 439 | MOUNTAIN_BERRY 13:08 | 3.9 h |
| D7  | 12 225 | MOUNTAIN_BERRY 10:00 | **7.0 h** |
| D8  | 12 902 | CROP_WATER (bin empty, no wait) | — |
| D9  | 15 191 | HARVEST_ROUTE ~06:20 | **~10.6 h** |
| D10 | 19 806 | CROP_WATER | ~2 h |

Clock rate is constant ≈ 15 f / in-game minute; 06:00→18:00 sleep ≈ 12 h ≈
12 000 f. So each day is ~12–20 k frames **whether or not any work happens
in it**. Roughly **half of every day, and ~90 % of a harvest day, is spent
standing still.**

## Mechanism

`multi_day_planner.py:593` — when `plan_day` finishes and the shipping bin
holds goods, the planner switches to phase `wait_shipping` and runs
`FarmShippingWaitTask` (`multi_day_planner.py:222`), which idles until hour
17 so the farm ShippingScene fires. It never asks whether more work is
available first. `plan_day` itself is a one-shot expansion of
`build_outdoor_day_phases_from_ram` (`day_plan_phases.py:556`) — once the
one ring is watered and no seed bag is in the pocket, the plan is empty and
the rest of the day evaporates.

Per `docs/SPRING_ECONOMY.md` §4 a full water pass on an 8-ring is ~1 800 f
≈ 2 in-game hours, and a grape leg ~3 100 f. **~6 idle hours/day ≈ 3 extra
ring waterings, or a second grape run plus a ring establish, every day, for
free.**

## The three compounding caps (all present in run11)

1. **One seed bag per day.** `BUY_SEEDS` bought `potato_seeds 0->1` on both
   D3 and D10 while the wallet held 250 G then 1 360 G. Planting is gated on
   `WorldProbe.pocket_has_plant_capacity` (`world_probe.py:145`), which is
   `farm_pond.pocket_plant_target(ram) is not None` — a **single** ring
   target. The whole establish pipeline is single-ring by construction.
2. **Two nav-proven ring sites**, so even unlimited bags have nowhere to go
   (see `docs/RING_SITES.md`, in progress).
3. **Idle afternoons** (above), so the extra rings would cost nothing to
   service even if they existed.

These multiply: fixing any one alone yields little. (1)+(3) without (2)
still tops out at 2 rings; (2) without (1) still buys 1 bag/day.

## Also visible in run11

- **Only 1 of 2 grapes ships, every single day** — D3–D7 all report
  `mountain grape 1/2 shipped; stopped early (shop window)`. The run then
  idles 4–7 h. Worth ~150 G/day ≈ +4 k G/spring; the shop-window cutoff
  should not apply on days with no shop hop, and never when the alternative
  is standing still.
- Grape wallet cap is gone. `GrapeDaySpec` (`day_phase_berry.grape_day_spec`)
  sets count/bail from day + `has_harvest` (D2: 1/10; harvest morning: 1/9;
  else 2/12). The old `BERRY_STOP_WALLET_G=700` kill from D8 was a proxy for
  the rr-20w.3.1 return-leg strand, which is fixed.
- **D9 harvested 7, not 8** (`harvested=7 shipped=7 skipped=0
  unreachable=0`) — one ring tile silently produced nothing. Either the
  8th tile was never sown (the centre-notch / bag-of-9 question) or it is
  mis-detected. −80 G/cycle.
- **D3 establish lost entirely** to `swap_preserve_hoe: carry slot swap
  timeout`; fix committed in `7d07444d` but unproven in a full run.

## Proposed change (not yet implemented)

Replace the one-shot `plan_day` → `wait_shipping` handoff with a **work
loop**: after `plan_day` reports done, re-expand
`build_outdoor_day_phases_from_ram` from live RAM; if it yields phases, run
them; only fall through to `FarmShippingWaitTask` when the expansion is
genuinely empty *and* hour < 17. Then let the same loop keep working after
17:00, since there is no forced bedtime — sleep is a policy choice at 18:00.

For that loop to have anything to do it needs (1) an N-bag purchase and
(2) an N-site ring catalogue. Sequence the work accordingly.

## The grape line: measured, and smaller than it looks

Attempted fix and what the emulator said (`logs/spring_d3_30/grapefix_d3_d9.log`).

`MountainGrapeShipTask.step` carried a hard-coded `hour >= 12` bail that
shadowed the `shop_bail_hour` field, so the field was dead and the default
10 abandoned the second grape the instant the first landed. Raising the gate
to 12 made things **worse**, not better:

1. grape 1 lands ~10:00, `10 < 12`, so a second loop starts and walks the
   full route back up to mountain `0x10`;
2. at 12:00 the same guard fires **mid-loop** and returns SUCCESS from
   `0x10`;
3. `NAV_FARM_EXIT` then fails `expected tilemap 0x00, got 0x10`, and
   `BUY_SEEDS`, `ENSURE_CROP_SEEDS`, `NAV_CROP` and `CROP_ESTABLISH` all
   cascade off the map lock.

D3 lost its seed purchase and its ring establish outright — strictly worse
than the idle afternoon it was meant to replace. The rule the code was
missing: **an abort that leaves the farmer off-farm is worse than not
starting.** The decision has to be pre-flight, taken while standing at the
bin, using the loop's *duration*, not the current hour alone. Timeout,
a failed second pick, and a walk-back that reports arrived while still
on `0x10` are the same rule — SUCCESS only on a farm tilemap.

Measured timings that constrain it: grape 1 lands **~10:00** from a 06:00
start; a loop is ~4 in-game hours. The leftover notes that "2 grapes +
shop does not fit" were wrong. ROM: `d3_mountain_grape_two.json` ships
0→300 at 13:12, then `Y1_D3_PostTwoGrape` buys potato 13:12→16:08.

The grapefix_d3_d9 failure was bail-9 on restock days (`hour >= 9` after
grape 1 at 10:00 never starts loop 2) plus SUCCESS off-farm when the
hour gate fired mid-loop. Shop days now use bail 12 as well. Harvest
mornings do **not** force a second loop — grapes stay an option after
crop work (count=1, bail 9).

Loop 2 of the proven 2-grape run returned to the same `(20,25)` stand
and still shipped. Treat the mountain as **300 G/day** on restock
mornings; harvest days keep one grape as an option.

## Non-claims

The idle-hour figures are read off phase completion times in the run11
journal, not instrumented. No change has been made to the planner yet. The
"~3 extra rings per idle afternoon" figure uses the *modelled* 1 800 f water
cost from SPRING_ECONOMY §4, which is itself uncalibrated.

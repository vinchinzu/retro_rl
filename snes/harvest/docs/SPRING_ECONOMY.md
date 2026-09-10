# Spring 1 economy — ROM model + optimizer

Goal: maximise gold shipped by Summer D1 from `Y1_D3_Morning`
(`run_to_day2 --state Y1_D3_Morning --end-of-spring`). This doc is the
analytical backbone for `harvest.planner.spring_opt`; proven runtime facts
still live in [STATUS.md](STATUS.md).

## 1. Crop growth (ROM: `NightlyFarmTilesCheck` @ `82A811`)

Every night each farm tile is scanned:

| Tile parity | Meaning | Night effect (spring, not raining) |
|-------------|---------|-------------------------------------|
| **odd** (e.g. `0x55`) | watered this day | `INC` → next stage (`0x56`) |
| **even** (e.g. `0x56`) | dry | **no change** — growth stalls, does *not* regress |
| any | raining (`FLAG196 & 0x0002`) | `INC` (free water) |
| odd, **summer** | spring crop in summer | `DEC` → dies |

Consequences:

- A crop advances **one stage per day it is watered**. Skipping a day only
  *stalls* it (no penalty) — so on a tight frame budget it is legal to water
  a subset of rings each day and let the rest wait.
- Rain is a free growth day for every planted tile.
- **6 waterings** take potato from planted-dry `0x54` to mature `0x60/0x61`
  (`(0x60-0x54)/2 = 6`). Matches `days_to_first_harvest = 6`.
- Harvest leaves the tile as watered-tilled `0x08` — **replant needs no
  re-hoe**, just a seed bag (`HOED_OR_PLANTED` now includes `0x08`).

## 2. Prices (ROM: `UnlinkedText` shipping-rate + seed-shop screens)

| Item | Seed bag | Tiles/bag | Ship /unit | Grow (waterings) | Regrow |
|------|----------|-----------|-----------|------------------|--------|
| Potato | 200 G | **9** (3×3; `$096B` counts to 9) | 80 G | 6 | no |
| Turnip | 200 G | 9 | 60 G | ~4 | no |
| Grape (mountain forage) | — | — | ~150 G | — | daily respawn |

Runtime currently plants an **8-ring** (centre notch is the plant stand);
the 9th centre tile is left unplanted — a standing +12.5 % if nav can reach it.

Per-ring spring economics (8 tiles, plant D=day, harvest D+6, replant on
harvest day):

```
potato ring value = harvests * 8 * 80  -  harvests * 200
  plant D3  -> harvest D9,D15,D21,D27         = 4 harvests -> 1760 G net
turnip ring (4d): plant D3 -> D7,D11,...,D27  ~6 harvests -> ~1680 G net
```

Potato wins per plant/harvest op and per bag; turnip wins first-cash speed
and value-per-watering (`480/4 = 120` vs potato `640/6 ≈ 107`). If watering
frames are the binding constraint, a turnip phase early can help.

## 3. Day budget — there is no forced bedtime

**The evening never ends.** You can water all night; the can refills at F0
and stamina refills at the hot spring, both unbounded. A wake→sleep cycle
can therefore do an arbitrary amount of work — the runtime's ~18:00
return-home is a *policy choice*, not a game limit. A crop still advances
exactly **one stage per sleep** no matter how much you do that day.

So plot count is **not** capped by a daily frame budget. The real caps:

1. **Maturity vs. the summer wipe.** A ring must be established early enough
   to mature before D30 (odd spring-crop tiles `DEC` = die in summer). Last
   useful potato plant ≈ **D24**.
2. **Seed capital timing** — bags are 200 G; early cash comes from grapes +
   the first harvests.
3. **Harvest pile-up** — at very high ring counts the mature rings on a big
   harvest day take longer than one (practical) evening to clear + replant +
   water, stalling growth. This is the "can't harvest all in one day" point.

`spring_opt` models this with `CostModel.evening_frames` — a *practical*
ceiling on how long one day's work should take in a real run (default
60 000 f ≈ 16 real minutes/day), not a forced bedtime. Raise it freely.

- Frame rate ≈ **14.5 f / in-game minute** while the clock runs (measured:
  2-grape run 6194 f over 06:00→13:12).
- Calendar constraints that remain:
  - Seed shop / shipping office **7:00–17:00, closed weekends & holidays**
    (`DATA16_B9CE24`). `buy_seed_hour` policy = 12 — the shop hop must be an
    early-morning errand.
  - Be on the farm at **17:00** for the ShippingScene; wallet credit is the
    following NightReset `AddMoney`.
  - **Sundays = D7, D14, D21, D28** (Spring D1 = weekday 1; `SUNDAY_WEEKDAY=0`).
    Shop closed. Saturdays (D6/D13/D20/D27) may also be shop-closed.

## 4. Measured frame costs (fill in as evidence accrues)

Source: `docs/tasks/rr-20w.3*`, campaign logs `logs/spring_d3_30/`.

| Action | Frames | Note |
|--------|--------|------|
| 2 mountain grapes (pick+return+ship) | ~6 200 | 06:00→13:12, ~3 100 f/grape |
| Seed shop round trip (farm→plaza→farm) | ~2 600 | 13:12→16:08 |
| Establish 8-ring (nav+hoe 8+plant 8) from farm pose | ~3 200 | incl. shed carry-swap |
| Water 8-ring (nav+select+8 tiles), can charged | ~1 800 | can 12→4 |
| Empty-can refill (fence-open + F0 + return) | ~2 500–4 000 | rr-3ae8 late-spring exhaustion |
| Harvest 8-ring → bin | TBD | measure from a mature-ring pin |
| Return home + sleep | ~1 000 | |

Can capacity = **20 charges** (1 tile/charge). Watering >20 tiles/day forces
a mid-water refill.

## 5. Optimisation framing

Decision variables per spring day D ∈ [3,30]:
- grape run? (0/1) — only worth it while cheaper than the marginal crop hour
- buy N seed bags? (shop open, cash ≥ 200N, D not Sunday)
- which rings to establish / harvest / water today
subject to Σ frame_cost(actions) ≤ daily_budget, cash ≥ 0.

Objective: maximise wallet at Summer D1 = Σ shipped·price − Σ seed cost.

`spring_opt` solves this by forward beam search over the 28-day horizon with
a per-ring growth state vector. Output: a `SpringPlan` (per-day action list)
that `multi_day_planner` can follow instead of the reactive heuristics.

### Optimiser output (`spring_opt`, infinite-evening model)

`uv run python -m harvest.scripts.spring_plan --sweep`

| rings | potato final G | potato gross shipped |
|-------|----------------|----------------------|
| 4 | 7 170 | 9 520 |
| 8 | 12 450 | 17 200 |
| 12 | 16 410 | 22 960 |
| 18 | **21 690** | **30 640** |
| 22+ | 22 130 | 31 280 (saturated) |

- **Returns scale to ~18–20 rings** (~$22 k final, ~$31 k gross) with the
  default `evening_frames=60 000`. Saturation is the harvest-pile-up point
  (§3.3), tunable via `evening_frames`. vs the current reactive campaign
  ~$1.5 k this is ~15×.
- Recommendation: **grapes D3–D6 to bootstrap the first 2–3 bags, then
  ramp potato rings as fast as capital + nav allow, replanting every ring
  the day it's harvested.** Stop planting new ground ≈ D24.
- The binding real-world constraint is **nav**: only 2 ring sites are
  currently execution-proven (`WEST_POCKET_PLANT_CENTER (13,28)` +
  `SECOND_POCKET_PLANT_CENTER (19,28)`). Unlocking more nav-reachable ring
  sites in the cleared farm is now the top lever — see §5 levers.
- Grape sensitivity: with `allow_grapes_through_day` unbounded the solver
  still likes 2 grapes/day early; sustained daily 2-grape yield past D6 is
  measured **once** and prior work capped it at `BERRY_STOP_WALLET_G=700`.
  Measure it if the grape line is pursued.

### Structural levers (biggest first)

1. **More nav-reachable ring sites.** Every extra ring that establish +
   water + harvest can actually execute is worth **~1 800 G/cycle × up to
   4 cycles ≈ 1 700 G net/spring**. The cleared farm has room for ~20; only
   2 sites are execution-proven. This is now the #1 lever — needs a
   ring-site catalogue in the cleared field (south of the y=31 fence too)
   with proven hoe/water/harvest stands.
2. **Replant cadence.** 1 cycle/ring → 4 cycles/ring. Blocked by
   `CROP_ESTABLISH → nav_pocket_hoe_stand` / `nav_hoe_ring_*` timeout
   after harvest (rr-20w.3.2, WIP).
3. **Grape return_to_bin** (rr-20w.3.1 residual) — the shared outbound
   force-run at mountain `(520,712)` pinned the return at ~(505,633).
   Fixed: plain waypoint + `MultiMapNavTask` run_direction stall guard.
4. **Grape discipline.** Grapes D3–D6 only, to fund the first bags.
5. **Centre tile.** ROM: a seed bag is 9 plants (`$096B`→9). The runtime
   8-ring leaves the centre — +12.5 % if the plant pass reaches the notch.
6. **Carry-swap robustness.** `swap_preserve_hoe` (SwapCarrySlotsTask) times
   out on the farm after BUY_SEEDS, so the bag never enters carry and
   CROP_ESTABLISH fails `select_carry_0x07`. Alternative: toss the can
   before the seed fetch and re-fetch it in the water pass.

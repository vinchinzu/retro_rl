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

## 3. Day budget

- NightReset sets the clock to **06:00**; the runtime heads home ~**18:00**
  (`BUY_SEEDS_WINDOW cutoff 18:08`). Usable window ≈ **12 in-game hours**.
- Frame rate ≈ **14.5 frames / in-game minute** (measured: 2-grape run
  6194 f over 06:00→13:12). ⇒ **≈ 10 400 usable frames/day** before the
  walk home + sleep (~1 000 f).
- Hard constraints:
  - Shipping office / seed shop hours **7:00–17:00, closed weekends+holidays**
    (`DATA16_B9CE24`). `buy_seed_hour` policy = 12.
  - Be on the farm (tilemap < 4) at **17:00** for the ShippingScene; wallet
    credit is the following NightReset `AddMoney`.
  - **Sundays = D7, D14, D21, D28** (Spring D1 = weekday 1; `SUNDAY_WEEKDAY=0`).
    Shop closed. Saturdays (D6/D13/D20/D27) may also be closed for the shop.

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

### Structural levers (biggest first)

1. **Replant cadence.** 1 cycle/ring → 4 cycles/ring is +1300 G/ring.
   Blocked by `CROP_ESTABLISH → nav_pocket_hoe_stand` timeout (rr-20w.3.2).
2. **Ring count.** 2 rings (16 tiles) is the current nav-proven cap. Each
   further nav-reachable ring in the y27–29 / x11–21 band is ~1760 G/spring.
3. **Grape discipline.** Grapes only D3–D4 to fund the first 1–2 bags, then
   stop (each grape-day costs a full crop rotation of frames).
4. **Centre tile.** 8→9 tiles/ring if the plant pass can reach the notch.
5. **Refill amortisation.** Keep total watered tiles ≤ 20 where possible, or
   schedule the refill on a light day.

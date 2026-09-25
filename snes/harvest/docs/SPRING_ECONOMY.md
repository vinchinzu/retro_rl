# Spring 1 economy - ROM model + optimizer

Goal: maximise gold shipped by Summer D1 from `Y1_D3_Morning`
(`run_to_day2 --state Y1_D3_Morning --end-of-spring`). This doc is the
analytical backbone for `harvest.planner.spring_opt`; proven runtime facts
still live in [STATUS.md](STATUS.md).

All ROM claims cite `HM-Decomp/src/code_banks/<bank>.asm:<line>` unless noted.
Community-guide claims are cited by URL and cross-checked against the ROM
(CONFIRMED-BY-ROM / CONTRADICTED-BY-ROM / UNVERIFIED, see §7).

## 0. CORRECTION - the "summer wipes spring crops" claim was wrong

A prior version of this doc claimed crops die at the Spring→Summer boundary
and that the last useful potato planting was ≈D24. **That is false.**

- `NightlyFarmTilesCheck`'s per-tile season dispatch
  (`bank_82.asm:2440-2458`, labels `CODE_82A8DA`/`CODE_82A8F0`) only `DEC`s
  (kills) a still-growing watered crop tile when `season == 2` (**Fall**).
  When `season == 3` (Winter) it does nothing (frozen). Otherwise - **season
  0 (Spring) or 1 (Summer)** - it `INC`s (grows). A potato/turnip ring
  planted in spring keeps growing across the Spring→Summer boundary with no
  penalty and matures normally in Summer.
- The only wholesale farm wipe is `MonthlyFarmTilesCheck` (`bank_82.asm:2091`),
  and it is called from exactly one place - the tail of
  `NightlyFarmTilesCheck` (`bank_82.asm:2526-2534`), gated on
  `season == 3 (Winter) AND day == 1`. It resets grass and all tilled/crop
  tiles (`0x1D-0x6F → 0x07` dry-tilled, `≥0x70 → 0x7A`) back to bare farm -
  **once a year, on Winter D1, never at the Spring/Summer boundary.**
- **What this changes:** there is no D24 planting cutoff from crop death.
  End-of-spring plantings are not stranded - they simply finish maturing in
  Summer (see §0.1 for the real constraint that replaces it).

### 0.1 The REAL constraint: potato/turnip can only be *planted* in Spring

This is new and matters more than the wipe myth it replaces. The seed-bag
planting code checks the season **before** writing the real crop tile:

- `ToolUsedPotatoSeeds` (`bank_82_toolused_subrutines.asm:846`): `LDA
  !season; BEQ` → uses the real potato tile id `$001B` only when
  `season == 0` (Spring); **any other season loads `$00EB` instead**
  (lines 858-862).
- `ToolUsedTurnipSeeds` (`:897`): identical pattern with tile id `$001C`,
  season-gate at lines 909-913.
- `ToolUsedCornSeeds` (`:742`) and `ToolUsedTomatoSeeds` (`:794`) are the
  mirror image: they check `CMP season,#$01` (Summer) and use their real
  tile ids (`$0019` corn, `$001A` tomato) only then (lines 755-759,
  807-811); off-season plants also degrade to `$00EB`.
- `$00EB` (≥`$A0`) is never touched by `NightlyFarmTilesCheck`'s dispatch -
  tiles `≥0xA0` are treated as out-of-bounds and skipped
  (`bank_82.asm:2340-2342`). So an off-season seed placement is a dead,
  inert tile: it never grows, never gets watered, and is never harvestable -
  **and it still consumes the seed-bag charge** once the 9th placement is
  reached (the `$096B` counter, see §2.3, doesn't know the plant failed).

**Consequence:** Potato and Turnip are Spring-only *plantable* crops; Corn
and Tomato are Summer-only *plantable* crops (out of scope for this doc, but
relevant to a future summer optimizer). You *can* plant/replant potato or
turnip any day **D1–D30** of Spring - the crop will simply mature in Summer
if planted late, per §0. You *cannot* start a **new** potato/turnip ring, or
**replant** a harvested tile with potato/turnip, once Summer begins (D31 /
Summer D1) - the game will happily let you try and will burn the bag.

For the stated objective ("gold shipped by **Summer D1**"), this reproduces
almost the same practical cutoff as the old (wrong) belief for a different
reason: a ring planted after D24 cannot both mature (+6d potato / +4d
turnip) and ship before Summer D1, so it scores zero *within this horizon*
- but it is **not wasted or killed**; it is simply out of scope for a
Spring-only objective. If the optimizer's horizon is ever extended a few
days into Summer, D25-D30 potato plantings become valuable again (§6).

## 1. Season & calendar (ROM)

- `!season` (`$7F1F19`) = **0 Spring, 1 Summer, 2 Fall, 3 Winter**. Confirmed
  independently three ways: `WeatherTomorrow`'s season branches
  (`bank_82.asm:1358-1401`, e.g. `CMP #$02` → Fall festival day-11 branch),
  `SetWeatherFlags`'s `.sunny` case (`bank_82.asm:1269-1272`, "no sun in
  summer" = `season==1`), and the Spring/Summer-only crop gates in §0.1.
- Day rolls at `!day == 0x1F` (31) → new month; season increments and wraps
  `0→1→2→3→0` with `!year++` on the wrap (`NightReset`, `bank_82.asm:364-374`).
  Each season is a fixed **30 days**.
- Weekday (`!weekday`, `$7F1F1A`) starts at `1` on a new game
  (`NewGameSetup`, `bank_83.asm:4533`) and cycles `0..6`, wrapping at 7
  (`bank_82.asm:355-360`). A community FAQ states Spring D1 is a Tuesday
  and shops are closed weekends (UNVERIFIED-BY-ROM - see §7); the ROM
  confirms the wrap arithmetic but I did not find the actual
  weekday→shop-open/closed gating code in this pass.
  **Correction to a prior citation:** the earlier doc cited
  `DATA16_B9CE24` as a "shop-hours table." It is not - that address holds
  plain dialogue text (font/character codes, `bank_B9.asm:1464`), reached
  only through a generic text-pointer array (`bank_83.asm:4142`). The real
  shop-hours logic is unlocated; treat "7:00-17:00, closed weekends" as
  community folklore, not an ROM-verified fact (§8 open questions).
- **Festival / blocked days found in `WeatherTomorrow`** (`bank_82.asm:1355-1584`,
  forces `!weather_tomorrow` to a festival id, no crop-damage check runs
  that day per `SetWeatherFlags`'s `.calm` no-op for weather≥6):
  | Day | Festival |
  |-----|----------|
  | Spring D22 | Flower Festival (`bank_82.asm:1360-1362`) |
  | Fall D11 | Harvest Festival (`:1376-1378`) |
  | Fall D19 | Egg Festival (`:1386-1388`) |
  | Winter D9 | Thanksgiving (`:1405-1407`) |
  | Winter D23 | Star Night Festival (`:1415-1417`) |
  | Winter D30 | New Year's (`:1425-1427`) |

  **No Spring or Summer festival other than Spring D22** was found in this
  routine. Whether the shop/shipping office actually closes on festival
  days was not verified (§8).

## 2. Crop growth & tile mechanics (ROM: `NightlyFarmTilesCheck` @ `bank_82.asm:2312`)

Every night, every farm tile (`GetTileIndex` scan of the 64×64 array) is
dispatched by its raw tile id:

| Tile id / range | Meaning | Night effect |
|---|---|---|
| `0x00-0x02` | bare/empty ground | `.addrandomtrash` weed roll (§2.1) |
| `0x03` | weed | left alone by this routine |
| `0x07` dry tilled / `0x08` watered tilled | tilled soil pair | `0x07`+rain → `0x08` (free water); `0x08` w/o rain → `0x07` (dries) (`:2349-2354`, `:2410-2432`) |
| `0x1E`/`0x1F` | freshly-planted seed pair | same dry/watered pair logic as tilled soil, **not** subject to the seasonal grow/die dispatch below |
| `0x20-0x6F`, odd (watered, not yet mature) | growing crop | season dispatch: **Fall `DEC` (dies)**, **Winter no-op (frozen)**, **Spring/Summer `INC` (grows)** - `CODE_82A8DA`/`CODE_82A8F0`, `bank_82.asm:2440-2458` |
| `0x20-0x6F`, even (dry, not yet mature) | growing crop | rain-only: `INC` (free water) if raining, else stalls - no seasonal death risk while dry |
| `0x39` tomato / `0x53` corn / `0x61` potato / `0x6F` turnip (all odd, "fully grown" exact matches, `bank_82.asm:2374-2391`) | **mature, watered** | routed to `.unwateredraincheck`: `DEC` to the even mature-dry id overnight unless raining (then stays watered, no further growth) - **never dies, never regresses further** (`:2420-2432`) |
| `0x70+`, `0x79`, `0x7C` | grass stages | separate grass growth/climate logic, not covered here |

Consequences (all four crop types, not just potato - confirmed by the four
explicit "fully grown" tile checks above):

- A crop advances **one growth stage per night it was watered** (by hand or
  by rain). Skipping a day only *stalls* it - no regression, no penalty.
- **Rain is a double win, not just free water.** If a growing tile is
  **dry** going into the night, rain simply waters it (`INC` once,
  `bank_82.asm:2410-2415/2460-2468`) - no growth that night, but it starts
  tomorrow already watered. If the tile is **already watered**, the season
  dispatch grows it a full stage (`INC`, landing on the even/dry id of the
  next stage) and then immediately re-enters the rain check, which waters
  it again (`INC` a second time) - so a rainy night on an already-watered
  crop is a full growth stage **and** it stays watered, saving the next
  day's watering action (`bank_82.asm:2450-2458` → falls through to
  `.wateredraincheck` → `.watertile`).
- Once mature, a crop is stable: it just toggles watered/dry overnight
  (`0x61⇄0x60` for potato) and never dies from age, season, or neglect.
- Harvest leaves the tile as watered-tilled `0x08` (per the runtime code) -
  replanting needs no re-hoe, just a seed bag.

### 2.1 Weeds cannot destroy a crop tile

`.addrandomtrash` (`bank_82.asm:2470-2490`) is reached **only** when the raw
tile value is `< 3` (`bank_82.asm:2335-2337`, `CMP #$03; BCS .notempty`) -
i.e. bare/empty ground, never a tilled, seeded, or growing/mature crop tile.
Gated further: skipped entirely in Fall/Winter (`season==2` or `3`), only
rolled on nights where `!day & 0x03 == 0` (every 4th day), and then a single
`GetRNG` byte must equal exactly `0` (~1/256 per eligible bare tile per
eligible night). **A weed can never appear on, or destroy, a tilled or
planted tile** - it only colonizes ground you haven't hoed yet. This fully
answers the "does a weed on a crop tile destroy the crop" question: no, by
construction the dispatch order never lets it reach one.

### 2.2 Watering can & stamina

- Watering-can capacity is **20** (`0x14`), confirmed at
  `ToolAnimationWateringCan` (`bank_82_toolanimation_subrutines.asm:163`):
  `LDA #$14; STA !watering_can_water` on refill (triggered when
  `PreCheckToolSuccess` reports "at a water source", state `2`). Each use
  costs exactly 1 charge (`DEC A`,
  `bank_82_toolused_subrutines.asm:1187`).
- One hoe swing costs **2 stamina** (`LDA #$FE` = -2, `ChangeStamina`,
  `bank_82_toolused_subrutines.asm:293-295`). One watering-can use also
  costs **2 stamina** (`:1220-1222`, identical `#$FE`).
- `!max_stamina` starts at **100** (`NewGameSetup`, `bank_83.asm:4549-4550`)
  and is fully refilled every `NightReset` (`bank_82.asm:459-461`,
  unconditional `LDA max_stamina; STA current_stamina` - confirms the
  existing doc's "no forced bedtime" framing: stamina is never a
  cross-day constraint, only a same-evening one). At 100 stamina / 2 per
  action that's **50 hoe-or-water actions before exhaustion** (exhaustion
  itself only slows the player; it was not found to block actions
  outright in this pass - see §8). I did not locate the specific
  hot-spring partial-restore-rate code in this pass (§8).

### 2.3 Seed bag = 9 plants, confirmed, and it's a *shared* counter

`$096B` is the in-progress plant count; it is identical for all four crop
types and increments/resets in the same pattern for each:
`ToolUsedCornSeeds` (`bank_82_toolused_subrutines.asm:769-775`),
`ToolUsedTomatoSeeds` (`:821-827`), `ToolUsedPotatoSeeds` (`:872-878`),
`ToolUsedTurnipSeeds` (`:923-929`) - all `INC $096B; CMP #$09; if ==9: STZ
$096B; DEC !seeds_<crop>_N`. So **9 plants per bag is a hard game limit**,
not a bot/nav limitation - the 9th (centre) tile genuinely exists and is
worth +12.5% if the plant pass reaches it.

**Newly-noted quirk:** `$096B` is a single shared RAM cell across all four
crop types. If the bot ever switches seed-bag type mid-count (e.g. 3
potato then 6 turnip), the 9th placement - whichever type it happens to be
- is the one that gets `STZ`'d and decrements *its own* `!seeds_X_N`, while
the other type's bag counter is never touched despite having been "used."
Not currently a practical risk (the runtime plants one type per ring pass)
but worth guarding against if crop-type interleaving is ever introduced.

## 3. Prices

### 3.1 Confirmed in ROM: `Items_Price_Table` (`bank_81.asm:3060`)

Used directly by the shipping-bin drop handler `Dropedonsaleplace`
(`bank_81.asm:2563-2597`) and by the shop-preview text path
(`bank_81.asm:2321-2340`): `X = item_id*2+1`, one byte read, `XBA`-extended
to 16-bit, added straight into `!shipping_moneyL/H`. Money is stored ×10
(comment confirmed independently at `bank_83.asm:5502` and
`bank_84.asm:4631`), so **displayed price = raw byte × 10**.

I located a contiguous 4-item block at `bank_81.asm:3062` (item ids 16-19)
whose decoded prices are **120, 100, 80, 60** G. I could not find an
explicit item-id→name table in this pass, but this exactly matches - value
for value, in the same relative order - the community-sourced prices for
**Corn (120), Tomato (100), Potato (80), Turnip (60)** (see §7), which is
strong circumstantial confirmation these four ids are Corn/Tomato/Potato/
Turnip in that order. Treat the id↔name mapping as high-confidence but not
independently ROM-proven.

**2026-09-10 extension - the same block continues into egg and milk.** Read
as 16-bit LE values (the table's first entry starts one byte after the label,
so pairs align on odd offsets), the eight entries from item 16 are:

```
12, 10, 8, 6,   5, 15, 25, 35     (x10 G)
corn tomato potato turnip | egg  milkS milkM milkL
```

The first four are the block already pinned above. The next four decode to
**50 / 150 / 250 / 350 G**, which matches the community figures for
**Egg (50)** and **Milk S/M/L (150/250/350)** value-for-value and in order -
the same circumstantial argument as §3.1, one block further along. This
upgrades those four rows from "ext, UNVERIFIED-BY-ROM" to
**ROM-corroborated (block-index inference), not ROM-decoded**. The id→name
table is still not located, so the caveat in §3.1 applies unchanged.

Note what is *not* here: this is the **ship-price** table. The **purchase**
price of a chicken or cow is not in it, and was not found in this pass -
see §12.

### 3.2 Full price table (ROM-value-confirmed + community, Spring-relevant items)

| Item | Seed cost | Ship price | Grow (first harvest) | Regrow | Plantable season |
|------|-----------|-----------|------------------------|--------|-------------------|
| Potato | 200 G | **80 G** (ROM `bank_81.asm:3062` item 18; ext-confirmed) | 6 days | no - replant | **Spring only** (`bank_82_toolused_subrutines.asm:858-862`) |
| Turnip | 200 G | **60 G** (ROM item 19) | 4 days | no - replant | **Spring only** (`:909-913`) |
| Tomato | 300 G (ext, UNVERIFIED-BY-ROM) | **100 G** (ROM item 17) | 7 days (ext) | yes, 3 days (ext) | **Summer only** (`:755-759`... `:807-811`) |
| Corn | 300 G (ext, UNVERIFIED-BY-ROM) | **120 G** (ROM item 16) | 11 days (ext) | yes, 3 days (ext) | **Summer only** (`:755-759`) |
| Grass/fodder | - | not pinned in ROM this pass | - | - | year-round |
| Egg | - | **50 G** ship (ROM-corroborated, item 20; see above) / 100 G peddler (ext) | - | - | - |
| Milk (S/M/L) | - | **150/250/350** ship (ROM-corroborated, items 21-23) / 200/300/400 peddler (ext) | - | - | - |
| Wild grape (forage) | free | ~150 G ship / 200 G peddler (ext, UNVERIFIED-BY-ROM; matches existing doc's runtime-measured ~150G) | - | daily respawn | year-round |
| Mushroom (forage) | free | ~150 G ship / 200 G peddler (ext, UNVERIFIED-BY-ROM) | - | - | year-round |

"ext" = external guide only, not independently decoded from ROM this pass -
see §7 for sourcing and confidence.

Per-ring spring economics (8 tiles, plant D=day, harvest D+6, replant on
harvest day) - unchanged from before, since the prices didn't move:

```
potato ring value = harvests * 8 * 80  -  harvests * 200
  plant D3  -> harvest D9,D15,D21,D27         = 4 harvests -> 1760 G net
turnip ring (4d): plant D3 -> D7,D11,...,D27  ~6 harvests -> ~1680 G net
```

Potato wins per plant/harvest op and per bag; turnip wins first-cash speed
and value-per-watering (`480/4 = 120` vs potato `640/6 ≈ 107`).

## 4. Weather & risk

`WeatherTomorrow` (`bank_82.asm:1355-1584`) rolls tomorrow's weather from
season-indexed chance tables (`RNGReturn0toA(N)` returns uniform `[0,N)`;
damage/weather "hits" on result `== 0`, so probability ≈ `1/N`):

| Table (`bank_82.asm:1580-1584`) | Spring | Summer | Fall | Winter |
|---|---|---|---|---|
| `Rain_Chance_Table` | **1/6 (~16.7%)** | 1/10 (10%) | 1/10 (10%) | 0 (never rains) |
| `Snow_Chance_Table` | 0 | 0 | 0 | 1/3 (~33%) |
| `Hurricane_Chance_Table` | **0 - impossible** | **1/30 (~3.3%)** | 0 | 0 |
| `Thunder_Chance_Table` (Year-1 only) | 0 | 1/30 | 0 | 0 |

**Rain never damages crops.** `DamageProbabilityTable` row "Rain"
(`bank_82.asm:344`) = `Fence 1/96, Grass 0, Crops 0` - confirmed via
`ClimateFarmDamageCheck`'s per-tile dispatch (`bank_82.asm:2164-2309`):
only the fence tile branch has a nonzero chance on a rain night. Same for
Snow (`Fence 1/64, Grass 0, Crops 0`).

**Hurricanes are a real, previously-unmodeled risk - but Spring-safe.**
Hurricanes can *only* occur when `season==1` (Summer) - the whole
thunder/hurricane branch of `WeatherTomorrow` is only reached when
`season==1` (`bank_82.asm:1397-1402`, `.notfall`→`.thunder` only falls
through for summer). When one hits, `ClimateFarmDamagePrep`
(`bank_82.asm:262-313`) calls `ClimateFarmDamageCheck` with the "Hurricane"
row (`bank_82.asm:346`): **Fence 1/8, Grass 1/16, Crops 1/4** - i.e. on a
hurricane night, **every tilled/crop tile independently has a 25% chance of
being wiped to bare ground** (`.tiledsoil` branch, `bank_82.asm:2273-2286`,
sets the tile to `0x02`). Holding the Turtle Shell item halves the
hurricane roll (doubles the RNG denominator, `bank_82.asm:1519-1526`), i.e.
~1/60 instead of ~1/30. Hurricanes are also suppressed on the last day of
summer (`day==30` skip, `bank_82.asm:1506-1507`) and on Year-1 Summer D29
(forced Thunder instead, no damage check that day per `SetWeatherFlags`'s
`.otherclimate`/no-op path).

**Practical read:** because Spring crops now provably survive into and
mature during Summer (§0), a hurricane during Summer is a genuine tail risk
for anything not yet harvested when Summer starts - expect roughly one
hurricane in an average 30-day summer (`P(≥1) ≈ 1-(29/30)^29 ≈ 61%`,
treating nights as independent, which the ROM code does), each destroying
~25% of standing tilled/crop tiles it touches. **This risk is irrelevant to
the stated "ship by Summer D1" objective** (nothing is still in the ground
past Summer D1 under that horizon) but is a hard blocker for any future
"push into Summer" extension of the optimizer - that extension should not
assume zero attrition on crops left standing into Summer.

Thunder (`weather==5`) sets `FLAG196=$0100`, which `ClimateFarmDamagePrep`
explicitly no-ops (`bank_82.asm:303-304`) - **cosmetic only, no farm
damage.**

One more edge case, flagged but out of scope: the Winter→Spring year
rollover itself runs an extra damage check with its own "New Year" row
(`Fence 0, Grass 1/64, Crops 1/32`, `bank_82.asm:372-378`, table row at
`:348`) - this fires the same night as `MonthlyFarmTilesCheck`'s full wipe,
so it's moot for anything already wiped, but note it exists if the wipe
logic is ever changed.

## 5. Day budget - there is no forced bedtime

Unchanged from before, and now further corroborated by the stamina finding
in §2.2 (unconditional full stamina refill every `NightReset`,
`bank_82.asm:459-461`, independent of how much stamina was spent):

**The evening never ends.** You can water all night; the can refills at the
water source and stamina refills every `NightReset`, both unbounded. A
wake→sleep cycle can therefore do an arbitrary amount of work - the
runtime's ~18:00 return-home is a *policy choice*, not a game limit. A crop
still advances exactly **one stage per sleep** no matter how much you do
that day.

So plot count is **not** capped by a daily frame budget. The real caps:

1. **Objective-horizon maturity, not death.** A ring must be established
   early enough to both mature and ship *before Summer D1* to count toward
   this optimizer's stated objective - last useful potato planting for
   this horizon is still ≈**D24** (`24+6=30`), turnip ≈**D26**
   (`26+4=30`) - but see §0.1: this is a horizon-boundary effect, not crop
   death, and disappears if the objective is ever extended into Summer.
2. **Seed capital timing** - bags are 200 G; early cash comes from grapes +
   the first harvests.
3. **Harvest pile-up** - at very high ring counts the mature rings on a big
   harvest day take longer than one (practical) evening to clear + replant
   + water, stalling growth. This is the "can't harvest all in one day"
   point.

`spring_opt` models this with `CostModel.evening_frames` - a *practical*
ceiling on how long one day's work should take in a real run (default
60 000 f ≈ 16 real minutes/day), not a forced bedtime. Raise it freely.

- Frame rate ≈ **14.5 f / in-game minute** while the clock runs (measured:
  2-grape run 6194 f over 06:00→13:12).
- Calendar constraints that remain (community-sourced, **not** ROM-verified
  this pass - see §7):
  - Seed shop / shipping office **7:00–17:00, closed weekends & holidays**.
    `buy_seed_hour` policy = 12 - the shop hop must be an early-morning
    errand. (The prior `DATA16_B9CE24` ROM citation for this was wrong -
    see §1 - the real gating code is unlocated.)
  - Be on the farm at **17:00** for the `ShippingScene`
    (`bank_82.asm:160-166`, `CMP hour,#17`, ROM-confirmed); wallet credit
    is the following `NightReset` `AddMoney` (`bank_82.asm:479-485`,
    ROM-confirmed).
  - **Sundays = D7, D14, D21, D28** if weekday `0` = Sunday and Spring D1 =
    weekday `1` (`NewGameSetup`, `bank_83.asm:4533-4535`, arithmetic
    ROM-confirmed) - the weekday-name↔Sunday mapping itself is UNVERIFIED.

## 6. Measured frame costs (fill in as evidence accrues)

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

Can capacity = **20 charges** (1 tile/charge) - now ROM-confirmed, see §2.2
(previously asserted without citation).

## 7. External guides - cross-checked against ROM

| Claim | Source | Verdict |
|---|---|---|
| Potato 80G ship / Turnip 60G ship / 200G seed each | [fogu.com crops](https://fogu.com/hm/snes/crops.php) | **CONFIRMED-BY-ROM** - matches `Items_Price_Table` items 18/19 (§3.1) |
| Corn 120G ship / Tomato 100G ship, 300G seed each, Summer-only | [fogu.com crops](https://fogu.com/hm/snes/crops.php) | Ship prices **CONFIRMED-BY-ROM** (items 16/17); Summer-only planting **CONFIRMED-BY-ROM independently** (§0.1); seed cost UNVERIFIED |
| "Potato is the best profit spring crop, beats turnip despite slower growth" | [GameFAQs guides](https://gamefaqs.gamespot.com/snes/562623-harvest-moon/faqs/5354), [GamerZenith](https://gamerzenith.com/guides/money-making-guide-hm-snes/) | Consistent with our per-ring math (§3.2) - UNVERIFIED as a ROM fact (it's a derived strategy claim, not a table) |
| Spring crops "likely do not survive the transition to summer" (inferred, not stated outright) | [fogu.com crops](https://fogu.com/hm/snes/crops.php) | **CONTRADICTED-BY-ROM** - see §0, this is exactly the wrong belief this doc is correcting |
| Shops closed on weekends; Day 1 is a Tuesday | [GameFAQs (LoudKing)](https://gamefaqs.gamespot.com/snes/562623-harvest-moon/faqs/5354) | UNVERIFIED-BY-ROM - weekday arithmetic checks out (§1) but the day-name mapping and shop-hours gate weren't located |
| Egg 50G ship/100G peddler; Milk S/M/L 150/250/350 ship, 200/300/400 peddler; Wild grape 150G ship/200G peddler; Mushroom 150G/200G; Golden egg 10 000G peddler-only | web search (Harvest Moon Wiki / Fandom, via search snippet - direct fetch of harvest-moon.fandom.com returned HTTP 402 in this session) | UNVERIFIED-BY-ROM this pass - grape figure matches the existing doc's own runtime-measured ~150G, which is reassuring but not an independent ROM check |
| A "Shipper vs. Peddler" dual price exists, peddler pays more | same as above | UNVERIFIED-BY-ROM - no peddler-specific code path was located this pass |
| Power Berry duplication glitch (tired-animation + screen-transition dupe) | [Speedrun.com HM1 forum](https://www.speedrun.com/hm1/forums/24xtq) | Noted, not judged - not investigated in ROM; flagged per the task's ask, not a money exploit directly (it dupes a stamina-boosting collectible) |

## 8. Unverified / open questions - be honest about the gaps

- **Shop/shipping-office open hours and weekend closure**: not located in
  this pass. The previous `DATA16_B9CE24` citation was wrong (it's dialogue
  text, `bank_B9.asm:1464`). The actual gating code (hour + weekday check
  around the general store / shipping bin) needs a fresh search, likely in
  `bank_84.asm` or the event-script interpreter rather than a flat data
  table.
- **Corn/Tomato seed cost** (300G per fogu.com) not independently decoded
  from `Items_Price_Table` or another ROM table this pass.
- **Grass/fodder, egg, milk, and forage (grape/mushroom) ship prices**: not
  independently decoded from ROM. The existing doc's ~150G grape estimate
  (from runtime measurement) lines up with the community figure, which is
  reassuring corroboration but still not a ROM table citation.
- **Item-id → name mapping for `Items_Price_Table`**: I did not find an
  explicit lookup table associating item ids 16-19 with "corn/tomato/
  potato/turnip" by name - the mapping is inferred from an exact 4-value
  price match against external sources. High confidence, not proof.
- **Hoe/watering-can hidden bonuses**: `ToolUsedHoe` (`bank_82_toolused_
  subrutines.asm:157`) contains RNG rolls for buried money bags (1G/5G,
  1/16 each) and Power Berries (1/64, capped at 2 lifetime) - noted for
  completeness, not modeled in the economy since they're small and
  incidental.
- **Exhaustion (`!exaustion_level`) gameplay effect**: `ChangeStamina`
  (`bank_81.asm:7590`) computes an exhaustion level 0-3 from remaining
  stamina fraction but I did not trace what in-game effect (movement
  speed? forced collapse?) it produces, or whether it ever *blocks*
  further tool use outright rather than just animating tiredness.
- **Hot-spring stamina restore rate**: not isolated from the generic
  `ChangeStamina` call sites in this pass - only the guaranteed full
  refill at `NightReset` (§2.2) is confirmed.
- Do **not** treat the §3.2 "ext" rows as ROM fact - they're included
  because they're needed for planning and are plausible, cross-checked
  community numbers, but the task instruction not to invent numbers means
  I'm flagging them rather than asserting them.

## 9. Ranked money levers (Spring-1, D3→D30 horizon)

1. **Establish potato/turnip rings as early as capital allows, replant
   every ring the day it's harvested, through D24 (potato) / D26
   (turnip).** This is unchanged and remains the dominant lever - see §3.2
   math (~220 G/tile/cycle net for potato).
2. **More nav-reachable ring sites** - every extra ring that establish +
   water + harvest can actually execute is worth ~1 800 G/cycle × up to
   4 cycles ≈ 1 700 G net/spring. Still the #1 *practical* (nav-limited)
   lever per `spring_opt` (§10) - only 2 sites are execution-proven today.
3. **Replant cadence** - 1 cycle/ring → 4 cycles/ring, currently blocked by
   a nav timeout after harvest (rr-20w.3.2). Worth as much as lever 2 if
   fixed.
4. **Grape discipline D3-D6** to bootstrap the first 1-2 seed bags before
   the first harvest lands (grapes have no ROM-confirmed price this pass,
   see §8, but the runtime-measured ~150G/grape line matches community
   figures).
5. **Centre tile** - ROM-confirmed 9-plant bags (§2.3); the runtime 8-ring
   leaves the centre unplanted, a standing +12.5% if the plant pass can
   reach the notch.
6. **Turnip-for-speed opening** - turnip's faster cycle (4d vs 6d) gives a
   higher value-per-watering (120 vs ~107 G/watering) and faster first
   cash if watering-frame budget, not seed capital, is the bottleneck
   early on.
7. *(New, low priority for this horizon)* **Do not fear a D25-D30
   potato/turnip planting for its own sake** - it is not wasted, it just
   won't ship before Summer D1 under the current objective. If the
   objective is ever extended a few days into Summer, this stops being
   dead weight and should be re-evaluated jointly with the Summer
   hurricane risk (§4).

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
  (§5.3), tunable via `evening_frames`. vs the current reactive campaign
  ~$1.5 k this is ~15×.
- Recommendation unchanged: **grapes D3–D6 to bootstrap the first 2–3
  bags, then ramp potato rings as fast as capital + nav allow, replanting
  every ring the day it's harvested.** Stop planting *new ground* ≈ D24 -
  now understood as an objective-horizon boundary, not a death cutoff
  (§0.1), which matters if the horizon ever moves.
- The binding real-world constraint is **nav**: only 2 ring sites are
  currently execution-proven (`WEST_POCKET_PLANT_CENTER (13,28)` +
  `SECOND_POCKET_PLANT_CENTER (19,28)`). Unlocking more nav-reachable ring
  sites in the cleared farm is the top lever.
- Grape sensitivity: with `allow_grapes_through_day` unbounded the solver
  still likes 2 grapes/day early; sustained daily 2-grape yield past D6 is
  measured **once**. There is no wallet cutoff; `GrapeDaySpec` sets count
  and bail hour from day + harvest (D2: 1/10; harvest morning: 1/9; else
  2/12). Measure the second loop if the grape line is pursued.

## 10. Structural levers (biggest first, unchanged priority order)

1. **More nav-reachable ring sites.** See §9.2.
2. **Replant cadence.** See §9.3. Blocked by
   `CROP_ESTABLISH → nav_pocket_hoe_stand` / `nav_hoe_ring_*` timeout
   after harvest (rr-20w.3.2, WIP).
3. **Grape return_to_bin** (rr-20w.3.1 residual) - the shared outbound
   force-run at mountain `(520,712)` pinned the return at ~(505,633).
   Fixed: plain waypoint + `MultiMapNavTask` run_direction stall guard.
4. **Grape discipline.** Grapes D3–D6 only, to fund the first bags.
5. ~~**Centre tile** (+12.5%).~~ **WITHDRAWN 2026-09-10 - this lever does
   not exist; it was a misreading of the ROM.** A bag is indeed 9 tool-use
   cycles (`$096B`, `bank_82_toolused_subrutines.asm:846-892` - independently
   re-read and confirmed), but the auto-plant offset table
   (`DATA8_8292FA`, `bank_82.asm:1888-1897`) resolves **slot 0 to offset
   (0,0) - the player's own standing tile**, with slots 1-8 mapping to the
   8 ring cells (S, SE, E, NE, N, NW, W, SW). The centre is deliberately
   left untilled *because it is the mandatory stand tile*, so cycle 1
   structurally cannot plant anything. **1 bag → exactly 8 tiles, with no
   wasted charge.** There is no 9th planting to reach.
6. **Carry-swap robustness.** `swap_preserve_hoe` (SwapCarrySlotsTask)
   times out on the farm after BUY_SEEDS, so the bag never enters carry
   and CROP_ESTABLISH fails `select_carry_0x07`. Alternative: toss the
   can before the seed fetch and re-fetch it in the water pass.

## 11. Cash flow - the day-by-day ledger (added 2026-09-10)

`optimize_spring` now reports every plan as a **cash-flow ledger**, not just
a final wallet. `SpringPlan.ledger()` prints one row per day, and every row
balances exactly:

```
wallet_end == wallet_start + (crop_income + berry_income) - (seeds + livestock)
```

Run it with `spring_plan --rings N --ledger`.

### 11.1 The two income lines have different timing

- **`crop_income`** is the shipping-bin credit that posts at the *next*
  morning's `NightReset AddMoney` (§2.2). Gold harvested on D14 cannot buy
  seeds on D14. The ledger shows this as `bin` on the harvest row and
  `crop` on the row after.
- **`berry_income`** (grapes) is same-day cash. This asymmetry is why the
  grape line matters out of proportion to its size: it is the only income
  that can fund a *same-day* seed run.

`CostModel.grape_value_g = 150` is **confirmed** by run12
(`mountain grape 2/2 shipped: shipping_money=150->300`). What is *not*
settled is the grape **count**: the runtime ships 1/day, not 2 (§11.4).
`spring_plan --sensitivity` has a grape-price row regardless.

### 11.2 What the ledger actually says: cash is not the constraint

Spring-only horizon (D3→D30), potato, `CostModel` calibrated against
`run12_short.log` (§11.4). "Today" = the runtime caps that actually hold -
one seed bag and one grape per day. "Caps lifted" = the same search with
`max_bags_per_day=4, grape_max_per_day=2`, i.e. what fixing the runtime
would buy.

| rings | Summer-D1 G (today) | crop in | berry in | cash-blocked | frame-blocked | Summer-D1 G (caps lifted) |
|-------|---------------------|---------|----------|--------------|---------------|---------------------------|
| 1  | 5 610 | 1 920 | 3 600 | 0 | 0 | - |
| 2  | 5 790 | 3 840 | 3 300 | 0 | 0 |  9 590 |
| 3  | 7 310 | 5 120 | 3 300 | 1 | 0 | - |
| 4  | 8 050 | 5 760 | 3 600 | 0 | 0 | 10 220 |
| 6  | 8 330 | 7 040 | 3 000 | 1 | 1 | 11 500 |
| 8  | 8 330 | 7 040 | 3 000 | 1 | 1 | 11 500 |

1. **CORRECTION (same session, after run13).** An earlier version of this
   section called the single-bag-per-day cap "the most expensive line in the
   ledger", worth ~+3 200 G. **That was wrong**, for two independent
   reasons found by auditing the runtime instead of the model:

   - The "+3 200 G" priced a 6-ring plan against a 1-bag cap, but only
     **2 ring sites are nav-proven** (`POCKET_PLANT_CENTERS`,
     `farm_pond.py:188-191`). Priced at the two sites that actually exist,
     fixing the buy loop is worth **+0 G** - with two rings there is nothing
     to spend a second bag on. It is dead weight until more sites land.
   - The grape half of the cap **was already fixed** in an uncommitted
     working-tree rewrite of `day_phase_berry.py`, and run13 proved it live.
     See `docs/tasks/rr-20w-run13-defects.md` §5.

   Priced at the 2 real ring sites (potato, spring-only horizon):

   | at the 2 nav-proven sites | Summer-D1 G | Δ |
   |---|---|---|
   | run12-era code (1 bag/day, 1 grape/day) | 5 790 | - |
   | + fix the `BuySeedsTask` buy loop | 5 790 | **+0** |
   | + the grape fix (already written, uncommitted) | 9 590 | **+3 800** |
   | + nav-prove a 3rd ring site, both fixes | 10 770 | +4 980 |

   **The ring-site count gates everything.** Seed-bag plumbing and extra
   frames are both worthless without somewhere to plant.

2. **The bag cap is four stacked limits, not one** - all must fall together
   for a multi-bag day: the one-shot `_bought` boolean
   (`buy_seeds.py:406-421`); `day_plan_decision.py:255-263` scheduling only
   one `BUY_SEEDS` phase per day; `crop_planner.py:251`
   `max_seed_bags: int = 1`; and `crop_establish.py:331-335` hardcoding
   `max_bags=1` across all three fallback tiers. None is a ROM limit - the
   buy routine (`bank_81.asm:719-723`) just increments `!seeds_potato_N`,
   saturating at 255.

3. **Whether crops or berries lead the ledger is regime-dependent**, so
   neither is safe to quote without naming the cap regime. Under run12-era
   code crops lead; with the grape fix live, berries lead again.

4. **Rings saturate at ~4-6 in the model, but the real farm has 2 sites.**
   The pre-correction "18-20 rings" came from a horizon running to Summer
   D30 *and* a search that sowed potatoes in Summer, which the ROM does not
   allow (§0.1, §13).

5. **The model cannot express reliability, and that is now its biggest
   gap.** run13 shows the grape phase delivering 2 grapes on about half its
   days and **failing outright on the rest** - six distinct failure modes,
   10 of 14 total run failures (`rr-20w-run13-defects.md` §1, §3, §6).
   `CostModel` has no success-rate term, so every figure in this section
   silently assumes phases that work. Adding one is pending the completed
   run13.

**All magnitudes in this section are provisional** until recalibrated from
the completed D3→D30 run13. The orderings above are structural and expected
to hold; the numbers are not.

### 11.4 Calibration provenance - and what is still not measured

`CostModel` defaults now come from `logs/spring_d3_30/run12_short.log`
(D3→D13, clean run, `uv run python -m harvest.scripts.spring_plan
--calibrate <log>`):

| field | default | evidence |
|-------|---------|----------|
| `sleep_hour` → `evening_frames` | 20:00 → 12 600 f | day-change deltas n=9, median **12 859** (11 535-16 199). The old 18:00 budgeted 10 800 and under-counted the day ~27 % |
| `home_sleep_f` | 2 450 | n=10 median; range 735-11 100 - **very noisy**, treat as weak |
| `grape_first_f` | 3 350 | n=6 median (2 733-7 200) |
| `shop_roundtrip_f` | 2 250 | n=3 median (2 239-2 411) |
| `harvest_ring_f` | 2 500 | n=2 median (370-4 650) - **n=2, low confidence** |
| `grape_value_g` | 150 | **confirmed**: log reads `shipping_money=150->300` for 2 grapes |
| `grape_max_per_day` | 1 | run12 shipped 1/2 on 4 of 6 berry days, 2/2 on 2, and lost D7-D9 entirely (one `pick: farm_to_path: pixel_stuck`, one 15:03 `BERRY_RUN_WINDOW` cutoff) |
| `max_bags_per_day` | 1 | every `BUY_SEEDS` in every log |
| `establish_*_f` | 3 000 / 1 800 | measured loop is tiny (80-286 f, n=3) but excludes a `NAV_CROP` walk that ranges 65-2 500 f; kept buffered |
| `grape_marginal_f`, `spa_refill_f`, `tool_uses_per_spa` | 1 800 / 2 400 / 40 | **still GUESSES** - never observed |

Known model-vs-ROM gaps this calibration exposed and did **not** fix:

- ~~**`RING_TILES = 8` vs 15 harvested - ~5 % optimistic.**~~
  **CORRECTED: `RING_TILES = 8` is right; this was a misdiagnosis.** Both
  rings are a full 3×3-minus-centre 8 (`plot_tiles`,
  `crop_geometry.py:194-205`; `PLOT_RING_SIZE = 8`, `crop_skills.py:56`),
  and establish only reports success once `count_ring_planted(...) >= 8`
  against live RAM. The shortfall is one specific tile, **(12,28)** on the
  west ring, and its cause is a **silent target drop in the watering
  reorder** - see `docs/tasks/rr-20w-run13-defects.md` §4. Crucially this is
  a *timing* loss (a tile left immature, so fewer completed replant cycles),
  **not** a per-cycle gold loss: mature crops never decay (§0), so a deferred
  harvest is delayed revenue, not destroyed revenue. The old flat "~5 %
  optimistic" framing conflated the two and should not be used.
- **Frame costs are interpolated**, not counted: the calibrator linearly
  interpolates phase boundaries between `[RUN]` heartbeats ~2 000 f apart.
  Right order of magnitude, not cycle-accurate.
- **Nothing past D13 is measured at all.** run12 is a `until=(0,13)` short
  run. D14-D30 - the replant-cycle steady state the whole plan depends on -
  has never been executed end-to-end.

### 11.3 Saving for a lump sum

`optimize_spring(min_cash_reserve=G)` (`spring_plan --reserve G`) is a floor
the plan may never *spend* below - seed buys must leave that much in hand.
It does not conjure the reserve; a plan that starts at 250 G still opens
below any meaningful floor. `SpringPlan.earliest_day_affording(amount,
keep_reserve_g=...)` answers the other half: the first day the ledger holds
that much realized cash. Both exist for §12.

## 12. Livestock (chicken) - stub, deliberately unpriced

`harvest.planner.livestock_econ` holds the purchase decision. The mechanism
is complete and tested (`tests/test_livestock_econ.py`); the inputs are not
measured, so `plan_purchase` returns "unknown, here is what to measure"
rather than a number. `spring_plan --chicken` prints it.

The decision, once priced, is:

```
daily_net = products_per_day * product_price - feed_cost - chore_frames * (G/frame elsewhere)
buy_day   = first day the ledger holds purchase_cost   (earlier is always better if daily_net > 0)
worth_it  = daily_net * (horizon - buy_day) > purchase_cost
```

The `chore_frames` term is the one that will decide it. §11.2 shows the
campaign is frame-blocked, not cash-blocked, so a chicken does **not** cost
"1500 G"; it costs 1500 G *plus* whatever the coop chores displace from an
evening that is already full. `CoopChoresTask` has never been frame-measured.

**What to measure, in priority order:**

1. `chore_frames_per_day` - time a `CoopChoresTask` cycle (feed + collect +
   ship) at 1 chicken and at 3, to get the fixed and marginal cost. Highest
   value: it is the term that flips the sign.
2. `purchase_cost_g` - not in `Items_Price_Table` (that is ship prices
   only, §3.1). The buy path is the Animal Shop; start at
   `ReplaceTilesAnimalShop` (`bank_81.asm:4844`) / `MapAnimalShop`
   (`src/maps/Maps_Graphics.asm:830`) and follow the purchase handler.
3. `products_per_day` - laying cadence for a fed adult, and whether rain or
   being left outdoors interrupts it.
4. `feed_cost_g_per_day` - bought feed vs. cut fodder. These are different
   economies: the second is a frame cost, not a gold cost, and therefore
   competes with the same budget as (1).

The egg **ship** price is already settled: 50 G, ROM-corroborated (§3.2).

## 13. Measuring into summer - the plan, not yet run

`Calendar.horizon_end` now defaults to **Spring D30**, because that is the
last day this model is grounded in measured data. Summer is fully modelled
and the search will run through it on request (`spring_plan
--through-summer`, which prints a caveat banner), but every such number is a
projection.

### 13.1 What the model already gets right about summer

- Spring-sown crops keep maturing through Summer - `NightlyFarmTilesCheck`
  INCs in seasons 0 *and* 1 (§0). There is no Spring→Summer wipe.
- Spring crops **cannot be sown** in Summer. `_plantable` enforces this now;
  the old model did not, which is where the inflated 18-20 ring / 22 k
  figures came from (§11.2). A `--through-summer` run today correctly shows
  the farm winding down after D30 with nothing to plant.

### 13.2 What is missing before a summer number means anything

1. **Corn/tomato are the whole point and are unrepresented.** They are
   Summer-only, regrow every 3 days, and ship at 120/100 G - a regrowing
   crop has completely different economics from potato's replant cycle
   (no seed cost per harvest, no establish frames per cycle). `CropSpec`
   already carries `regrow_days` for both; `spring_opt`'s `RingState`
   models a one-shot crop and would need a regrow branch. **Their seed cost
   (300 G) is external-only and not ROM-decoded** (§3.2) - decode it first,
   since it sets the whole capital cycle.
2. **No nav-proven summer ring sites, and no proven corn/tomato plant
   action.** Same blocker as §10.1, unsolved.
3. **Hurricanes are not modelled.** Summer-only, ~1/30 per night (~1/60
   with a Turtle Shell), and each tilled/crop tile independently has a 25 %
   wipe chance (§4). Over a 30-day summer that is roughly a coin flip on at
   least one hurricane, and it makes summer plans *risk-bearing* in a way
   spring plans are not. A summer optimiser that reports a single number
   without a distribution is lying. Model it as a scenario sweep first
   (`rain_days`-style: `hurricane_days`), then as an expectation.
4. **No measured summer frame costs.** Every `CostModel` figure is
   calibrated from spring campaign logs (D3-D11 of `run11_grapefix.log`).
   Day length, nav routes and shop hours in summer are assumed identical
   and have never been checked.

### 13.3 Order of work when summer is picked up

1. Finish spring: prove the replant cycle end-to-end on the ROM
   (`run12`), then calibrate `CostModel` against that run.
2. Decode corn/tomato seed cost from the shop path.
3. Add a regrow branch to `RingState` + a `hurricane_days` scenario to
   `Calendar`, with tests, before running any search.
4. Only then quote a summer number, and quote it as a distribution.

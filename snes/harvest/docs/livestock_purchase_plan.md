# Livestock purchase plan

Fixture editing is not a purchase. [adr/0006-clean.md](adr/0006-clean.md):
published rungs use controller input only. `livestock_builder` may write
money, feed, and slots so a later diff has something to compare. That state
is not a Clean buy.

## What the builder covers

`harvest/tools/livestock_builder.py` starts from `Y1_After_Buy_Potato` and
can write money, hay, chicken feed, cow feed, and slot 0 for one chicken
and one cow. Sheep stay blocked until a ROM dump or a live purchase shows a
separate structure. Do not assume this SNES build has sheep.

```bash
uv run python -m harvest.tools.livestock_builder --base Y1_After_Buy_Potato --verify
```

## Live buy, when it is in scope

Walk with search. Do not record a path BFS can close. Do not treat the
animal-shop menu, or a cross-map return to the farm, as a completed buy.
A buy needs the shop tilemap plus a wallet change and an animal-count
change. Playback that only returns to the origin is a miss.

## Controller verification, 2026-09-29 — rr-bu62.28

The shop charged **5,000G for a cow** and **1,000G for a chicken**. The
summer window recorded both wallet and count deltas; the counter-only chicken
probe ended on tilemap `0x24`, 4,090G → 3,090G, chickens 0 → 1, in 2,796
frames. Reports are under `recordings/rr-bu62.28-*.json`.

The shop compares `planted_grass` with 16 × the resulting total number of
animals. Grass counts immediately when planted; it need not be mature to buy.
The sickle cuts tile `0x79` into stored hay. Both animal chores use this hay.
Rain/snow, a purchase earlier that day, and full animal capacity also block
delivery. The dialogue's selected row is `$018F`; `$018A` is a text offset.

The daily planner establishes at least 32 pasture tiles from Summer D2 and
throughout Fall D2–D20, at most two bags per morning, and cuts feed when mature
grass is present. Grass bags never spend gold earmarked for the Summer seed
bags, the hen, or a funded cow (`pasture_reserve_g`). The purchase order
(`next_livestock`, Year 1, Summer and Fall, one animal a day, shop open and
delivery weather) is **hen first, then cows up to two**: the hen costs 1,000G
plus the 300G floor and 16 planted tiles, and every egg she lays can be
incubated into a free hen; a cow costs 5,000G plus the floor and 32 tiles and
earns nothing until its 21 child nights pass and the brush and milker are owned.
Eve at 250 hearts does not add the house fund on top of either price. The
carpenter payment waits until a cow exists. Four failed purchase routes in a
season stop buying (`include_animal_purchases`) but never stop chores.

Grass gate trap: the shop compares planted grass with 16 tiles per resulting
animal, counting every hatched chick. A 36-tile pasture therefore allows only
one hen plus one cow. With a flock of 12 the next cow needs 16 x (12 + cows + 1)
tiles, so a cow must be bought before the flock grows or the pasture must grow.

Chicken economy: A chicken costs 1,000G and requires 16 planted grass tiles.
Once acquired, it lays an egg daily (50G/day shipped), and eggs can be incubated
in the coop to hatch up to 12 chickens for free without further cash outlay,
creating a self-multiplying flock economy (up to 600G/day).

Fall mushroom runs: Daily normal-mushroom foraging on the mountain ships 150G/day.
Previous failures ("no loop can bank a grape past 16:00") were resolved by:
1. Settle hold on mountain entry (`0x10`–`0x13`) while coordinates are stale (`y <= 200`),
   preventing accidental step-down exits back to `0x0C`.
2. Registering tile `(20, 44)` in `NO_GO_TILES_BY_TILEMAP` for mountain maps,
   preventing the bot from pushing into the solid rock cliff face upon entry.
3. Adding dedicated reverse return routes `first_mushroom_to_mountain_exit` and
   `first_mushroom_to_shipping_bin` in `map_routes.py` so return navigation
   threads east down the slope rather than pathfinding through west cliffs.
4. Treating seasonal mountain tilemaps `(0x10, 0x11, 0x12, 0x13)` as equivalent
   in `tilemaps_match` and `_config_for_tilemap`, ensuring Fall mountain `0x12`
   loads `MOUNTAIN_WALKABLE` instead of falling back to farm tiles.

The rainy Spring D15 gate probe still does not reach the counter. Summer
funded purchases and fall mushroom shipping are verified separately. None of
these pin windows establish power-on readiness; see the living residual.

The purchase gates live in `harvest.planner.livestock_econ`. Laying cadence
and daily chore frames are still unmeasured, so no profitability verdict exists.

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

The daily planner now establishes at least 32 pasture tiles from Summer D2,
uses at most two bags per morning, and cuts feed when mature grass is present.
It schedules one cow, then one chicken, from Summer D14 when the shop is open,
delivery weather permits, 16 hay is stored, and the purchase leaves 300G.
These gates do not guarantee a D14 purchase: the verified funded window bought
the cow on D20 and the chicken on D23. Missing brush/milker ownership does not
prevent feeding and talking. Buying those tools and measuring milk output
remain future work.

Fall adds a daily normal-mushroom pickup/keep/return/ship phase. Three
consecutive pin mornings deposited 150G each and credited 450G overnight.
The phase avoids the poisonous mushroom and reads the live Eat/Don't eat menu.
The exact summer predecessor also shipped on Fall D1–D3, including after
fodder cutting. After the rr-bu62.29 visitor handoff fix, barn entry and both
feeds succeeded on 7/7 mornings (98,168 frames, zero rescues). Cow talking
still times out on Fall D3; feeding completes before that attention failure.

The rainy Spring D15 gate probe still does not reach the counter. Summer
funded purchases and fall mushroom shipping are verified separately. None of
these pin windows establish power-on readiness; see the living residual.

`harvest.planner.livestock_econ` still lacks production cadence and daily
frame-cost measurements. Its profitability verdict remains unavailable.

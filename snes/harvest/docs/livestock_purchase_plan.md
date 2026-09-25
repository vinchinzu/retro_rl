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

Prices, and whether a chicken is worth buying, are unmeasured. The stub is
`harvest.planner.livestock_econ`. It refuses to answer while inputs are
missing. Ship prices that are already pinned live in
[SPRING_ECONOMY.md](SPRING_ECONOMY.md).

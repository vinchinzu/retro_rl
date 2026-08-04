# Level 1 route — The Eagle

This route folds the external walkthrough into the emulator-verified room
graph. The walkthrough is a planning accelerator, not runtime assistance:
room IDs, transitions, object types, combat policies, and stop predicates were
confirmed in fceumm before inclusion in the natural-entry chain.

Primary planning source:
[Zelda Dungeon — Level 1: The Eagle](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-1-the-eagle/).
The source identifies the ordered encounters, optional Compass/Map/Bow and
Boomerang branches, Wallmaster key, Aquamentus, Heart Container, and first
Triforce shard. The ASCII layout in
[TheStarbird's GameFAQs guide](https://gamefaqs.gamespot.com/nes/563433-the-legend-of-zelda/faqs/36907)
was used to disambiguate the boss and Triforce room directions.

## Verified speed route

```text
0x73 entrance
  → E 0x74 carried key → W 0x73 → unlock N
  → 0x63 clear 3 Stalfos → N
  → 0x53 clear 5 Stalfos + key
  → W 0x52 clear 6 Keese → N
  → 0x42 clear 3 Gels, push switch block
  → W 0x41 hint room → E through 0x42
  → 0x43 clear 5 Gels (Map room; Map pickup skipped)
  → N 0x33 clear 3 Stalfos + key
  → N 0x23 clear 3 Goriyas + key
  → S 0x33 → S 0x43 → unlock E
  → 0x44 clear 3 Goriyas (Boomerang pickup skipped)
  → E 0x45 defeat Wallmasters + key
  → unlock N 0x35 Aquamentus + Heart Container
  → E 0x36, route around the lower Eagle wall, collect Triforce shard 1
```

The optional east branch from `0x53` is `0x54`: eight Keese and the Compass.
The optional Bow branch is west of `0x23`. Neither item is required by the
current completion policy. The route also skips the Map and Boomerang pickups
after clearing their rooms.

## Source correlation and live evidence

| Room | Walkthrough role | Live observation |
|------|------------------|------------------|
| `0x54` | Compass branch | 8 Keese; `RoomItemId=0x16` |
| `0x52` | Six-Keese continuation | 6 Keese; north transition to `0x42` |
| `0x42` | Gel switch room | 3 Gels; push at x≈112 opens the left door |
| `0x41` | Old Man hint | Dialog room; current route visits it before returning east |
| `0x43` | Map room | 5 Gels; `RoomItemId=0x17` |
| `0x33` | Stalfos key | 3 Stalfos; fixed key inventory increase |
| `0x23` | Goriya key / Bow branch | 3 Goriyas; fixed key; west branch skipped |
| `0x44` | Boomerang | 3 Goriyas; `RoomItemId=0x1D`; pickup skipped |
| `0x45` | Wallmaster key | 8 Wallmaster slots; fixed key; north boss door |
| `0x35` | Aquamentus / Heart | type `0x3D`; fireballs `0x55`; health `0x20→0x31` |
| `0x36` | Triforce shard 1 | lower opening at x≈112–128; `triforce & 0x01` |

The item labels for `0x16`, `0x17`, and `0x1D` are
walkthrough-correlated. They are intentionally not described as collected
inventory facts because the speed route skips those pickups.

## Acceptance

Run:

```bash
uv run python zelda_i/scripts/run_level1_complete.py --trials 2
uv run python zelda_i/scripts/run_level1_complete.py \
  --natural-entry --trials 2 --save-state
```

Both modes passed 2/2 on 2026-07-28. The natural run starts from power-on,
uses no state load or RAM write, and stops only after `ADDR_TRIFORCE & 0x01`.
Evidence is in `recordings/level1_complete_natural.json`; the report records
the exact entry class and every room-level stage result.

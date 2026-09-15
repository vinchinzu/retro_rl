# Pre-L1 0x6F corridor (4.5.1 geometry)

Living evidence for bombs at `shop_p7` 0x6F. Assisted, `--infinite-life`
(heart refill only). Power-on, no state load, no pokes. Scratch probe
`nes/zelda_i/scratch/probe_pre_l1_6f.py`. PNGs under gitignored
`recordings/scratch_pre_l1_6f/`. Do not STATUS. Cave buy not claimed.

`overworld/gathering.py` exists (sibling). Its `SHOP_P7_HOPS` is the
**dead** row-6 table. Do not silently overwrite — merge from this file.

## Verified

Natural entry, `boot_to_ready(first_playthrough=True)`, sword cave, then
`OverworldPathController` (`require_sword=True`, `farm_below_hearts=0`,
`evade=True`, `occupied_lane=True`, `need_rupees=20` scoop,
`max_farm_attempts=0`). Assist `progression_writes=0` `capacity_writes=0`.

Sword leftover (both trials): play **0x77** `(64,77)` mode 5 sword 1
rupees 0 bombs 0 hp `0x22` lo==hi (3/3). 749f. PNG `t1b_sword_final.png`.

### Live screens walked (t2, 3050f after sword, hits_taken=0, rupees 1)

```
0x77 --RIGHT align_y=140--> 0x78
0x78 --UP    align_x=48 --> 0x68
0x68 --UP    align_x=48 --> 0x58
0x58 --RIGHT y148-162  --> 0x59
0x59 --RIGHT y120-145  --> 0x5A
0x5A --RIGHT y130-150  --> 0x5B
0x5B --RIGHT y80-95    --> 0x5C
0x5C --RIGHT y120-140  --> 0x5D   [LEVEL2_5C_MAZE_WAYPOINTS]
0x5D --RIGHT y130-150  --> 0x5E
```

Notes: `hop_0_78` … `hop_8_5e`, `maze_start`, `maze_complete`. 16 evades.
Did not enter 0x79, 0x67, 0x68 door-repair, or L8 0x6D.

t2 leftover glance: play **0x5E** `(224,141)` mode 5 facing E sword 1
rupees 1 bombs 0 hp `0x22` lo==hi (3/3) tile 217. PNG `t2_hop_final.png`
(candle-shop sand pocket, cave north, tree wall east).

t1b leftover (row-6 halt): play **0x68** `(48,141)` mode 5 facing E sword 1
rupees 0 bombs 0 hp `0x22` lo==hi tile 197. 573f after sword, hits 0.
Notes `hop_0_78`, `hop_1_68`. PNG `t1b_hop_final.png`.

### Occupancy on 0x68 east (t3/t4, gathering.ShopP7WalkController)

t3 sliding-y goal `(EDGE_EAST_X, link_y)`: 3955f, 6 misses, leftover play
**0x68 `(48,213)`** `bush68_stand` (south trees). PNG `t3_final.png`.

t4 fixed goal `(EDGE_EAST_X, 149)`, playfield ymax: 565f, 17 misses,
leftover play **0x68 `(48,198)`** mode 5 sword 1 rupees 0 bombs 0 hp `0x22`
lo==hi (3/3) hits 0. Still the west sand column. The east octorok strip is
visible and disconnected — no 1px path from x=48.

## Sitting Progress (2026-09-14)

1. **South Hop Landed**:
   - `0x58` UP @ x=48 -> `0x59` RIGHT (y 148-162) -> `0x69` DOWN @ `align_x=120` (open band `[109, 131]`).
   - Replaced dead `0x68` EAST attempt with live corridor: `0x77` -> `0x78` -> `0x68` -> `0x58` -> `0x59` -> `0x69`.
2. **Row 6 Extension & Measured 0x6C Barrier**:
   - `0x69` -> `0x6A` -> `0x6B` walked cleanly at `align_y=141`.
   - Continuous spine `--through pre-l1` advances from power-on cleanly past `0x68`, `0x58`, `0x59`, `0x69`, `0x6A`, `0x6B`, to `room=0x6c` with 0 hits taken.
   - **0x6C EAST dead**: Entering 0x6C places Link into a walled west pocket (`x <= 64`). Solid vertical bush column at `x≈80` blocks through-access to 0x6D.
3. **Drop Mechanics & Clean Farming**:
   - Added `BOMB_DROP_OBJECT_TYPE = 0x60`, `BOMB_DROP_STATE = 0x00`, and `BOMB_DROP_STATES` in `ids.py` and `combat.py`.
   - `HeartFarmController` updated to collect bombs (`int(snap.bombs) < 8`).
   - `ShopP7WalkController` equipped with transit drop scooping: scoops hearts when hurt, bombs when < 4, and rupees when < 20 without restock-farm stalls (`max_farm_attempts=0`).

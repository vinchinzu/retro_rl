# Residual — rr-8t4.3 L7-C 0x0D walk-on **SOLVED** (2026-09-04)

Did not STATUS. Did not touch `docs/STATUS.md`, `level8/**`, L9, or harvest.
Did not `git push` or `git add -A`. No position pokes anywhere
(`position_writes=0`), no door/key/Triforce/Map/Whistle/candle/bomb/HC/mode/
facing writes, no state loads mid-run. OccupancyWalker still banned in 0x0D
(unit-tested). `MEASURED_POST_L7_EXIT.verified` stays **False**.
`make_aquamentus_heart_controller` / `make_level7_shard_leave_controller`
stay fail-closed; only `make_tip_stairs_controller` was flipped, on 2/2.

## What cracked it

The blocker was never the terrain — it was the measuring instrument plus a
stale guard.

1. **`$049E` `colliding_tile` is direction-sensitive**: it reports the tile
   Link is walking *into*, so every prior tile "sweep" of 0x0D was a lead,
   not a map. The room's real collision map is in **cart WRAM at `$6530`**,
   column-major, 32 cols x 22 rows of 8x8 tiles (stride 22);
   `env.get_ram()` index `0x800` is `$6000`. New read-only reader:
   `zelda_i.dungeon.tilemap` (`ascii_room`, `stair_cells`, `door_cells`,
   `link_cell`). Independently cross-checked against a 16x16 block
   classification of the rendered frame before it was trusted.
2. **`INLAND_X = (64, 192)`** in `scratch/probe_l7_room0d_push.py` — a
   wallmaster-grab guard carried over from the *uncleared* recon — excluded
   **both** of the room's only two corridors. The pin is cleared
   (`room_all_dead=4`, no wallmasters), so the guard was pure loss.

Measured map of play 0x0D (16x16 cells; `.` floor, `#` solid, `S` stairs,
`B` the pushable `0x68`; west door at `(16,144)` is outside the interior):

```
        x=  32  48  64  80  96 112 128 144 160 176 192 208
 y= 96       .   .   .   .   .   .   .   .   .   .   .   S   <- stairs (post-push)
 y=112       .   #   #   #   #   #   #   #   #   #   #   .
 y=128       .   .   .   .   .   .   .   .   .   .   #   .
 y=144       .   .   .   .   .   .   .   .   .   .   B   .
 y=160       .   .   .   .   .   .   .   .   .   .   #   .
 y=176       .   #   #   #   #   #   #   #   #   #   #   .
 y=192       .   .   .   .   .   .   .   .   .   .   .   .
```

It is a **ring**. `x=32` (west) and `x=208` (east) are the only crossings of
the `y=112` / `y=176` solid bands. Link's stored `y` is 11px above the cell
row his feet collide with (`y=93/109/125/141/157/173/189` -> cell rows
`96/112/128/144/160/176/192`), which is why UP from `(144,141)` parks at
`y=117` and not `y=125`.

## Glance (pin start, all trials)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2 (west), `room_all_dead=4`, keys 2, bombs 6,
candle 2, whistle 1, ladder 1, TF 0, 3 HC. `0x68` `(192,144)`.
`stair_cells()` **empty**. `door_cells()` = `[(16,144)]`.

## RAM claims (written before the first live trial)

Committed in `scratch/probe_l7_room0d_ring.py`'s docstring and mirrored in
`level7.stairs0d.RAM_CLAIM` before any run.

| id | claim | miss condition |
|----|-------|----------------|
| R1 | UP to `y=125`, LEFT to `x=32` along the `y=128` row | LEFT pins at `x>34` |
| R2 | UP the `x=32` column to `y=93` | UP pins at `y>=101` |
| R3 | pre-push `stair_cells()` is empty | stairs already present |
| R4 | RIGHT push puts the block quad at `(208,144)` and stairs at `(208,96)`; the "(208,96) block" is a RAM artifact | tile map shows the block quad at `(208,96)` |
| R5 | RIGHT along the `y=96` row onto `(208,93)` enters cellar 0x7B mode 9 | Link stands at `(208,93)` in play 0x0D with no mode change |

## Trials

Probe `scratch/probe_l7_room0d_ring.py` (raw legs), then
`scratch/probe_l7_room0d_stairs0d.py` (the shipped controller).

| tag | leg / dest `$EB`/mode/xy | frames | PNG | vs claim |
|-----|--------------------------|--------|-----|----------|
| `0d_ring_v1` push | block RAM `(208,96)`, **tile map quad `(208,144)`**, `stair_cells()` -> `[(208,96)]` | 182 | `0d_ring_v1_pushed.png` | **HIT** R3+R4 |
| `0d_ring_v1` west_column | play 0x0D `(32,125)` | 311 | `0d_ring_v1_west_column.png` | **HIT** R1 |
| `0d_ring_v1` north_column | play 0x0D `(32,93)` | 335 | `0d_ring_v1_north_column.png` | **HIT** R2 |
| `0d_ring_v1` east_top_row | **cellar `0x7B` mode 16 -> 9** `(208,93)`, settle `(192,93)` | 466 / 706 | `0d_ring_v1_cellar.png` | **HIT** R5 |
| `0d_ring_v2` | same, bit-identical | 466 / 706 | `0d_ring_v2_cellar.png` | **HIT** — probe **2/2** |
| `20260904_S1` first cut | `Level7Stairs0DController` FAILED `phase_push_stalled_64_145`, play 0x0D `(64,145)` | 701 | none retained (tag re-used by the fixed re-run) | **MISS** — see dead belief 4 |
| `20260904_S3` | **cellar `0x7B` mode 9**, settle `(192,93)` | 458 ctl | `20260904_S3_dest.png` | **HIT** R5 |
| `20260904_S4` | **cellar `0x7B` mode 9**, settle `(192,93)` | 458 ctl | `20260904_S4_dest.png` | **HIT** — controller **2/2** |
| `20260904_S5` chain | walk-on then the existing 2/2 `0x7B` B->A cross: play **`0x29`** `(96,157)` | 458 + 370 | `20260904_S5_cross.png` | **HIT** — no poke in the lineage |
| `20260904_S6` | **cellar `0x7B` mode 9**, fixture saved | 458 ctl | `20260904_S6_dest.png` | **HIT** |
| `20260904_S7` | **cellar `0x7B` mode 9**, settle `(192,93)` | 458 ctl | `20260904_S7_dest.png` | **HIT** — final code |
| `20260904_S8` | **cellar `0x7B` mode 9**, settle `(192,93)` | 458 ctl | `20260904_S8_dest.png` | **HIT** — final code **2/2** |

Every trial: `progression_writes=0` `capacity_writes=0` `deaths=0`
`position_writes=0`. Reports `recordings/0d_ring_v1.json`,
`0d_ring_v2.json`, `20260904_S1..S8.json`.

`20260904_S1` is the only MISS and it is a controller bug, not geometry —
it was fixed and then re-proved 2/2 (`S3`/`S4`), and once more (`S7`/`S8`)
after the last edit, so the shipped code is the code that ran green.

## Resolved discrepancy — "the block snaps to (208,96)"

**It does not.** The tile map across the push shows cols 26-27 / rows 4-5
going `74 76 / 75 77` -> `70 72 / 71 73` (a staircase at `(208,96)`) while
the block quad moves `(192,144)` -> `(208,144)`, a clean 16px RIGHT slide.
The `0x68` object's RAM `x`/`y` (`$7B`/`$8F`) is **repointed to the revealed
stairs** once the slide completes, which is what earlier sittings read.
`level7.path.ROOM_0D_BLOCK_AFTER_RIGHT` keeps the RAM-observed `(208,96)`
and is now documented as the artifact; `ROOM_0D_BLOCK_CELL_AFTER_RIGHT`
`(208,144)` and `ROOM_0D_STAIR_CELL` `(208,96)` are the tile-map truth.
(`room_all_dead` also reads garbage in this room post-push — 4/174/35/103
across one run — so do not gate on it after the push.)

## New dead beliefs

- Dead: `$049E` sweeps map a room. They are direction-sensitive; use
  `zelda_i.dungeon.tilemap` (`$6530`). The 2026-09-03 tilesweep note that
  "the y~101-115 band is solid across the whole room width" is right about
  `x=48..192` and **wrong at `x=32` and `x=208`**, which are floor.
- Dead: `x=192` is a column. Static blocks at `(192,128)` and `(192,160)`
  sandwich the pushable; the residual's "(192,133) plug tile 179" is just
  the static block at `(192,128)`. RIGHT-push-then-climb-east can never
  work — the pushed block seals `(208,144)` too.
- Dead: `x=32` is a wallmaster grab column. That was the **uncleared** recon
  (`INLAND_X`); from the cleared pin it is the main corridor. The real
  hazard there is the **west door at `(16,144)`** — travel west on the
  `y=128` row, not the `y=141` door row, or Link exits to 0x79.
- Dead (controller shape): a per-frame "align y, else press RIGHT" priority
  loop. Zelda re-snaps Link's `y` on horizontal movement, so it oscillates
  and never advances (`20260904_S1`: 700 frames, x 63 -> 64). Every leg must
  hold **one** cardinal until its own predicate, with a stall guard.
- Dead: the `(96,141)`/`(144,141)`/`(176,141)` "north-arm" and "plus-corner"
  model of this room. There is no plus and there are no statues in the
  cleared pin — the `_0D_STATUE_XY` cells are just floor. The five earlier
  MISS pins (`(144,117)`, `(176,117)`, `(144,157)`, `(96,157)`, `(64,157)`)
  are all simply the `y=112` / `y=176` solid bands, and all of them are
  reproduced exactly by the tile map.

## Landed

- `zelda_i/dungeon/tilemap.py` — read-only `$6530` room tile map reader
  (generalises to every room in the game). Never writes.
  Tests `tests/test_dungeon_tilemap.py` (9).
- `zelda_i/level7/stairs0d.py` — `Level7Stairs0DController`, single-cardinal
  phase machine `PUSH_UP -> PUSH_EAST -> PUSH_ALIGN -> PUSH_HOLD ->
  PEEL_WEST -> NORTH_ROW -> WEST_COLUMN -> NORTH_COLUMN -> EAST_TOP`, with a
  west-door guard and a per-leg stall guard. Dest is RAM (mode 9, `0x7B`).
  Tests `tests/test_level7_stairs0d.py` (13 cases). `level7/path.py` not grown
  (only comments/constants corrected).
- `make_tip_stairs_controller` **flipped** from fail-closed to
  `make_stairs0d_controller()` (2/2). `route_eligible` stays False,
  `evidence="fixture-live"`.
- Graph: `NOSE_CELLAR.ram_id` promoted `None -> 0x7B`, `evidence`
  `fixture-live`, `route_eligible=False`.
- Fixture `Level7Interior0DStairsWalkOnCellarFixture` — mode 9 cellar `0x7B`
  `(192,93)`, walk-on lineage, no position pokes. It supersedes the
  poke-derived `Level7Interior0DNoseCellarReconFixture` as the start of the
  `0x7B -> 0x29` cross (same pose, proved live in `20260904_S5`).
- Probes `scratch/probe_l7_room0d_ring.py`,
  `scratch/probe_l7_room0d_stairs0d.py`.
- Suite green: **698 passed** (`QT_QPA_PLATFORM=offscreen uv run pytest
  nes/zelda_i/tests -q`), up from 676.

## Not done / leftover

- L7-C prefix now runs `0x0C -> 0x0D -> 0x7B -> 0x29 -> 0x2A -> 0x2B` with
  **no position poke anywhere**, but only as fixture-lineage; the Survival
  `--through level7` chain is still blocked upstream (bait/pond,
  `rr-8t4.4`). Do not mark `route_eligible`.
- `make_aquamentus_heart_controller` / `make_level7_shard_leave_controller`
  stay fail-closed. `MEASURED_POST_L7_EXIT.verified` stays False.
- Not attempted: LEFT/UP/DOWN pushes of the `0x68` (RIGHT works, so the
  alternatives were not needed); the south ring (`y=192` row then the east
  column) is implemented in `probe_l7_room0d_ring.py --ring south` but was
  never run — after a RIGHT push the block seals `(208,144)`, so that ring
  needs a LEFT push.
- The `$6530` reader is only calibrated on L7 `0x0D` and the L7 `0x29`
  leftover; other rooms/levels are unverified (overworld untried).

Next leftover: Survival-lineage L7 (bait/pond `rr-8t4.4`), or apply the tile
map to the remaining unmapped rooms instead of `$049E` sweeps.

# Residual — rr-n91a / rr-8t4.3 L7-C 0x0D walk-on (2026-09-04)

Did not STATUS. Did not poke doors/TF/position. Did not touch `level8/**`,
`AGENTS.md`, harvest, or parent `rr-8t4.3` (still in_progress). Spine
`make_tip_stairs_controller` / `make_aquamentus_heart_controller` /
`make_level7_shard_leave_controller` stay fail-closed. Cellar-cross dest
0x29 (below) left untouched.

## 2026-09-04 sitting — north-corridor UP from y=141 MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, keys 2, bombs 6, candle 2, TF 0.
`0x68` `(192,144)` state 0.

### RAM claim (before run)

From pin to `(144,141)` (known live), UP toward y≤99 between upper
plus-corners `(128,125)`/`(160,125)`. Miss if UP pins at y≥117.

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_NU` col | play 0x0D `(142,141)` tile 119 | 63 | `20260904_NU_nu_144_141.png` | **HIT** `(144,141)` |
| `20260904_NU` UP x=144 | play **0x0D** m5 `(144,117)` tile **179** | 122 stuck | `20260904_NU_nu_after_up_144.png` | **MISS** y≥117 |
| `20260904_NU176` col | play 0x0D `(174,141)` tile 119 | 87 | `20260904_NU176_nu_176_141.png` | **HIT** `(176,141)` |
| `20260904_NU176` UP x=176 | play **0x0D** m5 `(176,117)` tile **179** | 146 stuck | `20260904_NU176_nu_after_up_176.png` | **MISS** y≥117 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports `recordings/20260904_NU.json`, `20260904_NU176.json`. Probe
`--north-up`. y≤99 not entered. No push-then-walk-on. Did not retry
DOWN or plug clips. `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: UP from `(144,141)` between upper plus-corners reaches y≤99.
  Pins **`(144,117)` tile 179** (same class as west `(48,117)`).
- Dead: UP from `(176,141)` (west of 0x68) reaches y≤99. Pins
  **`(176,117)` tile 179**.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — plus-corner x=144 DOWN MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, keys 2, bombs 6, candle 2, TF 0.
`0x68` `(192,144)` state 0. PNG `recordings/l7_0d_walkon_20260904_start.png`.

### RAM claim (before run)

From pin `(63,149)` to `(96,141)` (known live), RIGHT to `(144,141)`,
DOWN between plus-corners (statues x=128 and x=160 at y=157) toward
y=165. Miss if DOWN pins at y≤157 or x slides off 144.

### Trial `20260904_PC`

| step | dest $EB/mode/xy | frames | PNG | vs claim |
|------|------------------|--------|-----|----------|
| y=141 gap | play **0x0D** m5 `(94,141)` tile 119 | 27 | `20260904_PC_pc_96_141.png` | **HIT** |
| `(144,141)` | play 0x0D `(142,141)` tile 119 | 63 | `20260904_PC_pc_144_141.png` | **HIT** |
| DOWN x=144 | play **0x0D** m5 `(144,157)` tile **178** | 117 stuck | `20260904_PC_pc_after_down.png` | **MISS** y≤157; x held 144 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Report `recordings/20260904_PC.json`. Probe `--plus-corner`. Did not
retry Trial B. No dest fixture. `NOSE_CELLAR.ram_id` stays None.

### New dead belief

- Dead: x=144 between plus-corner statues `(128,157)` / `(160,157)` is a
  walkable DOWN corridor to y≥160. Moving Link pins **`(144,157)` tile
  178**. x does not slide off 144.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — Trial A y=141 lure + Trial B plug clip MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, `room_all_dead=4`, keys 2, bombs 6, candle 2,
whistle 1, ladder 1, TF 0. `0x68` `(192,144)` state 0.
PNG `recordings/l7_0d_walkon_20260904_start.png`.

### RAM claim (before Trial A)

From pin `(63,149)`, walk to `(96,141)` via y=141 gap (UP/RIGHT, never
DOWN at x=63), then DOWN to `(96,165)` (known lure), then RIGHT along
y=162-165 toward x=192, then UP to the 0x68 south face. Miss if
`(96,165)` is not reached, or RIGHT pins at x≤176, or south face
unreachable.

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_A` gap | play **0x0D** m5 `(94,141)` tile 119 | 27 | `20260904_A_a_96_141.png` | **HIT** y=141 gap |
| `20260904_A` lure | play **0x0D** m5 `(96,157)` tile **178** | 81 stuck DOWN | `20260904_A_a_96_165.png` | **MISS** never `(96,165)` |
| `20260904_B` push | play 0x0D `(177,141)` block **(208,96)** s2 | — | `20260904_B_b_pushed.png` | RIGHT push as known |
| `20260904_B` LEFT+UP | play 0x0D `(191,133)` tile 118 | 1 | `20260904_B_b_clip_LEFT.png` | still y=133 |
| `20260904_B` RIGHT+UP | play 0x0D `(192,133)` tile 179 | 1 | `20260904_B_b_clip_RIGHT.png` | no move |
| `20260904_B` grade | play **0x0D** m5 `(192,133)` | — | `20260904_B_b_plug.png` | **MISS** boxed y≥133 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports: `recordings/20260904_A.json`, `recordings/20260904_B.json`.
Probe `--trial-a` / `--trial-b`. No dest fixture. No Trial C (south face
never stood). `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: DOWN from y=141 gap `(96,141)` reaches lure `(96,165)`. Pins
  **`(96,157)` tile 178** (plus SW diamond). The y=141 gap itself is live.
- Dead: one-frame LEFT+UP / RIGHT+UP at the `(192,133)` plug after the
  verified RIGHT push crosses north onto the x=192-204 column. LEFT+UP
  slides 1px west, y stays 133; RIGHT+UP is tile 179 no-move.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`. Do not leave the post-push pin as leftover.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — 0x0D walk-on south-of-gap / south-strip MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2 (LEFT), `room_all_dead=4`, keys 2, bombs 6,
candle 2, whistle 1, ladder 1, TF 0, food 0, 3 HC. `0x68` `(192,144)`
state 0. 3× bubble `0x2b`. PNG:
`recordings/l7_0d_walkon_20260904_start.png`. `route_eligible=false`.

### RAM claim (written before first live trial)

From this pin, DOWN to y=162 (south-of-gap band between unwalkable
y136-158 and south wall y~166), then RIGHT along y=161-164 to x=192.
Claim: that band is walkable floor to the 0x68 south-face column. Miss if
y never enters 161-164, or RIGHT pins at x<=176 (old gap pin).

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_sog` | play **0x0D** m5 `(64,157)` tile **178** | 48 stuck DOWN | `recordings/20260904_sog_sog_after_down.png` | **MISS** y never entered 161-164 |
| `20260904_ss` | play **0x0D** m5 `(64,157)` tile **178** | 48 stuck DOWN | `recordings/20260904_ss_ss_after_down.png` | **MISS** y never reached 180 (L9 y=189 one-shot) |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports: `recordings/l7_0d_walkon_20260904_sog.json`,
`recordings/l7_0d_walkon_20260904_ss.json`. Probe:
`scratch/probe_l7_room0d_squeeze.py --south-of-gap` / `--south-strip`.

Did not UP-push (south face unreachable). Did not retry west-above-band
y<=99. Did not save a dest fixture. Did not flip spine factories. Did
not fill `MEASURED_POST_L7_EXIT`. `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: y=161-164 (between gap y136-158 and south wall y~166) is reachable
  by DOWN from pin `(63,149)`. Moving Link pins **`(64,157)` tile 178**
  (SW diamond). PNG `20260904_sog_sog_after_down.png`.
- Dead: L9 `room30_stairs_step` y=189 south-strip is reachable from
  `(63,149)` in 0x0D. Same pin; ROM south door is WALL (code 1). One shot,
  not looped.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)`, tile 118, doors=2, `room_all_dead=4`,
Candle **2**, TF **0**, keys 2, bombs 6, whistle 1, food 0, ladder 1,
3 hearts. Block `0x68` `(192,144)` state 0. `route_eligible=false`.
(Trials left Link at `(64,157)`; leftover is the uncleared-push pin.)

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

---

# Residual — rr-n91a L7-C 0x7B B→A dest 0x29 (2026-09-04)

Did not STATUS. Did not poke doors/TF/position. Did not touch `level8/**`,
`AGENTS.md`, harvest, or parent `rr-8t4.3` (still in_progress). Spine
`make_tip_stairs_controller` / `make_aquamentus_heart_controller` /
`make_level7_shard_leave_controller` stay fail-closed.

## Glance (pin start)

`Level7Interior0DNoseCellarReconFixture`: L7 mode **9** cellar **`$EB=0x7B`**
`(192,93)` tile 111, keys 2, bombs 6, candle 2, TF 0. 4× keese 0x1B hp=0.
`route_eligible=false`. Existing disclosed poke is in fixture provenance;
no new position pokes.

## RAM claim (written before first live trial)

From this pin, floor-cross left, first settled play `$EB` is **0x29**
(ROM AttrA). Miss if dest is 0x0D (CheckSubroom AttrB / UP on the source
ladder). Never UP at `x>=$80`.

## Dest 0x29 — live 2/2

Policy: DOWN the east/source column, floor LEFT to `x=48`, UP left ladder.
Never UP on the right/source ladder. OccupancyWalker banned. Inverse of
`cellar_cross_dir` (L6 A→B east then UP).

| tag | dest | pose | controller frames | total frames |
|-----|------|------|-------------------|--------------|
| `20260904_C1` | play **0x29** | `(96,157)` | 396 | 856 |
| `20260904_C2` | play **0x29** | `(96,157)` | 396 | 856 |

Reports: `recordings/l7_7b_cellar_cross_20260904_C1.json`,
`recordings/l7_7b_cellar_cross_20260904_C2.json`.
Pin: `Level7Interior29PreBossReconFixture`. Goriya 0x05/0x06 in 0x29.
`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.

C1 first miss: `pit_tile_250` at west floor `(48,189)` — false L8 y=141
trap. Policy now only fails 250 at the y=141 ledge, not the west ladder.

## Suffix (C1, after dest 0x29)

- **0x29 bomb-E → 0x2A** 1/1. Survival bomb-count top-up 6→8 at the
  verified gate (`apply_owned_inventory`, never `max_bombs`). Used 8→7.
  Dest play **0x2A** `(32,141)` west mouth. Pin
  `Level7Interior2AAquamentusReconFixture`.
- **0x2A Aquamentus** 1/1. Reused `Level1AquamentusController`
  ALIGN/FACE/ATTACK/DODGE/COLLECT_HEART, `tank_hits=True`, room aliased
  from live `$EB=0x2A` (not L1 0x35). Type **0x3D** at entry. Fireballs
  0x55 not on the west-mouth census (spawn hp=0); one damage event in
  0x2A while tanking. Sword-only. HC **3→4**. Leftover `(120,141)`.
- **0x2A E-shutter → 0x2B** 1/1. Play **0x2B** `(16,141)` west mouth.
  Pin `Level7Interior2BTriforceReconFixture`. Diamond floor; DOWN at
  x=16 does not move. South-around `(32,141)→(32,189)→(120,189)→(128,141)`
  collects the shard. Fanfare mode 18, then OW.
- **OW leftover (fixture-lineage, 1/1, not Survival):** play **0x42**
  `(96,93)` mode 5, TF **0x40**, candle 2, whistle 1, keys 2, bombs 7,
  HC 4. Pin started at TF 0 so this is **not** the Survival `0x7F`
  packet. `MEASURED_POST_L7_EXIT.verified` stays False.
  `PostLevel7Handoff.verified` untouched.

## Rooms live / selected min

L7 remaining selected suffix: cellar `0x7B` → play `0x29` → boss `0x2A`
→ tf `0x2B` = **4/4 fixture-live** (dest 0x29 is 2/2; 0x2A/0x2B/OW are
1/1 from C1). Prefix is live through TIP_OF_NOSE `0x0D`. `NOSE_CELLAR.ram_id`
stays None (no 0x0D walk-on). Graph ram_ids: PRE_BOSS `0x29`, AQUAMENTUS
`0x2A`, TRIFORCE `0x2B`, evidence `fixture-live`, `route_eligible=false`.

## Not done / leftover

- **0x0D south-face squeeze / walk-on of cellar 0x7B is still OPEN.**
  0x0D is not dead. Parent `rr-8t4.3` keeps that TAS.
- Spine factories stay fail-closed. Do not wire the poke-spawned cellar
  cross onto `level7-red-candle` / `level7_complete`.
- Survival OW leave packet (TF `0x7F`, 8+1 HC) is unmeasured. Do not
  fill `MEASURED_POST_L7_EXIT` from the TF-0 recon pin.

Next leftover: 0x0D walk-on (parent), or Survival-lineage shard leave
once the poke-fixture TF=0 is no longer the start.

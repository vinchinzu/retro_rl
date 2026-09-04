# L8 Wave A handoff — Lion

Did not STATUS-promote. Did not claim spine-green. Did not poke candle/doors/TF/magic key.

## Sitting — 2026-09-03 (L7→L8 seam + L8 0x4E→0x3E)

Development-only. `natural_entry=false`, `route_eligible=false` throughout.
No STATUS promotion, no handoff `verified=true`, no push.

### Track A — L7-pond → OW 0x6D, 2/2 byte-identical (rr-6o7.4)

The old `L7_POND_TO_LEVEL8_BUSH_HOPS` reverse-waypoint (OccupancyWalker)
approach never crossed 0x52 (flooded occupancy misses). Rebuilt
`Level7PondToLevel8BushController._extra_hop_action` from the **live**
per-screen point lists of `OverworldToLevel7PondController` (the forward
`0x77→…→0x42` pond walk), reversed screen by screen:

```
0x42 (OW_L7Pond south shore, 112,221) --DOWN--> 0x52
  0x52 boulder field: (128,61) top -> DOWN to the y~85 corridor -> LEFT to
    the x~48 sand column -> DOWN to y~189 bottom -> RIGHT east edge
0x52 --RIGHT--> 0x53   (0,189) west -> RIGHT -> x~192 pillar UP to y~141 -> RIGHT
0x53 --RIGHT--> 0x54   (0,141) west -> RIGHT to x~64 -> DOWN
0x54 --DOWN--> 0x64    (64,61) north -> DOWN the x~64 column to y~141 -> RIGHT
0x64 --RIGHT--> 0x65   (0,141) west -> RIGHT along the y~141 river ford to
                        x~112 -> UP  (north gap is x~112, NOT the west bank)
0x65 --UP--> 0x55      (112,221) south -> UP the x~112 sand spit to y~133 -> RIGHT
0x55 --RIGHT--> 0x56   (0,133) west -> RIGHT along y~133 to x~224 -> DOWN to
                        y~157 -> RIGHT
0x56 --RIGHT--> 0x57   straight RIGHT at y~157
0x57 --RIGHT--> 0x58   straight RIGHT at y~157
0x58 --RIGHT--> 0x59   join the already-live LEVEL8_BUSH_HOPS corridor
0x59 → 0x5A → 0x5B → 0x5C (maze) → 0x5D → 0x6D
  0x5B: added a local climb that holds x~16 (the inherited "y>100 -> UP"
        corridor climb pins Link on the x=0 wall at a y~100 snag).
```

Evidence: `probe_l7_exit_to_l8_bush.py --from-state OW_L7Pond`, tags
`A18`/`A19` — **2/2 byte-identical**: `success=True`, `frames=4980`,
identical transition-frame list, settled **OW 0x6D `(48,61)` tile `0x00`**
($EB=0x6D), `deaths=0`, `progression_writes=0`, `capacity_writes=0`.
Bombs `0→4` from natural Octorok drops along the corridor (pickup, not a
poke). `route_eligible=false`, `natural_entry=false`,
`evidence="fixture-live-prefix"`, hop-engine seam unchanged (no new
sibling dispatcher). `overworld.py` now 931 lines.

**Refilled pond handled:** the natural `Level7Entrance` exit refills the
pool and strands Link in the `y~85–100` top strip of `0x42` — probe
`s42` reachability sweep: **nothing south of y~107 is reachable**, Link
cannot walk around the pool. Controller keeps that fail-closed
(`42_refilled_pond_straight_down_dead`, now fires anywhere in the strip).
The 2/2 route starts from the disclosed `OW_L7Pond` south-shore pose
(the route-review-recommended start), where `DOWN` scrolls straight to
`0x52` below the pool.

0x6D census (settled from the route, idle): Link at `(48,61)` (arrival
from 0x5D south @ x≈48; only candle-free exit is back UP@x≈48 → 0x5D).
Objects: 4× `0x04` HP32 (red Octoroks) + 1× `0x5B` HP192 + 1× `0x64`
HP240 — the last two are almost certainly transition residuals (same
pattern as the 0x4E `0x2B` HP240 residuals), not live enemies.

Did **not** attempt the Red-Candle burn / L8 entry (rr-6o7.1, still
fail-closed).

### Track B — L8 0x4E → 0x3E north key door, 2/2 (rr-6o7.2 groundwork)

`probe_l8_4e_north.py` from `Level8InteriorReconFixture`: reused the
confirmed `0x7E → clear 0x6E → bomb-N 0x5E → clear 5 blue Darknuts /
center key (9→10) → shutter-N → settled 0x4E (120,205)` policy
**unchanged**, then ONE attempt at the **north key door** from `0x4E`.
The mixed `0x4E` census was **not** cleared (first-departure guards kept
around the generic combat helpers).

**2/2 byte-identical** (tags `B1`/`B2`, `frames=2918`):

- north door is a **locked key door** — one natural key spent, **keys
  10 → 9**; **bombs 7 → 7** (unchanged, no bombs)
- settled play room **`0x3E`** at `(120,205)`, `$EB=0x3E`, facing DOWN,
  `cur_opened_doors=4` / `open_doorway_mask=4` (south, the door Link
  came through)
- `0x3E` census: **6× type `0x0C` HP 128** (blue Darknuts),
  `room_item_id=0x03`, `room_obj_count=6`, `room_all_dead=0`
- `deaths=0`, `progression_writes=0`, `capacity_writes=0`, no combat or
  further key/bomb use in `0x3E`
- Survival refilled 17 filled units across 12 damage events
  (`0x6e`:6, `0x5e`:8, `0x4e`:3 — all pre-door, from the mixed census
  while aligning to the door). Bomb selection stayed normal pause input.

Resume point: **`Level8Interior3EReconFixture`** (`.state` gitignored per
repo convention; `.provenance.json` committed). Disclosed writes: **NONE**
— the only inventory delta vs `Level8InteriorReconFixture` is the natural
key spend `10→9` at the north door. Loads at L8 `0x3E (120,205)`, keys 9,
bombs 7, TF `0x7F`, Candle 2, 6 blue Darknuts.

Recon wiring: `level8/dungeon.py` `Level8InteriorRoomRecon` +
`LEVEL8_INTERIOR_0X3E_RECON` (evidence `live_recon_fixture`,
`route_eligible=False`). **Not** a `DungeonRoomSpec`, **not** on
`L8_THROUGH`; `LEVEL8_HYPOTHESIS_ROOMS` room_ids untouched
(`hypothesis_room_ids_unobserved()` still true). Live-prefix test:
`test_level8_interior_0x3e_recon_is_fixture_only_not_route_eligible`.

### Dead beliefs burned — 2026-09-03

- **Reverse pond route via OccupancyWalker waypoints** — never crossed
  0x52; the walker's grid modelled every cell blocked (263 misses).
  Replaced with deterministic reversed forward-trace handlers.
- **0x42→0x52 north gap at x≈128** — wedges Link on the boulder wall;
  the gap is **x≈112** and must be x-aligned on the 0x42 south shore
  first.
- **0x52 y≈93 corridor descends anywhere** — it is a horizontal
  dead-end; the only descent is the **x≈48 column** off the **y≈85**
  corridor (not y≈93, not x≈32).
- **0x65 north exit to 0x55 on the west bank (x≈40–64)** — a false
  reading from staging Link on the wrong bank; the live forward trace
  crosses **straight at x≈112** (river ford at y≈136–144, then UP).
- **`POND_42_REFILLED_DEAD_POSE` is a single (112,93) point** — the
  strand is the whole `y≈85–100` strip; Link now lands at `(96,93)`
  from the natural exit and the guard fires strip-wide.
- **0x4E north door is an open shutter** (hypothesis-chain label
  "shutter-N" then "key-N") — it is a **locked key door**; one key is
  spent 0x4E→0x3E.
- **`Level7Entrance` can route around the refilled pool** — reachability
  sweep proves it cannot; that start is fail-closed, not a route.

Still open / not attempted this sitting: 0x6D Red-Candle burn + L8 entry
(rr-6o7.1); 0x3E onward (clear 6 blue Darknuts, bomb-N toward 0x2E);
four-head Gleeok body type; measured post-L7 fanfare leave.

## Fixture-live continuation — 2026-09-03

This section supersedes the older sitting boundary below.  It remains
development-only: every run starts from a disclosed poked fixture and is
`natural_entry=false`, `route_eligible=false`.

### Poked interior start

`Level8InteriorReconFixture` was derived from
`Level8EntranceReconFixture` at settled L8 room `0x7E`, TF `0x7F`.  The
builder is `scratch/build_level8_interior_recon_fixture.py`; its provenance
records every write.  It grants only recon resources: Magical Sword `1→3`,
bombs `0→8` (within the existing capacity of 8), Bow `0→1`, wooden arrows
`0→1`, rupees `0→255`, and keys `0→9`.  It does **not** write Magic Key,
Triforce, room/screen, door flags, health, or heart capacity.

### Live boundary chain

Luna sub-agent probes established this serial chain:

```text
0x7E --UP--> 0x6E Manhandla
  --clear, sword only--> cleared 0x6E
  --one bomb north, 8→7--> 0x5E
  --clear 5 blue Darknuts 0x0C--> cleared 0x5E
  --center key pickup--> keys 9→10
```

- `0x7E→0x6E` settled at `(120,205)`; census observed Manhandla type
  `0x3C`.  `recordings/l8_7e_to_6e_up_trial_20260903_v2.json`.
- The input-only Manhandla clear stayed in `0x6E`; bombs remained 8.
  `recordings/l8_6e_manhandla_clear_20260903.json`.
- Bomb-UP from `(120,105)` spent exactly one bomb and settled `0x5E` at
  `(120,189)`.  `recordings/l8_6e_to_5e_bomb_north_20260903.json`.
- After two settle frames, `0x5E` spawned five blue Darknuts (`0x0C`, HP
  128 each).  The guarded sword-only clear left bombs at 7 and opened the
  north shutter (`open_doorway_mask=0x04`).
  `recordings/l8_5e_spawn_clear_key_20260903_v2.json`.
- The natural center item `0x19` raised keys `9→10` at `(120,133)`.
  `recordings/l8_5e_key_pickup_center_20260903.json`.

All completed runs reported deaths 0, `progression_writes=0`, and
`capacity_writes=0`.  Survival refill was the only runtime assist.  Normal
pause input changed the selected B item from Candle to bombs; there was no
RAM selection write.

### North-shutter result: predicted `0x4E` confirmed

One guarded replay from `Level8InteriorReconFixture` confirmed the next hop:

```text
0x7E -> clear 0x6E -> bomb-N 0x5E -> clear/pick up key
  -> open shutter N -> settled 0x4E (120,205)
```

The prediction was recorded before the final act as play `0x5E`,
`x=120±4`, open north shutter, then UP to expected play `0x4E` with keys and
bombs unchanged.  It passed at frame 2,322; a 120-frame input-free census
ended at frame 2,442.  Final inventory was keys `10`, bombs `7` / capacity
`8`, Magical Sword `3`, Bow `1`, wooden arrows `1`, Red Candle `2`, Magic
Key `0`, TF `0x7F`, and three hearts full (`health=0x22`).  The final room
had item id `0x0F`, door bytes `cur_opened_doors=0`,
`open_doorway_mask=0`, and eight live occupants:

- 2× `0x2B`, HP 240 (invulnerable mover residuals)
- 1× `0x0C`, HP 128
- 2× `0x0B`, HP 64 (Darknuts)
- 3× `0x30`, HP 112 (Gibdos)

No combat or key/bomb use occurred in `0x4E`.  Runtime integrity was deaths
`0`, direct RAM/controller writes `0`, state loads after start `0`,
`progression_writes=0`, and `capacity_writes=0`.  Survival restored 14 filled
heart units across 10 damage events (13 refill writes; `0x6E`: 6 units,
`0x5E`: 8 units).  Bomb selection remained normal pause input (`4→1`), not
a selection write.  This is still `natural_entry=false` and
`route_eligible=false`.

Evidence:

- `scratch/probe_l8_5e_north.py`
- `recordings/l8_5e_north_fixture_20260903_v1.json`
- `recordings/l8_5e_north_20260903_v1_first_settled_destination_f2322_L8_s4e_m5.png`
- `recordings/l8_5e_north_20260903_v1_final_census_f2442_L8_s4e_m5.png`

Next boundary: from this replayed `0x4E` arrival, test only the north key door
to predicted `0x3E`, stopping at its first settled destination.  Do not clear
the mixed `0x4E` census unless live input proves the key door is unavailable;
on that miss, stop and replan.  Do not rerun the now-confirmed `0x5E→0x4E`
policy unchanged.

The early HP-zero `0x0C` census was pre-activation state, not an empty room;
idle settling produced five HP-128 blue Darknuts.  Keep the first-departure
guard around generic combat helpers: an unguarded attempt walked back
through the open south door.

## Fixture-live sitting — 2026-09-03

The newly available `Level8BushWithCandleFixture` is a disclosed poked start,
not a predecessor checkpoint (`natural_entry=false`, `route_eligible=false`).
Three screenshot-first trials observed Red-Candle use but no mode-16 mouth and
therefore hit the sitting halt:

1. `(144,93)`, face RIGHT, push UP: no mouth after the 800-frame burn budget.
2. `(144,93)`, face RIGHT, push RIGHT: no mouth after one resolved fire/push
   window.
3. target `(128,93)`, face RIGHT, push RIGHT: no mouth; controller finished at
   `(132,93)` while realigning, so this is a destination diagnostic for the
   x≈128 region rather than proof of an exact x=128 fire pose.

Evidence: `recordings/l8_entry_luna_attempt{1,2,3}.json` plus their initial,
sample, and final PNGs. Attempt 3 used Survival refill and reported deaths 0,
`progression_writes=0`, and `capacity_writes=0`. Do not repeat these policies
unchanged. The broad position-poke sweep in scratch is not route evidence and
must not replace the predict-one/act-one loop.

Offline ROM decode corrects the old walkthrough-grid placeholders, but remains
static hypothesis until a live entry observes RAM:

```text
entry 0x7E -> N 0x6E -> bomb-N 0x5E -> shutter-N 0x4E
  -> key-N 0x3E -> bomb-N 0x2E -> key-N blue Gohma 0x1E
  -> east 0x1F (Magical Key branch)

return 0x3E -> east 0x3F -> cellar 0x2F left -> 0x4C
  -> bomb-N four-head Gleeok 0x3C -> shutter-N shard 0x2C
```

Static LevelInfo anchors: entry `0x7E`, Triforce room `0x2C`, prior TF mask
`0x7F`, cellars `0x2F/0x0F/0x6F`, boss `0x3C`. Static room data says boss
item `0x1A` and shard item `0x1B`; the four-head body type is still not live.
Do not register `DungeonRoomSpec` rows or make the canonical controllers move
from these values alone.

## chapter id and evidence label

- `rr-6o7.1` L8-A Red-Candle bush entry + topology: **hypothesis** (canonical seam fail-closed). Isolated 0x6D burn is **fixture-live**, `route_eligible=false`. Live trial **blocked** (no ROM, no `Level8BushOW`/`OW_6D`, no `l8_*.png`/`l8_bush_recon.json` in this worktree).
- `rr-6o7.2` L8-B Magical Key + L8-C four-head Gleeok factories: **hypothesis**. Room IDs unobserved. Controllers idle-fail.

## predecessor

Measured post-L7 leave is **UNMEASURED**. `PostLevel7Handoff.verified=false`. Controllers must not move without `handoff.complete()` except isolated bush recon on 0x6D.

Expected predecessor (L7 leave, not yet supplied): overworld play, TF `0x7F`, Candle 2, full hearts, deaths 0. Screen/x/y/keys/bombs/rupees/B-slot/Whistle/Food/Rod/Bow/arrows unknown.

Bush screen facts (prior assisted recon, not this sitting): enter 0x6D only from 0x5D south @ x≈48; walkable left corridor x≈32–56 + mid sand y≈88–96 east to x≈144; only open exit without candle is UP @ x≈48 → 0x5D.

## required inventory (Candle 2, TF 0x7F, Bow)

- Candle 2 (natural Red Candle from L7). Do not put the 60R Blue Candle shop on the mainline.
- TF exactly `0x7F`.
- Bow + wooden arrows from the cumulative route (L8 blue Gohma needs 3 shots). No L6 wooden-arrow poke. No L6 room `0x1C` check.
- Magical Key is the selected L8 investment (L9 key bottleneck). Book/Map/Compass omitted.

## internal stages

Public through (unchanged): `level8-entry` → `level8-magic-key` → `level8`.

- `level8-entry`: `level8_post_l7_to_bush`, `level8_select_red_candle` (pause/input only), `level8_burn_bush_enter`.
- `level8-magic-key`: `level8_north_manhandla_bomb`, `level8_darknut_key_up`, `level8_blue_gohma`, `level8_magic_key_stairs`.
- `level8`: `level8_return_passage`, `level8_four_head_gleeok`, `level8_heart_shard_leave`.

Hypothesis Magic-Key walk (grid labels only; RAM ids `None`):

```
entry --UP--> north_manhandla --BOMB UP--> darknut_key
  --KEY UP--> shutter_darknuts --KEY UP--> blue_darknuts
  --BOMB UP--> map_manhandla --KEY UP--> blue_gohma
  --RIGHT--> magic_key_stairs
```

Hypothesis Gleeok suffix:

```
blue_gohma --DOWN×2--> blue_darknuts --KILL RIGHT--> passage_east
  --STAIRS--> pols_west --BOMB UP--> gleeok --UP--> triforce
```

## endpoint predicates

- `level8-entry` / `level8_entry_live`: observed L8 entry room, play mode, TF `0x7F`, Candle 2, natural burn/transition (`ADDR_CANDLE_USED` then mode-16 mouth). Fails while `topology.entry_room` is None.
- `level8-magic-key` / `level8_magic_key_natural`: `ADDR_MAGIC_KEY` increases naturally, TF still `0x7F`, at RAM-observed magic-key room. Ledger records incoming/outgoing keys and bombs.
- `level8` / `level8_triforce_0x80`: TF `0x7F→0xFF` (bit `0x80`), Magic Key owned, one natural heart-container increase, full hearts. Settled post-fanfare leave is `UNOBSERVED_LEVEL8_CLEAR`. Deaths 0 / zero state loads are runner contracts, not yet measurable.

Burn-budget exhaust while still on 0x6D is **failure**, never success. Canonical `BurnLevel8BushController` also refuses an unverified aim.

## inventory deltas

| Gate | TF | Candle | Magic Key | Hearts | Keys/bombs |
|------|----|--------|-----------|--------|------------|
| entry | stay `0x7F` | stay 2 | 0 | unchanged | recorded, not invented |
| magic-key | stay `0x7F` | 2 | 0→1 natural | unchanged | exact before/after recorded |
| level8 | `0x7F→0xFF` | 2 | owned | containers +1, full | recorded; Magical Key removes later key spend |

Blue Candle shop remains `BLUE_CANDLE_FALLBACK_ENABLED=False`. `ADDR_SELECTED_ITEM` is never written in the L8 lane modules (pause cycle only).

## dead beliefs (especially burn tile)

- **(136, 93) face/push RIGHT** never opened a mouth (historical dense walkable burns). Not an executable target.
- Exhausting the burn budget on 0x6D is not entry success (frozen `overworld.py` still has that false positive; canonical path does not use it).
- Four-head Gleeok body type is **not** assumed `0x45` (L4=`0x43`, L6=`0x44`, L8=`None` until live).
- Walkthrough grid hex is not a dungeon room ID.
- Book of Magic is not on the minimum full-clear route.

One new hypothesis (fixture-live, `verified=false`, `route_eligible=false`): stand **(144, 93)** at the sampled east walkable limit, face **RIGHT**, push **UP** (mouths are mode-16 UP). Formed from documented walkable raster, not from PNGs (absent here). No live trial this sitting.

## fixture provenance route_eligible=false

`Level8BushOW` / `OW_6D` / `l8_walkable.png` / `l8_bush_6d.png` / `l8_bush_recon.json` are cited prior evidence, **not present** in this worktree. Isolated `IsolatedBushReconController` may step on 0x6D without a complete L7 handoff; it still fails closed without candle, without B-slot candle, or when the burn budget expires on 0x6D. Nothing here is route-eligible or spine-green.

## files changed

- `nes/zelda_i/level8/dungeon.py` — hypothesis door graph, ledger, fail-closed stops
- `nes/zelda_i/level8/path.py` — Magic-Key / blue-Gohma / four-head Gleeok blockers
- `nes/zelda_i/level8/hops.py` — named chapter factories; public through names unchanged
- `nes/zelda_i/level8/bush.py` — isolated 0x6D recon (not wired into `L8_THROUGH`)
- `nes/zelda_i/docs/LEVEL8_ROUTE.md`
- `nes/zelda_i/tests/test_level8_entry.py`
- `nes/zelda_i/docs/tasks/l8-handoff.md` (this file)

Not written: `level8/overworld.py` (frozen), `level8/entry.py` (499, no new knob), `spine/survival.py`, RAM/assist, L6/L7/L9, `STATUS.md`, `.beads`. `QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests/test_level8*.py nes/zelda_i/tests/test_hygiene_architecture.py -q` → 28 passed.

## stitch notes for L9: TF 0xFF, Magic Key, heart +1, post-fanfare OW leftover UNMEASURED

When L8 later clears, L9 should inherit: TF exactly `0xFF`, Magical Key owned, Bow + arrows, Candle 2, one extra heart container vs L7 leave, full hearts, deaths 0, **post-fanfare overworld leftover UNMEASURED** (do not invent screen/x/y). L9 must not assume L8 leftover `0x6D`. Do not grant Magic Key. Do not skip L8 Magical Key on the cumulative path: it is the L9 key-bottleneck investment.

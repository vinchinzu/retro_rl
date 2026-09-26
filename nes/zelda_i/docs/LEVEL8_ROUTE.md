# Level 8 — The Lion (route notes)

Route notes for this dungeon. The live sitting is the gathering prefix in [PRE_L1.md](PRE_L1.md). Do not treat this file as the current plan, and do not write STATUS from it.

> **Probe names below are historical provenance labels, not paths.** The
> one-shot probe CLIs under `scratch/` were deleted 2026-09-07 once their
> beads closed; the measurements they produced live on in the constants and
> route tables this document describes. Git history is the restore path.



Status: **Survival history only (2026-09-05).** Clean STATUS stops at
Level 6. The next Clean Level 8 check is `rr-npv.4`, blocked on Level 7
(`rr-rgum`) and on the 0x3E/0x4C bomb patch (`rr-awh6`). The 2026-09-05
`--through level8` 1/1 below was Survival:
`set_state=0`, first quest: entry (`rr-6o7.1`), Magical Key (`rr-6o7.2`,
power-on 2/2) and the four-head Gleeok suffix (`rr-6o7.3`) all pass.  Link
settles OW `0x6D` `(96,93)` mode 5, TF `0xFF`, Magical Key 1, heart
containers 10, deaths 0, progression/capacity writes 0.  No inventory
assist since 2026-09-24 (rr-doua): the post-L7 walk splits at 0x54 to buy a
20R bomb pack at 0x44 (L7 leaves 4, L8 bombs 0x6E, 0x3E and 0x4C and throws
at 0x3E's Darknuts), and the carried key plus 0x5E's open the two key
doors. Interior
`RoomHopSpec` cardinals and the 0x3C heart walk are leftover-relative
(`door_band_goal`; heart dest is RAM slot 19 + hc bit).  See
[`docs/tasks/rr-npv.4-residual.md`](tasks/rr-npv.4-residual.md).  The measured
leave is `level9.dungeon.MEASURED_POST_L8_HANDOFF`.

The sections below are the recon history that got here; most predate the
power-on greens and keep their `fixture-live` / `route_eligible=false`
labels.

Wave A has a fail-closed cumulative seam in `level8/{entry,dungeon,hops,spine}.py`.
This is implementation structure, not route evidence. The public chapter
targets are `level8-entry`, `level8-magic-key`, and `level8`; all three remain
red until their natural predecessor and exact live endpoints are supplied.

**Provenance vocabulary used below.** *fixture-live* = observed in RAM/PNG on
a real emulator run that started from a **disclosed development checkpoint**
(poked inventory / poked stand), never from a natural walk; every such claim
carries `natural_entry=false` and `route_eligible=false`. *source* =
walkthrough only. *assisted* = live but under the Survival assist contract
from the power-on start screen. Nothing on this page is a Clean claim and
nothing here promotes `STATUS.md`.

Planning sources:

- [Zelda Dungeon — Level 8: The Lion](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-8-the-lion/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- Assist: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) (Survival infinite-life only)

Walkthrough claims that are emulator-verified are marked; source-only claims
stay labeled. **No Clean STATUS claims.**

## Overworld

### Door / bush screen (live)

| Claim | Source | Live |
|-------|--------|------|
| Bush pocket screen | walkthrough path decode | **`0x6D`** (assisted) |
| Mouth under lone bush | walkthrough | **opened** — mode 5→16 (fixture-live, rr-6o7.1) |
| Entry room id | — | **`0x7E`**, Link `(120,205)` facing UP (fixture-live) |
| Triforce bit | walkthrough | `0x80` (source) |

The entry-room row is `LIVE_RECON_LEVEL8_TOPOLOGY` in `level8/dungeon.py`
(`evidence="live_recon_fixture"`, `route_eligible=False`) and
`Level8EntranceReconFixture.provenance.json`
(`natural_entry=false`, `development_only=true`). It is **not** a route claim:
the burn was fired from a poked stand on a poked-candle fixture, not from the
still-unmeasured natural post-L7 leave.

Dead-end geometry (live, `OW_6D` / `Level8BushOW`):

- Enter **0x6D** only from **0x5D south @ x≈48**.
- **Walked** corridor (assisted recon): left column **x≈32–56** + mid sand
  channel **y≈88–96** east to **x≈144** (see `recordings/l8_walkable.png`).
- **Standable** area is much larger: the 2026-09-03 teleport-and-settle probe
  kept **732** tiles (729 distinct coordinates due to 3 duplicate stand samples)
  out of a 32×23 sampled grid
  (`logs/level8_6d_walkable_positions.json`). Standable is **not** reachable —
  it only says Link stops drifting on that tile, not that he can walk there.
  The `(136,93)` burn aim sits inside the *walked* channel; the `x≥184` mouth
  stands do not, and have never been walked to.
- Only open screen exit found without candle: **UP @ x≈48 → 0x5D**.
- Raster + UP pushes without candle: **no** mode-16 mouth (expected).

Evidence: `recordings/l8_bush_recon.json`, `l8_bush_6d.png`,
`custom_integrations/.../Level8BushOW.state`, `OW_6D.state`.

### Bush burn recipe (fixture-live, rr-6o7.1 / rr-u9js)

**Corrected 2026-09-04. The old "(144,93) face RIGHT push UP" hypothesis and
the old "(136,93) RIGHT is a dead aim" claim are both REFUTED — do not revive
either.** The refutation is a 5856-trial live sweep,
`logs/level8_bush_burn_sweep.json` (732 standable OW-`0x6D` tiles × 8
facing/push pairs, run from `Level8BushWithCandleFixture`: Candle 2 + B-slot 4
+ TF `0x7F` poked, Link teleported to each tile).

Outcomes: 5846 `no_effect`, 3 `link_death`, **7 `mouth_mode16`**. Every mouth
stand fires *and pushes in the same direction*:

| Stand `(x,y)` on `0x6D` | Facing | Push | Sweep outcome |
|--------------------------|--------|------|---------------|
| `(120, 93)` | RIGHT | RIGHT | mode-16 mouth |
| `(128, 93)` | RIGHT | RIGHT | mode-16 mouth |
| **`(136, 93)`** | **RIGHT** | **RIGHT** | mode-16 mouth (the recorded aim) |
| `(160, 77)` | DOWN | DOWN | mode-16 mouth |
| `(184, 93)` | LEFT | LEFT | mode-16 mouth |
| `(192, 93)` | LEFT | LEFT | mode-16 mouth |
| `(200, 93)` | LEFT | LEFT | mode-16 mouth |

`(144, 93)` **is** in the swept tile set and was tried all 8 ways with the
candle actually used — all eight are `no_effect`. One secret tile, several
approach angles; `(144,93)` is not one of them.

Caveat: the swept stands are teleport-and-settle *standable*, not walked. Only
`(120/128/136, 93)` lie inside the walked sand channel, so **`(136,93)` is the
only aim a natural approach can currently use**; the LEFT and DOWN stands are
recon curiosities until someone walks there.

Two separate facts, kept apart on purpose:

- **Mouth open (mode 16)** — established by the sweep, 7 stands above.
  `entry_room` is `null` on all seven sweep rows: the sweep itself never
  carried Link into level 8.
- **Entry room `0x7E`** — established by `capture_level8_entrance_fixture.py`
  replaying **`(136,93)` facing RIGHT, push RIGHT** only, reaching live
  `level==8`, `mode==5`, screen `0x7E` at `(120,205)` 111 frames after the
  push. The other six stands have **not** been carried through to a room id.

**Entry does not complete on UP.** After the mouth appears, keep sending the
same `push_direction` you fired with; pushing UP leaves Link on `0x6D`
(rr-i6hq). `level8/entry.py` `BurnLevel8BushController` now sends
`target.push_direction` in its ENTER phase (source-only reading of the current
file, not re-run live here).

Recorded as `level8.entry.LIVE_RECON_BUSH_BURN_TARGET`
(`link_x=136, link_y=93, facing="RIGHT", push_direction="RIGHT", tolerance=4,
evidence="live_recon_fixture", verified=True, route_eligible=False`). It is
**not** a spine default: `continue_level8_spine` still defaults to
`UNVERIFIED_BUSH_BURN_TARGET`; the recon aim only arrives through the explicit
opt-in bundle `level8.spine.LIVE_RECON_L8_OVERRIDES`.

### Verified walk (assisted) — start → bush 0x6D

Source “right×4 …” collides with **0x79 rocky pocket**. Live detour reuses the
L1 north lane and L2 door corridor + **0x5C maze**:

```text
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
  E@y≈155 → 0x59 E → 0x5A E → 0x5B
  (climb to y≈88) E@y80–95 → 0x5C
  [maze: east @y≈88 → channel → east @y≈128] → 0x5D
  S@x≈48 → 0x6D  (Lion bush pocket)
```

Hop table + controller: `level8.overworld.LEVEL8_BUSH_HOPS`,
`OverworldToLevel8Controller` (maze waypoints =
`overworld.graph.LEVEL2_5C_MAZE_WAYPOINTS`).

### Fixture-live L7-pond → bush 0x6D (rr-6o7.4, `route_eligible=false`)

Not the natural post-L7 leave (still unmeasured). From the disclosed
`OW_L7Pond` south-shore pose, the reverse of the live forward pond walk:

```text
0x42 S→ 0x52 (boulder field: y~85 corridor, x~48 column, y~189 bottom)
  E→ 0x53 (x~192 pillar to y~141) E→ 0x54 (x~64) S→ 0x64 (x~64 column to y~141)
  E→ 0x65 (y~141 river ford to x~112) U→ 0x55 (x~112 spit to y~133)
  E→ 0x56 (y~133 row, step to y~157 at x~224) E→ 0x57 E→ 0x58
  E→ 0x59 → 0x5A → 0x5B → 0x5C [maze] → 0x5D S→ 0x6D
```

Hop table + controller: `level8.overworld.L7_POND_TO_LEVEL8_BUSH_HOPS`
(9 reversed pond hops, then `LEVEL8_BUSH_HOPS[3:]` from `0x59`) driven by
`Level7PondToLevel8BushController` — a subclass of `OverworldToLevel8Controller`
with `burn_bush=False`, `enter_dungeon=False`, `route_eligible=False`,
`evidence="fixture-live-prefix"`. 2/2 byte-identical
(`probe_l7_exit_to_l8_bush.py --from-state OW_L7Pond`, frames 4980,
settled 0x6D `(48,61)`). Natural `Level7Entrance` exit refills the pool
and fails closed (Link cannot walk around it — `probe s42` reachability);
that dead pose is pinned as `POND_42_REFILLED_DEAD_POSE = (112, 93)`.

Isolated `probe_level8_entry.py` pruned. The L8 seam **is** attached to the
shared continuous spine now (source-only reading of `spine/survival.py`, not
re-run live here): `SPINE_THROUGH` includes `L8_THROUGH`, and after a
successful `level7` suffix `run_survival_spine` calls `continue_level8_spine`.
Reaching the seam is not greening it — the default `handoff` is
`UNMEASURED_POST_L7_HANDOFF`, so `level8-entry` fails on
`post_l7_handoff_unmeasured` until rr-8t4.3 measures the real L7 leave:

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
```

Mid-path fixtures used during recon: `OW_5B`, `OW_5C`, `OW_5D`, `OW_6A`,
`OW_6B`, `OW_6C` (0x6C is a **side pocket** UP-only to 0x5C — **not** on the
bush route).

### Fallback item — Blue Candle

The cumulative mainline does **not** require this shop. It inherits the
naturally earned Red Candle (`ADDR_CANDLE == 2`) from Level 7. The Blue Candle
shop/farm is retained only as a disabled, route-ineligible fallback for an
unexpected Candle-0 handoff; normally that mismatch fails the L7 contract.

| Field | Value |
|-------|--------|
| RAM | `ADDR_CANDLE = 0x065B` (1=blue, 2=red) |
| B-item cursor | `ADDR_SELECTED_ITEM = 0x0656` — live candle pos **`4`** |
| Once-per-screen | `ADDR_CANDLE_USED = 0x0513` (0 ready / 1 used; leave screen to reset blue) |
| Source price | **60 rupees** (Blue Candle, merchant caves) |
| Also works | Red Candle from L7 (source; multi-use per screen) |
| Assist | **inventory poke forbidden** for Clean / published assisted STATUS |

#### Shop (live, rr-ccx) — first-quest **O-6** / screen **`0x5E`**

| Field | Live |
|-------|------|
| Map id | GameFAQs **O-6** regular shop (Shield 160 / Key 100 / Candle 60) |
| OW path | **verified assisted** `CANDLE_SHOP_HOPS` (see below) |
| Cave mouth | **UP @ x≈112** on 0x5E (mode 16→11); approach y≈77 |
| Cave fixture | **`CandleShop5E`** — mode **11**, screen **`0x5E`**, spawn xy≈(112,213) |
| Merchant | type **`0x78`** @ (120, 128) |
| Left item | type `0x40` @ x≈**72** — Magical Shield **160R** |
| Mid item | type `0x40` zone x≈**120** — Key **100R** (touch y≈149 drains R) |
| Right / candle | touch ≈(**152, 149**) — Blue Candle **60R** → `ADDR_CANDLE=1`, R−60 |
| Rupees in fixture | **0** (buy needs farm; poke-R recon only) |
| False lead | IGN “N of start then W” → live **0x67** no west corridor |

##### Verified walk (assisted) — start → shop cave 0x5E

Reuses L8 bush corridor through **0x5C maze** + **0x5D**, then **east**
(not south into bush):

```text
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
  E@y≈155 → 0x59 E → 0x5A E → 0x5B
  (climb y≈88) E@y80–95 → 0x5C [maze] → 0x5D
  E@y≈130–150 → **0x5E**  (enter y≈141)
  cave: UP @ x≈112 → mode 11
```

Hop table + controller: `level8.overworld.CANDLE_SHOP_HOPS`,
`OverworldToCandleShopController` (door_x=`CANDLE_SHOP_CAVE_X`).

Isolated `probe_level8_entry.py` pruned. Shop hops live on
`OverworldToCandleShopController`. The shop lane is deliberately **outside**
`L8_THROUGH` — it is not attached to the continuous spine:

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
```

##### Buy interaction (live geometry; needs ≥60R)

1. Settle cave bottom (dialog timer →0; ~120f idle OK).
2. **UP** the stairs until `link_y ≤ 150`.
3. **RIGHT** along y≈149 until `link_x ≥ 152` (do **not** linger on mid
   x≈120 if you only have 60–99R — Key costs 100 and will drain all R).
4. Touch right zone → `ADDR_CANDLE=1`; rupees drain async (−60).
5. Exit cave **DOWN** to OW 0x5E (post-buy residual for runner).

Constants: `CANDLE_BUY_X/Y`, `CANDLE_SHOP_PRICE`, pedestals
`CANDLE_SHOP_ITEM_*_X`.

##### Rupee farm sketch (residual — not automated)

| Idea | Notes |
|------|--------|
| Path Octoroks | Screens **0x59–0x5E** (type `0x03`); `RUPEE_FARM_SCREENS_SKETCH` |
| Pre-shop farm | Clear 0x5A/0x5B/0x5E before cave enter until `rupees ≥ 60` |
| Fixture | `CandleShop5E` has **0R** — farm **before** or reload after farm state |
| Clean rule | No RAM poke for rupees/candle on published tracks |

#### Candle recon (`rr-q8a` + shop path `rr-ccx`)

- Inventory poke (`poke_candle_for_recon`: candle + selected=4 + clear used)
  shows candle on HUD B-slot (`recordings/l8_pos4.png`).
- Pressing B with candle selected sets **`0x0513=1`** (engine accepts use).
- **Superseded 2026-09-04 (rr-u9js).** The old bullets here said the mouth was
  never opened, that **(136,93)** face/push RIGHT was a *dead aim*, and that
  the live hypothesis was **(144,93)** face RIGHT push **UP**. All three are
  wrong. `logs/level8_bush_burn_sweep.json` opens the mode-16 mouth at
  `(120/128/136,93)` facing+push RIGHT, `(184/192/200,93)` LEFT and
  `(160,77)` DOWN, and shows `(144,93)` `no_effect` on all 8 facing/push
  combinations with the candle used. See **Bush burn recipe (fixture-live)**
  above for the corrected table.
- The old failure was a *targeting* failure, not a candle failure: the dense
  walkable burns swept tiles that do not front the secret mouth.
- `Level8BushOW.state` / `OW_6D.state` are present in this worktree, and the
  candle-ready stage state `Level8BushWithCandleFixture.state` was built on top
  of them (poked Candle 2 + B-slot 4 + TF `0x7F` + stand `(48,90)`).
- Shop OW + cave path **assisted green** (`recordings/l8_shop_path.json`);
  natural 60R + buy still residual.

**Remaining recon blocker:** the natural Red-Candle handoff. The bush burn no
longer blocks anything — its aim, mouth and entry room are fixture-live. What
is still missing is the **measured post-L7 leave** (rr-8t4.3): the settled OW
screen/x/y and exact inventory Link actually has when the L7 fanfare returns
him to the overworld. Until that lands, every L8 claim on this page stays
`natural_entry=false` / `route_eligible=false`.

The old start-based controller remains recon-only. A burn-budget expiry on
`0x6D` is not entry success; the canonical `BurnLevel8BushController` fails
whenever its budget expires without live L8 play, even if Link is still
controllable on `0x6D`.

## Interior (source → live)

Walkthrough grid labels live in `level8.dungeon.LEVEL8_HYPOTHESIS_ROOMS`.
**Every `room_id` is `None` until RAM observes it** — `entry` is now the one
exception (`0x7E`, `evidence="live_recon_fixture"`). Do not promote any of
these labels into `DungeonRoomSpec` rows; `LEVEL8_ROOM_SPECS` is still empty.

### Fixture-live interior chain 0x7E → 0x3E (rr-6o7.1 / rr-6o7.2)

`natural_entry=false`, `route_eligible=false`, **not** on `L8_THROUGH`. The
run starts from `Level8InteriorReconFixture`, which is
`Level8EntranceReconFixture` plus *disclosed* recon resources poked in
(`sword→3`, `bombs→8`, `bow→1`, `arrows→1`, `rupees→255`, `keys→9`; Magic Key,
Triforce, room and door flags untouched). So the room ids, gates and censuses
below are live RAM, but the *resources* used to reach them are not earned.

```text
0x7E --free UP--> 0x6E --clear, BOMB N--> 0x5E --clear, key, shutter UP--> 0x4E
  --north KEY door UP--> 0x3E   (stop; 2/2 byte-identical, 2918 frames)
```

Name collision warning: dungeon room `0x5E` below is **not** the Blue Candle
shop overworld screen `0x5E` above. Dungeon rooms here are `level==8`.

| Room | Gate in | Live census on arrival | `room_item_id` | Cost |
|------|---------|------------------------|----------------|------|
| `0x7E` entry | candle mouth (OW `0x6D`) | none live; `room_all_dead` set | `0x03` | — |
| `0x6E` | open north doorway, no key | 5× `0x3C` manhandla HP64 (+ `0x56` projectiles) | `0x0F` | sword-only clear |
| `0x5E` | **bomb** the `0x6E` north wall from stand `(120,105)` | 5× `0x0C` HP128 (+ `0x55` statue fireballs) | `0x19` small_key | bombs 8→7; clear, then natural key pickup at `(120,141)` keys 9→10 |
| `0x4E` | open shutter UP from `0x5E` at `x=120±4` | 2× `0x2B` HP240, 1× `0x0C` HP128, 2× `0x0B` darknut HP64, 3× `0x30` gibdo HP112 — **not cleared** | `0x0F` | — |
| `0x3E` | **north KEY door** UP from `0x4E` | 6× `0x0C` HP128, `room_all_dead=0`, `open_doorway_mask=4` | `0x03` | keys 10→9 |

Every settle above is `mode 5` with Link at `(120,205)` facing UP.
Integrity on the recorded pair: deaths 0, `progression_writes` 0,
`capacity_writes` 0, `direct_ram_writes` 0, `state_loads_after_start` 0; the
B-slot change before the bomb was a **natural pause-menu selection**
(`ram_selection_write: false`, cursor seen `4,4,4,4,4,4,1`).

Recorded as `level8.dungeon.LEVEL8_INTERIOR_0X3E_RECON`
(`Level8InteriorRoomRecon`, `evidence="live_recon_fixture"`,
`route_eligible=False`) and `LEVEL8_INTERIOR_ROOM_RECON`. Evidence:
`recordings/l8_5e_north_fixture_20260903_v1.json`,
`recordings/l8_4e_north_fixture_B1.json` / `_B2.json`, the
`l8_5e_north_B*_*.png` frame captures, and the fixtures
`Level8InteriorReconFixture` / `Level8Interior3EReconFixture` with their
`.provenance.json`.

A `0x3E →` north continuation is recorded in `LEVEL8_INTERIOR_ROOM_RECON`
(`0x3E` bomb-north → `0x2E` → north key door → `0x1E`). This page's table
above still stops at `0x3E`; the next live-recon paragraph is the 0x1E kill.

### Fixture-live 0x1E kill → 0x1F (rr-gw0x)

From `Level8Interior1EReconFixture` (fixture-only, `natural_entry=false`,
`route_eligible=false`). Kill the one 0x1E body — RAM type `0x33` HP96,
ids.py `gohma_red`; **no colour asserted**, never `0x34` — on the L6
eye-open rising edge (`RAM 0x03C7` leaving `0xC0`). 2/2 byte-identical
(probe `l8_1e_gohma` D1b/D2, 1731 frames): three connecting wooden arrows
(HP 96→64→32→0), nine shots loosed (rupees 255→246), then ONE east
kill-clear shutter → first settled L8 play `0x1F` `(16,141)`, keys 8→8,
bombs 6→6. `cur_opened_doors` rose `0x04→0x0D` (RIGHT bit); PNG east went
black; `open_doorway_mask` stayed `0x04` and is not the shutter stop.
`BLUE_GOHMA_ARROWS_REQUIRED = 3` is now the live connect count. 0x1F
arrival census: 2× `0x16` pols_voice HP160, 2× `0x0C` HP128, 2× `0x0B`
darknut HP64, centre stairs sprite `0x68` at ~(96,144) (not population),
`room_item_id=0x03`, Magic Key still 0. Not on `L8_THROUGH`.

### Spine-green 0x1F stairs → Magical Key cellar 0x0F (rr-6o7.2, DONE)

**2026-09-05: `level8_magic_key_stairs` is live and wired; `--through
level8-magic-key` power-on 1/1** (`l8mk_poweron_v3`, `set_state=0`).  All 4
magic-key stages spine-green.  `Level8MagicKeyStairsController` (clear the
0x1F diamond census → route out via the open x=192 lane to the y=93 band →
hold DOWN to slide the centre `0x68` south → centre stairs → cellar 0x0F
DOWN→RIGHT→UP→LEFT pickup loop → two-ladder return) settles play `0x1F`
`(96,157)` carrying `ADDR_MAGIC_KEY` 0→1, TF `0x7F`, 0 writes, 0 deaths.
`MEASURED_LEVEL8_ENTRY_TOPOLOGY.magic_key_room = 0x1F`.  Iteration harness:
`scripts/magic_key_lab.py`.

Fixture-recon history (from `Level8Interior1FReconFixture`, `natural_entry=
false`, `route_eligible=false`): E1 no-clear south-face UP on the west `0x68` was
hitstun-blocked (same as L7 0x1A). Sword-clear of the mixed census, then
the west `0x68` `(96,144)` slides DOWN to `(96,160)` and the vacated gap
walks onto the centre stairs `(128,141)`. 2/2 byte-identical (probe
`l8_1f_magic_key` E2/E3, 9074 frames): natural `ADDR_MAGIC_KEY` 0→1 in
mode-9 cellar `$EB=0x0F` leftover `(136,141)`, keys 8→8, bombs 6→6, TF
`0x7F`. Two-ladder cellar (west/east ladders, pit). F1 cardinal DOWN at
the pad did not move (south brick). F2 LEFT+DOWN at `(160,141)` is pit
tile 250. Return 2/2 (probe `l8_0f_cellar_return` F3/F4, 588 frames):
RIGHT to east `x=176`, LEFT+DOWN, floor LEFT, UP `(48,93)` -> first
settled play **`0x1F` `(96,157)`**, keys 8→8, bombs 6→6, MK 1, TF `0x7F`.
Not Gleeok `0x3C`. `topology.magic_key_room` stays unset. Not on
`L8_THROUGH`. Pin `Level8Interior1FReturnedReconFixture`.

### Fixture-live 0x1F west door → cleared 0x1E (rr-6o7.2)

From `Level8Interior1FReturnedReconFixture`. G1 OccupancyWalker 1px LEFT
grade false-missed 2px dungeon steps and boxed at `(88,157)` tile 118
(walkable floor); west door still open. Cardinal LEFT past `0x68` (`x<=80`),
y-align, LEFT push. G2/G3 2/2 (probe `l8_1f_west`, 275 controller frames /
335 with census): first settled play **`0x1E` `(208,141)`** east mouth,
keys 8→8, bombs 6→6, MK 1, TF `0x7F`. Arrival census empty (Gohma already
dead). `0x55` statue fireballs spawn after idle — pin at arrival, not after
knockback. Not Gleeok `0x3C`. `make_gleeok_passage_controller` stays
fail-closed. Not on `L8_THROUGH`. Pin `Level8Interior1EWestReconFixture`.
Do not start the fight.

### Fixture-live 0x1E south door → cleared 0x2E (rr-6o7.2)

From `Level8Interior1EWestReconFixture`. Occupancy empty-grid BFS first
dir is DOWN along x=208 into the SE statue; OccupancyWalker 1px-grade
already false-missed 2px dungeon steps on the west hop, so occupancy was
not used live. Cardinal x-align LEFT to 120, DOWN push. H2/H3 2/2 (probe
`l8_1e_south`, 264 controller frames / 324 with census): first settled
play **`0x2E` `(120,77)`** north mouth, keys 8→8, bombs 6→6, MK 1, TF
`0x7F`. Arrival census empty (Manhandla already dead). `room_item_id=0x17`
map still on the floor. RAM doors `0x0C` (UP+DOWN). Not Gleeok `0x3C`.
`make_gleeok_passage_controller` stays fail-closed. Not on `L8_THROUGH`.
Pin `Level8Interior2ESouthReconFixture`. Do not start the fight. Do not
chain a second DOWN into `0x3E`. Next gate is hyp DOWN toward `0x3E`.

### Fixture-live 0x2E south door → cleared 0x3E (rr-6o7.2)

From `Level8Interior2ESouthReconFixture`. Occupancy still banned (1px-grade
false-misses 2px dungeon steps). Cardinal DOWN along already-aligned
x=120; statues at ~x=96 and x=144 y~141, center aisle passes between
them. I1/I2/I3 2/2 (probe `l8_2e_south`, 255 controller frames / 315 with
census): first settled play **`0x3E` `(120,93)`** north mouth, keys 8→8,
bombs 6→6, MK 1, TF `0x7F`. Arrival census empty (already cleared inbound).
Walking x=120 picked up map `0x17` (`ADDR_MAP` 0→`0x80`); incidental, not
a detour. Arrival RAM doors `0x0C` (UP+DOWN); idle later raises RIGHT
(`0x0D`) as the already-cleared east shutter. Not Gleeok `0x3C`, not
cellar `0x0F`. `make_gleeok_passage_controller` stays fail-closed. Not on
`L8_THROUGH`. Pin `Level8Interior3ESouthReconFixture`. Do not start the
fight. Do not chain RIGHT into passage_east. Next gate is hyp RIGHT toward
`0x3F`.

### Fixture-live 0x3E east shutter → cleared 0x3F (rr-6o7.2)

From `Level8Interior3ESouthReconFixture`. Occupancy still banned
(1px-grade false-misses 2px dungeon steps). Idle until the RIGHT door
bit (arrival doors `0x0C`, idle raises `0x0D` as the already-cleared
east shutter), stay on the north band (y≈93–109) past the x=144 statue,
y-align to the east mouth, RIGHT push. J1/J2 2/2 (probe `l8_3e_east`,
328 controller frames / 388 with census): first settled play **`0x3F`
`(32,141)`** west mouth, keys 8→8, bombs 6→6, MK 1, TF `0x7F`. Arrival
census empty. `room_item_id=0x00`. Arrival RAM doors `0x02` (LEFT). Not
Gleeok `0x3C`, not cellar `0x0F`. `make_gleeok_passage_controller` stays
fail-closed. Not on `L8_THROUGH`. Pin
`Level8Interior3FEastReconFixture`. Do not start the fight. Do not chain
STAIRS into cellar `0x2F`. Next gate is hyp STAIRS toward cellar `0x2F`.

### Fixture-live 0x3F stairs → cellar 0x2F (rr-6o7.2)

From `Level8Interior3FEastReconFixture`. Occupancy still banned. Visual
stairs at ~(192,141) are tile `0x77` (K1/K2 timeout; same decorative
hole as L6 0x3A). Live CheckWarp is tile `0x71` at `(193,141)`. x-first
RIGHT along y=141 toward `(208,93)` crosses it and idles. K3/K4 2/2
(probe `l8_3f_stairs`, 585 controller frames / 645 with census): first
settled **mode 9 cellar `$EB=0x2F` `(208,141)`** tile `0x71`, keys 8→8,
bombs 6→6, MK 1, TF `0x7F`. `position_writes=0`. Not Gleeok `0x3C`, not
MK cellar `0x0F`. `make_gleeok_passage_controller` stays fail-closed.
Not on `L8_THROUGH`. First mode-9 frame is unloaded (black HUD, 0x3F
residuals). After ~100f idle the two-ladder cellar paints; leftover
settles `(192,93)` tile `0x6F` on the east/source ladder (S1/S2). Pin
`Level8Interior2FCellarReconFixture` is that **settled** leftover.

### Fixture-live 0x2F cellar-cross → play 0x4C (rr-6o7.2)

From settled `Level8Interior2FCellarReconFixture`. Occupancy still
banned. DOWN the east/source ladder, floor LEFT to x=48, UP west.
Never UP on the east ladder (returns play `0x3F`). P1/P2 2/2 (probe
`l8_2f_cross`, 356 controller frames / 476 with load+census): first
settled play **`0x4C` `(112,125)`** by the centre stairs, keys 8→8,
bombs 6→6, MK 1, TF `0x7F`. Arrival census empty; `room_item_id=0x19`
small key still on the floor. Enemies spawn after idle — pin at
arrival. Not Gleeok `0x3C`, not source `0x3F`.
`make_gleeok_passage_controller` stays fail-closed. Not on
`L8_THROUGH`. Pin `Level8Interior4CWestReconFixture`. Do not start the
fight. Do not bomb-N toward hyp `0x3C`. Next gate is hyp bomb-N
`0x3C`.

### Fixture-live 0x4C bomb-N → Gleeok 0x3C (rr-6o7.2, census only)

From `Level8Interior4CWestReconFixture`. Occupancy still banned. Spawn
pocket is diamond-locked N/S; RIGHT is the centre stairs (N1/N2 miss).
N3 LEFT at y=117 to `(64,117)`, N4 DOWN to `(64,157)`, N5/N6 east
column x=176 then north wall `(120,93)`. N7/N8 2/2 (probe
`l8_4c_north`, 1037 controller frames / 1127 with census): first
settled play **`0x3C` `(120,189)`** south mouth, bombs 6→5, keys 8→8,
MK 1, TF `0x7F`. Arrival census empty; `room_item_id=0x1A` heart
container. After 90f idle: **one body type `0x45` HP160 at `(124,111)`**
plus `0x56` fireball residual. **Census only — do not fight.**
`GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45` is live RAM (not a ROM assumption);
`assumed_0x45` stays False. `make_gleeok_passage_controller` stays
fail-closed. Not on `L8_THROUGH`. Arrival pin
`Level8Interior3CNorthReconFixture`.

### Fixture-live 0x3C four-head Gleeok south-stand (rr-5eb2)

From `Level8Interior3CNorthReconFixture` play `0x3C` `(120,189)`. Clone
L6 `gleeok18` south-stand (`STAND_DY=22`, bare UP then UP+A, fireball
dodge manhattan ≤14). Body sensor type **`0x45`**. OccupancyWalker
banned. F1/F2 body-gone f5029, then missed HC (mid-room and `(48,157)`
stood 4px short). F5dump room-treasure slot 19 (`$83/$97`) = `(32,192)`.
F6/F7 2/2 byte-identical (probe `l8_3c_gleeok`, 5124 controller / 5184
with census): type `0x45` absent, hc 3→4 (`health` `0x22`→`0x33`),
leftover play **`0x3C` `(32,181)`** tile 118, doors `12` (UP+DOWN; north
shutter RAM-open), keys 8, bombs 5, MK 1, TF `0x7F`. `room_item_id`
stays `0x1A` after pickup. `saw_0x46` mid-fight. ghp samples at 250f
still read 160 until type-gone. Deaths 0, `progression_writes=0`
`capacity_writes=0`, no HP poke. Pin
`Level8Interior3CKillReconFixture`. Not on `L8_THROUGH`.

### Fixture-live 0x3C north shutter → TF 0x2C (rr-6o7.2)

From `Level8Interior3CKillReconFixture` play `0x3C` `(32,181)`. Do not
DOWN (south bomb hole `0x4C`). OccupancyWalker banned. T1 boxed west-wall
UP `(32,133)` tile 179. T2/T3 2/2 (probe `l8_3c_north`, 296 controller
/ 356 with census): first settled play **`0x2C` `(120,205)`** south
mouth, `room_item_id=0x1B` triforce, doors 0, keys 8, bombs 5, MK 1, TF
still `0x7F`, hc 4. ROM 0x2C matched live; lock is the trial.
`assumed_0x2c=False`. Pin `Level8Interior2CTriforceReconFixture`.

Same probe T5/T6: UP x=120 onto the shard (43f), fanfare mode 18, idle
to OW **`0x6D` `(96,93)`** mode 5, TF **`0xFF`**. OW leftover is
**fixture-lineage**, not a Survival-true post-L8 handoff (this pin is
not Survival-from-L7). Pin `Level8PostShardOWReconFixture`.
`make_gleeok_passage_controller` stays fail-closed. Not on `L8_THROUGH`.

**Object type `0x0C` is not registered** in `dungeon/ids.py` — live censuses
print it as `unknown_object_0x0c`. What the evidence establishes is
**type `0x0C`, HP 128**, appearing 5× in `0x5E`, 1× in `0x4E` and 6× in `0x3E`,
alongside the registered `0x0B` `darknut` at HP 64 in the same `0x4E` room.
The **blue** colour is a walkthrough correlation, not a registered claim; do
not write "blue Darknut" as if RAM said so.

Mapping the live chain onto the hypothesis grid (`0x7E`=`entry`,
`0x6E`≈`north_manhandla`, `0x5E`≈`darknut_key`, `0x4E`≈`shutter_darknuts`,
`0x3E`≈`blue_darknuts`) is **plausible but unverified** — the grid names stay
hypotheses and no `room_id` beyond `entry` has been written into the table.
The **gates do not line up**: the source min route below spends a key going
`darknut_key → shutter_darknuts`, but live `0x5E → 0x4E` is a free open
shutter, and the live key spend is one door later (`0x4E → 0x3E`). Treat the
source gate list as a sketch, not a key budget.

Minimum Magical Key route (Book/Map/Compass omitted, source):

```text
entry --UP--> north_manhandla --BOMB UP--> darknut_key
  --KEY UP--> shutter_darknuts --KEY UP--> blue_darknuts
  --BOMB UP--> map_manhandla --KEY UP--> blue_gohma
  --RIGHT--> magic_key_stairs
```

Four-head Gleeok suffix (after Magical Key):

```text
blue_gohma --DOWN×2--> blue_darknuts --KILL RIGHT--> passage_east
  --STAIRS--> pols_west --BOMB UP--> gleeok --UP--> triforce
```

| Room / feature | Enemies / notes | Live |
|----------------|-----------------|------|
| Entry | live: room `0x7E`, `(120,205)`, item `0x03`, no live objects | **fixture-live** (`route_eligible=false`) |
| Manhandla early | live `0x6E` north of entry: 5× `0x3C` HP64, sword-only clear. West of entry is the source Book detour, unvisited | **fixture-live** (`0x6E` only) |
| Book of Magic (staircase) | `ADDR_BOOK=0x0661` | **omitted** on min route |
| Darknut / keys / Compass / Map | live `0x5E` (5× `0x0C`, small key `0x19`), `0x4E` (mixed), `0x3E` (6× `0x0C`); Compass/Map omitted | **fixture-live** through `0x3E` |
| Gohma (blue, 3 arrows) | live `0x1E` type **`0x33` HP96** (ids: L6 red; colour not asserted). 3 connecting wooden arrows, 9 shots loosed. East kill-clear → `0x1F`. **no L6 poke** | **fixture-live** (`route_eligible=false`) |
| Magical Key (staircase) | live cellar `$EB=0x0F` MK 0→1; return 2/2 play `0x1F` `(96,157)`. West door 2/2 back to cleared `0x1E` `(208,141)`. Not `0x3C` | **fixture-live** (`route_eligible=false`) |
| Boss Gleeok 4-head | Heart → TF; live body `0x45` kill in `0x3C`; north shutter → live TF room **`0x2C`**, shard `0x1B`, TF `0xFF` | **fixture-live kill+shard** (`route_eligible=false`) |

Items optional for credits (source). TF bit **`0x80`**. Magical Key is the
deliberate L9 key-bottleneck investment.

## Boss / Triforce

- Boss: **Gleeok (4 heads)** — fixture-live type `0x45` kill in `0x3C`.
- `ADDR_TRIFORCE == 0xFF` after shard 8 — fixture-lineage OW `0x6D` `(96,93)`, not Survival-true leave.

## Checkpoints

All L8 recon fixtures below carry `development_only: true`,
`natural_entry: false`, `route_eligible: false` and the standard
`acceptance_warning` in their `.provenance.json`. None may be used as a route
start.

| State | Provenance |
|-------|------------|
| `Level8BushOW` / `OW_6D` | Assisted settle on bush screen; **no candle** |
| `OW_5D` | Live south of maze; path parent of 0x6D / west of shop |
| `OW_5C` | North corridor entry from 0x5B |
| `BFS_5E` / `OW_5E` | Live OW on shop screen (west edge entry y≈141) |
| `CandleShop5E` | Cave on **0x5E** mode 11 (0R; buy residual) |
| `CandleOwned` | **not created** (natural buy residual) |
| `Level8BushWithCandleFixture` | Stage 1 of the burn recon: from `Level8BushOW`, pokes Candle 2 + B-slot 4 + TF `0x7F` + stand `(48,90)` on `0x6D` (settles `(48,93)`) |
| `Level8EntranceReconFixture` | **First live L8 interior.** Burn `(136,93)` RIGHT/RIGHT from the above; L8 play `0x7E` `(120,205)`, mode 5, TF `0x7F`. Replaces the old “`Level8Entrance` not created” row |
| `Level8InteriorReconFixture` | `Level8EntranceReconFixture` + disclosed recon resources (sword 3, bombs 8, bow 1, arrows 1, rupees 255, keys 9); Magic Key / TF / rooms / doors untouched |
| `Level8Interior3EReconFixture` | L8 play `0x3E` `(120,205)`, keys 9, bombs 7, 6× `0x0C` HP128 alive |
| `Level8Interior1EReconFixture` | L8 play `0x1E` `(120,205)`, keys 8, bombs 6, one body type `0x33` HP96 |
| `Level8Interior1FReconFixture` | L8 play `0x1F` `(16,141)`, keys 8, bombs 6, rupees 246, Magic Key 0; mixed 0x16/0x0C/0x0B plus centre stairs |
| `Level8InteriorMKReconFixture` | Magical Key pad leftover: L8 mode-9 cellar `0x0F` `(136,141)`, keys 8, bombs 6, rupees 247, Magic Key **1** |
| `Level8Interior1FReturnedReconFixture` | Cellar return leftover: L8 play `0x1F` `(96,157)`, MK 1, keys 8, bombs 6 |
| `Level8Interior1EWestReconFixture` | West-gate leftover: L8 play `0x1E` `(208,141)` east mouth, MK 1, keys 8, bombs 6; Gohma already dead |
| `Level8Interior3ESouthReconFixture` | South-gate leftover: L8 play `0x3E` `(120,93)` north mouth, MK 1, keys 8, bombs 6; east shutter still metal at arrival |
| `Level8Interior3FEastReconFixture` | East-gate leftover: L8 play `0x3F` `(32,141)` west mouth, MK 1, keys 8, bombs 6; stairs on the raised east platform, do not walk on |
| `Level8Interior2FCellarReconFixture` | Settled mode-9 cellar `0x2F` `(192,93)` tile `0x6F` east ladder, MK 1, keys 8, bombs 6; two ladders + pit + 4 keese HP0 |
| `Level8Interior4CWestReconFixture` | Cellar-cross leftover: L8 play `0x4C` `(112,125)`, MK 1, keys 8, bombs 6; `room_item` `0x19` key on floor; do not bomb-N |
| `Level8Interior3CNorthReconFixture` | Bomb-N leftover: L8 play `0x3C` `(120,189)` south mouth, MK 1, keys 8, bombs 5; Gleeok `0x45` HP160 after idle; fight pin |
| `Level8Interior3CKillReconFixture` | Post-kill leftover: L8 play `0x3C` `(32,181)`, MK 1, keys 8, bombs 5, hc 4; body `0x45` gone; north shutter RAM-open |
| `Level8Interior2CTriforceReconFixture` | TF-room leftover: L8 play `0x2C` `(120,205)` south mouth, `room_item` `0x1B`, TF still `0x7F` |
| `Level8PostShardOWReconFixture` | Fixture-lineage OW `0x6D` `(96,93)` after shard; TF `0xFF`; **not** Survival-true post-L8 leave |
| `Level8Entrance` (canonical route state) | **still not created** — needs the measured post-L7 leave, not a poked stand |

## Scaffold modules

| Path | Role |
|------|------|
| `level8/overworld.py` | **Live, not frozen.** Bush + shop hops, `L7_POND_TO_LEVEL8_BUSH_HOPS` and `Level7PondToLevel8BushController` (rr-6o7.4, 2/2 to `0x6D`). Start-based burn false-positive stays recon-only |
| `level8/entry.py` | Canonical measured post-L7 approach, natural pause selection, fail-closed Red Candle burn; holds `LIVE_RECON_BUSH_BURN_TARGET` |
| `level8/bush.py` | Isolated 0x6D fixture-live burn recon; `route_eligible=false` |
| `level8/dungeon.py` | Hypothesis door graph + exact stop predicates; `LIVE_RECON_LEVEL8_TOPOLOGY` (entry `0x7E`) and `LEVEL8_INTERIOR_0X3E_RECON`; `LEVEL8_ROOM_SPECS` still empty |
| `level8/gleeok.py` | Four-head south-stand fight in `0x3C` (live type `0x45`); heart slot 19 (idle if missing) |
| `level8/triforce.py` | `0x3C` north shutter dest hop (live `0x2C`) + shard walk-on |
| `level8/path.py` | Fixture-live 0x1F west, 0x1E/0x2E south, 0x3E east doors; four-head wraps `level8.gleeok`; Gleeok-passage still fail-closed |
| `level8/hops.py` | Fresh chapter/controller factories and three `SpineHop` rows |
| `level8/spine.py` | `L8_THROUGH`, `L8_STOPS`, `continue_level8_spine`, opt-in `LIVE_RECON_L8_OVERRIDES` |
| `level8_bush_burn_sweep` | Producer of `logs/level8_bush_burn_sweep.json` (5856 trials) |
| `capture_level8_entrance_fixture` | Reproduces the `(136,93)` RIGHT/RIGHT burn into live `0x7E` |
| `probe_l8_5e_north` / `probe_l8_4e_north.py` | The `0x7E→0x4E` and `0x4E→0x3E` fixture replays |
| `probe_l8_1f_magic_key` | The `0x1F` stairs → Magical Key cellar `0x0F` fixture replay |
| `probe_l8_0f_cellar_return` | Two-ladder return `0x0F` → play `0x1F` |
| `probe_l8_1f_west` | Play `0x1F` west door → cleared `0x1E` |
| `probe_l8_3e_east` | Play `0x3E` east shutter → cleared `0x3F` |
| `probe_l8_3f_stairs` | Play `0x3F` stairs walk-on → cellar `0x2F` |
| `probe_l8_2f_settle` | Idle-settle unloaded mode-9 `0x2F` (~400f) |
| `probe_l8_2f_cross` | Settled `0x2F` east-ladder cross → play `0x4C` |
| `probe_l8_4c_north` | Play `0x4C` bomb-N → Gleeok `0x3C` (census pin) |
| `probe_l8_3c_gleeok` | Play `0x3C` south-stand `0x45` kill + heart `(32,192)` |
| `probe_l8_3c_north` | Play `0x3C` north shutter → TF `0x2C` + shard → OW `0x6D` |
| `probe_l7_exit_to_l8_bush` | L7-pond → `0x6D` geometry lane |
| Isolated `probe_level8_entry.py` | pruned; Composer `scripts/run_survival_spine.py` |
| `docs/LEVEL8_ROUTE.md` | This file |

### Wave A handoff contract

The integrator must provide all of the following before the default seam can
move: settled post-L7 OW screen/x/y, exact keys/bombs/rupees/hearts/B-slot,
Whistle/Food/Rod/Bow/arrows, Candle 2, full health, TF exactly `0x7F`, and a
live-derived hop table ending at `0x6D`. The bush burn's tile / facing / push
are now supplied (`LIVE_RECON_BUSH_BURN_TARGET`, fixture-live), but the
**natural `ADDR_CANDLE_USED` transition from a walked pose** is still owed.
The Magical Key room, boss room, Triforce room and post-fanfare leave remain
`None`; the entry room is fixture-live `0x7E`, not a promoted route anchor;
walkthrough grid positions are hypotheses.

Fixture work may fill controller behavior while keeping
`route_eligible=false`. Only cumulative recomposition from the natural L7
predecessor may promote the handoff/topology/endpoint contracts.

## Evidence

- `recordings/l8_bush_recon.json` — 0x6D settle, exits, path string
- `recordings/l8_5d_DOWN_48_sc6d.png` — live 0x5D→0x6D
- `recordings/l8_bush_6d.png` — bush pocket
- `recordings/l8_walkable.png` — pink walkable samples on 0x6D
- `recordings/l8_pos4.png` — HUD candle after recon poke (selected=4)
- `recordings/l8_candle_burn_enter.json` — burn recon summary (entrance false)
- `recordings/l8_candle_shop66.json` — earlier 0x66 cave probes (not shop)
- `recordings/l8_shop_path.json` — assisted start→0x5E hop success
- `recordings/l8_5e_cave_x112.png` / `l8_5e_cave_enter.png` — cave mouth
- `recordings/l8_buy_ok_repro.png` — recon buy @ (152,149) with poked 80R
- `recordings/l8_*_exits.json` — pocket maps (0x6B/0x6C/0x5C)

Fixture-live evidence added 2026-09-03/04 (all `route_eligible=false`):

- `logs/level8_bush_burn_sweep.json` — **5856-trial** live burn sweep on OW
  `0x6D`; 7 mode-16 mouth stands, `(144,93)` `no_effect` ×8
- `logs/level8_6d_walkable_positions.json` — the 732 teleport-standable tiles
  the sweep iterated (729 distinct coordinates due to 3 duplicates; standable, not walked)
- `custom_integrations/.../Level8EntranceReconFixture.provenance.json` — the
  `(136,93)` RIGHT/RIGHT recipe and the first live L8 interior (`0x7E`)
- `custom_integrations/.../Level8BushWithCandleFixture.provenance.json`,
  `Level8InteriorReconFixture.provenance.json`,
  `Level8Interior3EReconFixture.provenance.json` — the disclosed poke lists
- `recordings/l8_5e_north_fixture_20260903_v1.json` — `0x7E → 0x4E` chain
- `recordings/l8_4e_north_fixture_B1.json` / `_B2.json` — the same chain plus
  the `0x4E → 0x3E` north key door, 2/2 byte-identical, 2918 frames
- `recordings/l8_5e_north_B1_*.png` — per-transition frames, including
  `..._first_settled_destination_f2322_L8_s4e_m5.png` (`0x4E`) and
  `..._key_north_pushed_f2768_L8_s3e_m5.png` (`0x3E`)

## Next

1. **Blocker.** Receive the measured natural post-L7 leave/inventory
   (rr-8t4.3) and derive its hop table through live `0x5C` geometry to
   **`0x6D`**. Nothing else can un-red an L8 chapter.
2. From a naturally-earned Red-Candle state on `0x6D`: replay the recorded
   burn — stand **`(136, 93)`**, face **RIGHT**, push **RIGHT**, and keep
   pushing **RIGHT** through the mouth (do **not** switch to UP). Then confirm
   live `0x7E` from a *walked* pose. `(144, 93)` is refuted; do not revive it.
3. Re-walk `0x7E → 0x6E → 0x5E → 0x4E → 0x3E` on naturally earned bombs/keys
   (the recorded chain runs on disclosed recon resources), then continue past
   `0x3E` toward Magical Key → Gleeok / shard rooms without promoting source
   room IDs. Magical Key stays on the min route; Book stays off.
4. Keep the 60R Blue Candle farm/shop as fallback-only, outside `L8_THROUGH`.
5. Do not promote Clean and do not touch `STATUS.md`; all L8 fixture evidence
   remains `natural_entry=false` / `route_eligible=false`.

---

## Gleeok family model (merged from `docs/tasks/rr-5eb2-gleeok-model.md`, 2026-09-07)

Live-measured, not walkthrough. The three Gleeok fights differ only in
these fields. The cleanup campaign that was going to fold them into one
`BossSpec` is retired. This table is the note.

| Field | L4 (2-head) | L6 (3-head) | L8 (4-head) |
|-------|-------------|-------------|-------------|
| Body object type | `0x43` | `0x44` | **`0x45`** (live RAM, not a ROM guess) |
| Room | `0x13` | `0x18` | `0x3C` |
| Start HP | ≈160 | — | **160** |
| Body pose | x≈124, y≈111 | `(124, 111)` | `(124, 111)` |
| Stand target | `(body.x, body.y + 22)` | same | same |
| Measured fight | ~3,649 f assisted | 2,848 f to body-gone | 5,124 controller f, body-gone f5029 |
| `MAX_FRAMES` | 20,000 | 20,000 | 20,000 |
| Reward | HC `0x1A` mid-room | — | HC in treasure slot 19 at `(32, 192)`, hc 3→4 |
| Post-kill leftover | — | — | `(32, 181)`, doors 12 (UP+DOWN) |

**Shared across all three** (already in `dungeon/gleeok.py`):
- Detached head type `0x46`; fireball residual `0x56`.
- South-stand policy: face **UP + A**; fireball dodge at manhattan **≤14**
  horizontal. Body type is dungeon-specific; `0x46` / `0x56` are shared.
- Dead when the **body type is absent** — heads and fireballs may linger.
- **Do not chase `0x46` while the body type remains** (L4 `rr-vdnc`).
- Bombs do **not** damage Gleeok.
- Four-head is attrition, not new geometry: more attached heads → more `0x56`
  attempts → more `0x46` kites after detaches.

`dungeon/ids.py` carries `GLEEOK_OBJECT_TYPE=0x43`,
`GLEEOK_3HEAD_OBJECT_TYPE=0x44`, `GLEEOK_HEAD_OBJECT_TYPE=0x46` — and **no
4-head constant**; L8 defines `GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45` locally.
Fold that into `ids.py` during Phase 2.6.

**Sword damage:** `SwordDamagePoints-1[Items]` with `Items=3` (Magical Sword)
= `0x40` (64) per connected slash, sword state fully extended (`0x02`) —
9 hits vs 28 with the wooden sword. Relevant to Phase 5.5 (Magical Sword,
12 HC) and to any Clean-pass fight budget.

**Also live-measured here:** L8 Gohma is type **`0x33` HP 96** in `0x1E`
(the walkthrough's `0x34` is wrong).

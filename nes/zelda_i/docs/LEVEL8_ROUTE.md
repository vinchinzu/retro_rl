# Level 8 — The Lion (route notes)

Status: **PARTIAL** — assisted OW bush path green; shop OW path **green**
(rr-ccx). The bush burn recipe and the first L8 interior rooms are now
**fixture-live** (`natural_entry=false`, `route_eligible=false`). The
cumulative Red-Candle route still needs **the measured post-L7 leave**
(rr-8t4.3, unmeasured), so no L8 chapter may green. The 60R shop path is
fallback-only.

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
  kept **732** tiles out of a 32×23 sampled grid
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

### Fixture-live 0x1F stairs → Magical Key cellar 0x0F (rr-6o7.2)

From `Level8Interior1FReconFixture` (fixture-only, `natural_entry=false`,
`route_eligible=false`). E1 no-clear south-face UP on the west `0x68` was
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
Do not start the fight. Next gate is hyp DOWN toward `0x2E`.

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
| Boss Gleeok 4-head | Heart → TF; body type **unobserved** (not assumed `0x45`) | no |

Items optional for credits (source). TF bit **`0x80`**. Magical Key is the
deliberate L9 key-bottleneck investment.

## Boss / Triforce

- Boss: **Gleeok (4 heads)** — source only.
- `ADDR_TRIFORCE & 0x80` after shard 8 — source only.

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
| `Level8Entrance` (canonical route state) | **still not created** — needs the measured post-L7 leave, not a poked stand |

## Scaffold modules

| Path | Role |
|------|------|
| `level8/overworld.py` | **Live, not frozen.** Bush + shop hops, `L7_POND_TO_LEVEL8_BUSH_HOPS` and `Level7PondToLevel8BushController` (rr-6o7.4, 2/2 to `0x6D`). Start-based burn false-positive stays recon-only |
| `level8/entry.py` | Canonical measured post-L7 approach, natural pause selection, fail-closed Red Candle burn; holds `LIVE_RECON_BUSH_BURN_TARGET` |
| `level8/bush.py` | Isolated 0x6D fixture-live burn recon; `route_eligible=false` |
| `level8/dungeon.py` | Hypothesis door graph + exact stop predicates; `LIVE_RECON_LEVEL8_TOPOLOGY` (entry `0x7E`) and `LEVEL8_INTERIOR_0X3E_RECON`; `LEVEL8_ROOM_SPECS` still empty |
| `level8/path.py` | Fixture-live 0x1F west door; fail-closed Magic-Key / Gleeok-passage / four-head Gleeok factories |
| `level8/hops.py` | Fresh chapter/controller factories and three `SpineHop` rows |
| `level8/spine.py` | `L8_THROUGH`, `L8_STOPS`, `continue_level8_spine`, opt-in `LIVE_RECON_L8_OVERRIDES` |
| `scratch/level8_bush_burn_sweep.py` | Producer of `logs/level8_bush_burn_sweep.json` (5856 trials) |
| `scratch/capture_level8_entrance_fixture.py` | Reproduces the `(136,93)` RIGHT/RIGHT burn into live `0x7E` |
| `scratch/probe_l8_5e_north.py` / `probe_l8_4e_north.py` | The `0x7E→0x4E` and `0x4E→0x3E` fixture replays |
| `scratch/probe_l8_1f_magic_key.py` | The `0x1F` stairs → Magical Key cellar `0x0F` fixture replay |
| `scratch/probe_l8_0f_cellar_return.py` | Two-ladder return `0x0F` → play `0x1F` |
| `scratch/probe_l8_1f_west.py` | Play `0x1F` west door → cleared `0x1E` |
| `scratch/probe_l7_exit_to_l8_bush.py` | L7-pond → `0x6D` geometry lane |
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
  the sweep iterated (standable, not walked)
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

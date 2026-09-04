# L7 sitting leftover (rr-8t4.3, 2026-09-03)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.3` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — L7-C 0x0D: Step-1 south-face squeeze re-tried & NO-GO; Step-2 poke cellar is a DEAD END back to 0x0D (rr-8t4.3)

MAIN tree, branch `main`. No capability pokes. One disclosed position
poke (`ADDR_LINK_X/Y` -> `(204,88)`) for the Step-2 recon fixture only,
logged with before/from/to. `route_eligible=false` everywhere. Chapter
factories fail-closed. `PostLevel7Handoff.verified` untouched. No graph
`ram_id` promotion. No spine edits. Tests 500/500.

### Step 1 (TAS south-face UP-push): NO-GO, reconfirmed from 4 fresh vectors

New tool `scratch/probe_l7_room0d_squeeze.py` (`--sweep --squeeze
--around-east --wiggle --from-east`). Pin `Level7Interior0DClearedReconFixture`.

- Full poke-read tile sweep y=136..189: the x176-188 diamond mass is
  y≈101..135 then a **GAP y136..158 that point-pokes as floor**, then the
  south wall y≈166..180, then the bottom door strip y≈182..188.
- **That y136..158 "floor" is not walkable by a moving Link.** Every
  approach to the block's south-face stand `(192,160)` pins Link at
  **`(176,155..157)` tile 178/119**:
  - west corridor east-hug at y=156 → pin.
  - east pocket `(192,141)` straight DOWN x=192 → `(176,157)`.
  - around the block's east side, DOWN x=204 / x=200 / x=196 then LEFT →
    `(176,157)` every column (Link slides west off x≥196).
  - sub-pixel diagonal wiggle (RIGHT+DOWN / DOWN+RIGHT alternation) at the
    pin → nets `(176,155)`, zero eastward progress.
- The UP push cannot be executed. `level9/stairs.py` `room03`/`room30`
  recipe stays the template if the pin is ever cracked; it was not.

### Step 2 (disclosed one-poke recon fixture): reaches 0x7b mode 9 but it is a DEAD END

New tools `scratch/probe_l7_room0d_cellarpoke.py` (`--scan`, `--poke X Y`,
`--save`) and `scratch/probe_l7_nosecellar_tail.py`
(`--sweep --manual --topsweep --cellar`).

- Without the RIGHT push, **no poke pixel trips CheckWarps** (stairs not
  revealed) — `(208,93)` from the task spec stays room 0x0D mode 5
  (`0d_pokewarp_v1`). The task's assumed `(208,93)` no-push recipe is dead.
- **With** the real `0x68` RIGHT push (block parks `(208,96)` state 2,
  reveals stair tiles `0x70-0x73` at x≈192-208 y≈93-100), a single poke to
  **`(204,88)`** + a couple idle frames → mode 16 → settles **screen
  `0x7b` mode 9** (`mode_name` "dungeon_underworld_passage"), census
  **4× keese `0x1b`**, `room_item_id=3`. **Reproduced 2/2** (`0d_cp_scan`,
  `0d_cp_p1`). Live `$EB` for NOSE_CELLAR-as-reached = **`0x7b`**.
  Bombs/keys/candle unchanged (6/2/2). Saved
  `Level7Interior0DNoseCellarReconFixture` (provenance:
  `route_eligible:false`, `fixture_only`, `development_only`, the one poke
  logged, DEAD-END flagged).
- **The 0x7b passage is wired back to 0x0D, not forward.**
  `provenance.state.next_room = 13` (0x0D). The room only loads after a
  ~400-frame idle; then Link walks a narrow x=192 top channel (y≈63-93)
  and the bottom floor strip (y≈189, reached via the right x≈200 vertical
  corridor). Walking **UP** the top channel at **any x** (topsweep x=32..208,
  8 separate envs) → mode-10 stairs-exit → **room `0x0D` at `(96,157)`**.
  The bottom strip has no UP transition at any column (swept x=208..16).
  The keese never activate (hp=0), `room_all_dead` never resolves.
- Conclusion: `(204,88)` lands Link on the **return** stair leg. The
  **forward** NOSE_CELLAR staircase is the tile the parked `0x68` sits ON
  — sealed by the RIGHT push, exactly as the 2026-09-03 sitting below
  established. Only the unsolved south-face UP push clears the forward
  stair.

### Downstream L7-C tail: NOT reached

`NOSE_CELLAR → PRE_BOSS → AQUAMENTUS → TRIFORCE → OW-leave` all remain
unobserved. `MEASURED_POST_L7_EXIT` in `level7/entry.py` **NOT filled** —
no genuine evidence. `Level1AquamentusController` reuse **not attempted**
(never reached a live boss room). No `level1/finish.py` change.

### Dead beliefs burned (2026-09-03)

- Dead: the y≈136..158 poke-read "floor" gap under the x176-188 diamond
  mass is walkable by a moving Link — pins at `(176,155..157)` from the
  west corridor, the east pocket, around-east x196-204, and diagonal
  wiggle (`probe_l7_room0d_squeeze.py`).
- Dead: going around the `0x68` east side (x196-204) DOWN then LEFT to
  reach `(192,160)` — Link slides to `(176,157)` every column.
- Dead: the task-spec `(208,93)` no-push position poke reaches a cellar —
  it stays room 0x0D mode 5 (stairs not revealed without the push).
- Dead: the `0x7b` cellar reached by the `(204,88)` poke leads to
  PRE_BOSS — it is `next_room=0x0D`; UP from the top channel (every x)
  exits mode-10 back to `0x0D (96,157)`. It is a return-only passage as
  reached; the forward leg is under the parked block.
- Dead: the `0x7b` cellar is navigable immediately on arrival — it needs
  a ~400f idle to finish loading; before that Link is frozen except UP
  and poking him around causes hurt/death (mode 8/17).

### Resume point

The `0x0D` → forward-NOSE_CELLAR walk-on is still the OPEN BLOCKER and is
now the ONLY path: the position-poke stand-in provably lands on the
return leg, so it cannot unblock the downstream recon. Next options:
1. Crack the south-face `(192,160)` UP push (frame-perfect; the y156
   corridor pin at `(176,157)` is the wall — needs a mechanic we don't
   have, or proof it is vanilla-impossible and the ROM predecessor of
   NOSE_CELLAR is a *different* room's staircase).
2. Re-derive NOSE_CELLAR's real predecessor from L7 ROM stair-list data
   (mirror of `level9/stairs.py` `LEVEL9_STAIR_PAIRS`) — 0x0D may not be
   it, or 0x0D's stair may be a within-room shortcut (exit `(96,157)`
   supports that reading).
3. Integrator: a heavier disclosed stand-in (e.g. poke `$EB`/`next_room`)
   is the only way to unblock PRE_BOSS recon, and is out of scope here.

Fixture `Level7Interior0DClearedReconFixture` unchanged. New fixture
`Level7Interior0DNoseCellarReconFixture` saved (DEAD-END, recon only).
New scratch: `probe_l7_room0d_squeeze.py`, `probe_l7_room0d_cellarpoke.py`,
`probe_l7_nosecellar_tail.py`.

## 2026-09-03 — L7-C 0x0D block −48 snap is REAL (matches L9 room30/03); walk-on NO-GO (rr-8t4.3)

Follow-up to the sitting below, per coordinator: diff the `0x68` push
physics against a known-good push room. No pokes advanced state (position
pokes for tile recon only). `route_eligible=false`.

### Root cause of the `(192,144)→(208,96)` −48px y-snap: **real in-game
behaviour, NOT a bug in our code.**

- It is LoZ's "secret staircase block" relocation. **`level9/stairs.py`
  has this exact mechanic solved and route-eligible in two rooms:**
  - `room30_stairs_step` / `room30_block_secret_open` — after the push the
    block sits at `x >= 0xC0, y <= 0x70`; Link then stands **exactly at
    `(0xD0,0x60) = (208,96)`** → CheckWarps → cellar. `ROOM30_BLOCK_REST =
    (0x60,0x90)` i.e. the block starts at the mirror of our `(192,144)`.
  - `room03_stairs_step` — block pushed UP to `y <= 0x80`, Link walks a
    "slot" at **`ROOM03_SLOT_Y = 133`** (the same y as `0x0D`'s plug) to a
    normal floor stand `(128,141)`.
- Frame trace of our push: block slides smoothly E 1px/2f `(193,144)…
  (207,144)`, then **one frame** → `(208,96)`, `state 0→2`. That single-
  frame teleport at push-completion IS the ROM's reveal routine parking
  the block. Stair tiles `0x70-0x73` genuinely appear at x≈196-208
  y≈96-100. Object x/y read straight from RAM `$007B`/`$008F` — no sim
  layer, no walker prediction.
- Room `0x1A`'s `0x68` in the same fixture lineage does a normal one-tile
  push → no global block corruption.

### Go / no-go on the `0x0D` walk-on: **NO-GO as reachable.**

In L9 room30/03 the reveal opens a walkable slot/column to the warp cell.
In `0x0D` it does not:
- **RIGHT push** (the only reachable block face): plug at `(192,133)` tile
  `0xB3` stays solid — not bombable (3 bombs, `0d_ne_v1`), stays solid for
  the whole 32f slide (`0d_race192_v1`). The x=192 and x=208 columns stay
  sealed post-push (`0d_ecol_RIGHT`).
- **UP push** (L9's recipe): **cannot be executed.** The block's south
  face stand `(192,160)` is unreachable — the x176-188 diamond mass
  (tiles `0xB0-0xB3`, solid for a moving Link at y≈157 even though a point-
  poke at y=156 reads floor) walls the y≈156 corridor
  (`0d_pushUP_v2/v3/v4`, Link pins at `(176,157)` tile `0xB1` every time);
  from the east pocket `(192,141)` the block at `(192,144)` blocks
  southward travel; the bottom strip (y184-188) + the x=192 column south
  of the block are isolated. Any nudge relocates the block to `(208,96)`.
- Graph: `NOSE_CELLAR`'s only entrance is UP from `TIP_OF_NOSE` — so this
  IS on the critical path, not optional.

### Recommendation (not a code fix)

1. The `0x0D` south-face approach may be a **1px-tight vanilla squeeze** at
   y≈156 — worth a frame-perfect / TAS attempt to stand `(192,160)` and
   UP-push (L9 room03 recipe). If the block then parks like L9 and opens
   the plug/slot at y=133, the walk-on is live.
2. Otherwise: **integrator call** on a disclosed one-write recon fixture
   (position poke to `(208,93)`, `route_eligible=false`, `fixture_only`,
   write logged) purely to unblock NOSE_CELLAR→PRE_BOSS→AQUAMENTUS→shard→
   OW-leave recon, with the `0x0D` walk-on flagged as an open blocker.

Reused knowledge, no lib change: `level9/stairs.py` `room30_stairs_step` /
`room03_stairs_step` are the template if the south-face approach is cracked.
New scratch flags: `probe_l7_room0d_ne.py --push-dir UP|RIGHT --e-column`.

## 2026-09-03 — L7-C 0x0D NE staircase pocket is fully sealed; NOSE_CELLAR = 0x7b (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder. No state-
advancing `ADDR_LINK_X/Y` writes (position pokes used for tile recon only,
disclosed below). `route_eligible=false`. Chapter factories stay fail-
closed. `PostLevel7Handoff.verified` stays false. No invented OW leave.

**Pin was** `Level7Interior0DClearedReconFixture` — L7 play `0x0D`
`(63,149)` `room_all_dead=1`, Candle 2, TF 0, keys 2, bombs 6, whistle 1,
ladder 1, `0x68` at `(192,144)`.

### New recon (all `deaths=0`, `progression/capacity writes=0`)

**NOSE_CELLAR destination observed: `$EB=0x7b` mode 9 (cellar), census
4× keese `0x1b`.** Reproduced 2× (`0d_nap_v1`, plus the sweep_0d poke run).
This retires the dead belief "do not treat dump mode-16 screen `0x7b` as
cellar" — it IS the cellar, but the transition was reached by a **position
poke to `(208,93)`**, NOT a walk-on. Not promotable. `NOSE_CELLAR.ram_id`
stays `None`.

**The `0x0D` staircase tiles are real and revealed by the RIGHT push.**
Fine tile sweep (`sweep_0d.py`, poke-read of `colliding_tile`, 2px grid):
- Stair tiles `0x70-0x73` appear at **x≈196-208, y≈96-100** ONLY after the
  `0x68` RIGHT push (absent on the cleared/un-pushed fixture).
- The push still snaps `0x68` `(192,144)→(208,96)` (16px E slide then a
  −48 y snap), block `state=2`. Reconfirmed 2/2 (`0d_nap_v1/v2`).

**The staircase pocket is geometrically sealed — exhaustively confirmed.**
- A **full-width solid band at y≈101–115** spans x≈32–188 (tiles
  `0xB0-0xB3` diamond = **never bombable**). Present before AND after the
  push; the push does not change it.
- Above the band (y≤99): open floor corridor x≈32–204 that connects to the
  staircase at its east end.
- Below the band: open floor; plus an isolated `0xB0-0xB3` mass at
  x≈176–188 / y≈117–135.
- The **x≈192–204 sub-column is floor from y≈97 down to y≈131** (it would
  be the walk-up path onto the staircase) but it is walled off:
  west neighbour x≈188 is diamond `0xB0-0xB3` for y≈117–131, and a
  **1-tile diamond plug at (192,133)** seals it from the reachable
  `(192,136–141)` east pocket. The plug is tile `0xB3` — bombs at
  `(192,136)`/`(192,135)` facing UP do **not** open it (`0d_ne_v1`,
  3 bombs).

### Dead beliefs burned (2026-09-03)

- Dead: the `y≈101–117` band has a walkable notch anywhere — swept x=32→204
  at 2px, every column solid through the band (`sweep_0d`, cleared + pushed).
- Dead: the west wall (x≈44–48) is a vertical corridor to the top band —
  `0x2b` residuals patrol x=32 y≈93–123 but Link pins at `(48,117)` tile
  `0xB3` (`0d_wn_v1`). The residuals are stuck ABOVE the band, not proof of
  a corridor.
- Dead: bombing the `(192,133)` plug from the south opens the x=192 column
  (`0d_ne_v1`, tile stays `0xB3`).
- Dead: racing UP the **x=192** column during the 32f block slide —
  `(192,133)` tile `0xB3` solid for the whole slide (`0d_race192_v1`).
  (x=176 race was already dead.)
- Dead: `--bomb-stair` reach targeting (`0d_bs_v1`) — `_reach` cannot hit
  `(192,136)` reliably, wasted 4 bombs; use the `0d_ne_v1` `_bomb_up` path.

### Assessment / options for next session

The block-push-reveals-staircase mechanic as MODELLED leaves the staircase
in a pocket with **no walk-on route** (non-bombable diamond geometry on
every side). Either:
1. The `0x68` RIGHT-push physics in the harness are wrong (the −48 y snap
   to `(208,96)` is not vanilla one-tile block behaviour) — the real game
   may move the block one tile to `(208,144)` and open the x=192 column /
   drop a staircase in the reachable area. Worth diffing block-push
   handling vs a known-good L1/L5/L9 push room.
2. NOSE_CELLAR (`0x7b`) is entered from a **different room's** staircase,
   not `0x0D`'s — check `0x1D` / `0x1C` / neighbours for a down-stair.
3. Integrator call: allow a disclosed one-write position-poke recon
   fixture at `(208,93)` purely to unblock the NOSE_CELLAR → PRE_BOSS →
   AQUAMENTUS → shard → OW-leave recon downstream, clearly
   `route_eligible=false` / `fixture_only=true`, with the `0x0D` walk-on
   left as an open blocker.

`Level7Interior0DClearedReconFixture` is unchanged (block un-pushed).
No new fixture saved (would have covered the hole or required a
forbidden poke).

**Leftover glance:** L7 play **`0x0D` mode 5** `(63,149)`,
`room_all_dead=1`, Candle **2**, TF **0**, keys 2, bombs 6, whistle 1,
food 0, ladder 1, 3 hearts. Block `0x68` `(192,144)`.
`route_eligible=false`.

**Resume point:** the `0x0D` → NOSE_CELLAR walk-on is an OPEN BLOCKER.
Try option 1 (block-push physics diff) or option 2 (alt entry room)
before another walk-on sweep of `0x0D`. Dest is `$EB=0x7b` mode 9 (keese).
Then NOSE_CELLAR far-side → PRE_BOSS bomb-E → AQUAMENTUS (verify census
`0x3D`+`0x55` live) → shard `0x40` → measure settled OW leave.

## 2026-09-03 — L7-C 0x0D NE hole unwalkable during 16px slide (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder.
`route_eligible=false`. Chapter factories stay fail-closed.
`PostLevel7Handoff.verified` stays false. No invented OW leave.

**Pin was** `Level7Interior0DClearedReconFixture` — L7 play `0x0D`
`(63,149)`, `room_all_dead=1`, Candle 2, TF 0, keys 2, bombs 6,
`0x68` at `(192,144)`.

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 16px RIGHT, release on start, race UP x=176 | north-arm **`(176,117)`** while block still y=144 | ~24f to the hole's west cell; snap still `(208,96)` at ~32f | **2/2** (`0d_race_v1` / `0d_race_v2`) |
| RIGHT+UP / clips from `(176,117)` during uncovered window | still `(176,117)` tile 179 | hole not walkable even before the 0x68 parks | `0d_race_v2` / `0d_nb_v2` |
| east pocket RIGHT toward x=208 | boxed `(192,141)` tile 177 | L6 east-column UP is not this room | `0d_ec_v1` |

Stairs dest `$EB` / mode **still unobserved**. Stair graphic is visible
NE; the `0x68` parks on it. L6-shaped CheckWarp `(208,93)` is tile
`0x75` under the parked block and does not mode-9 (`0d_poke_v2`;
position poke is recon only, not a walk-on). `(192,93)` holding UP
flashes tile `0x71` and still stays play `0x0D`. Do not treat dump
mode-16 screen `0x7b` as cellar (offscreen sweep, not reproduced).

Dead beliefs dated:
- Dead: vacated `(192,144)` is the stair tile.
- Dead: racing the 32f slide into the NE hole from `(176,117)` —
  RIGHT+UP is tile 179 while the block is still at `(205,144)`.
- Dead: L6 east-column `(208,93)` tile `0x71` is this room's warp —
  `(208,93)` is tile `0x75` / 119 with the 0x68 parked.
- Dead: north-arm center UP reaches the y=93 band (tile 179 at y=117
  across x=128..176).

Leftover stays the **cleared** pin (block unpushed). Do not leave a
post-push state with the `0x68` covering the hole. Do not commit
`Level7InteriorNoseCellarReconFixture` (still play `0x0D`, not cellar).

**Leftover glance:** L7 play **`0x0D` mode 5** `(63,149)`,
`room_all_dead=1`, Candle **2**, TF **0**, keys 2, bombs 6, whistle 1,
food 0, ladder 1, 3 hearts. Block `0x68` `(192,144)`.
`route_eligible=false`.

Next: another walk-on onto the NE hole (not L6 east-column, not
north-arm UP). Then NOSE_CELLAR far-side → PRE_BOSS.

# L7 sitting leftover (rr-8t4.3, 2026-09-03 archive)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.3` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — L7-C 0x0D 16px RIGHT slide (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder.
`route_eligible=false`. Chapter factories stay fail-closed.
`PostLevel7Handoff.verified` stays false. No invented OW leave.

**Pin was** `Level7Interior0DClearedReconFixture` — L7 play `0x0D`
`(63,149)`, `room_all_dead=1`, Candle 2, TF 0, keys 2, bombs 6,
`0x68` at `(192,144)`.

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 0x0D 16px RIGHT on `0x68` (stand 176,144, release on start) | block **`(208,96)`** state=2 | autonomous 1px/2f along y=144 through x=207; 16th pixel snaps north onto the NE stair hole | **2/2** (`0d_push_v20` / `0d_push_v22`) |

Stairs dest `$EB` / mode **still unobserved**. Stair graphic is visible
in the NE after the slide; the `0x68` parks on it. CheckWarps idles at
`(192,141)` / `(176,125)` / `(144,141)` do not warp. East pocket north
limit y=133 at x=192; north-of-plus east limit x=176 (tile 176/177).
Bombs on those walls do not open. UP/LEFT faces of the block are
unreachable (south/east diamond). Aquamentus / shard / OW leave not
reached.

Dead beliefs dated:
- Dead: holding RIGHT is what sends the block to `(208,96)` — release
  on the first 1px and it still completes the 16px slide and snaps.
- Dead: vacated `(192,144)` / `(192,141)` is the stair tile
  (CheckWarps x=192 y=141 tile 177, no mode 9).
- Dead: y=117 diamond bar is the only north wall — east pocket can
  reach y=136 at x=192; the stair hole still sits behind x=176 / y=133.

Leftover stays the **cleared** pin (block unpushed). Do not leave a
post-push state with the `0x68` covering the hole.

**Leftover glance:** L7 play **`0x0D` mode 5** `(63,149)`,
`room_all_dead=1`, Candle **2**, TF **0**, keys 2, bombs 6, whistle 1,
food 0, ladder 1, 3 hearts. Block `0x68` `(192,144)`.
`route_eligible=false`.

Next: enter the NE stair hole without parking the `0x68` on it, or
find another warp pose. Then NOSE_CELLAR far-side → PRE_BOSS.

# L7 sitting leftover (rr-8t4.3, 2026-09-03 archive)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.3` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — L7-C 0x0D kill 5 wallmasters (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder.
`route_eligible=false`. Chapter factories stay fail-closed.
`PostLevel7Handoff.verified` stays false. No invented OW leave.

**Pin was** `Level7Interior0DInteriorReconFixture` — L7 play `0x0D`
`(176,141)`, Candle 2, TF 0, keys 2, bombs 6, whistle 1, food 0, ladder 1,
wooden sword (Rod=0).

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 0x0D kill 5 wallmasters (nudge x≈52 y=117, slash LEFT at x=48, peel inland) | play **`$EB=0x0D`** `room_all_dead=1` `(63,149)` | plus-corner `0x27` peel to west wall one-at-a-time (not statues); x=32 any y grabs; bubbles shove | **2/2** (`0d_wm_v10` / `0d_cleared`) |

RIGHT-push `0x68` `(192,144)` after clear **slides the block to `(208,96)`**
(state=2). Stairs dest `$EB` / mode **unobserved** (y=117 diamond bar walls
the north half; vacated tile is not stairs). Aquamentus / shard / OW leave
not reached.

Dead beliefs dated:
- Dead: plus-corner `0x27` are invuln statues — they are the 5 spawners;
  they sit until a predecessor dies, then peel to the west/south wall.
- Dead: skip parked `x<=16` spawners (that's the live hand).
- Dead: inland-only patrol at x≥64 spawns them (need a west-wall nudge
  x≈42–52).
- Dead: y=141 centre / east-pocket UP reaches the north band (tile 179
  at y=117 across x=64..192).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room0DClearController` / `room_0d_clear_step`
- `level7/hops.py`: `make_room0d_clear_controller`
- `level7/graph.py`: `TIP_OF_NOSE` role updated (5 killable 0x27;
  push dest still unobserved)
- tests: `test_room_0d_clear_peels_west_grab_and_arrives_on_all_dead`

New dest fixture: `Level7Interior0DClearedReconFixture` (gitignored
`.state`) — play `0x0D` `(63,149)` `room_all_dead=1`, `0x68` still at
`(192,144)`, Candle 2, TF 0. `route_eligible=false`.

**Leftover glance:** L7 play **`0x0D` mode 5** `(63,149)`,
`room_all_dead=1`, Candle **2**, TF **0**, keys 2, bombs 6, whistle 1,
food 0, ladder 1, 3 hearts. Block `0x68` `(192,144)`.
`route_eligible=false`.

Next: from cleared leftover, get stairs without overshooting the `0x68`
to `(208,96)`. Then NOSE_CELLAR far-side → PRE_BOSS bomb-east →
AQUAMENTUS. Stay off x=32/208. Do not poke TF.

# L7 sitting leftover (rr-8t4.3, 2026-09-03 archive)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.3` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — L7-C 0x0C bomb-east to TIP_OF_NOSE 0x0D (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder.
`route_eligible=false`. Chapter factories stay fail-closed.
`PostLevel7Handoff.verified` stays false. No invented OW leave.

**Pin was** `Level7Interior0CReconFixture` — L7 play `0x0C` `(120,205)`
S mouth, Candle 2, TF 0, keys 2, bombs 7, selected 5, 3× `0x31`.

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 0x0C bomb-E east-around `(120,165)→(200,165)→(200,141)→(208,141)` RIGHT | play **`$EB=0x0D`** `(32,141)` W mouth | wallmaster `0x27` + bubbles `0x2b` + `0x68` at `(192,144)`; bombs 7→6 | **2/2** (`0c_be_v2`/`v3`) |

Dead beliefs dated:
- Dead: 0x0C y=141 centre RIGHT — boxed `(128,141)` tile 181 (`0c_be_v1`).
- Dead (so far): RIGHT-push the `0x68` from its west face `(176,144)` while
  `room_all_dead=0` (block never moves). Wiki: kill **5 wallmasters** then
  push the mid-right block RIGHT. West mouth `x=32` is a grab trap →
  entrance `0x79`. Plus-corner `0x27` at `(128/160,125/157)` never move and
  take no sword/bomb damage (spawn markers). South face of the `0x68` is
  diamond-walled (tile 179 at `(192,181)` / `(184,181)`).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `L7_ROOM0C_EAST_BOMB` / `L7_ROOM0C_EAST_APPROACH`
- `level7/hops.py`: `make_room0c_east_bomb_controller`
- `level7/graph.py`: `TIP_OF_NOSE ram_id=0x0D`; `DODONGOS_BOSS_PATH`
  RIGHT-bomb `verification=fixture-live`
- tests live-prefix += `TIP_OF_NOSE: 0x0D`

New dest fixture: `Level7Interior0DReconFixture` (west mouth; grabby).
Optional interior pin `Level7Interior0DInteriorReconFixture` `(176,141)`.

**Leftover glance:** L7 play **`0x0D` mode 5** `(32,141)` W mouth,
Candle **2**, TF **0**, keys 2, bombs 6, selected 1 (bombs), whistle 1,
food 0, ladder 1, 3 hearts. Wallmasters + `0x68`. `route_eligible=false`.

Next: from interior leftover, spawn/kill 5 wallmasters (stay off walls),
then RIGHT-push `0x68` `(192,144)` for stairs. Wooden sword only (rod=0
on this recon pin). Do not hug x=32 / x=208.

# L7 sitting leftover (rr-8t4.3, 2026-09-03 archive)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.3` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — L7-C cellar return through forced Digdogger (rr-8t4.3)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs / ladder.
`route_eligible=false`. L7-C chapter stages stay fail-closed.
`PostLevel7Handoff.verified` stays false. No invented OW leave screen.

**Pin was** `Level7Interior4AReconFixture` — L7 cellar `$EB=0x4A` mode 9
`(136,141)`, Candle 2 NATURAL, keys 3, bombs 8, food 0, whistle 1,
ladder 1, TF 0, keese `0x1b`.

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 0x4A west-ladder stairs return | play **`$EB=0x1A` mode 5** `(96,157)` | RIGHT y=141 to east column, LEFT+DOWN drop, LEFT x=48, UP tile 111 | **2/2** (`4a_ret_v7`/`v8`; `Room4AReturnController` `4a_ctl_v1`/`v2` 557f) |
| 0x1A bomb-E south-around `(208,141)` RIGHT | **`$EB=0x1B`** `(32,141)` W mouth | goriya `0x05`; bombs 8→7 | **2/2** (`1a_be_v1`/`v2`) |
| 0x1B KEY-E y=141 RIGHT | **`$EB=0x1C`** `(16,141)` W mouth | digdogger `0x38` HP 240 + statue `0x55`; keys 3→2 NATURAL | **2/2** (`1b_ke_v2`/`v3`) |
| 0x1C Whistle shrink + sword + KILL-CLEAR N | **`$EB=0x0C`** `(120,205)` S mouth | pause-select recorder=5 (seen 1→4→5, no `$0656` poke), 12×B 0x38→3×0x18 HP128, sword; dest 3× `0x31` | **2/2** (`1c_wh_v3`/`v4`; fixture `v5`) |

Dead beliefs dated:
- Dead: walk off the candle pad at y=141 as the stairs return (tile 243).
  Cardinal UP from the pad is also stuck. Return is east-column LEFT+DOWN
  then west-ladder UP `(48,93)`.
- Dead: 0x0C y=141 centre RIGHT reaches the east bomb wall — boxed at
  `(128,141)` tile 181 (`0c_be_v1`). East-around like 0x58 is next.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room4AReturnController` / `room_4a_return_step`,
  `L7_ROOM1A_EAST_BOMB`, `Room1BKeyEastController` / `room_1b_key_east_step`
- `level7/hops.py`: `make_room4a_return_controller`,
  `make_room1a_east_bomb_controller`, `make_room1b_key_east_controller`
- `level7/graph.py`: `CANDLE_PUSH` RIGHT-bomb `verification=fixture-live`;
  `GORIYA_PRE_DIG ram_id=0x1B`; `FORCED_DIGDOGGER ram_id=0x1C`;
  `DODONGOS_BOSS_PATH ram_id=0x0C`; matching KEY / KILL_CLEAR exits
  `verification=fixture-live`. `RED_CANDLE_CELLAR` UP now fixture-live.
- tests live-prefix += `GORIYA_PRE_DIG:0x1B`, `FORCED_DIGDOGGER:0x1C`,
  `DODONGOS_BOSS_PATH:0x0C`

0x1C fight is **probe-2/2** (no HopController yet; pause-select is
env-scripted like L5 `select_b_item_menu`). Chapter
`make_forced_digdogger_controller` stays fail-closed.

New dest fixtures (gitignored `.state` + provenance JSON):
`Level7Interior1AReturnedReconFixture`, `Level7Interior1BReconFixture`,
`Level7Interior1CReconFixture`, `Level7Interior0CReconFixture`.

**Leftover glance:** L7 play **`0x0C` mode 5** `(120,205)` S mouth,
Candle **2**, TF **0**, keys 2, bombs 7, selected 5 (recorder), whistle 1,
food 0, ladder 1, 3× `0x31`. `route_eligible=false`.

Next: 0x0C bomb-east (east-around the y=141 tile-181 mass) → TIP_OF_NOSE.

# L7 sitting leftover (rr-8t4.2, 2026-09-03)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.2` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — 0x1A block push + Red Candle cellar (rr-8t4.2)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs. `route_eligible=false`.
Not on `level7_red_candle_chapter_stages`. Chapter
`RedCandlePickupController` stays fail-closed.

**Pin was** `Level7Interior1AReconFixture` — L7 play `0x1A` `(32,141)`.

Verified (`deaths=0`, `progression/capacity writes=0`):

| walk | dest | notes | evidence |
|------|------|-------|----------|
| 0x1A kill-clear (NE goriya too) then 0x68 UP `(96,144)→(96,128)` then `(136,141)` stairs | cellar **`$EB=0x4A` mode 9** `(128,141)` | drop floor, east ladder `(176,141)`, LEFT onto pad; **ADDR_CANDLE 0→2 NATURAL** at `(135,141)` | **2/2** (`1a_push_v16` / `v18`) |

Dead beliefs dated:
- Dead: L5 south-face UP while a goriya still lives NE of the plus
  (`room_all_dead=0`, block never moves). After the last 0x05 at
  `(160,100)` dies, `room_all_dead` goes nonzero and the L5 recipe
  `(96,162)` UP slides the block.
- Dead (this sitting): walk off the candle pad at y=141 to return
  upstairs (tile 243). Leftover stays cellar `0x4A` `(135,141)` mode 9
  candle 2. Return UP is a later hop.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room1ACandleController`
- `level7/hops.py`: `make_room1a_candle_controller`
- `level7/graph.py`: `RED_CANDLE_CELLAR ram_id=0x4A evidence=fixture-live`;
  `CANDLE_PUSH` DOWN `verification=fixture-live`
- tests live-prefix += `RED_CANDLE_CELLAR: 0x4A`

New dest fixture: `Level7Interior4AReconFixture` (cellar leftover
candle=2). Stairs return to 0x1A still unobserved.

# L7 sitting leftover (rr-8t4.2, 2026-09-03 archive)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.2` stays
`in_progress`. Residual is this file. Did not `bd export` / push.

## 2026-09-03 — MAP 0x18 bomb-north through CANDLE_PUSH 0x1A (rr-8t4.2)

Did not poke `ADDR_CANDLE` / TF / doors / max_bombs. `route_eligible=false`
on every new fixture. Not on `level7_red_candle_chapter_stages`.

**Pin was** `Level7Interior18ReconFixture` — L7 play `0x18` `(120,189)`.

Verified this sitting (`deaths=0`, `progression/capacity writes=0`):

| walk | dest `$EB` | entry | census / notes | evidence |
|------|-----------|-------|----------------|----------|
| `0x18` bomb-N stand `(120,93)` UP | `0x08` | `(120,189)` S mouth | diamond cross, `0x35` cluster; bombs 7→6; doors UP bit | **2/2** (`18_bn_v2/v3`) |
| `0x08` bomb-E south-band `(208,141)` RIGHT | `0x09` | `(32,141)` W mouth | goriya `0x05`+`0x06`, water north, `room_item 0x0f`; bombs 6→5 | **2/2** (`08_be_v2/v3`) |
| `0x09` kill-clear then DOWN | `0x19` | `(120,93)` N mouth | diamond floor, goriya `0x05`; bombs 5→8 natural goriya drops; south shutter **KILL_CLEAR** (doors bit stays LEFT) | **2/2** (`09_down_v2/v3`) |
| `0x19` bomb-E south-around `(208,141)` RIGHT | `0x1A` | `(32,141)` W mouth | 4-diamond plus + `0x68` `(96,144)`; bombs 8→7 | **2/2** (`19_be_v5/v6`) |

Dead beliefs dated:
- Dead: `0x09` DOWN is OPEN on spawn — v1 blocked `(120,189)` tile 170;
  kill-clear first (v2/v3).
- Dead: `0x19` east column drop from y=93 / centre y=141 — NE boxed, y=141
  diamond at x≈112. South-around `(96,141)→(96,189)→(208,189)→(208,141)`.
- Dead (so far): L5-style `0x68` push UP from `(96,162)` moves the L7
  candle block. It does not. Cellar still unobserved.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `L7_ROOM18_NORTH_BOMB` / `L7_ROOM08_EAST_BOMB` /
  `L7_ROOM19_EAST_BOMB`, `Room09DownController` / `room_09_down_step`.
- `level7/hops.py`: `make_room18_north_bomb_controller`,
  `make_room08_east_bomb_controller`, `make_room09_down_controller`,
  `make_room19_east_bomb_controller`.
- `level7/graph.py`: `HIDDEN_RUPEES ram_id=0x08`, `GORIYA_POST_RUPEE
  ram_id=0x09`, `WEST_LOCK_SKIP ram_id=0x19`, `CANDLE_PUSH ram_id=0x1A`,
  all `evidence=fixture-live`; matching BOMB / KILL_CLEAR exits
  `verification=fixture-live`.
- `tests/test_level7_dungeon.py` live-prefix += those four ram_ids.
  `tests/test_level7_hops.py` unit tests for `room_09_down_step` + bomb
  factories.

New dest fixtures (gitignored `.state` + provenance JSON):
`Level7Interior08ReconFixture`, `Level7Interior09ReconFixture`,
`Level7Interior19ReconFixture`, `Level7Interior1AReconFixture`.

# L7 sitting leftover (rr-8t4.2, 2026-09-04)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.2` stays
`in_progress`. Residual is this file, not `rr-tne2-residual.md`.
Did not run `bd`/`git` writes.

**Pin:** `Level7Entrance` — L7 play `0x79` `(120,205)` south mouth, mode 5,
whistle=1 (poked), food=0, TF=0, keys=0, bombs=0, 3 HC. Not the L6-leave
packet. Do not start at Hungry Goriya. Do not poke Food/Whistle/doors/TF.

## This sitting (2026-09-04): recon fixture + `0x6B` east GREEN 2/2

**New disclosed recon fixture — `Level7InteriorReconFixture`.**
`scratch/build_level7_interior_recon_fixture.py` **walks** the fixture-live
`0x79 → 0x69 → 0x6A → 0x6B` chain from `Level7Entrance`
(`EntryNorthDoorController` → `Room69EastController` → `Room6AEastController`,
no `set_state`/teleport), clears the six `0x6B` goriya `0x05`, then discloses
exactly three fixture writes: `ADDR_FOOD` `$065D` 0→1, `ADDR_BOMBS` `$0658`
0→8 (`min(8, max_bombs)`; `ADDR_MAX_BOMBS` read never written), `ADDR_KEYS`
`$066E` 0→4. Settled leftover: L7 play `0x6B` `(136,109)` mode 5, TF 0,
Candle 0, Whistle 1, `deaths=0`, `progression_writes=capacity_writes=0`.
Provenance `Level7InteriorReconFixture.provenance.json`: `development_only:
true`, `natural_entry: false`, `route_eligible: false`, `fixture_only: true`,
every write recorded with before/from/to. Traverse ran under the standard
`UnlimitedHealthAssist` (disclosed in notes as a traversal aid, not a
fixture write; final health byte unchanged at `0x22`). **Not** on any spine.

**`0x6B` (`GORIYA_HINT`) east is GREEN 2/2.** From the fixture,
`scratch/probe_l7_room6b_onward.py --dir RIGHT`: `0x6B` `(16→136,109)` →
ride the `y=109` band east past the central X of diamond blocks → drop the
east column to `y=141` → push the OPEN east doorway → **live dest `$EB=0x6C`
`(16,141)` west mouth, mode 5**, arrived at frame 283, `deaths=0`,
`progression/capacity writes=0`, byte-identical on both trials
(`recordings/6b_right_v1.json` / `6b_right_v2.json`). `0x6C` census:
`0x38` (digdogger-family) + `0x55` (statue/fireball projectile) — this is
`DIGDOGGER_1` (source: RIGHT → `DIGDOGGER_1`, the skippable whistle-split
spur). Graph: `DIGDOGGER_1` promoted `ram_id=0x6C` `evidence=fixture-live`;
`GORIYA_HINT` RIGHT → `DIGDOGGER_1` `verification=fixture-live`.

**`0x6B` LEFT backtrack GREEN 2/2.** `--dir LEFT` → `0x6A` `(224,141)` east
mouth, mode 5, keese `0x1b` present, byte-identical
(`recordings/6b_left_v1/v2.json`). Graph: `GORIYA_HINT` LEFT → `KEESE`
`GateKind.OPEN` `verification=fixture-live`.

**`0x6B` UP is BLOCKED 2/2.** `--dir UP` → Link pins at `(128,93)` on the
`y=93` band, no room change, on both trials
(`recordings/6b_up_v1/v2.json`). A straight centre-x UP push does **not**
reach `OLD_MAN_NOSE` even with the six goriya cleared — the north exit (if
any) is not on the `y=93` centre band. `OLD_MAN_NOSE` stays hypothesis.

**Wired (fixture-live, `route_eligible=false`, NOT on the executable
chapter chain):**
- `level7/path.py`: `room_6b_east_step` + `Room6BEastController`
  (`spec_id="level7_room6b_east"`), `east_of_room6b_ram_id()`, `ROOM_6B*`
  geometry constants. Mirrors `Room6AEastController`.
- `level7/hops.py`: `make_room6b_east_controller` (+ back-filled
  `make_room69_east_controller` / `make_room6a_east_controller` for the
  earlier already-green legs). None are in `level7_red_candle_chapter_stages`
  yet — that chain is still `entry_first_door → hungry_goriya (fail-closed)
  → tip_stairs → red_candle`.
- `level7/graph.py`: `DIGDOGGER_1` `ram_id=0x6C`; `GORIYA_HINT` exits
  re-annotated (LEFT/RIGHT `fixture-live`, UP dead-belief note).
- `tests/test_level7_dungeon.py`: live-prefix test extended with
  `DIGDOGGER_1: 0x6C`.

## 2026-09-05 — EAST mainline extended: `0x6C`→`0x6D`, `0x6B`→`0x5B`, `0x69` west bomb

**Row-6 corridor is `0x69 – 0x6A – 0x6B – 0x6C – 0x6D` (W→E, OPEN doors).**
`0x6D` is the east **dead-end** (STALFOS_KEY). The candle mainline branches
**west of `0x6A`**: `0x69` has a **BOMB wall on its west side → `0x68`**.
(The earlier "LEFT + bomb-UP through `0x6A`" belief is dead — `0x6A` has NO
north exit; the `0x6A` top wall *is* the `y=93` band. `0x69` is the source
`GORIYA_BOMB_HUB`.)

Verified this sitting (all from `Level7InteriorReconFixture`, `deaths=0`,
`progression/capacity writes=0`, `route_eligible=false`):

| walk | dest `$EB` | entry | census / reward | evidence |
|------|-----------|-------|-----------------|----------|
| `0x6B` RIGHT | `0x6C` | `(16,141)` W mouth | digdogger `0x38` + statue `0x55` | **2/2** (`6b_right_v1/v2`) |
| `0x6C` RIGHT | `0x6D` | `(16,141)` W mouth | stalfos `0x2a`, reward `room_item_id 0x19` small_key | **2/2** (`6c_right_v1/v2`) |
| `0x6B` UP (x≈118, y=93) | `0x5B` | `(120,205)` S mouth | bubble `0x40` + statue `0x50` | **2/2** (`6b_north_dest_v1/v2`, frame 189) |
| `0x69` LEFT **bomb** (stand ~`(44,141)` face LEFT) | `0x68` | mid-transition `(188,141)` | interior unobserved; `cur_opened_doors` LEFT bit sets | **2/2** (`69_branch_v2/v3`) |

Dead-ends confirmed 2/2:
- `0x6D` (STALFOS_KEY): RIGHT/UP/DOWN blocked, only LEFT → `0x6C` (`6d_v1/v2`).
  *(Walking over the key in `0x6D` bumped recon `keys` 4→5 — natural pickup.)*
- `0x5B` (OLD_MAN_NOSE): N/E/W/S all walled at `y=141` from the south entry
  (`5b_v1`); the "secret in the tip of the nose" hint room, NOT the mainline.
- `0x69` NORTH: precise x-sweep `104..156` on the `y=93` band — all solid,
  no notch (`69_branch_v2/v3`).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room6BNorthController` / `room_6b_north_step`
  (`level7_room6b_north`), `Room6CEastController` / `room_6c_east_step`
  (`level7_room6c_east`), `Level7BombWall` + `L7_ROOM69_WEST_BOMB`
  (`room=0x69 stand=(44,141) face=LEFT opens_to=0x68`),
  `north_of_room6b_ram_id()` / `east_of_room6c_ram_id()`.
- `level7/hops.py`: `make_room6b_north_controller`,
  `make_room6c_east_controller`, `make_room69_west_bomb_controller`
  (returns `dungeon.bomb_wall.BombWallController(wall=L7_ROOM69_WEST_BOMB,
  level=7)`).
- `level7/graph.py`: `OLD_MAN_NOSE ram_id=0x5B`, `STALFOS_KEY ram_id=0x6D`,
  `KEESE_TRAPS ram_id=0x68`, all `evidence=fixture-live`; `MOLDORMS`
  (== `GORIYA_BOMB_HUB`) gains LEFT-bomb→`KEESE_TRAPS` and DOWN→`ENTRY`
  exits; `DIGDOGGER_1` RIGHT→`STALFOS_KEY` and `GORIYA_HINT` UP→`OLD_MAN_NOSE`
  promoted `fixture-live`.
- `tests/test_level7_dungeon.py`: live-prefix test + `OLD_MAN_NOSE:0x5B`,
  `STALFOS_KEY:0x6D`, `KEESE_TRAPS:0x68`.

## 2026-09-05 (cont.) — branch chain `0x69`→`0x68`→`0x58`; two more recon fixtures

Continuing the branch. All from the recon-fixture chain, `deaths=0`,
`route_eligible=false`:

| walk | dest `$EB` | census / notes | evidence |
|------|-----------|----------------|----------|
| `0x69` west **bomb** → `0x68` | `0x68` KEESE_TRAPS | dark; **4 blade traps `0x49`** (corners) + 4 keese `0x1b`; `open_doorway_mask=13` | **2/2** (`69_branch_v2/v3`) |
| `0x68` **UP** (align x=120) → `0x58` | `0x58` DODONGOS_UPGRADE | dodongo-family `0x31`, `room_item_id 0x0f`, dark; Link spawns bottom `(120,205)` | **2/2** (`68_up_v1/v2`) |

New disclosed recon fixtures (both `development_only`/`fixture_only`/
`route_eligible:false`/`natural_entry:false`, provenance records the WALK
chain; only a `poke_bombs(8)` count top-up before the `0x69` bomb, no
`max_bombs`/Candle/TF/door/health/capacity writes):
- **`Level7Interior68ReconFixture`** — settled 0x68 `~(208,93)`, keys 4,
  bombs 7, Food 1.
- **`Level7Interior58ReconFixture`** — settled 0x58 `(120,205)`, keys 4,
  bombs 7, Food 1.

**`0x58` layout (blind y-band sweep, `58_map_v1`):** dark, dodongo `0x31`.
North of `y≈88` is a **narrow x=120 corridor** (the door channel toward
`0x48`). `y=93` clear x=32..179; `y=109..189` open on the **east half**
(x≈92/116 → 208), west half walled. **No RIGHT transition on any band** —
Link reaches x=208 everywhere but does not cross. `cur_opened_doors`
flipped to `8` (RIGHT) and `keys` 4→3 during the sweep → there is a
**locked/kill-gated EAST door** (mainline → GORIYA_COMPASS) that needs the
dodongo `0x31` killed and/or a key. `0x58` also reaches **`0x48`** (bubble
`0x40` + `0x4f`, key-gated — the optional BOMB_UPGRADE `0x48`, NOT
mainline; the earlier probes' "LEFT→0x48" was really the north x=120
channel + knockback).

Wired: `level7/path.py` `Room68NorthController` / `room_68_north_step`
(`level7_room68_north`), `north_of_room68_ram_id()`, `ROOM_68*` consts;
`level7/hops.py` `make_room68_north_controller`; `level7/graph.py`
`DODONGOS_UPGRADE ram_id=0x58` + `KEESE_TRAPS` UP→`DODONGOS_UPGRADE`
`fixture-live`; test live-prefix += `DODONGOS_UPGRADE:0x58`.

## 2026-09-05 (cont. 2) — `0x58`→`0x59` (GORIYA_COMPASS)

**`0x58` EAST → live `$EB=0x59` (GORIYA_COMPASS)** — **2/2 byte-identical**
(`58_east_v2/v3`). `0x59` `(16,141)` W mouth, mode 5, census **goriya
`0x05` + `0x06`**. **OPEN door, keys unchanged** (the earlier "gated"
belief was wrong — the 58-map key-drop was a *north* door to `0x48`).

Key `0x58` facts:
- The 3× `0x31` (hp 240) are **invulnerable roamers** — sword + 7 bombs did
  nothing (`58_clear_v1`). Do **not** try to clear the room; dodge them.
- `0x58` has a **central structure** walling `y=141` west of `x~129`. The
  east door route climbs the east-open column: `(120,165) → (200,165) →
  (200,141) → push RIGHT`. `room_item_id 0x0f` stays uncollected (behind
  the block / not on the critical path).
- New fixture **`Level7Interior59ReconFixture`** (settled `0x59` `(16,141)`,
  keys 4 / bombs 7 / Food 1; disclosed: bombs count top-up only).

Wired: `level7/path.py` `Room58EastController` (`level7_room58_east`),
`east_of_room58_ram_id()`, `ROOM_58*` consts; `level7/hops.py`
`make_room58_east_controller`; `level7/graph.py` `GORIYA_COMPASS ram_id=0x59`
+ `DODONGOS_UPGRADE` RIGHT→`GORIYA_COMPASS` and DOWN→`KEESE_TRAPS`
`fixture-live`; test live-prefix += `GORIYA_COMPASS:0x59`.

## 2026-09-05 (cont. 3) — `0x59`→`0x49` (GORIYA_BUBBLE); **LADDER blocker**

**`0x59` UP → live `$EB=0x49` (GORIYA_BUBBLE)** — **2/2 byte-identical**
(`59_up_v2/v3.json`, arrived frame 2329; `59_ctl_v3/v4` = the wired
`Room59UpController` 2/2, arrived frame 1618). From
`Level7Interior59ReconFixture`: kill-clear the goriya `0x05`/`0x06` (sets
`cur_opened_doors` bit 3 = **UP**, `open_doorway_mask`→10; boxes Link at
`(48,125)`). The route around the central mass (fills ~`x100..190` /
`y118..165`) is a **perimeter waypoint micro**: rise the west side to the
`y~100` open band → west to `x~44` → rise to the `y~93` top band → cross
east to `x=120` → push UP (pre-push `(118,93)`). `0x49` `(120,205)` S
mouth, mode 5, census **goriya `0x05` + keese `0x1b` + bubble residual
`0x2b`** (+ transient boomerang `0x5c`). keys 4 / bombs 7 / Food 1
unchanged, `deaths=0`, `progression/capacity writes=0`.

New disclosed recon fixture **`Level7Interior49ReconFixture`** — settled
`0x49` `(120,205)`, keys 4 / bombs 7 / Food 1. **No fixture writes at all**
(counts already fine); `development_only`/`fixture_only`/
`route_eligible:false`/`natural_entry:false`.

**`0x49` (GORIYA_BUBBLE) is fully mapped — and the mainline is BLOCKED here
without the Stepladder.**
- Exactly two doors: **DOWN → `0x59`** (back, 1/1) and **UP → DIGDOGGER_2**
  (source). LEFT and RIGHT are **walled 2/2** (Link pins at `x=32` / `x=208`
  on `y=141`).
- The UP door **bit opens** on the goriya kill-clear, but a **full-width
  horizontal water moat** (~`y120`, colliding tile **`0xF4`**) walls the
  entire room — fine x-sweep `x=16..224` step 4, **every column blocked at
  `y=133`**, no land bridge anywhere.
- **Diagnostic (one-off, no fixture saved):** poking `ADDR_LADDER 0x0663`
  →1 lets Link walk straight north across the moat to the `x=120` door
  threshold (tile 118) and into the top half. **The moat is a Stepladder
  gate.**
- The recon-fixture chain descends from the `Level7Entrance` poke pin
  (Whistle poked, Food/TF/keys/bombs minimal) and **carries no Ladder**
  (`ADDR_LADDER`=0). The disclosed-write budget is Food/bombs/keys *count*
  top-ups only — a Ladder poke is a **capability write, out of scope**.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room59UpController` / `room` consts `ROOM_59*`,
  `north_of_room59_ram_id()`. Phases clear→rise1→west→rise2→cross→push,
  mirrors `Room58EastController`.
- `level7/hops.py`: `make_room59_up_controller`.
- `level7/graph.py`: `GORIYA_BUBBLE ram_id=0x49 evidence=fixture-live`;
  `GORIYA_COMPASS` UP→`GORIYA_BUBBLE` `KILL_CLEAR` `verification=fixture-live`;
  `GORIYA_BUBBLE` DOWN→`GORIYA_COMPASS` `fixture-live`, UP→`DIGDOGGER_2`
  annotated with the moat/Ladder note (dest still hypothesis).
- `tests/test_level7_dungeon.py`: live-prefix += `GORIYA_BUBBLE:0x49`.

## 2026-09-03 — ladder poke + candle mainline through MAP (`rr-8t4.2`)

Did not STATUS-promote. Did not edit `STATUS.md`. Did not poke
`ADDR_CANDLE` / TF / doors / max_bombs. `route_eligible=false` on every
new fixture. Not on `level7_red_candle_chapter_stages`.

**Disclosed recon fixture `Level7Interior49LadderReconFixture`.** From
`Level7Interior49ReconFixture` (L7 play `0x49` `(120,205)` S mouth, keys 4
/ bombs 7 / Food 1 / Candle 0 / Whistle 1 / `ADDR_LADDER=0`): one write
`{field:ladder, address:0x0663, from:0, to:1}`. Provenance:
`development_only:true`, `fixture_only:true`, `route_eligible:false`,
`natural_entry:false`. The Stepladder is an L4 item the real Survival
route already earned; this recon pin never carried L4 inventory.

Verified this sitting (`deaths=0`, `progression/capacity writes=0`):

| walk | dest `$EB` | entry | census / notes | evidence |
|------|-----------|-------|----------------|----------|
| `0x49` UP (ladder, x=120) | `0x39` | `(120,205)` S mouth | digdogger `0x38` + statue `0x55`; skip fight | **2/2** (`49_up_v1/v2`; `Room49UpController` `49_ctl_v4/v5` 5491f) |
| `0x39` LEFT (OPEN) | `0x38` | `(208,141)` E mouth | goriya `0x05`/`0x06`, diamond floor, compass `room_item 0x0f` uncollected | **2/2** (`39_left_v2/v3`; `Room39LeftController` `39_ctl_v1/v2` 364f) |
| `0x38` KEY-UP | `0x28` | `(120,205)` S mouth | GRUMBLE GRUMBLE NPC `0x36` + bubble `0x40`; keys **4→3** natural | **2/2** (`38_up_v6/v7`; `Room38UpController` `38_ctl_v1/v2` 7722f) |
| `0x28` bait feed + UP | `0x18` | `(120,189)` | MAP `room_item 0x17`; goriya+keese+bubble; **Food 1→0 NATURAL** | **2/2** (`28_feed_v1/v2`; probe only — no HopController yet) |

Stepladder notes (dead beliefs dated):
- Dead: strafe on the `0x49` water. The ladder crosses on the facing axis
  only (`49_ctl_v1` pinned at `(64,117)` holding RIGHT). Align x on land,
  hold UP across, then align on the north band `y=109`. Kill keese
  (AliveRule.TYPE, HP stays 0) or the north door stays shut.
- Dead: `0x39` SW statue corridor. `(48,189)` boxes (tile 151). Centre
  column `x=120` UP to `y=141`, then LEFT.
- Dead: `0x38` centre UP. Interior `y=149` diamond row blocks UP at
  `x=120` / `104` / `88` / `200`. Recollect the east mouth pocket
  `x=208`, rise to `y=93`, cross to `x=120`, KEY-UP.

Hungry Goriya: already-owned Bait B-slot **6** (selected_item poke of an
owned item, Food byte untouched). Walk to `(120,141)`, tap B. NPC `0x36`
despawns, Food 1→0, north door opens, dest MAP `0x18`. HUD B-slot falls
back to whistle=5 after consume.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room49UpController` / `room_49_up_step`,
  `Room39LeftController` / `room_39_left_step`, `Room38UpController` /
  `room_38_up_step`.
- `level7/hops.py`: `make_room49_up_controller`,
  `make_room39_left_controller`, `make_room38_up_controller`.
- `level7/graph.py`: `DIGDOGGER_2 ram_id=0x39`, `GORIYA_PRE_HUNGRY
  ram_id=0x38`, `HUNGRY_GORIYA ram_id=0x28`, `MAP ram_id=0x18`, all
  `evidence=fixture-live`; matching UP/LEFT exits `verification=fixture-live`.
- `tests/test_level7_dungeon.py` live-prefix += those four ram_ids.
  `tests/test_level7_hops.py` unit tests for the three new step functions.

New fixtures (gitignored `.state` + provenance JSON):
`Level7Interior49LadderReconFixture`, `Level7Interior39ReconFixture`,
`Level7Interior38ReconFixture`, `Level7Interior28ReconFixture`,
`Level7Interior18ReconFixture`.

**Stepladder added to `LEVEL7_ROUTE.md` required capabilities.**

## Onward toward Red Candle — resume point

**Pin:** `Level7Interior4AReconFixture` — L7 cellar **`$EB=0x4A` mode 9**
`(135,141)` (or `(136,141)`), **Candle 2 NATURAL**, keys 3, bombs 8,
Food 0, Whistle 1, Ladder 1, TF 0, keese `0x1b`. `route_eligible=false`.

L7-B Red Candle pickup is **2/2 fixture-live**. Next (not this sitting):
stairs return UP to `0x1A` (pad does not walk off at y=141; tile 243).
Then L7-C forced Digdogger. Do **not** poke `ADDR_CANDLE`. Do not STATUS.

Fixture chain to regenerate `.state` files (all gitignored): parent
`Level7InteriorReconFixture` (`build_level7_interior_recon_fixture.py`) →
`Level7Interior68ReconFixture` (`probe_l7_room68_onward.py --save-fixture`)
→ `Level7Interior58ReconFixture` (`probe_l7_room58_onward.py --save-fixture`)
→ `Level7Interior59ReconFixture` (`probe_l7_room58_east.py --save-fixture`)
→ `Level7Interior49ReconFixture` (`probe_l7_room59_up.py --save-fixture`)
→ `Level7Interior49LadderReconFixture`
(`build_level7_interior49_ladder_fixture.py`, ADDR_LADDER 0→1)
→ `Level7Interior39ReconFixture` (`probe_l7_room49_up.py --save-fixture`)
→ `Level7Interior38ReconFixture` (`probe_l7_room39_left.py --save-fixture`)
→ `Level7Interior28ReconFixture` (`probe_l7_room38_up.py --save-fixture`)
→ `Level7Interior18ReconFixture` (`probe_l7_hungry.py --save-fixture`)
→ `Level7Interior08ReconFixture` (`probe_l7_room18_bomb_north.py --save-fixture`)
→ `Level7Interior09ReconFixture` (`probe_l7_room08_onward.py --save-fixture`)
→ `Level7Interior19ReconFixture` (`probe_l7_candle_chain.py --save-fixture` 0x09 DOWN)
→ `Level7Interior1AReconFixture` (`probe_l7_room08_onward.py --room 0x19 --save-fixture`)
→ `Level7Interior4AReconFixture` (`probe_l7_candle_push.py --push UP --save-fixture`).
Optional dest pins: `Level7Interior78ReconFixture` (`probe_l7_room68_down.py
--save-fixture`), `Level7Interior48ReconFixture` (`probe_l7_room58_north.py
--save-fixture`).

## 2026-09-03 — L7-B side rooms: `0x68` DOWN + `0x58` north

Did not walk `0x49` UP. Did not poke `ADDR_LADDER` / `ADDR_CANDLE` /
`ADDR_MAX_BOMBS`. Candle-mainline narrative above is unchanged.

| walk | dest `$EB` | entry | census / reward | evidence |
|------|-----------|-------|-----------------|----------|
| `0x68` DOWN | `0x78` ROPES_KEY | `(120,77)` N mouth | ropes `0x28` + floor `small_key 0x19`; keys 4 (not picked); dead-end | **2/2** (`68_down_v3/v4`, `68_ctl_v1/v2` frame 297/298) |
| `0x58` UP KEY | `0x48` BOMB_UPGRADE | `(120,205)` S mouth | bubble `0x40` + `0x4f`; old-man "I BET YOU'D LIKE TO HAVE -100"; keys 4→3; `max_bombs` stays 8 | **2/2** (`58_north_v2/v3`, `58_ctl_v1/v2` frame 337/338) |

`0x68` DOWN: blade traps `0x49` in the four corners. OccupancyWalker
poisoned the grid on knockback (v2 stood `(174,149)`, 12 misses). Waypoint
micro: peel west to `x=160` (off the east trap column), drop `y=141`
(between trap rows `y~93`/`y~189`), align `x=120`, push DOWN. If knocked
onto the `y~189` trap row off-x, rise first.

`0x58` north: 3× invuln `0x31` (hp 240) — dodge, do not kill. A central
2-block mass walls the `x=120` column around `y=141`. OccupancyWalker to
`(120,93)` boxed at `(122,165)` (25 misses). East-around: climb `y=165`,
RIGHT `x=160`, UP `y=93`, align `x=120`, push the KEY door (natural key
spend). `0x48` is a **dead-end** 100-rupee bomb-capacity old-man room;
do **not** write `max_bombs` (`capacity_writes=0`).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room68DownController` / `room_68_down_step`
  (`level7_room68_down`), `Room58NorthController` / `room_58_north_step`
  (`level7_room58_north`), `south_of_room68_ram_id()` /
  `north_of_room58_ram_id()`.
- `level7/hops.py`: `make_room68_down_controller`,
  `make_room58_north_controller`.
- `level7/graph.py`: `ROPES_KEY ram_id=0x78`, `BOMB_UPGRADE ram_id=0x48`,
  both `evidence=fixture-live`; `KEESE_TRAPS` DOWN and `DODONGOS_UPGRADE`
  UP KEY `verification=fixture-live`.
- `tests/test_level7_dungeon.py`: live-prefix += `ROPES_KEY:0x78`,
  `BOMB_UPGRADE:0x48`.

New dest fixtures (disclosed writes: none; `development_only` /
`fixture_only` / `route_eligible:false` / `natural_entry:false`):
- `Level7Interior78ReconFixture` — settled `0x78` `(120,77)`, keys 4 /
  bombs 7 / Food 1 / `max_bombs` 8.
- `Level7Interior48ReconFixture` — settled `0x48` `(120,205)`, keys 3 /
  bombs 7 / Food 1 / `max_bombs` 8.

## Dead beliefs

1. Dead: `0x6B` UP at centre-x (x=128) reaches `OLD_MAN_NOSE` — the notch
   is at **x≈118**; centre-x is solid (blocked 2/2 at `(128,93)`).
2. Dead: `0x6C` (`DIGDOGGER_1`) is a "skippable spur" off the mainline —
   the room **is** on the mainline (only the digdogger *fight* is
   whistle-skippable); it leads to the STALFOS_KEY dead-end.
3. Dead: the candle mainline is "LEFT + bomb-UP through `0x6A`". `0x6A` has
   no north exit. The branch is the **`0x69` west bomb wall → `0x68`**.
4. Dead: `0x6A` north has a bombable wall (9 bomb attempts x=64..192 on the
   `y=93` band, which is the north wall — nothing opened; those attempts
   also mis-positioned Link, but the `0x69` west-bomb branch is now the
   confirmed route, so `0x6A`-north is retired).
5. Dead: OccupancyWalker to the `0x68` south / `0x58` north door — blade
   traps and `0x31` knockback poison the grid (stood `(174,149)` / boxed
   `(122,165)`). Waypoint micros. Also dead: `0x58` north is a straight
   `x=120` walk from the south mouth — a central 2-block mass walls that
   column around `y=141`; east-around first.

## Prior sittings (still standing)

- `0x6A` east GREEN 2/2 (`Room6AEastController`, `recordings/
  l7_room6a_east_room6a_east_v2/v3.json`): rise west column `y=93`, cross,
  drop east column `x=200`→`y=141`, push OPEN doorway plane `x=224`. Room
  unlit (Candle 0); keese `0x1b` never block. Dead: `0x6A` `y=141` centre
  band (walled `x=48` tile `0xB1`); dead: `0x6A` east is KEY/KILL_CLEAR
  (it is OPEN).
- `0x69` east GREEN 2/2 (`east_route_step`: rise `y=109`, RIGHT `x=204`,
  DOWN `y=141`, push `x=208`). Dead: `cur_opened_doors & RIGHT` as a
  walkability test — it never sets on an OPEN L7 doorway; per-pixel
  occupancy boxes Link in after four graded misses (use waypoints).

---

## L7 chapter handoff (integrator reference)

`route_eligible=false` on every fixture. Shared spine still fail-closed at
`level7_pond_drain_entry`. Public `--through` targets unchanged:
`level7-entry`, `level7-red-candle`, `level7`.

---

## L7-A — topology + entry preparation

- **chapter id:** `rr-8t4.1` / `level7-entry`
- **evidence label:** mixed. Offline graph + Bait/post-L6 interface =
  **hypothesis**. Start-based `0x53→0x52` inland-left micro =
  **fixture-live**. Post-L6 bait prefix `0x22→0x32→0x33→0x23→0x24→0x25` =
  **fixture-live**. Pond `0x42` overworld screen **reached 2/2 geometry-only**
  (`OverworldToLevel7PondController` from `PostSwordStart`). Recon drain
  with an `ADDR_WHISTLE` poke enters L7 play **`0x79` `(120,205)`**
  (`Level7Entrance` pin). Natural drain from the L6 leave is still
  unobserved (`rr-8t4.4`). Stop at fixture-live; integrator owns
  natural-segment / spine-green.

- **exact predecessor (updated 2026-09-02, Phase 1):** the L6 fanfare leave is
  **measured and verified** — OW `0x22` `(112,125)` TF `0x3F` keys 2 bombs 8
  rupees 42, `selected_item=2` (arrows), Whistle 1 Food 0 Candle 0, 8 HC full
  (`--through level6-exit` 2/2, `l6_exit_ow.json` + `l7p1_l6exit.json`,
  byte-identical). Carried as `MEASURED_POST_L6_EXIT`, a shared
  `zelda_i.overworld.stitch.OverworldHandoff`, **`verified=True`**
  (`route_eligible=false`). The deleted `(120,221)` / 80R / White-Sword poke
  loadout was never a fanfare. The live L6 interior residual play `0x09`
  `(56,109)` Rod=0 is not an L7 start.

- **required inventory/capabilities:** TF `0x3F`, Whistle ≥1, Rod ≥1, Bow ≥1,
  sword. Food 0 until natural 60R Bait at shop hyp `0x34`. Candle remains 1
  until Red Candle. Geometry-only pond mapping may run without Whistle.

- **ordered internal stage names and controller factories:**
  1. `level7_post_l6_overworld` — `make_post_l6_overworld_controller(handoff, hops)`
     (`OverworldHandoff` gate + default `hops=POST_L6_TO_BAIT_HOPS`). Handoff
     `verified=True` since Phase 1, so it **walks `0x22→0x25` green**
     (`path_complete`, phase DONE, 1577f, `writes=0`) on a continuous power-on.
     The `_left_mouth` latch keeps the `0x22` mouth-tile spawn from tripping
     the re-entry refusal on frame 1.
  2. `level7_bait_purchase` — **Survival:** `make_survival_bait_purchase_controller`
     (`l7_hops(survival=True)`) — one disclosed `ADDR_FOOD` write (`$065D`→1)
     in place of the natural 60R buy, since the natural L6→shop OW route is a
     mountain-locked pocket (bead `rr-8t4.4`); `SPINE_L7_RUPEE_RETOPUP` still
     tops the owned rupee **count** 42→60 for the cost paid; passes green.
     **Clean:** `make_bait_purchase_controller` still fails closed at
     `bait_shop_geometry_unobserved` (shop `0x34` geometry / cave xy / buy
     policy all still unobserved). Disclosed in `docs/ASSIST_CONTRACT.md`.
  3. `level7_pond_drain_entry` — `make_pond_entry_controller()`
     (still fail-closed on the spine). Recon drain from `OW_L7Pond` with a
     disclosed `ADDR_WHISTLE` poke is **green** (`drain_v2`, L7 play `0x79`
     `(120,205)`). A natural-whistle controller from the L6 leave is `rr-8t4.4`.

  Recon-only (not a spine stage): `OverworldToBaitShopController` from
  `Level6ExitOverworld`; `OverworldToLevel7PondController` from
  `PostSwordStart` (no Whistle).

- **exact endpoint predicate:** `level7_entry_stop` — live L7 play in
  **`0x79`**, TF `0x3F`, Whistle and Food owned, `route_eligible` and
  `evidence` in `{natural-segment, spine-green}`. Room id is live
  (`fixture-live`) but evidence is not in that set, so the predicate
  still fails closed.

- **expected inventory deltas:** Food 0→1. No TF change. Keys/bombs unchanged
  on OW. Candle stays 0 (Blue Candle not on the mainline; Red Candle is the L7
  dungeon item). Disclosed **Survival** writes at `level7_bait_purchase`:
  rupee **count** 42→60 (`SPINE_L7_RUPEE_RETOPUP`) + one `ADDR_FOOD`→1
  (`SurvivalBaitPurchaseController`, bead `rr-8t4.4`). No rupee-grant /
  Whistle / door / TF writes. Clean does the Food gain via a natural buy
  (still fail-closed until `rr-8t4.4`).

- **known dead beliefs / first missed RAM claim:**
  - Dead: `hop10_ay` DOWN from `0x53` `(224,173)` (v9). Inland-left first.
  - Dead: start-`0x77` pond walk as the spine post-L6 path.
  - Dead: OW `0x22` as proven L6 leave; live L6 prefix `0x09` as L7 start.
  - Dead: `(112,125)` on `0x22` is a dead spot — it *is* the measured leave
    (the mouth tile). Re-entry only fires on a fresh UP into it / mode 16;
    the `_left_mouth` latch handles the spawn-on-mouth case.
  - Dead: `0x32` `(120,61)` `off_north` DOWN (`l7_bait_from_l6`). Corridor is x=112.
  - Dead: `0x33` RIGHT @ y=141 → `0x34` (`l7_bait_32ax` leftover `(208,141)`).
  - Dead: `0x24` DOWN at `(16,189)` (`l7_bait_33up`), `(160,189)`
    (`l7_bait_24belt`, north-ladder x), `(208,189)` (`l7_bait_24se` SE).
  - Dead: occupancy xmin=14 west pocket `(0,141)`; SW occupancy box `(25,181)`.
  - Dead: `0x24↓0x34` (south wall sealed at x=16 / 160 / 208).
  - Current leftover (Phase 1, continuous power-on): play `0x25` `(0,141)`
    west mouth, `level7_post_l6_overworld` **green** through it
    (`l7p1_entry_v2.json`).
  - **2026-09-02 recon (`scratch/{sweep_25_armos,probe_24_to_shop,
    probe_33_south_to_shop}.py`, `--from-state Level6ExitOverworld`):**
    `0x33`/`0x23`/`0x24`/`0x25` are a mountain-bounded desert pocket with **no
    south exit** — `0x25` south walled (only hidden exit north x≈208→`0x15`,
    wrong way); `0x24` south walled at every x, 10-Armos sweep reveals no shop
    stair (top-right = Bracelet); `0x33` south walled at x∈{120,160,208},
    `0x33→0x34` RIGHT still walled. External grid confirms L6=C3=`0x22`,
    Bracelet=E3=`0x24`, Bait shop=E4=`0x34`. Source `U L×3 U×3` enters `0x34`
    walking **north from `0x44`**; the shop stair is the top-row-middle Armos
    *on `0x34`*.
  - `0x32` south edge also solid (x∈{56,80,96,192}); `0x32` exits are only
    NORTH→`0x22` and EAST→`0x33`. **The whole `0x22/0x32/0x33/0x23/0x24/0x25`
    region is a mountain-locked pocket** — no southward route to the row-4/5
    band with pond `0x42` / shop `0x34`.
  - **Retire** the `0x25→0x35→0x34` plan and the `0x23/0x24/0x25` detour
    entirely. The shop `0x34` and pond `0x42` are both entered walking north
    out of the **western forest band** (`LEVEL7_POND_APPROACH_HOPS`
    territory, reached from *start*): `…0x64→0x54→0x44↑0x34`,
    `…0x54→0x53→0x52→0x42↑`. Route-owner decision: buy Bait before L6, or
    loop the post-L6 pocket exit north/west around the mountains, or take a
    longer post-L6 leg. The spine's `0x22→…→0x25` "green" walk is a dead spur.
    **Tracked as bead `rr-8t4.4`** (natural L6→shop OW route). Until it lands,
    the Survival spine sets Food directly (`SurvivalBaitPurchaseController`,
    disclosed in `docs/ASSIST_CONTRACT.md`).

- **L7 start pin — captured (recon).** `Level7Entrance.state`: L7 play
  **`0x79` `(120,205)`** south mouth, whistle=1 (poked), food=0, TF=0,
  3 HC. Drain recipe: `OW_L7Pond` → poke `$065C` + B-slot 5 → 12×B →
  idle ~240f → walk dry bed; stairs `(96,132)` tile 114. Harness:
  `scratch/probe_l7_pond_drain.py`. `LEVEL7_ENTRY_STOP.screen=0x79`
  `evidence=fixture-live` `route_eligible=false` — spine still fail-closed.
  Not natural-entry (PostSwordStart + whistle poke, no L5/L6 TF). Natural
  whistle-carrying pond leftover is still blocked by the `0x22` mountain
  pocket (`rr-8t4.4`) and the missing post-L5 OW checkpoint.

- **Why the pond leftover had whistle=0 (not a failed L5 pickup).** The
  geometry walk is `PostSwordStart` → `LEVEL7_POND_HOPS` (`0x77→…→0x42`).
  It never enters L5, so `$065C` stays 0 and TF stays 0. On the Survival
  spine the Recorder **is** earned: `attach_level5_whistle_suffix` /
  `--through level5-whistle` 1/1, and `--through level6-exit`
  (`l7p1_l6exit.json` `final.whistle=1`, `MEASURED_POST_L6_EXIT`). There is
  no `Level5ExitOverworld`; whistle-owning pins are L5-interior
  (`Level5WhistleFrom77` cellar `0x04`). Post-L5 OW settle is Lost Hills
  `0x0B`, then L6. Post-L6 fanfare dumps Link in the `0x22` mountain
  pocket with Whistle 1 but **no south path** to pond `0x42`. So: pickup
  works; the pond walk that exists does not go through L5; the tape that
  owns Whistle cannot walk to the pond. Natural leftover = leave L5 to
  OW and take the west-forest hops to `0x42` **before** L6.

- **fixture provenance:** Phase 1 is a **continuous power-on** tape, not a
  save-state fixture. `--through level7-entry` (`recordings/l7p1_foodfix.json`)
  drives power-on → measured L6 fanfare exit → `level7_post_l6_overworld`
  green `0x22→0x25` → `level7_bait_purchase` green (Survival Food fixture) →
  fail-closed at `level7_pond_drain_entry`. `set_state=0`, `deaths=0`. Pond
  recon remains `recordings/l7_dnp_pond_53.json` leftover play `0x52` `(112,181)`.
  Entry pin: `Level7Entrance` / `OW_L7Pond` (`l7_pond_drain_drain_v2.json`).

- **files changed (Phase 1) / public target:** `level7/entry.py` (verified
  handoff + `_left_mouth` latch + `SurvivalBaitPurchaseController`),
  `level7/hops.py` (`survival=` swap), `level7/spine.py`
  (`SPINE_L7_RUPEE_RETOPUP`, `l7_hops(survival=True)`),
  `spine/survival.py` (`spine_final_fields(ram)`, `topup_owned_rupees`,
  `rupee_retopup`), `dungeon/ops.py` (`poke_rupees`, `poke_food`,
  `apply_owned_inventory(rupees=)`), `assist.py` (`poke_food` re-export),
  `scripts/run_survival_spine.py`, `tests/test_level7_hops.py`,
  `tests/test_dungeon_ops.py`, `docs/{LEVEL7_ROUTE,ASSIST_CONTRACT}.md`,
  `docs/tasks/{l7-handoff,ow-handoff,rr-tne2-residual}.md`.
  Public target: **`level7-entry`** (green through `level7_bait_purchase` on
  the Survival spine).

---

## L7-B — entry through Red Candle

- **chapter id:** `rr-8t4.2` / `level7-red-candle`
- **evidence label:** **fixture-live** row-6 corridor
  `0x79→0x69→0x6A→0x6B→0x6C→0x6D` + `0x6B`→`0x5B` spur + branch
  `0x69`─bomb→`0x68`─UP→`0x58`─EAST→`0x59`─UP→`0x49`─UP(ladder)→`0x39`─LEFT→
  `0x38`─KEY-UP→`0x28`(Hungry Goriya, Food 1→0)─UP→`0x18`(MAP)─bomb-N→
  `0x08`─bomb-E→`0x09`─KILL-S→`0x19`─bomb-E→`0x1A`(CANDLE_PUSH)─stairs→
  `0x4A` Red Candle cellar (ADDR_CANDLE 0→2 NATURAL); side rooms
  `0x68` DOWN `0x78` and `0x58` KEY-UP `0x48`. Cellar return to 0x1A
  still unobserved.
- **predecessor:** `Level7Entrance` pin — L7 play `0x79` `(120,205)`. Inventory
  is the poke loadout (Whistle 1, Food 0, TF 0), **not** the L6-leave packet.
  Hungry Goriya still needs Food; isolate with the recon fixture or `rr-8t4.4`.
- **required:** Food ≥1 (Hungry Goriya), Whistle, bombs for wall skips, keys
  for three locks (skip fifth via bombs)
- **recon fixture:** `Level7InteriorReconFixture` (`scratch/
  build_level7_interior_recon_fixture.py`) — WALKS `0x79→0x6B` from
  `Level7Entrance`, clears the six `0x6B` goriya, then discloses `ADDR_FOOD`
  0→1, `ADDR_BOMBS`→8, `ADDR_KEYS`→4. Settled L7 play `0x6B` `(136,109)`
  TF 0 Candle 0 Whistle 1. `development_only`/`fixture_only`/
  `route_eligible:false`/`natural_entry:false`; no Candle/Whistle/TF/door/
  room/health/capacity writes. NOT on any spine.
- **stages / factories:**
  1. `level7_entry_first_door` — `make_entry_first_door_controller()`
     (live north OPEN → `$EB=0x69`; `route_eligible=false`).
  1a. `level7_room69_east` — `Room69EastController` /
     `make_room69_east_controller()` clears the `0x69` goriyas, walks the
     **OPEN** east doorway to live `$EB=0x6A` `(16,141)` (2/2, 3,809f,
     `writes=0`). Not a spine stage.
  1b. `level7_room6a_east` — `Room6AEastController` /
     `make_room6a_east_controller()` walks the unlit `0x6A` KEESE room west
     mouth → **OPEN** east doorway to live `$EB=0x6B` `(16,141)` (2/2, 482f,
     `writes=0`, `deaths=0`). Waypoint: rise `y=93`, cross, drop
     `x=200`→`y=141`, push `x=224`. Not a spine stage.
  1c. `level7_room6b_east` — `Room6BEastController` /
     `make_room6b_east_controller()` (2/2) walks `0x6B` west mouth → OPEN
     east doorway to live `$EB=0x6C` (`DIGDOGGER_1`, `(16,141)`, 283f).
     Ride `y=109` east past the central X, drop `x=200`→`y=141`, push
     `x=224`. **Assumes `0x6B` goriya cleared upstream.**
  1d. `level7_room6c_east` — `Room6CEastController` /
     `make_room6c_east_controller()` **(new 2026-09-05, 2/2)** walks `0x6C`
     west mouth → east door to live `$EB=0x6D` (`STALFOS_KEY`: stalfos
     `0x2a` + small_key `0x19`, a **dead-end**). Ride `y=141`; a digdogger
     bump nudges Link through.
  1e. `level7_room6b_north` — `Room6BNorthController` /
     `make_room6b_north_controller()` **(new 2026-09-05, 2/2)** walks `0x6B`
     west mouth → OPEN north notch at **x≈118** → live `$EB=0x5B`
     (`OLD_MAN_NOSE`: bubble `0x40` + `0x50`, a **dead-end** hint spur).
  1f. `level7_room69_west_bomb` — `make_room69_west_bomb_controller()`
     **(new 2026-09-05, 2/2)** = `dungeon.bomb_wall.BombWallController(wall=
     L7_ROOM69_WEST_BOMB, level=7)`: `0x69` west BOMB wall (stand
     `(44,141)` face LEFT) → live `$EB=0x68` (`KEESE_TRAPS`: dark, 4 blade
     traps `0x49` + 4 keese). **This is the candle-path branch.** Needs
     bombs + bomb on B. Not a spine stage.
  1g. `level7_room68_north` — `Room68NorthController` /
     `make_room68_north_controller()` **(new 2026-09-05, 2/2)** walks `0x68`
     (align x=120 on top band) → OPEN north door → live `$EB=0x58`
     (`DODONGOS_UPGRADE`: 3× invuln `0x31`, `room_item_id 0x0f`). Not a
     spine stage.
  1h. `level7_room58_east` — `Room58EastController` /
     `make_room58_east_controller()` **(new 2026-09-05, 2/2)** climbs the
     `0x58` east-open column `(120,165)→(200,165)→(200,141)` → OPEN east
     door → live `$EB=0x59` (`GORIYA_COMPASS`: goriya `0x05`/`0x06`).
     Dodges the 3× invuln `0x31`. Not a spine stage.
  1i. `level7_room68_down` — `Room68DownController` /
     `make_room68_down_controller()` **(new 2026-09-03, 2/2)** walks `0x68`
     OPEN south door → live `$EB=0x78` (`ROPES_KEY`: ropes `0x28` + floor
     `small_key 0x19`, a **dead-end**). Peel `x=160`, drop `y=141`, push
     DOWN. Not a spine stage.
  1j. `level7_room58_north` — `Room58NorthController` /
     `make_room58_north_controller()` **(new 2026-09-03, 2/2)** east-arounds
     the `0x58` central mass, KEY north door (keys 4→3) → live `$EB=0x48`
     (`BOMB_UPGRADE`: bubble `0x40` + `0x4f`, 100-rupee old-man
     bomb-capacity **dead-end**). Do not write `max_bombs`. Not a spine stage.
  1k. `level7_room59_up` — `Room59UpController` /
     `make_room59_up_controller()` **(2/2)** 0x59 kill-clear + perimeter
     → live `$EB=0x49`. Not a spine stage.
  1l. `level7_room49_up` — `Room49UpController` /
     `make_room49_up_controller()` **(new, 2/2)** 0x49 goriya+keese clear
     then UP across the water moat at x=120 (Stepladder) → live `$EB=0x39`.
     Requires `ADDR_LADDER=1` on the recon fixture. Not a spine stage.
  1m. `level7_room39_left` — `Room39LeftController` /
     `make_room39_left_controller()` **(new, 2/2)** skip Digdogger, centre
     column then LEFT → live `$EB=0x38`. Not a spine stage.
  1n. `level7_room38_up` — `Room38UpController` /
     `make_room38_up_controller()` **(new, 2/2)** east-pocket rise x=208
     then KEY-UP (keys 4→3) → live `$EB=0x28` Hungry Goriya. Not a spine
     stage. Hungry feed+UP to MAP `0x18` is probe-2/2 only.
  1o. `level7_room18_north_bomb` — `make_room18_north_bomb_controller()`
     **(new, 2/2)** 0x18 bomb-UP stand (120,93) → live `$EB=0x08`.
     Recon-wired only.
  1p. `level7_room08_east_bomb` — `make_room08_east_bomb_controller()`
     **(new, 2/2)** 0x08 south-band bomb-RIGHT → live `$EB=0x09`.
     Recon-wired only.
  1q. `level7_room09_down` — `make_room09_down_controller()` **(new, 2/2)**
     0x09 kill-clear + south shutter → live `$EB=0x19`. Recon-wired only.
  1r. `level7_room19_east_bomb` — `make_room19_east_bomb_controller()`
     **(2/2)** 0x19 south-around bomb-RIGHT → live `$EB=0x1A`.
     Recon-wired only.
  1s. `level7_room1a_candle` — `make_room1a_candle_controller()`
     **(new, 2/2)** 0x1A kill-clear + 0x68 UP + stairs → cellar `$EB=0x4A`
     mode 9, ADDR_CANDLE 0→2 NATURAL. Recon-wired only. Chapter
     `RedCandlePickupController` stays fail-closed.
  2. `level7_entry_to_hungry_goriya` — `make_entry_to_goriya_controller()`
     (fails `hungry_goriya_requires_food` if Food=0; else room unobserved)
  3. `level7_tip_of_nose_stairs` — `make_tip_stairs_controller()` (blocker + ledger notes)
  4. `level7_red_candle_pickup` — `make_red_candle_controller()`
     (`ADDR_CANDLE` 1→2 natural; room unobserved)
  Executable chapter chain (`level7_red_candle_chapter_stages`) is unchanged:
  `entry_first_door → entry_to_hungry_goriya (fail-closed) → tip_stairs →
  red_candle_pickup`. Stages 1a–1n are recon-wired only.
- **endpoint:** `level7_red_candle_stop` — Candle==2, TF `0x3F`, Whistle
  retained, Food==0, exact live room. Room id `None` → fail closed.
- **resume point:** L7-C took the cellar return. Leftover is now
  `Level7Interior0CReconFixture` play `0x0C` `(120,205)` Candle 2.
  See the rr-8t4.3 sitting at the top of this file.
- **expected deltas:** Food 1→0 at Hungry Goriya; Candle **0→2** at cellar
  (Blue Candle was never bought).
  Ledger hyp net (dungeon): keys +1+1−1−1+1−1, bombs several −1 wall skips.
  Prefer bomb north of Map over locked east (fifth lock).
- **dead beliefs:** fifth lock required; source RAM room ids as stop specs;
  N path is Moldorms (live dest `0x69` is goriya `0x05`); `0x69`/`0x6A`/`0x6B`
  east are key/kill-clear doors (all OPEN — `cur_opened_doors` stays 0 on
  every live L7 doorway, but a **bombed** door DOES set the bit — `0x69`
  west); per-pixel occupancy boxes Link in — use waypoints; `0x6B` UP at
  centre-x (x=128) reaches `OLD_MAN_NOSE` (notch is x≈118); `0x6C`/`0x6D`
  are on the mainline as far as the STALFOS_KEY dead-end (`0x6C` fight is
  whistle-skippable, room is not); candle mainline is "bomb-UP through
  `0x6A`" (`0x6A` has no north exit — the branch is `0x69` **west** bomb →
  `0x68`); `0x49` UP without Stepladder; strafe on the `0x49` moat;
  `0x38` centre-column UP (y=149 diamond wall — east pocket x=208);
  `0x09` DOWN is OPEN without kill-clear; `0x19` east drop from y=93;
  `0x68` UP while a goriya still lives (`room_all_dead=0`).
- **fixture:** `Level7InteriorReconFixture` (recon only). `route_eligible=false`.
- **public target:** **`level7-red-candle`**.

---

## L7-C — Red Candle through shard leave

- **chapter id:** `rr-8t4` clear / `level7`
- **evidence label:** **fixture-live** through TIP_OF_NOSE `0x0D`
  kill-5 `room_all_dead=1` (cellar return `0x4A→0x1A` bomb-E `0x1B`
  KEY-E `0x1C` whistle+kill N `0x0C` bomb-E `0x0D` kill-5). Nose-cellar
  / Aquamentus / shard / OW leave still **hypothesis**.
- **predecessor:** L7-B cellar `Level7Interior4AReconFixture` (Candle 2,
  Food 0, Whistle 1). Recon TF is 0 (poke-loadout chain), not the L6
  packet `0x3F`.
- **required:** Whistle (forced Digdogger), Candle 2, incoming heart
  container count recorded
- **stages / factories:**
  1. `level7_forced_digdogger` — `make_forced_digdogger_controller()`
     (still fail-closed). Recon-wired: `make_room4a_return_controller`
     2/2, `make_room1a_east_bomb_controller` 2/2,
     `make_room1b_key_east_controller` 2/2. 0x1C fight is probe-2/2 only.
  2. `level7_aquamentus_heart` — `make_aquamentus_heart_controller()`
  3. `level7_shard_and_settled_leave` — `make_level7_shard_leave_controller()`
- **endpoint:** `level7_complete_stop` — TF `0x7F`, Candle 2, Whistle ≥1,
  hearts +1 and full, measured settled OW leave. Leave screen `None` → fail closed.
- **expected deltas:** TF `0x3F→0x7F` (`0x40`), heart containers +1, full
  hearts, deaths 0. Post-fanfare OW leftover **UNMEASURED**.
- **dead beliefs:** walk off candle pad y=141 as stairs return; 0x0C
  y=141 centre RIGHT (tile 181 at x=128); 0x0D west mouth is safe (grab
  trap); 0x0D `0x68` RIGHT-pushes while wallmasters live; plus-corner
  `0x27` are invuln statues (they are the 5 spawners); holding RIGHT is
  what sends the block to `(208,96)` (release-on-start still snaps);
  vacated `(192,141)` is the stair tile. Boss type Aquamentus still
  hypothesized (verify).
- **fixture:** `Level7Interior0DClearedReconFixture` leftover play `0x0D`
  `(63,149)` `room_all_dead=1` Candle 2. `route_eligible=false`.
- **resume point:** enter the NE stair hole without parking the `0x68`
  on it. Then NOSE_CELLAR / PRE_BOSS / AQUAMENTUS.
- **public target:** **`level7`**.

---

## Stitch notes for L8

Expected L7 leave (still unmeasured, do not invent):

- TF `0x7F` (bits through shard 7)
- Candle 2 (Red)
- heart containers +1 vs L7 incoming, full `lo==hi`
- Whistle retained; Food 0 after Hungry Goriya
- post-fanfare OW leftover still **UNMEASURED** (screen, x/y, keys, bombs,
  rupees, selected). L8 must keep `PostLevel7Handoff.verified=false` until
  this packet is measured.

Seam for integrator: `L7_THROUGH`, `L7_STOPS`, `continue_level7_spine`.
`spine/survival.py` is yours. Wave A did not attach it.

Did not STATUS-promote.

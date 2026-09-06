# Residual — rr-6o7.2 L8-B Magical Key

**Spine bead:** `rr-6o7.2` (`in_progress`). Acceptance is power-on
`--through level8-magic-key` from the L8-entry leftover (`0x7E` `(120,205)`
TF `0x7F`).  `rr-6o7.1` is closed 2/2.  Do not STATUS.  Do not push unless
asked.

## 2026-09-05 (cont.) — `level8_magic_key_stairs` live + wired; 4/4 stages

`Level8MagicKeyStairsController` promoted from the fail-closed stub to the
live controller and wired (`make_magic_key_stairs_controller` →
`make_magic_key_stairs_live_controller`; `MEASURED_LEVEL8_ENTRY_TOPOLOGY.
magic_key_room = 0x1F`).

**Iteration harness:** `scripts/magic_key_lab.py` (`--pin` builds
`MagicKeyStairsEntryLive` from a power-on `--through level8-magic-key` stop
at the 0x1F frontier; bare run drives only the stairs controller from it,
with a per-60f block/pose trace + `ascii_room` dumps — mirrors
`[[gohma-iteration-pins]]`).

**What was wrong / fixed** (`level8/magic_key.py`):
- The `_push_policy` used E1's rejected "stage west / south-face UP" line.
  From the power-on 0x1F arrival the `GenericDungeonRoomController` clear
  patrol leaves Link boxed **south-east of the centre diamond** (`(144,165)`,
  tile 178 wall to the west) — the old line wedged there for its whole
  budget.  0x1F is a 5-block diamond (blocks at x∈{96,112,128,144,160},
  rows y∈{112..176}; x≥176 and x≤80 columns fully open; tilemap dump).
  New `_push_policy`: route out via the open **x=192** vertical lane up to
  the **y=93** north band (the diamond's top block at `(128,112)` only
  clears above y~109), west to the `0x68` column (x=96), then hold **DOWN**
  so Link shoves the block **south** into the open `(96,160)` cell — the
  probe_l8_1f_magic_key E2 slide direction, not E1's UP.
- `_cellar_policy` no longer idles at the pad hoping the key spawns; it
  walks the full **DOWN→RIGHT→UP→LEFT** pickup loop around the pit
  (waypoints `(entry_x,189) (176,189) (176,141) (136,141)`, frame-paced at
  180f like `_cellar_walk`), then hands to `magic_key_cellar_return_step`
  once `ADDR_MAGIC_KEY` rises.
- `slid` detection widened to `abs(by - b0y) >= 8` (either direction).

**magic_key_lab result** (`recordings/l8mk_stairs_try5.json`, power-on pin,
Survival assist): **success**, MK 0→1, back in play `0x1F` `(96,157)`,
5811f / 34000, TF `0x7F`, keys 1, bombs 15, `writes=0`, deaths 0.
Phases: clear ~2760f → north-lane push (block `(96,144)`→`(96,160)`) →
centre stairs → cellar `0x0F` pickup + two-ladder return ~1620f.

Suite **853 → 856** (`test_level8_cellar.py` +3 live-controller cases;
`test_level8_{north_column,east,south}.py` stub asserts updated).

**Power-on `--through level8-magic-key`: 1/1 GREEN** (`l8mk_poweron_v3`,
`set_state=0`, first playthrough).  All 4 magic-key stages spine-green:

```
level8_north_manhandla_bomb  ok  1848f
level8_darknut_key_up        ok  9185f
level8_blue_gohma            ok  2712f
level8_magic_key_stairs      ok  9229f   MK 0->1, writes=0, phase cellar
final: L8 play 0x1F (96,157) m5  keys 1  bombs 15  rupees 56  TF 0x7F
```

deaths 0, progression_writes 0, capacity_writes 0.  `inventory_assist` is
only `SPINE_L8_RETOPUP` (bombs->16 / keys->2 count top-up, ASSIST_CONTRACT).
The stairs stage ran 9229f from the live arrival vs 5811f from the lab pin
(RNG differs) -- comfortably inside the 34000 budget.  **2/2 byte-consistent**
(`l8mk_poweron_v4`: stairs 9229f, MK 0->1, writes 0, deaths 0, same final
fields).

## 2026-09-05 — 3/4 magic-key stages spine-green; only `level8_magic_key_stairs` left

Power-on `--through level8-magic-key` before this sitting failed on the
**first** magic-key stage (`level8_north_manhandla_bomb`): the cumulative
spine arrives at the L8 entry with **bombs=0 / keys=1** (the L7 leave carries
none, the burn spends the trip), and the fixture-live north-column
controllers assume a healthy dungeon inventory.

**After this sitting** (`recordings/l8mk_v2.json`, power-on, `set_state=0`):

```
level8_burn_bush_enter        ok  321f   level8_live_entry
level8_north_manhandla_bomb   ok  1848f  arrived_0x5e
level8_darknut_key_up         ok  9185f  arrived_0x1e     <- 0x3E blocker fixed
level8_blue_gohma             ok  2712f  arrived_0x1f (16,141)  14 shots
level8_magic_key_stairs       -- blocked_unverified (fail-closed stub)
final: L8 0x1F (16,141) m5  keys 1  bombs 15  rupees 55  B=arrows  TF 0x7F
```

3 of the 4 magic-key stages are now spine-green from power-on.  The only
remaining blocker for `--through level8-magic-key` is the fail-closed
`level8_magic_key_stairs` stub (the `Level8MagicKeyStairsController` cellar
pickup, below).

### Landed

1. **`SPINE_L8_RETOPUP`** (`level8/spine.py`) — the sanctioned ASSIST_CONTRACT
   bomb/key count top-up, applied before `level8_north_manhandla_bomb` and
   `level8_darknut_key_up` (mirrors L2/L3/L7).  `topup_owned_inventory`
   writes bombs→16 / keys→2 + B-slot bombs; keys→2 is enough because 0x5E's
   natural key pickup (+1) lands before the two key doors (0x4E→0x3E,
   0x2E→0x1E).  Verified: `level8_north_manhandla_bomb` now **passes** on the
   power-on spine (`recordings/l8mk_topup.json`, `l8mk_gohma.json`).

2. **`Level8BlueGohma1EController`** (`level8/magic_key.py`, new) — promotes
   `scratch/probe_l8_1e_gohma.py`.  The Level-6 `0x1C` eye-edge arrow policy
   (RAM `0x03C7`) re-aimed at the live `0x1E` body (RAM type `0x33`; colour
   not asserted).  Naturally owned bow + wooden arrows, B cycled through the
   pause menu (`PauseSelectController`, no `$0656` write), no L6 `0x1C`
   detour, no arrow poke.  Kill → RIGHT kill-clear shutter (`cur_opened_doors`
   RIGHT-bit rising edge, same byte as L6 post-Gleeok) → play `0x1F`.
   Fixture check `Level8Interior1EReconFixture`: success, 3145f, 24 shots.
   **Power-on-green** (`l8mk_v2.json`: 2712f, 14 shots, arrived 0x1F
   `(16,141)`).  Wired as stage 3 (`make_blue_gohma_controller`).

3. **`Level8MagicKeyStairsController`** (`level8/magic_key.py`, new — WIP,
   **NOT wired**; `make_magic_key_stairs_controller` stays fail-closed).
   Promotes `probe_l8_1f_magic_key`.  Fixture-checked from
   `Level8Interior1FReconFixture` (`probe_l8_darknut_clear_check --stages
   stairs`, s1-s8):
   - **works**: sword-clear the `0x1F` mixed census
     (`GenericDungeonRoomController`, pols voice `0x16` in
     `type_only_enemy_types` — it hops at hp=0 and otherwise hitstuns the
     push, E1); stage west, drop fully south of the centre `0x68`, latch
     `south_reached`, hold UP to slide it (the earlier oscillation back to
     the staging lane on a small northward push-drift was the stall);
     walk the revealed `(128,141)` stairs and descend to cellar `0x0F`
     (`0x0f@2008` reached).
   - **OPEN**: the cellar `0x0F` key pickup.  The centre stairs Link rides
     down are a warp tile — any step that stays on them re-warps to play
     `0x1F` (`l8_mk_cellar_rewarped_no_key`).  `probe_l8_1f_magic_key` E2
     (`recordings/l8_1f_magic_key_fixture_20260904_E2.json`) shows the real
     path is a full loop: from `(128,141)` **DOWN to y≈189** (south is open
     at the entry x, only *brick at the pad x=136*), **RIGHT to x≈176**,
     **UP to y≈141**, **LEFT to x≈136** → key at f567 → then
     `magic_key_cellar_return_step` from `(136,141)`.  The waypoint version
     of this (s6) crept y185→151 over 5k frames pressing UP at x=176 — too
     slow / wall-fighting.  Needs the actual E2 waypoint cadence and a
     bigger cellar budget.  When it lands: set
     `MEASURED_LEVEL8_ENTRY_TOPOLOGY.magic_key_room = 0x1F` (settled play
     frame after the two-ladder return; `CELLAR_RETURN_DEST`) and re-wire
     `make_magic_key_stairs_controller`.

4. **`level8_darknut_key_up` 0x3E bomb-north approach — FIXED** (was the
   power-on blocker before the fixes below landed: `l8mk_topup.json` /
   `l8mk_gohma.json` both dead-deterministic at 14192f, end 0x3E `(104,149)`
   — the old **8k `BombWallController` timeout**, not a clear timeout).  The
   5-6 HP128 blue-darknut (`0x0C`) fights are high-variance; on a bad
   shield-RNG power-on run the patrol clear leaves Link off-centre and south
   of the `0x3E` statue row (~y=141), and the bare y-first `(120,109)`
   approach rammed a statue.  Fixes (`level8/north_column.py`):
   `BOMB_NORTH_APPROACH_3E` → `((120,157),(120,109))` (rally the statue-free
   centre column south then north); `_bomb` `BombWallController` 8k→16k;
   `_DARKNUT_CLEAR_FRAMES = 16_000` for `CLEAR_5E_SPEC`/`CLEAR_3E_SPEC` (a
   parametrised `_sword_clear_spec`; Manhandla rooms keep 5k);
   `Level8DarknutKeyController` 30k→55k, `Level8NorthManhandlaController`
   20k→26k.  `l8mk_v2.json`: `level8_darknut_key_up` **passes** on power-on,
   9185f, arrived 0x1E.

### DONE — `Level8MagicKeyStairsController` cellar `0x0F` key pickup

Closed 2026-09-05 (cont.): live + wired, power-on 1/1.  See the sitting note
at the top of this file.  `scripts/magic_key_lab.py` is the iteration harness
(`--pin` → `MagicKeyStairsEntryLive`, bare run drives only stage 4).

## Frontier

`rr-6o7.2` acceptance is **MET**: `--through level8-magic-key` power-on 1/1
(`l8mk_poweron_v3`), MK room `$EB` live `0x1F`, `ADDR_MAGIC_KEY` 0→1 natural,
TF `0x7F`, Candle 2, Bow + wooden arrows, keys 1 / bombs 15, deaths 0,
progression/capacity 0.  Planner may close `rr-6o7.2` and STATUS.

Next is **`rr-6o7.3`** (L8-C, `--through level8`): compose the Gleeok suffix
`0x1F → 0x1E → 0x2E → 0x3E → 0x3F → cellar 0x2F → 0x4C → bomb-N 0x3C →
Gleeok kill + heart → north shutter 0x2C → shard 0x80 → measured post-L8 OW
leave`.  Every hop in that chain is already fixture-live 2/2
(`docs/LEVEL8_ROUTE.md`); the work is promoting them to live spine stages via
`level8/suffix.py` + a measured post-fanfare OW leftover, then
`PostLevel8Handoff`.  `make_gleeok_passage_controller` /
`make_shard_leave_controller` are still fail-closed.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0, position_writes 0,
tf_poke 0.  Inventory assist: `SPINE_L8_RETOPUP` bomb/key count top-up only
(ASSIST_CONTRACT), listed in every run report's `inventory_assist`.

## Suite

853 passed, 3 deselected (`uv run python -m pytest nes/zelda_i/tests`).

---

## 2026-09-04 audit sitting (no live hops; suite 676 passed)

Bug eval of `level8/**` + spine rows. Four findings, all fixed, each with a
regression test. No live run, no new hop claim, no re-walk of `0x3E`→OW.

1. `level8/triforce.py` `Level8Shard2CController.arrived` greened on **any**
   state already carrying TF `0x80`, with no room/mode check — a post-shard
   pin (`Level8PostShardOWReconFixture`, TF `0xFF`) reported a zero-input
   `success` with `evidence="fixture-live"`. Now `tf_in` latches on the first
   stepped frame and a pre-set bit fails `l8_shard_already_taken`; success is
   a rising edge (or fanfare). Reported as `tf_in`.
2. `level8/gleeok.py` success disjunct `item_gone = room_item_id != 0x1A`
   could green the fight with no heart container (F6/F7 evidence says the id
   *stays* `0x1A` after the pickup, so it never fired live). Now it needs the
   watched falling edge (`saw_heart_item`).
3. `level8/hops.py` `magic_key_ok` never passed `magic_key_before`, so the
   chapter stop degraded to "owns a Magical Key". Added a `SpineHop.before`
   hook that latches `ADDR_MAGIC_KEY` at the top of the chapter; the stop now
   needs 0→1.
4. `level8/gleeok_entry.py` `RAM_CLAIM` (published as the stage `policy`)
   claimed the bomb stand `(120,105)`; the executed `BOMB_NORTH_STAND` is
   `(120,93)` (`(120,105)` is 0x3E's stand and overshoots the 0x4C alcove).
   String corrected.

Verified clean: `GLEEOK_FOUR_HEAD_OBJECT_TYPE == 0x45` everywhere; no L8 code
or doc revives the refuted `(144,93)` aim (`bush.py` keeps it only as
`REFUTED_BUSH_AIM`); every measured budget is inside its `max_frames`.

New: `level8/suffix.py` — the ordered Gleeok-suffix composition
(`LEVEL8_SUFFIX_GATES`, eight rows: the seven live gates plus the mode-9
`0x2F` no-input settle that separates the K3/K4 stairs arrival `(208,141)`
from the P1/P2 cross pin `(192,93)`). It is inert: `suffix_stages` returns
`()` for every lineage that is not both `route_eligible` and
`natural_predecessor`, and the only exported lineage
(`FIXTURE_LINEAGE_LEVEL8_SUFFIX`) is neither, so `_clear_stages` still yields
the fail-closed `level8_return_passage` row.
`make_gleeok_passage_controller` unchanged. `L8_THROUGH` not greened;
`--through level8` still stops at `level8_entry_live`. Even a composable
lineage cannot green it — `level8_clear_stop` still needs a complete
`Level8ClearEndpoint`, and the fixture-lineage OW pin may not fill one
(`tests/test_level8_suffix.py`).

## Frontier pin

`Level8PostShardOWReconFixture` — **fixture-lineage**, not Survival-true
post-L8 leave (this pin is not Survival-from-L7). OW play `$EB=0x6D`
`(96,93)` mode 5, keys 8, bombs **5**, Magic Key **1**, TF **`0xFF`**,
bow 1, arrows 1, rupees 247, Magical Sword 3, Candle 2, B = bombs,
hc **4** (`health=0x33`). L8 bush screen, south of the bush.

Dest hop 2/2: probe `l8_3c_north` T2/T3, 296 controller / 356 census,
play **`0x2C` `(120,205)`**, `room_item=0x1B`. Shard+OW 2/2: T5/T6, 43
shard frames, 936 env, TF `0x7F|0x80=0xFF`. `position_writes=0`.
`tf_poke=False`.

Also pinned: `Level8Interior2CTriforceReconFixture` (pre-shard TF room).

## How we got here

From `Level8Interior3CKillReconFixture` play `0x3C` `(32,181)`. Do not
DOWN (`0x4C`). OccupancyWalker not used.

- T1: UP west-wall boxed `(32,133)` tile 179.
- T2 dest=None / T3 dest `0x2C`: `NORTH_BAND_Y=141` x-align then UP
  center aisle. Live dest **`0x2C`**.
- T4: shard pickup, 400f fanfare idle still mode 18.
- T5/T6: `OW_IDLE=2500`, OW `0x6D` `(96,93)`.

Policy in `level8/triforce.py`. Factories `make_north_3c_controller`
(dest live `0x2C`) and `make_shard_2c_controller`.
`route_eligible=False`. `assumed_0x2c=False`.
`make_gleeok_passage_controller` stays fail-closed. Not on `L8_THROUGH`.

### Prior leftover (0x3C post-kill, now predecessor)

`Level8Interior3CKillReconFixture` — L8 play `$EB=0x3C` `(32,181)`,
keys 8, bombs 5, MK 1, TF `0x7F`, hc 4, body `0x45` gone, doors 12.

## Next live boundary

Measured Survival-true post-L8 OW leave still needs the L7→L8 power-on
predecessors (`rr-8t4.3`, `rr-6o7.1`). Do not treat
`Level8PostShardOWReconFixture` as that handoff. Do not poke MK / TF /
doors / position.

## Remaining L8

1. Power-on `--through level8` still blocked on `rr-8t4.3` then
   `rr-6o7.1` then composing this suffix.
2. `rr-6o7.3`: Survival-true post-L8 OW leave (this leftover is
   fixture-lineage only).

## Rooms live / selected min

L8 selected min-through (MK chapter + Gleeok suffix, Book/Map/Compass omitted):
`0x7E 0x6E 0x5E 0x4E 0x3E 0x2E 0x1E 0x1F 0x0F` + `passage_east pols_west gleeok 0x3C triforce 0x2C` = **13**.
Live unique: those nine plus `0x3F` `0x2F` `0x4C` `0x3C` **`0x2C`**.
Unvisited play: **0**. **L8 13/13 rooms visited including TF.** Leftover
is fixture-lineage OW `0x6D`.

L9 selected Magical Key min (Red Ring `0x07` out, plus Ganon/Zelda): **28**.
Live dest hops: `0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52 0x42 0x32` = 10.
Poked `0x76` settle only. **L9 10/28 = 36%** (11/28 = 39% if counting poked `0x76`).

L7 suffix **4/4 fixture-live** (`rr-n91a` closed): dest `0x29` 2/2, then
`0x2A`/`0x2B`/OW 1/1 from the poke-cellar pin. 0x0D walk-on still open
on `rr-8t4.3`. Survival OW leave (TF `0x7F`) unmeasured.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0, position_writes 0,
tf_poke 0.

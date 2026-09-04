# Residual — rr-6o7.2 L8-B Magical Key (OW 0x6D after L8 shard, fixture-lineage)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

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

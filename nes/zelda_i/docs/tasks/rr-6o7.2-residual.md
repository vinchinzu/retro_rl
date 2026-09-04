# Residual — rr-6o7.2 L8-B Magical Key (OW 0x6D after L8 shard, fixture-lineage)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

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

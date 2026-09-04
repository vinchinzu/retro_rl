# Residual — rr-6o7.2 L8-B Magical Key (0x0F two-ladder return)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only: `natural_entry=false`, `route_eligible=false`.
Do not STATUS. Nothing was pushed. HEAD `de9b4c0f`, origin `97852034`,
main is 74 local commits ahead.

This sitting researched the return. No live hop. No `level8/cellar.py`, no
`scratch/probe_l8_0f_cellar_return.py`. `probe_l8_1f_magic_key.py`
`_cellar_return` is still a no-op that records `deferred: two_ladder_return`.

Skills: `zelda-session`, `zelda-assisted-route`, `predict-path`.

## First command

```bash
bd ready -l zelda_i -l spine
# rr-6o7.2 is already in_progress. Claim it. Do not start rr-6o7.3 / rr-5eb2 live.

QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scratch/probe_l8_0f_cellar_return.py \
  --from-state Level8InteriorMKReconFixture \
  --tag 20260904_F1 --infinite-life
```

The probe file does not exist yet. Write it. Drive the one-frame policy from
`level8/cellar.py` (also unwritten) so unit tests and the live loop share
the same step.

## Frontier pin

`Level8InteriorMKReconFixture` (custom_integrations, SHA256 in provenance).
L8 mode-9 cellar `$EB=0x0F` `(136,141)` facing 2, tile **36** (pad, not pit).
Keys 8, bombs 6, bow 1, arrows 1, rupees 247, Magical Sword 3, Candle 2,
Magic Key **1**, TF `0x7F`, B = arrows. `room_item_id=0x0B`. Four HP-0 keese
`0x1B` (not census). `next_room=0x1F`. `returned_to_play=false`.

Inbound 2/2: `probe_l8_1f_magic_key` E2/E3, 9074 frames, from
`Level8Interior1FReconFixture`. Sword-clear 0x1F, west `0x68` `(96,144)`
slides DOWN to `(96,160)`, centre stairs `(128,141)`, MK 0→1 on the pad.

**Dead:** E1c naive LEFT at y=141 leftover `(112,141)` tile **250**.

## First live prediction (write this in the report before the run)

From `(136,141)` mode 9:

1. Cardinal **DOWN** off the pad to floor `y=189`.
2. **LEFT** on the floor to `x=48`.
3. **UP** west ladder to `(48,93)`.
4. Hold UP until colliding tile ∈ `0x70..0x73`, then idle. Tile `0x6F` at
   `(48,93)` does not CheckWarp.
5. Stop at first settled **play** (mode 5, not transitioning). Record `$EB`,
   xy, doors, census. Idle 60f. Do not swing.

Never LEFT or RIGHT while `y < 189-2` and not already on `x=48`.
`goto()` is x-first: never `goto(48, 93)` from the pad in one shot. Waypoints
only: `(136,189)` then `(48,189)` then `(48,93)`.

If colliding tile is 250, that is the miss. Halt with RAM/PNG.

If cardinal DOWN from the pad does not decrease y (~40f) or the tile is 243
(L7 candle-pad analog) or 250 (L6 east-column analog), halt. Do not silently
switch. Declared follow-up only: L1/L7 east-drop from the same leftover
`(136,141)`, RIGHT to east column (inbound used `x=176`; L1 uses `192`),
LEFT+DOWN, floor LEFT, UP `(48,93)`. Cardinal DOWN at `(192,141)` is also
tile 250.

OccupancyWalker is banned on this hop. Default `OccupancyGrid` treats unknown
cells as free and BFS walks the pit. Hardcoded one-frame, same as L1
`bow_pickup` EXIT, L6 `exit75`, L7 `room_4a_return_step`.

## Dest (do not invent `$EB`)

Do not assume Gleeok `0x3C`. That is ROM LevelInfo boss, several hops later,
via a **different** cellar (`0x2F`, 0x3F ↔ 0x4C). See
`docs/tasks/rr-5eb2-gleeok-model.md`.

Best current hypothesis for first play after the west ladder: **mode 5,
`$EB=0x1F`** (stairs origin, fixture `next_room=0x1F`, L1 both-ladders return
to origin). If RAM disagrees, the RAM is the census.

Walkthrough Gleeok suffix after Magical Key (hypothesis, not live):

```text
0x1F LEFT -> 0x1E --DOWN x2--> 0x3E --KILL RIGHT--> passage_east (hyp 0x3F)
  --STAIRS cellar 0x2F--> pols_west (hyp 0x4C) --BOMB UP--> gleeok (hyp 0x3C)
  --UP--> triforce (hyp 0x2C)
```

Do not start the Gleeok fight. `GLEEOK_FOUR_HEAD_OBJECT_TYPE` stays `None`.
ROM claims body `0x45`; L8 Gohma already broke a source type (`0x34` vs live
`0x33`).

## Wiring after 2/2

New file `nes/zelda_i/level8/cellar.py` (path.py is 237 lines and owns
fail-closed factories; `north_column.py` is 0x7E→0x1E only).

Keep all of these fail-closed / unset:

- `make_magic_key_stairs_controller` (`UnverifiedLevel8PathController`)
- `make_gleeok_passage_controller` (stage `level8_return_passage`)
- `topology.magic_key_room` on `LIVE_RECON_LEVEL8_TOPOLOGY`
- `L8_THROUGH` greening
- `test_magic_key_and_gleeok_stages_are_named_and_blocked` requires every
  MK and Gleeok spine stage to fail. Do not wire the cellar return into
  those factories.

Fixture-live factory, `route_eligible=False`, dest filled from live `$EB`.
Unit tests: leftover `(136,141)` emits DOWN not LEFT/RIGHT; floor `(100,189)`
LEFT; west `(48,189)` UP; tile 250 idles as fail. Port
`test_bow_pickup_does_not_right_into_pit_at_y141` and
`test_room_4a_return_step_east_drop_then_west_ladder`.

Keys 8→8, bombs 6→6 unless a new gate spends one. MK stays 1. TF stays
`0x7F`. deaths 0, progression_writes 0, capacity_writes 0. Survival refill
only. Leave proof is RAM + `zelda_i.screen_glance`, not MP4.

## Remaining L8 (after this hop)

`rr-6o7.2` still open until power-on. Fixture-live can continue:

1. This hop: cellar 0x0F → first settled play. Stop.
2. Walk the Gleeok suffix from that leftover. One room or one gate per
   sitting. Do not fight Gleeok.
3. `rr-5eb2`: live census of body/head types and HP, then the fight from
   the written model (south-stand dy=22, Magical Sword 9 hits
   160+96+96+96, do not chase `0x46`, budget 20000f). Close `rr-5eb2`
   only on live types.
4. `rr-6o7.3`: four-head Gleeok, heart, TF `0x80`, measure post-L8 OW leave.

True `--through level8-entry` / `level8-magic-key` / `level8` stay named and
red. Default `UNMEASURED_POST_L7_HANDOFF` fails closed. Blockers:

- `rr-8t4.3` L7-C Aquamentus / shard / measured post-L7 leave (blocked on
  `rr-8t4.2` Hungry Goriya and `rr-n91a` 0x0D stairs decode)
- `rr-6o7.1` natural bush 0x6D burn from that leave

Durable north-column `0x7E→0x1E` already exists in `level8/north_column.py`.
Gohma / MK stairs / Gleeok factories stay fail-closed.

## Integrity

No poke of Magic Key, TF, doors, Gleeok HP, arrows, or rupees. Do not grant
Map/Book/Compass.

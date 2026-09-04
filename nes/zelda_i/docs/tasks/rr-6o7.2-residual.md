# Residual — rr-6o7.2 L8-B Magical Key (play 0x1F after cellar return)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

## Frontier pin

`Level8Interior1FReturnedReconFixture` — L8 play `$EB=0x1F` `(96,157)`,
mode 5, keys 8, bombs 6, Magic Key **1**, TF `0x7F`, bow 1, arrows 1,
rupees 247, Magical Sword 3, Candle 2, B = arrows. `room_item_id=0x03`.
Centre `0x68` stairs sprite still at `(96,144)` (not census). West door
visible. Census empty of live enemies (0x1F was sword-cleared inbound).

2/2 byte-identical: probe `l8_0f_cellar_return` F3/F4, 588 frames, 528
controller frames. Dest is play **0x1F**, not Gleeok `0x3C`.

## How we got here

From `Level8InteriorMKReconFixture` cellar `0x0F` `(136,141)` tile 36.

- F1: cardinal DOWN stuck 40f (south of pad is brick).
- F2: LEFT+DOWN at `(160,141)` is pit tile 250. Stay RIGHT until `x>=174`.
- F3/F4: RIGHT to east `x=176`, LEFT+DOWN, floor LEFT, UP `(48,93)` until
  tiles `0x70..0x73`. Policy in `level8/cellar.py`. OccupancyWalker banned.

`make_magic_key_stairs_controller` and `make_gleeok_passage_controller`
stay fail-closed. `topology.magic_key_room` unset. `L8_THROUGH` not greened.
Factory `make_magic_key_cellar_return_controller` dest=`0x1F`,
`route_eligible=False`.

## Next live boundary

From play `0x1F` `(96,157)` walk the Gleeok suffix. One gate. Do not start
the Gleeok fight. Do not poke MK / TF / doors / Gleeok HP.

Hypothesis (not live): LEFT into cleared `0x1E`, DOWN×2 to `0x3E`, RIGHT
kill-clear to passage_east (hyp `0x3F`), STAIRS cellar `0x2F` to hyp `0x4C`,
bomb-N hyp `0x3C`. Record `$EB`. `GLEEOK_FOUR_HEAD_OBJECT_TYPE` stays None.
ROM `0x45` is not a live type.

Keys 8→8, bombs 6→6 unless a new gate spends one.

## Remaining L8

1. This leftover: 0x1F west / Gleeok suffix, one gate per sitting.
2. `rr-5eb2`: live Gleeok census, then the written south-stand model.
3. `rr-6o7.3`: heart, TF `0x80`, post-L8 OW leave.
4. Power-on still blocked on `rr-8t4.3` then `rr-6o7.1`.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

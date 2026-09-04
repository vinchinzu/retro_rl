# Residual — rr-6o7.2 L8-B Magical Key (play 0x1E after 0x1F west gate)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

## Frontier pin

`Level8Interior1EWestReconFixture` — L8 play `$EB=0x1E` `(208,141)`,
mode 5, keys 8, bombs 6, Magic Key **1**, TF `0x7F`, bow 1, arrows 1,
rupees 247, Magical Sword 3, Candle 2, B = arrows. `room_item_id=0x03`.
Arrival census empty (Gohma already dead). `cur_opened_doors=5`
(RIGHT+DOWN), `open_doorway_mask=0`. East mouth of cleared 0x1E.
`0x55` statue fireballs spawn after idle — leftover is arrival, not
post-knockback.

2/2 byte-identical hop: probe `l8_1f_west` G2/G3, 275 controller frames
(335 with census). Dest is play **0x1E**, not Gleeok `0x3C`.

## How we got here

From `Level8Interior1FReturnedReconFixture` play `0x1F` `(96,157)`.

- G1: OccupancyWalker 1px LEFT grade false-missed 2px dungeon steps.
  Boxed `(88,157)` tile 118 (walkable floor). West door still open.
- G2/G3: cardinal LEFT until `x<=80` (west of 0x68), y-align 141, LEFT
  push. Policy in `level8/path.py` `west_1f_step`. OccupancyWalker banned
  for this hop.

`make_magic_key_stairs_controller` and `make_gleeok_passage_controller`
stay fail-closed. `topology.magic_key_room` unset. `L8_THROUGH` not greened.
Factory `make_west_1f_controller` dest=`0x1E`, `route_eligible=False`.

## Next live boundary

From play `0x1E` `(208,141)` continue the Gleeok suffix. One gate. Do not
start the Gleeok fight. Do not poke MK / TF / doors / Gleeok HP.

Hypothesis (not live): DOWN from cleared 0x1E into 0x2E, DOWN again to
0x3E, RIGHT kill-clear to passage_east (hyp 0x3F), STAIRS cellar 0x2F to
hyp 0x4C, bomb-N hyp 0x3C. Record `$EB`. `GLEEOK_FOUR_HEAD_OBJECT_TYPE`
stays None. ROM `0x45` is not a live type.

PNG of arrival: north and south doors visible. RAM doors 0x05 is
RIGHT+DOWN only. Prefer the DOWN bit that is already open.

Keys 8→8, bombs 6→6 unless a new gate spends one.

## Remaining L8

1. This leftover: 0x1E south / Gleeok suffix, one gate per sitting.
2. `rr-5eb2`: live Gleeok census, then the written south-stand model.
3. `rr-6o7.3`: heart, TF `0x80`, post-L8 OW leave.
4. Power-on still blocked on `rr-8t4.3` then `rr-6o7.1`.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

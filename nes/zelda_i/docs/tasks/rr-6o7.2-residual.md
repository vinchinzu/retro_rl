# Residual — rr-6o7.2 L8-B Magical Key (play 0x2E after 0x1E south gate)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

## Frontier pin

`Level8Interior2ESouthReconFixture` — L8 play `$EB=0x2E` `(120,77)`,
mode 5, keys 8, bombs 6, Magic Key **1**, TF `0x7F`, bow 1, arrows 1,
rupees 247, Magical Sword 3, Candle 2, B = arrows. `room_item_id=0x17`
(map still on the floor, not picked up). Arrival census empty
(Manhandla already dead). `cur_opened_doors=12` (UP+DOWN),
`open_doorway_mask=0`. North mouth of cleared 0x2E.

2/2 byte-identical hop: probe `l8_1e_south` H2/H3, 264 controller frames
(324 with census). Dest is play **0x2E**, not Gleeok `0x3C`.

## How we got here

From `Level8Interior1EWestReconFixture` play `0x1E` `(208,141)`.

- G1 occupancy skipped: empty-grid BFS first dir is DOWN along x=208
  into the SE statue; OccupancyWalker 1px-grade already false-missed
  2px dungeon steps on the west hop.
- G2/G3: cardinal x-align LEFT to 120, DOWN push. Policy in
  `level8/path.py` `south_1e_step`. OccupancyWalker not used live.

`make_magic_key_stairs_controller` and `make_gleeok_passage_controller`
stay fail-closed. `topology.magic_key_room` unset. `L8_THROUGH` not greened.
Factory `make_south_1e_controller` dest=`0x2E`, `route_eligible=False`.

## Next live boundary

From play `0x2E` `(120,77)` continue the Gleeok suffix. One gate. Do not
start the Gleeok fight. Do not poke MK / TF / doors / Gleeok HP.

Hypothesis (not live): DOWN from cleared 0x2E into 0x3E (already cleared
inbound), then RIGHT kill-clear to passage_east (hyp 0x3F), STAIRS cellar
0x2F to hyp 0x4C, bomb-N hyp 0x3C. Record `$EB`.
`GLEEOK_FOUR_HEAD_OBJECT_TYPE` stays None. ROM `0x45` is not a live type.

PNG of arrival: north mouth, south door black/open. RAM doors 0x0C is
UP+DOWN. Prefer the DOWN bit that is already open. Map 0x17 is optional
and omitted on the min route.

Keys 8→8, bombs 6→6 unless a new gate spends one.

## Remaining L8

1. This leftover: 0x2E south / Gleeok suffix, one gate per sitting.
2. `rr-5eb2`: live Gleeok census, then the written south-stand model.
3. `rr-6o7.3`: heart, TF `0x80`, post-L8 OW leave.
4. Power-on still blocked on `rr-8t4.3` then `rr-6o7.1`.

## Rooms live / selected min

L8 selected min-through (MK chapter + Gleeok suffix, Book/Map/Compass omitted):
`0x7E 0x6E 0x5E 0x4E 0x3E 0x2E 0x1E 0x1F 0x0F` + `passage_east pols_west gleeok 0x3C triforce` = **13**.
Live unique: those nine (MK 9/9 including cellar). Unvisited: 4.
**L8 9/13 = 69%.** Leftover is a revisit of live `0x2E`.

L9 selected Magical Key min (Red Ring `0x07` out, plus Ganon/Zelda): **28**.
Live dest hops: `0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52 0x42 0x32` = 10.
Poked `0x76` settle only. **L9 10/28 = 36%** (11/28 = 39% if counting poked `0x76`).

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

# Residual — rr-6o7.2 L8-B Magical Key (play 0x3E after 0x2E south gate)

**Spine bead:** `rr-6o7.2` (`in_progress`). Do not close it. Acceptance is
power-on `--through level8-magic-key`, still blocked on `rr-8t4.3` and
`rr-6o7.1`. Fixture-live only. Do not STATUS. Do not push unless asked.

## Frontier pin

`Level8Interior3ESouthReconFixture` — L8 play `$EB=0x3E` `(120,93)`,
mode 5, keys 8, bombs 6, Magic Key **1**, TF `0x7F`, bow 1, arrows 1,
rupees 247, Magical Sword 3, Candle 2, B = arrows. `ADDR_MAP=0x80`
(L8 map picked up incidentally walking the 0x2E aisle; not a detour).
`room_item_id=0x03`. Arrival census empty (0x0C bodies already dead).
`cur_opened_doors=12` (UP+DOWN), `open_doorway_mask=0`. North mouth of
cleared 0x3E. Idle later raises RIGHT (`doors 13=0x0D`, mask `0x0C`) as
the already-cleared east shutter — pin at arrival, not after idle.

2/2 byte-identical hop: probe `l8_2e_south` I1/I2 (I3 save), 255
controller frames (315 with census). Dest is play **0x3E**, not Gleeok
`0x3C`, not cellar `0x0F`.

## How we got here

From `Level8Interior2ESouthReconFixture` play `0x2E` `(120,77)`.

- OccupancyWalker not used live: 1px-grade already false-missed 2px
  dungeon steps on the west hop.
- I1/I2/I3: cardinal DOWN along already-aligned x=120. Policy in
  `level8/path.py` `south_2e_step`. Statues at ~x=96 and x=144 y~141;
  center x=120 passed between them. Map 0x17 on the aisle was walked
  over (`ADDR_MAP` 0→0x80); recorded, not a detour.

`make_magic_key_stairs_controller` and `make_gleeok_passage_controller`
stay fail-closed. `topology.magic_key_room` unset. `L8_THROUGH` not greened.
Factory `make_south_2e_controller` dest=`0x3E`, `route_eligible=False`.

## Next live boundary

From play `0x3E` `(120,93)` continue the Gleeok suffix. One gate. Do not
start the Gleeok fight. Do not poke MK / TF / doors / Gleeok HP.

Hypothesis (not live): RIGHT from cleared 0x3E into passage_east (hyp
`0x3F`), STAIRS cellar 0x2F to hyp 0x4C, bomb-N hyp 0x3C. Record `$EB`.
`GLEEOK_FOUR_HEAD_OBJECT_TYPE` stays None. ROM `0x45` is not a live type.

PNG of arrival: north mouth, two mid-row statue blocks, east shutter
still metal at arrival; after idle the east doorway is black/open.
Prefer the RIGHT bit that idle raises. Do not chain into Gleeok.

Keys 8→8, bombs 6→6 unless a new gate spends one.

## Remaining L8

1. This leftover: 0x3E east / passage_east (hyp 0x3F), one gate per sitting.
2. `rr-5eb2`: live Gleeok census, then the written south-stand model.
3. `rr-6o7.3`: heart, TF `0x80`, post-L8 OW leave.
4. Power-on still blocked on `rr-8t4.3` then `rr-6o7.1`.

## Rooms live / selected min

L8 selected min-through (MK chapter + Gleeok suffix, Book/Map/Compass omitted):
`0x7E 0x6E 0x5E 0x4E 0x3E 0x2E 0x1E 0x1F 0x0F` + `passage_east pols_west gleeok 0x3C triforce` = **13**.
Live unique: those nine (MK 9/9 including cellar). Unvisited: 4.
**L8 9/13 = 69%.** Leftover is a revisit of live `0x3E`.

L9 selected Magical Key min (Red Ring `0x07` out, plus Ganon/Zelda): **28**.
Live dest hops: `0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52 0x42 0x32` = 10.
Poked `0x76` settle only. **L9 10/28 = 36%** (11/28 = 39% if counting poked `0x76`).

L7 suffix **4/4 fixture-live** (`rr-n91a` closed): dest `0x29` 2/2, then
`0x2A`/`0x2B`/OW 1/1 from the poke-cellar pin. 0x0D walk-on still open
on `rr-8t4.3`. Survival OW leave (TF `0x7F`) unmeasured.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

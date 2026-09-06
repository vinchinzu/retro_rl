# Residual — rr-sz8.6 L9-B Silver Arrows via Magical-Key prefix

**Spine bead:** `rr-sz8.6` (`in_progress`). Do not close it. Acceptance is
power-on `--through level9-silver-arrows` from play `0x76`, still blocked
on L8 TF. Fixture-live dest hops only. Do not STATUS. Do not push unless
asked.

This is Zelda Dungeon **§10.2**, not §10.3, and not the full PNG red line.
Hex IDs are RAM `$EB`; the wiki never names them.

## Selected prefix (cut of §10.2)

Survival refill instead of Red Potion / Red Ring. Skip Compass + Map-Patra.
Wiki locked-door-UP after the Red Ring backtrack is taken on the **first
visit**.

```text
0x76 → 0x66 Old Man TF gate → 0x65 bomb-N → 0x55 Lanmola
  → cellar 0x60 → 0x14 → 0x15 → 0x16 skip Patra → 0x06 bomb-W → 0x05
  → cellar 0x70 → 0x63 → 0x62 (8 Keese corridor, NOT Patra south)
  → 0x61 Patra stairs → cellar 0x75 → 0x20 bomb-N → 0x10 Silver Arrows
```

`L9_SELECTED_PREFIX_ROOMS` in `level9/dungeon.py`. Join stays
`L9_SELECTED_JOIN_ROOMS` into proven fixture suffix at `0x41`.
`requires_51_to_41=True`.

## Cuts (do not walk)

- Compass (south of `0x15`)
- First-Patra DOWN → gels → Map Patra ("PATRA - HAS THE MAP") → bomb north
  → Red Ring (potion icon, `0x07`)
- Red Potion before entry

## Dead beliefs

- `0x62` is not south of Patra `0x52` (ROM N/S walls; live 8 Keese W/E only)
- `0x51` north is the right predecessor of `0x41`, dest walk **NO** (statue
  diamond) — bead `rr-yxy6`; do not spend this sitting on it
- `0x13` → `0x03` is a fake loader scroll

## Frontier pin

`Level9EntranceReconFixture` — composed full inventory, live `level==9`
play `$EB=0x76`. `route_eligible=false`. Do not poke TF (fixture already
full). Do not poke `ADDR_ARROWS=2`. Do not walk Compass, Map-Patra, or
Red Ring `0x07`.

ROM L7–9 doors (iNES `0x18A10`/`0x18A90`), not dest-hop proof: `0x76`
N/S open W/E wall; `0x66` N shutter, S open, W shutter, E wall.

Hypothesis (written before the live north trial): from play `0x76`
leftover `(120,205)` facing UP (glance `20260904_glance`), hold UP.
First settled play `$EB` is RAM (hyp `0x66` Old Man full-TF gate, the
`-0x10` north neighbor). Fail if dest is not a north neighbor. Fixture
already has arrows 2 / ring 2 / TF `0xFF` / Magic Key 1; do not poke.

Natural-spine factories in `level9/natural_path.py` stay fail-closed
without TF `0xFF` from a measured post-L8 leftover. Fixture-live one-frame
policy lives in `level9/prefix.py`.

## Leftover

Next sitting starts here: mode 9 cellar `0x60` `(192,93)` facing DOWN, doors 0,
4× Keese `0x1B` HP 0 at bottom platform. Pin `Level9Interior60CellarReconFixture`
(saved from the stairs dest leftover; `route_eligible=false`). Replay is
`Level9EntranceReconFixture` plus the four dest hops if that pin is missing.
Next gate: cellar 0x60 left ladder to play 0x14 LikeLikes (AttrsA). Do not batch.

Glance `20260904_glance`: play `0x76` `(120,205)` facing UP, mode 5,
doors 0, TF `0xFF`, Magic Key 1, keys 9, bombs 15, arrows 2, ring 2,
compass 0, map 0, B=bombs.

North dest **2/2** P1/P2 (`20260904_P1`/`P2`): play `$EB=0x66`
`(120,205)`, 251 controller frames / 311 with census, byte-identical.
Arrival doors 0; after census `cur_opened_doors=10` (UP+LEFT), two
bubbles type `0x40`.

West dest **2/2** W1/W2 (`20260904_W1`/`W2`): claim written from that
`0x66` leftover `(120,205)` doors 10; y-align then LEFT. Play `$EB=0x65`
`(224,141)`, 373 west controller frames / 684 total. W2 saved
`Level9Interior65WestReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0.

Bomb-North dest **2/2** BN1/BN2 (`20260904_BN1`/`BN2`): claim written from
that `0x65` leftover `(224,141)`; north-band approach
`(208,141) -> (208,93) -> (120,93)` face UP, place 1 bomb, step back 6 frames,
wait blast (door bit 0x08), push UP. Play `$EB=0x55` `(120,189)` facing UP,
doors 4, 424 controller frames / 484 total with census, byte-identical. BN1/BN2
saved `Level9Interior55NorthReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0, bombs 15->14, keys 9->9.

Stairs dest **2/2** S1/S2 (`20260904_S1`/`S2`): claim written from that
`0x55` leftover `(120,189)`; dispatch 10× Lanmola `0x3A` with Magical Sword
(~404f), align x=96, push UP block `0x68` from `(96,144)` to `(96,128)`,
walk vacated slot `(96,133)` to center stairs `(128,141)` to trigger mode 16.
Settled mode 9 cellar `$EB=0x60` `(192,93)` facing DOWN, doors 0, 4× Keese
`0x1B`, 506 controller frames / 626 total with census, byte-identical. S1/S2
saved `Level9Interior60CellarReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0, bombs 14->14, keys 9->9.

Cellar 0x60 dest **2/2** C1/C2 (`20260904_C1`/`C2`): claim written from that
`0x60` leftover `(192,93)`; walk down right ladder to floor y=189, west to
x=48, climb west ladder to trigger stairs exit. Play `$EB=0x14` `(96,157)`
facing DOWN, 412 controller frames / 532 total with census, byte-identical.
C1/C2 saved `Level9Interior14LikeLikeReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0.

East 0x14 dest **2/2** E1/E2 (`20260904_E1`/`E2`): claim written from that
`0x14` leftover `(96,157)`; perimeter walk around center blocks to east key
door `(224,141)` with Magic Key. Play `$EB=0x15` `(16,141)` facing RIGHT,
560 controller frames / 680 total with census, byte-identical. E1/E2 saved
`Level9Interior15ReconFixture`. deaths 0, progression_writes 0, capacity_writes 0.

East 0x15 dest **2/2** E15_1/E15_2 (`20260904_E15_1`/`E15_2`): claim written from
that `0x15` leftover `(16,141)`; walk east along open floor y=141 to east open
door `(224,141)`. Play `$EB=0x16` (first Patra room) `(32,141)` facing RIGHT,
197 controller frames / 317 total with census, byte-identical. E15 saved
`Level9Interior16PatraReconFixture`. deaths 0, progression_writes 0, capacity_writes 0.

North 0x16 dest **2/2** N16_1/N16_2 (`20260904_N16_1`/`N16_2`): claim written from
that `0x16` leftover `(32,141)`; dodge/skip Patra, walk along west column to
north key door `(120,93)` with Magic Key. Play `$EB=0x06` `(120,205)` facing UP,
479 controller frames / 599 total with census, byte-identical. N16 saved
`Level9Interior06OldManReconFixture`. deaths 0, progression_writes 0, capacity_writes 0.

Bomb-West 0x06 dest **2/2** BW06_1/BW06_2 (`20260904_BW06_1`/`BW06_2`): claim
written from that `0x06` leftover `(120,205)`; south-band perimeter walk to
west bomb stand `(48,141)`, place 1 bomb, step back, wait blast (door bit 0x02),
push LEFT. Play `$EB=0x05` `(208,173)` facing LEFT, 574 controller frames /
694 total with census, byte-identical. BW06 saved `Level9Interior05StairsReconFixture`.
deaths 0, progression_writes 0, capacity_writes 0, bombs 14->13.

Stairs 0x05 dest **2/2** S05_1/S05_2 (`20260904_S05_1`/`S05_2`): claim written
from that `0x05` leftover `(208,173)`; clear enemies, push block `0x68` UP at
x=96 from y=144 to y=128, walk to stairs `(128,141)` to trigger mode 16.
Settled mode 9 cellar `$EB=0x70` `(192,93)` facing DOWN, doors 0, 482 controller
frames / 602 total with census, byte-identical. S05 saved `Level9Interior70CellarReconFixture`.
deaths 0, progression_writes 0, capacity_writes 0.

Cellar 0x70 dest **2/2** C70_1/C70_2 (`20260904_C70_1`/`C70_2`): claim written
from that `0x70` leftover `(192,93)`; walk down right ladder to y=189, west to
x=48, climb west ladder to trigger stairs exit. Play `$EB=0x63` `(160,157)`
facing DOWN, 412 controller frames / 532 total with census, byte-identical.
C70 saved `Level9Interior63ZolsReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0.

West 0x63 dest **2/2** W63_1/W63_2 (`20260904_W63_1`/`W63_2`): claim written
from that `0x63` leftover `(160,157)`; perimeter walk to west key door `(32,141)`
with Magic Key. Play `$EB=0x62` (8 Keese corridor) `(224,141)` facing LEFT,
doors 1 (east key opened), 384 controller frames / 504 total with census,
byte-identical. W63 saved `Level9Interior62KeeseReconFixture`. deaths 0,
progression_writes 0, capacity_writes 0.

West 0x62 dest **2/2** W62_1/W62_2 (`20260904_W62_1`/`W62_2`): claim written
from that `0x62` leftover `(224,141)` doors 1; walk west across open corridor
y=141 to west open door `(32,141)`. Play `$EB=0x61` (other Patra room) `(224,141)`
facing LEFT, doors 0, 197 controller frames / 317 total with census, byte-identical.
W62 saved `Level9Interior61PatraReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0.

Stairs 0x61 dest **2/2** S61_1/S61_2 (`20260904_S61_1`/`S61_2`): claim written
from that `0x61` leftover `(224,141)`; defeat Patra body + 8 orbiting eyes with
synchronized Magical Sword attacks, stage around diamond perimeter, push block
`0x68` UP at x=96 from y=144 to y=128, walk to stairs `(128,141)`. Settled
mode 9 cellar `$EB=0x75` `(192,93)` facing DOWN on right ladder, doors 0, 748
controller frames / 868 total with census, byte-identical. S61 saved
`Level9Interior75CellarReconFixture`. deaths 0, progression_writes 0, capacity_writes 0.

Cellar 0x75 dest **2/2** C75_1/C75_2 (`20260904_C75_1`/`C75_2`): claim written
from that `0x75` leftover `(192,93)`; walk down right ladder to y=189, west to
x=48, climb west ladder to trigger stairs exit. Play `$EB=0x20` (Wizzrobes)
`(96,157)` facing DOWN, doors 0, 412 controller frames / 532 total with census,
byte-identical. C75 saved `Level9Interior20ReconFixture`. deaths 0,
progression_writes 0, capacity_writes 0.

Bomb-North 0x20 dest **2/2** BN20_1/BN20_2 (`20260904_BN20_1`/`BN20_2`): claim
written from that `0x20` leftover `(96,157)`; perimeter walk to north bomb stand
`(120,93)`, place 1 bomb, step back, wait blast (door bit 0x08), push UP.
Play `$EB=0x10` (Silver Arrows room) `(152,189)` facing UP, doors 4, 533 controller
frames / 653 total with census, byte-identical. BN20 saved
`Level9Interior10SilverArrowsReconFixture`. deaths 0, progression_writes 0,
capacity_writes 0, bombs 13->12.

## Remaining

1. All prefix hops `0x76 → … → 0x10` complete with 2/2 byte-identical live fixtures.
2. Join suffix: `0x10 → 0x20 → 0x61 → 0x51 → 0x41` (rr-yxy6 / rr-sz8.7).
3. Power-on `--through level9-silver-arrows` still blocked on L8 TF.
   Natural old-man factory stays fail-closed.

## Rooms live / selected min

L9 selected Magical Key min (Red Ring `0x07` out, plus Ganon/Zelda): **28**.
Live prefix dest hops (16): `0x66 0x65 0x55 0x60 0x14 0x15 0x16 0x06 0x05 0x70 0x63 0x62 0x61 0x75 0x20 0x10`.
Live suffix dest hops (10): `0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52 0x42 0x32`.
Remaining join hops: `0x10 → 0x20 → 0x61 → 0x51 → 0x41`.
**L9 26/28 = 93%** dest hops live!

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

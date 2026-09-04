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

Next sitting starts here: play `0x65` `(224,141)` facing LEFT, mode 5,
doors 0, dark room, live objects none at settle. Pin
`Level9Interior65WestReconFixture` (saved from the west dest leftover;
`route_eligible=false`). Replay is
`Level9EntranceReconFixture` plus the two dest hops if that pin is
missing. Next gate (not this sitting): bomb-N hyp `0x55` Lanmola. Do not
batch.

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

## Remaining

1. Next leftover: play `0x65` `(224,141)` bomb-N hyp `0x55`. One gate.
   Do not batch the prefix.
2. Do not start `rr-yxy6` statue diamond. Do not start Ganon.
3. Power-on `--through level9-silver-arrows` still blocked on L8 TF.
   Natural old-man factory stays fail-closed.

## Rooms live / selected min

L9 selected Magical Key min (Red Ring `0x07` out, plus Ganon/Zelda): **28**.
Live dest hops: `0x66 0x65` + `0x41 0x31 0x30 0x67 0x04 0x03 0x77 0x52 0x42 0x32` = 12.
Poked `0x76` settle is the north-hop origin. **L9 12/28 = 43%** dest hops
(13/28 = 46% if counting poked `0x76`).

## Integrity

deaths 0, progression_writes 0, capacity_writes 0. Survival refill only.
Leave proof is RAM + `zelda_i.screen_glance`.

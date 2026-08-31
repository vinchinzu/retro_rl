# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered power-on tip is cleared L6 dark wizz/key room
(play `0x29`) leftover `(47,157)` tile 119. `--through level6-south29`
from the west leftover is **BLOCKED 6/6** (prior 3 + this sitting 3).
East leftover south29 remains **BLOCKED 4/4**. y=157 SW squeeze cannot
leave. Bead `rr-tne2` stays open. Do not STATUS-promote.

Walkthrough room (Zelda Dungeon L6 018): orange-sand moat around a green
tile island, key in the center, north door from the Map room, south door
into Vires (019). Live leftover hugs the **SW corner of that island**.

## Recovered prefix (unchanged through south19)

Prefix through skip-Map `0x09` clear was already 1/1. Stairs09 east-of-block
through south19 still 1/1.

| through | tag | leftover | keys |
|---------|-----|----------|------|
| level6-stairs09 | l6_stairs09_recompose | mode 9 `0x75` `(208,93)` | 3 |
| level6-rod | l6_rod_recompose | mode 9 `0x75` `(136,141)` Rod=1 | 3 |
| level6-exit75 | l6_exit75_recompose | play `0x09` `(192,141)` | 3 |
| level6-south09 | l6_south09_recompose | play `0x19` `(120,77)` | 3 |
| level6-south19 | l6_south19_recompose | dark play `0x29` `(120,77)` | 3→2 |
| level6-clear29 | l6_clear29_west_recompose | play `0x29` `(56,157)` | 2→3 |

Census/inventory still match: TF `0x1F`, bombs=8, Bow=1, Rod=1, map `0x0A`,
health `0x66` lo==hi, deaths/state-load/progression/capacity writes 0. Key
deficit is still one versus historical; do not top up. East door of this
room sealed (`open_doorway_mask` 12 = north+south). Do not rerun the east
leftover `(184,144)`.

## clear29 west leftover — 1/1 (unusable for south29)

`--through level6-clear29 --tag l6_clear29_west_recompose` 1/1, hop 2,194f.
West-aisle patrol + LEFT peel; fight only x<64. Final play `0x29` `(56,157)`
tile 244, keys=3. Occupancy 449 misses / 13 blocked during clear. South from
this leftover is BLOCKED 6/6. Next sitting may reshape the leftover to
y<=133 (historical `(55,133)`); do not chase patrol `(48,157)`.

## south29 — east leftover BLOCKED 4/4 (do not retry)

All four east-leftover trials `--through level6-south29 --tag l6_south29_recompose`
0/1 from leftover `(184,144)`. Do not retry any of these, and do not retry
the prior-sitting RIGHT+DOWN clip from leftover `(55,133)`.

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| prior | RIGHT+DOWN clip, occupancy x=120 @ y=141 | `(184,144)` `south_up_halt` 4000f misses 0 | clip assumed west leftover |
| 1 | LEFT on y=144 to x=120 | `(184,141)` `south_up_halt`; LEFT is the island (tile 244) | wall-slide 144→141 |
| 2 | DOWN clip to south band y=181 | `(184,165)` `south_peel` 4000f tile 244 | east aisle walks 144→165 then boxes |
| 3 | occupancy dest `(120,189)` | `(184,149)` `south_up_halt` | occupancy false-misses the DOWN the clip completed |
| 4 | DOWN to measured box y=165, then LEFT | `(184,165)` `south_up_halt`; LEFT solid | y=165 is not a west aisle |

## south29 — west leftover BLOCKED 6/6 (do not retry)

Walkthrough 018 is an orange-sand moat around a green island; live leftover
hugs the **SW corner** (brown tile 244). LEFT unsticks into sand tile 119.
The gap west of the island at y=157 is only x=41–47; occupancy xmin x=40
is the **west wall**, not a northbound moat. Do not run another south29
from `(56,157)` / `(47,157)` / `(40,157)`.

Prior sitting (BLOCKED 3/3 from leftover `(56,157)`):

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| west 1 | occupancy dest `(120,189)`, forbid UP | `(57,157)` `south_up_halt` 4000f; DOWN/RIGHT solid, then occupancy wanted UP | SW face. `forbid_up` stood |
| west 2 | UP-peel `clip_buttons=UP` along x=56 | `(56,157)` 4000f tile 244, misses 0, reason `west_up_peel` | **UP is also solid** at the leftover |
| west 3 | LEFT-peel to x=48, x-align occupancy | `(47,157)` 4000f tile 119; `miss_f9_UP_48_157` | LEFT 56→47. UP at `(48,157)` is the island south face |

This sitting (BLOCKED 3/3; dest still play `0x39`; keys=3 rod=1 TF=`0x1F`
health `0x66` lo==hi; deaths/state-load/progression/capacity 0):

| # | policy / tag | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| west 4 | LEFT to x=40 then occupancy UP (`l6_south29_west_wall`) | `(40,157)` 4000f tile 244; `miss_f14_UP_40_157` / RIGHT@42 / DOWN@40 then `south_stand` | **x=40 is the west wall.** UP/RIGHT/DOWN all 244. Sand was x=41 tile 119 |
| west 5 | LEFT to sand x=41 then occupancy DOWN (`l6_south29_west_sand`) | `(40,157)` 4000f tile 244; `miss_f13_DOWN_40_157` | DOWN at x=41 **slides 41→40** into the wall. y never moved |
| west 6 | LEFT to sand x=47 then occupancy DOWN (`l6_south29_west_sand47`) | `(47,157)` 4000f tile 119; `miss_f9_DOWN_48_157` / LEFT@45 / DOWN@47 | **DOWN is solid at x=47–48.** Same leftover as west 3, other cardinal |

Parked `SOUTH29_SPEC` is west 6. Do not restore west 1–5, east-box LEFT,
occupancy-DOWN from `(184,144)`, or the `(55,133)` RIGHT+DOWN clip.

```bash
# Next sitting: reshape clear29 so leftover is not the y=157 SW squeeze.
# Historical west leftover (55,133) is north of this block. Do not chase
# patrol (48,157). Do not top up keys. Do not retry south29 from y=157.
```

## Key deficit (still open; do not top-up)

Ended L6 compass keys=4 vs historical 5. Gap already at L4 `0x40`. clear58
had no natural key. room09 KEY-UP spent 4→3. south19 KEY-DOWN spent 3→2
(historical 4→3). clear29 floor key 2→3 (historical 3→4). Do not hide it
with a key top-up.

## Remaining L6 after south29 (not this sitting)

Phase 1: south29 (west leftover BLOCKED 6/6; east leftover BLOCKED 4/4) →
settle39 → clear39 → east39 → settle3a → clear3a.

Phase 2: un-dedicate `stairs3a-warp` / `cellar08` / `south1d` / `west2d` /
`north2c` so they compose from clear `0x3A`. Do not start Phase 2 until
clear3a is green.

Phase 3: natural heart in `0x1C`, north `0x0C`, TF `0x20`.

Phase 4–5: measured `set_state` audit, then `--through level6`.

## Parallel L7/L8/L9/OW (fixture-live only)

Wave A handoffs: `docs/tasks/{l7,l8,l9,ow}-handoff.md`. Public `--through`
names unchanged. Do not attach `continue_level7/8/9_spine` until L6 leave is
measured. `route_eligible=false`. Post-dungeon OW leftovers remain
UNMEASURED (do not treat this poke fixture as a measured fanfare leave).

Operator-directed poke fixture `Level6ExitOverworld` (source `L6Probe_22`):
OW play `0x22` `(120,221)` facing north, TF `0x3F`, Rod=1, Bow=1, Whistle=1,
Food=0, Candle=1, sword=2, rupees=80, keys=3, bombs=8, hearts `0xAA` 11/11
full. Loadout is `POST_L6_EXIT_LOADOUT`. Dead belief: standing `(112,125)`
on `0x22` is the cave mouth (mode 16 → L6). Spine still fail-closed
(`HYPOTHESIZED_POST_L6_EXIT.verified=false`). Next L7 hop from this
fixture: walkthrough bait shop `0x34` (Armos top-middle, 60R), then pond.

## Non-claims

- Did not STATUS-promote or overwrite Clean M5.
- Did not close `rr-tne2` or reach Gohma / TF `0x20`.
- L6 sitting did not poke doors, Map, or capacity. The L7 poke fixture
  disclosed TF/Rod/Bow/Whistle/sword/rupee writes; Food stayed 0.
- Did not retouch maze-west, 0x40, stairs09, or the cellar walk.
- Did not retry west-aisle stairs09.
- Did not retry the `(55,133)` RIGHT+DOWN clip or any east-leftover south29.
- Did not run a fourth south29 from leftover `(56,157)` after the prior
  sitting's 3 reds. This sitting added west 4–6 (wall / sand-41 / sand-47)
  and halted at 3. Did not retry west 1–3 or any east-leftover south29.
- Did not push.

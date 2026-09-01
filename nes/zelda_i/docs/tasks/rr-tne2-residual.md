# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered prefix through south19 is still 1/1. `clear29`
reshape leftover is **1/1** play `0x29` `(63,133)` keys 2→3. `south29`
from that leftover is **1/1** play `0x39` `(120,93)` hop 282f. West
south29 from y=157 remains **BLOCKED 6/6**; east south29 **BLOCKED 4/4**.
Do not retry those. Bead `rr-tne2` stays open. Do not STATUS-promote.

Walkthrough room (Zelda Dungeon L6 018): orange-sand moat around a green
tile island, key in the center, north door from the Map room, south door
into Vires (019). Recovered enter leftover is the north mouth `(120,77)`.
The SW island corner `(56,157)` cannot leave.

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
| level6-clear29 | l6_clear29_north_handoff | **BLOCKED 3/3** (prior sitting) | — |
| level6-clear29 | l6_clear29_north_inland | play `0x29` `(63,133)` | 2→3 |
| level6-south29 | l6_south29_right_down | play `0x39` `(120,93)` | 3 |

Census/inventory still match: TF `0x1F`, bombs=8, Bow=1, Rod=1, map `0x0A`,
health `0x66` lo==hi, deaths/state-load/progression/capacity writes 0. Key
deficit is still one versus historical; do not top up. East door of this
room sealed (`open_doorway_mask` 12 = north+south). Do not rerun the east
leftover `(184,144)`.

## clear29 west leftover — 1/1 (unusable for south29)

`--through level6-clear29 --tag l6_clear29_west_recompose` 1/1, hop 2,194f.
West-aisle patrol + LEFT peel; fight only x<64. Final play `0x29` `(56,157)`
tile 244, keys=3. Occupancy 449 misses / 13 blocked during clear. South from
this leftover is BLOCKED 6/6.

## clear29 north-inland reshape — 1/1

Fresh power-on `--through level6-clear29 --tag l6_clear29_north_inland`
1/1, hop 3,509f, end_frame 208,204. Prefix through south19 still 1/1.
DOWN inland from `(120,77)`, LEFT at y=109, full-room chase, leftover
invariant. Final play `0x29` `(63,133)` keys=3 last_live=0 max_live=5
occupancy misses 0. Glance: play `0x29` `(63,133)` rod=1 keys=3 TF=`0x1F`
bombs=8 Bow=1 health `0x66` lo==hi. Deaths / state-load / progression /
capacity writes 0. 8px east of historical `(55,133)`; still x<64 y<=133.
Do not restore reds 1–3. Do not retry south29 from y=157.

Prior sitting `--tag l6_clear29_north_handoff` BLOCKED 3/3 (do not retry):

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| 1 | west-only chase, never y>133, leftover invariant | `(56,117)` timeout 15000f last_live=2 keys=3 | 3/5 killed; last two never entered west-north. PNG: blue+orange south of island |
| 2 | west-only + combat y=141 south_hold | `(56,117)` timeout 15000f last_live=2 keys=3 | Same box. Link never reached y=141; empty west-live stands |
| 3 | full-room chase, LEFT while x>=64 and y<=109 | `(120,77)` timeout 15000f last_live=5 keys=2 misses 0 | **LEFT at north mouth y=77 is the door channel.** Never inland. PNG: spawn, key still on island, 2×blue+orange north |

Live leftover `(63,133)` is ON tile 244 (west-arm north face). North
around is dated (see south29 reds 1–2 below). RIGHT+DOWN along that
face is the live hop. Occupancy seed of plus+SW encoded dated UP from
`(63,133)` and was never wired into `SOUTH29_SPEC`; deleted.

```bash
# Next sitting: --through level6-settle39 from leftover (120,93).
# Do not LEFT at y=77. Do not west-only chase. Do not top up keys.
# Do not retry south29 from y=157. Do not restore UP / LEFT+UP @ y=133.
```

## south29 from (63,133) — 1/1 after two north-peel reds

Different approach was north around the plus. Both peels died on tile 244
at y=133 (no path to y=109). Third knob: RIGHT+DOWN, the remaining open
axis. `--through level6-south29 --tag l6_south29_right_down` **1/1**, hop
282f, end_frame 208,486. Dest play `0x39` `(120,93)`. Glance: play `0x39`
`(120,93)` rod=1 keys=3 TF=`0x1F` bombs=8 Bow=1 health `0x66` lo==hi.
Deaths / state-load / progression / capacity writes 0.

| # | tag / policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| 1 | `l6_south29_north_moat` cardinal UP to y=109 | `(64,133)` timeout 4000f tile 244 `south_stand` | **UP is solid.** Slid 63→64 onto the west face; occupancy LEFT-miss boxed |
| 2 | `l6_south29_left_up` LEFT+UP off the face | `(56,133)` timeout 4000f tile 244 `north_peel` | LEFT 63→56; **y never moved.** UP solid on the whole 244 row |
| 3 | `l6_south29_right_down` RIGHT+DOWN along 244 | play `0x39` `(120,93)` hop 282f | **1/1.** Same waist path as historical v4 from `(55,133)` |

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

Prior sitting continued (BLOCKED 3/3 more; dest still play `0x39`; keys=3
rod=1 TF=`0x1F` health `0x66` lo==hi; deaths/state-load/progression/capacity 0):

| # | policy / tag | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| west 4 | LEFT to x=40 then occupancy UP (`l6_south29_west_wall`) | `(40,157)` 4000f tile 244; `miss_f14_UP_40_157` / RIGHT@42 / DOWN@40 then `south_stand` | **x=40 is the west wall.** UP/RIGHT/DOWN all 244. Sand was x=41 tile 119 |
| west 5 | LEFT to sand x=41 then occupancy DOWN (`l6_south29_west_sand`) | `(40,157)` 4000f tile 244; `miss_f13_DOWN_40_157` | DOWN at x=41 **slides 41→40** into the wall. y never moved |
| west 6 | LEFT to sand x=47 then occupancy DOWN (`l6_south29_west_sand47`) | `(47,157)` 4000f tile 119; `miss_f9_DOWN_48_157` / LEFT@45 / DOWN@47 | **DOWN is solid at x=47–48.** Same leftover as west 3, other cardinal |

Live `SOUTH29_SPEC` is RIGHT+DOWN from `(63,133)` (1/1 into `0x39`).
Do not restore west 1–6, east-box LEFT, occupancy-DOWN from `(184,144)`,
cardinal UP @ y=133, or LEFT+UP @ y=133. Do not retry south29 from y=157.
Do not restore an occupancy seed that BFS-es UP off tile 244.

## Key deficit (still open; do not top-up)

Ended L6 compass keys=4 vs historical 5. Gap already at L4 `0x40`. clear58
had no natural key. room09 KEY-UP spent 4→3. south19 KEY-DOWN spent 3→2
(historical 4→3). clear29 floor key 2→3 (historical 3→4). Do not hide it
with a key top-up.

## Remaining L6 after south29 (not this sitting)

Phase 1: settle39 → clear39 → east39 → settle3a → clear3a.

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
(`HYPOTHESIZED_POST_L6_EXIT.verified=false`).

Bait-shop approach (`OverworldToBaitShopController`,
`--from-state Level6ExitOverworld`; 3-red off this sitting). Fixture-live
prefix `0x22→0x32→0x33→0x23→0x24→0x25`. `0x24→0x25` RIGHT @ y=141 is live
(`l7_bait_25` 1/1 1,438f). Leftover play `0x25` `(0,141)` west mouth;
Food=0, rupees=81; `success=true`, `route_eligible=false`. Dead: `0x32`
`(120,61)` DOWN; `0x33` RIGHT @ y=141; `0x24` DOWN at `(16,189)`,
`(160,189)` (north-ladder x), and `(208,189)` (SE). Also do not retry
occupancy xmin=14 west pocket `(0,141)` or SW occupancy box `(25,181)`.
Next sitting: inland RIGHT off `0x25` `(0,141)`, then DOWN toward `0x35`
(SW south ladder on the PNG). Do not LEFT back to `0x24`. Armos tap /
60R buy / pond `0x42` after 0x34.

## Non-claims

- Did not STATUS-promote or overwrite Clean M5.
- Did not close `rr-tne2` or reach Gohma / TF `0x20`.
- L7 sitting did not set `verified=true` or attach the spine.
- Did not poke Food / rupees / Whistle. Did not enter the 0x22 cave mouth.
- 3-red off this sitting; did not retry the three prior dead cells.
- 0x24↓0x34 abandoned (south mountain). 0x34 still unobserved.
- L6 sitting did not poke doors, Map, or capacity. The L7 poke fixture
  disclosed TF/Rod/Bow/Whistle/sword/rupee writes; Food stayed 0.
- Did not retouch maze-west, 0x40, stairs09, or the cellar walk.
- Did not retry west-aisle stairs09.
- Did not retry the `(55,133)` RIGHT+DOWN clip or any east-leftover south29.
- Did not run a fourth south29 from leftover `(56,157)` after the prior
  sitting's 3 reds.
- Did not restore west-only chase (clear29 reds 1–2) or LEFT at north
  mouth y=77 (clear29 red 3). Did not retry south29 from y=157.
- Did not restore cardinal UP or LEFT+UP at y=133 after they boxed on
  tile 244. Did not run settle39 this sitting.
- Gut: deleted unused `room29_south_grid` (dated UP from `(63,133)`) and
  dead `clip_xmin`/`clip_xmax` (west LEFT-peel). clear29 leftover is
  `settle_fight(..., success=clear29_handoff_ok)`.
- Did not push.

## Residual — rr-tne2 L6 Survival compose (stairs / Gohma / TF `0x20`)

**Status:** operator 0x3A position cheat is **live**. `--through
level6-stairs3a-warp` **1/1** dest mode 9 `0x08`. `--through
level6-cellar08` **1/1** west mouth returns play `0x3A` `(96,157)` (two-mouth
same-room cellar, not Gohma). `--through level6-center3a` v1 red: center
hole is **not** CheckWarp (timeout `(112,141)` tile 118). `--through
level6-east3a` **BLOCKED 3/3**; y-align to `(96,141)` **live**; RIGHT 0px
tile 118. Do **not** east3a v4. Walk-on stairs hops stay BLOCKED 3/3; do
not v4. Bead `rr-tne2` still open. TF still `0x1F`. Bow=0 arrows=0. Do
not fight Gohma. Do not poke ADDR_BOW/ADDR_ARROWS. Do not KEY-UP `0x09`.

**Pin:** cellar08 leftover play `0x3A` `(96,157)` rod=1 keys=4 bombs=8
bow=0 arrows=0 TF=`0x1F` map=`0x0A`. Center hole revealed; east door PNG
open; west door open. Warp used **one** Link-position write
`(112,149)→(208,93)`. Tape `l6_cellar08_continuous_v2` 220,267f.
PNG: `nes/zelda_i/recordings/l6_cellar08_continuous_v2_final.png`.

Do **not** STATUS-promote. Do not overwrite Clean M5. Glance leave with
`zelda_i.screen_glance` — no MP4. `--no-video` on spine CLIs. Occupancy
halt at first **new** miss. Failed hops now publish leftover on
`ControllerStageResult.report()["leftover"]`; `grade_controller` /
`grade_stage_report` return that leftover even when glance misses is
non-empty. Specs: `CLEAR_3A`, `CELLAR08_LEAVE` (play 0x3A (96,157)),
`STAIRS3A_DEST` (mode 9 cellar 0x08). The next hop starts from leftover
still, not a re-clear of 3a. Walk-on stairs remain BLOCKED.

### Already green (do not re-prove)

Closed L1–L5 Survival spine. L6 hops through `--through level6-clear3a`
1/1 leftover play `0x3A` `(144,141)`. `--through level6-stairs3a-warp`
1/1 dest mode 9 `0x08` `(208,93)` position_writes=1 hop 86f tape 219,735f.
`--through level6-cellar08` 1/1 play `0x3A` `(96,157)` hop 532f tape
220,267f. Dedicated skipped: stairs3a-neunder/neclip/ne71/ne/71, west39*,
clear39-west, south28, aisle28, south38, clear38-south, bomb38-south,
east38*, west38, exit-ow, clear28-south, east28, west28, aisle-west28.
north39 / inland29 / west19 / south18 stay green north-leave (skipped).

### Current hop (`level6-east3a`) — BLOCKED 3 serial reds

| Field | Live |
|-------|------|
| Start | cellar08 play `0x3A` `(96,157)` |
| v1 | occupancy UP `(96,157)` 2px halt `(96,155)` tile 118 |
| v2 | y-align live to `(96,143)`; RIGHT 0px occupancy_halt tile 119 |
| v3 | y-align live to `(96,141)`; RIGHT 0px occupancy_halt tile 118 |

All stayed play `0x3A`; deaths/progression/capacity 0; bow=0 arrows=0;
position_writes=1 (warp only); no state load. No east3a v4. Dated: x=96
y=141 RIGHT 0px (west of center hole). East door PNG-open, not reached.

### Prior hop (`level6-center3a`) — red 1/3, not this checkbox

Walk onto center hole from spit. Hole idle/UP yo-yo `(112,140↔141)` tile
118/119 still mode 5 timeout 4000f. Center hole is **not** CheckWarp.
Do not center3a v2.

### Prior hop (`level6-cellar08`) — 1/1

West mouth UP of cellar `0x08` is play `0x3A` `(96,157)` (center-stairs
spit). Both mouths of 0x08 are in 0x3A. Not the Gohma passage.

### Prior hop (`level6-stairs3a-warp`) — 1/1

Live center 0x68 push, one `ADDR_LINK_X`/`Y` write `(112,149)→(208,93)`.
Dest mode 9 room `0x08` leftover `(208,93)` tile 113. Assist:
position_writes=1, progression/capacity/door/inventory/TF writes 0,
mid_run_state_load=false, seamed=false.

### Historical reds (not this checkbox)

`--through level6-stairs3a-neunder` 3 serial reds. `--through
level6-stairs3a-neclip` 3 serial reds. `--through
level6-stairs3a-ne71` 3 serial reds. `--through level6-stairs3a-ne` 3
serial reds. `--through level6-stairs3a-71` 3 serial reds. `--through
level6-stairs3a` 3 serial reds. west39-reband / upclip / west39 /
clear39-west / aisle-west28 / west28 / east28 / west38 / east38-lane /
east38 / bomb38-south / clear38-south / south38 / aisle28 / south28 /
exit-ow 3 serial reds. stairs18 v1–v5 red.

### Next action — blocked

- **No east3a v4.** y=141 at x=96 RIGHT is sealed (tile 118, west of the
  hole). Retarget from offline geometry: south-around the hole (y≈157)
  then RIGHT, then y=141 to the east mouth. New file. Do not walk Gohma.
  Do not poke bow/arrows.
- Do not start stairs3a* v4 / west39* v4 / center3a v2. Do not KEY-UP
  `0x09`. Do not take 0x29. Do not invent Gohma.

Do not rerun this checkbox unchanged.

### Non-claims

- Did not STATUS-promote
- Did not overwrite Clean M5
- Did not poke doors/keys/bow/arrows/undiscovered items
- Did not grant Map/Whistle
- Did not write room/door/inventory/TF/capacity/facing/mode (only one
  disclosed 0x3A Link-position pair)
- Did not load state
- Did not KEY-UP `0x09`
- Did not close `rr-tne2`
- Did not fight Gohma
- Did not start east3a v4
- Did not start center3a v2
- Did not start stairs3a* v4
- Did not start L7 fixture-only work

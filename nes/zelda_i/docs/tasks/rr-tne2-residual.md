## Residual — rr-tne2 L6 Survival compose (stairs / Gohma / TF `0x20`)

**Status:** `--through level6-west39-reband` **1/3 red**. Occupancy
DOWN first new miss `DOWN (127,133)` tile **118**. Dest stayed `0x39`.
Reband DOWN at dated leftover `(125,133)` **live** (`reband_125_133_tile=118`;
x slid 125→127, y stayed 133 — y-dead). west39 / clear39-west /
west39-upclip / aisle-west28 / west28 stay BLOCKED 3/3 (no v4). Bead
`rr-tne2` still open. Predecessor `--through level6-clear3a` **1/1**
do not re-prove. stairs3a / south28 / aisle28 / south38 / clear38-south
/ bomb38-south / east38 / east38-lane / west38 / exit-ow / east28 /
west28 / aisle-west28 stay dedicated reds. north39 / inland29 / west19 /
south18 stay green north-leave (skipped as this through). Do **not**
start west39 v4, clear39-west v4, west39-upclip v4, aisle-west28 v4,
west28 v4, east28 v4, west38 v4, or any 0x38 mouth v4. Do not KEY-UP
`0x09`. Do not CheckWarp. Do not invent Gohma. Do not bomb. Do not
take 0x29.

**Pin:** clear3a leftover play `0x3A` `(144,141)` rod=1 keys=4 bombs=8
bow=0 arrows=0 TF=`0x1F` map=`0x0A`. PNG:
`nes/zelda_i/recordings/l6_clear3a_continuous_v1_final.png` — west door
open, center 0x68 unpushed. Tape 219,649f. Do not grant Map. Do not poke
doors/keys/bow/arrows.

Do **not** STATUS-promote. Do not overwrite Clean M5. Glance leave with
`zelda_i.screen_glance` — no MP4. `--no-video` on spine CLIs. Occupancy
halt at first miss ([predict-path](../../../../.grok/skills/predict-path/SKILL.md)).

### Already green (do not re-prove)

Closed L1–L5 Survival spine. L6 hops through `--through level6-clear3a`
1/1 leftover play `0x3A` `(144,141)`. North-leave through south18 1/1
is a different through (skipped here). Split: `survival_spine.py` 712 /
`level6_spine.py` 773 / `level6_spine_suffix.py` 638 /
`level6_west39.py` 334 / `level6_clear39_west.py` 393 /
`level6_west39_upclip.py` 456 / `level6_west39_reband.py` 484. Dedicated
skipped: stairs3a, south28, aisle28, south38, clear38-south,
bomb38-south, east38, east38-lane, west38, exit-ow, clear28-south
(0x38 trap), **east28 (3/3)**, **west28 (3/3)**, **aisle-west28 (3/3)**,
**west39 (3/3)**, **clear39-west (3/3)**, **west39-upclip (3/3)**. Unit
tests passed (hygiene + through-list after clear3a is west39-reband +
leftover occupancy LEFT not UP/RIGHT/B + leftover LEFT miss replan not
halt + 0x3A `(32,93)` west_align DOWN not UP + dest 0x39 reclear then
y=141 LEFT not north + dated `(144,109)` LEFT+DOWN clip not occupancy
DOWN + dated `(142,141)` / `(139,141)` LEFT+DOWN clip not occupancy
LEFT + dated `(136,141)` LEFT+UP not LEFT+DOWN + occupancy LEFT on clip
band y=138 + halt first new miss `(133,138)` + dated `(133,133)`
LEFT+DOWN not occupancy LEFT / not LEFT+UP + occupancy LEFT on new band
y=136 + halt first new miss `(130,136)` + dated `(130,133)` LEFT+UP not
LEFT+DOWN + occupancy LEFT on clip band y=130 + halt first new miss
`(127,130)` + dated leftover `(125,133)` DOWN not occupancy LEFT / not
LEFT+DOWN / not LEFT+UP + OccupancyWalker LEFT on y=141 after reband +
halt first new miss `(125,141)` + dest RAM ≠ `0x3A`/`0x29`/`0x39` +
fail 0x29 / 0x09 / cellar / backtrack 0x3A + no bomb + ignore 0x2B).

### Prior hop (`level6-west39`) — BLOCKED 3/3 (skip; no v4)

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | play mode 5, dest **≠ `0x3A` and ≠ `0x29`** — **not reached** |
| v1 | leftover LEFT miss f2 `(144,141)` tile **119**. Occupancy halt. |
| v2 | west mouth `(32,93)` tile 200 occupancy_stand timeout. Dest `0x3A`. |
| v3 | west_align entered play `0x39` `(208,141)` f14864. Reclear started. Occupancy WEST miss `DOWN (144,109)` tile **118**. Dest stayed `0x39`. |

### Prior hop (`level6-clear39-west`) — BLOCKED 3/3 (skip; no v4)

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | play mode 5, dest **≠ `0x3A` and ≠ `0x29` and ≠ `0x39`** — **not reached** |
| v1 | occupancy y=141 LEFT `(142,141)` tile **119** (0px). Halt. |
| v2 | LEFT+DOWN clip live; LEFT 0px `(139,141)` tile **119**. Halt. |
| v3 | LEFT+DOWN clip live; LEFT 0px leftover `(136,141)` tile **117**. Halt. West mouth undated. |

### Prior hop (`level6-west39-upclip`) — BLOCKED 3/3 (skip; no v4)

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | play mode 5, dest **≠ `0x3A` and ≠ `0x29` and ≠ `0x39`** — **not reached** |
| v1 | LEFT+UP at `(136,141)` live. Occupancy y=133 LEFT `(133,133)` tile **116**. Halt. |
| v2 | LEFT+DOWN at `(133,133)` live; occupancy LEFT `(130,133)` tile **116**. Halt. |
| v3 | LEFT+UP at `(130,133)` **y-dead** (~5px west, y stayed 133). Occupancy y=133 LEFT leftover `(125,133)` tile **118**. Halt. Dest stayed `0x39`. |

### This hop (`level6-west39-reband`) — 1/3 red

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | play mode 5, dest **≠ `0x3A` and ≠ `0x29` and ≠ `0x39`** (RAM — not invented), rod=1, TF=`0x1F` — **not reached** |
| v1 | `l6_west39_reband_continuous_v1` hop **18,786f** tape 238,435f prefix 219,649f. v3 enter live: west_align/west_push arrived play `0x39` `(208,141)` keys=4. Reclear started `reclear_39_207_141`. LEFT+DOWN clips dated `(144,109)` / `(142,141)` / `(139,141)` onto y=141. LEFT+UP clip at `(136,141)` tile 117 **live**. LEFT+DOWN clip at `(133,133)` tile 116 **live**. LEFT+UP clip at `(130,133)` tile 116 **live**. Reband DOWN at dated leftover `(125,133)` tile 118 **live** (`reband_125_133_tile=118`; x slid 125→127, y stayed 133 — **y-dead**). Occupancy **DOWN** first new miss: `DOWN (127,133)` tile **118**. Occupancy **halt**. Dest stayed `0x39`. Did not occupancy LEFT at y=133. Did not reach y=141 door band. |

Glance v1: mode 5, room `0x39`, `(127,133)`, TF=`0x1F`, rod=1 keys=4
bombs=8 bow=0 arrows=0 map=`0x0A`, health `0x66` lo==hi, deaths 0,
`cur_opened_doors=9`, `open_doorway_mask=0`, tile 118. PNG:
`nes/zelda_i/recordings/l6_west39_reband_continuous_v1_final.png` — dark
0x39, Link still on y=133 just east of the statue (2px east of the
upclip v3 leftover), west mouth PNG-open not reached, no Vire in the
still. Continuous session, seamed=false, mid_run_state_load=false,
progression_writes=0, capacity_writes=0. stairs3a not attached. Did
not enter `0x29`. Did not start west39 v4. Did not start
clear39-west v4. Did not start west39-upclip v4.

Dated:

- Occupancy LEFT leftover `(144,141)` miss f2 stuck 0px tile **119**
  (f1 tile 118). v1 west39 halt. This hop replanned (no leftover halt)
  then west_align entered `0x39` `(208,141)` keys=4.
- west39 v3 occupancy WEST first miss `DOWN (144,109)` tile **118**.
  This hop **clipped** LEFT+DOWN onto y=141.
- clear39-west v1–v3 occupancy y=141 LEFT `(142,141)` / `(139,141)` /
  `(136,141)` tiles 119/117. This hop **clipped** LEFT+DOWN at
  `(142,141)` / `(139,141)`, then **LEFT+UP** at `(136,141)` tile 117
  (east39 reverse of RIGHT+UP). Clip notes `upclip_136_141_tile=117`;
  bands `upclip_band_140` … `upclip_band_133`.
- upclip v1 miss: occupancy y=133 LEFT `(133,133)` tile **116** 0px.
  This hop **clipped** LEFT+DOWN.
- upclip v2 miss: occupancy y=133 LEFT `(130,133)` tile **116**. This
  hop **clipped** LEFT+UP (y-dead).
- upclip v3 miss: occupancy y=133 LEFT `(125,133)` tile **118**. This
  hop **rebanded** cardinal DOWN (not occupancy LEFT). Reband **live**
  then y-dead: x slid 125→127, y stayed 133. Occupancy DOWN halt
  leftover `(127,133)` tile **118**. West mouth of 0x39 undated.
  `cur_opened_doors=9` (N+E); west bit not set.
- Did not KEY-UP north to `0x29`. Did not take 0x29. Did not CheckWarp
  0x3A stairs. Isolated BFS unused.

### Historical reds (not this checkbox)

`--through level6-west39-upclip` 3 serial reds (halted; occupancy LEFT
`(125,133)` tile 118). `--through level6-clear39-west` 3 serial reds
(halted; occupancy LEFT `(136,141)` tile 117). `--through
level6-west39` 3 serial reds (halted; v3 occupancy DOWN `(144,109)`
tile 118). `--through level6-aisle-west28` 3 serial reds (halted;
occupancy DOWN x=120 boxed north diamond `(120,93)` tile 178).
`--through level6-west28` 3 serial reds (halted; occupancy LEFT along
north diamond boxed x=96). `--through level6-east28` 3 serial reds
(halted; east mouth tile **223** no-op). `--through level6-west38` 3
serial reds (halted; west mouth tile **222** no-op). `--through
level6-east38-lane` 3 serial reds (halted; east tile 223). `--through
level6-east38` 3 serial reds (halted; boxed south of y=141).
`--through level6-bomb38-south` 3 serial reds (halted; consume ≠ wall).
`--through level6-clear38-south` 3 serial reds (halted; reclear analog
**false**). `--through level6-south38` 3 serial reds (halted).
`--through level6-aisle28` 3 serial reds (halted; no v4). `--through
level6-south28` 3 serial reds (halted; no v4). `--through
level6-exit-ow` 3 serial reds (halted; no v4). stairs3a 3 serial reds
(push yes, warp no). stairs18 v1–v5 red (north hole of 0x18 is
decorative, not mode 9).

### Next action (not this worker)

- **1/3 red** on `level6-west39-reband`. Occupancy DOWN first new miss
  `DOWN (127,133)` tile **118**. Dest stayed `0x39`. Reband DOWN at
  `(125,133)` **live** (x slid 125→127, y unchanged — **y-dead**, then
  occupancy halt). West mouth undated. Do **not** occupancy LEFT at
  y=133 (dated upclip v3). Do **not** occupancy DOWN at `(125,133)`
  (dated this v1). Halt at 3 serial reds on this checkbox.
- Do not occupancy DOWN at `(144,109)` (dated, clipped). Do not
  occupancy LEFT at `(142,141)` / `(139,141)` / `(136,141)` (dated,
  clipped). Do not occupancy LEFT at `(133,133)` (dated upclip v1,
  clipped). Do not occupancy LEFT at `(130,133)` (dated upclip v2,
  clipped LEFT+UP y-dead). Do not occupancy LEFT at `(125,133)`
  (dated upclip v3, rebanded this v1). Do not KEY-UP north from 0x39.
  Do not take 0x29. Ignore 0x2B. Do not bomb. Do not CheckWarp 0x3A
  stairs.
- `level6-west39` / `level6-clear39-west` / `level6-west39-upclip` /
  `level6-west28` / `level6-east28` / `level6-west38` /
  `level6-east38` / `level6-east38-lane` / `level6-bomb38-south` /
  `level6-south38` / `level6-clear38-south` / `level6-south28` /
  `level6-aisle28` / `level6-aisle-west28` / `level6-exit-ow` /
  `level6-stairs3a` / `level6-clear28-south` stay dedicated. Do not
  start those v4.
- Isolated BFS banned. Do not poke doors/keys/bow/arrows. Do not KEY-UP
  `0x09`. Do not invent Gohma. Do not poke bomb counts.

### Non-claims

- Did not STATUS-promote
- Did not overwrite Clean M5
- Did not poke doors/keys/bow/arrows/undiscovered items
- Did not grant Map/Whistle
- Did not CheckWarp
- Did not KEY-UP `0x09`
- Did not close `rr-tne2`
- Did not start west28 v4
- Did not start aisle-west28 v4
- Did not take 0x29
- Did not bomb
- Did not enter `0x38`
- Did not start west39 v4
- Did not start clear39-west v4
- Did not start west39-upclip v4
- Did not occupancy LEFT at y=133 (dated upclip v3; rebanded DOWN instead)
- Did not start east28 v4, west38 v4, east38-lane v4, east38 v4, bomb38-south v4, clear38-south v4, south38 v4, aisle28 v4, south28 v4, or clear28-south v2

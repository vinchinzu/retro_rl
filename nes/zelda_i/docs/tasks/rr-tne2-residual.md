## Residual — rr-tne2 L6 Survival compose (stairs / Gohma / TF `0x20`)

**Status:** `--through level6-stairs3a-71` **1 red** (v1). Occupancy
halt `(114,149)` tile **116** with **misses still 1**. Dest stayed
`0x3A`. LEFT+DOWN clip from leftover `(144,141)` tile 118 **live**
(x 144→114, y 141→149). Center 0x68 **unpushed** `(112,144)`. Did not
reach tile `0x71`. Did not idle on tile 119. Did not hold-UP past the
hole. Bead `rr-tne2` still open. Predecessor `--through level6-clear3a`
**1/1** do not re-prove. west39-reband / west39-upclip / west39 /
clear39-west / stairs3a stay dedicated reds (no v4). Do **not** start
west39-reband v4, west39-upclip v4, stairs3a v4. Do not take 0x29. Do
not fight Gohma. Do not poke ADDR_BOW/ADDR_ARROWS. Do not KEY-UP `0x09`.
Do not walk east door unarmed.

**Pin:** clear3a leftover play `0x3A` `(144,141)` rod=1 keys=4 bombs=8
bow=0 arrows=0 TF=`0x1F` map=`0x0A`. PNG:
`nes/zelda_i/recordings/l6_clear3a_continuous_v1_final.png` — west door
open, center 0x68 unpushed. Tape 219,649f. Do not grant Map. Do not poke
doors/keys/bow/arrows.

Do **not** STATUS-promote. Do not overwrite Clean M5. Glance leave with
`zelda_i.screen_glance` — no MP4. `--no-video` on spine CLIs. Occupancy
halt at first **new** miss ([predict-path](../../../../.grok/skills/predict-path/SKILL.md)).

### Already green (do not re-prove)

Closed L1–L5 Survival spine. L6 hops through `--through level6-clear3a`
1/1 leftover play `0x3A` `(144,141)`. North-leave through south18 1/1
is a different through (skipped here). Split: `survival_spine.py` 713 /
`level6_spine.py` 779 / `level6_spine_suffix.py` 645 /
`level6_stairs3a_71.py` 554. Dedicated skipped: stairs3a, west39-reband,
west39-upclip, west39, clear39-west, south28, aisle28, south38,
clear38-south, bomb38-south, east38, east38-lane, west38, exit-ow,
clear28-south, east28, west28, aisle-west28. north39 / inland29 /
west19 / south18 stay green north-leave (skipped as this through). Unit
tests passed (hygiene + through-list after clear3a is stairs3a-71 +
leftover LEFT+DOWN clip not UP + tile 0x71 still-stand not UP + tile
119 sidestep not idle + east-door fail + dest 0x29 / 0x09 / 0x39 fail
+ occupancy halt first new miss + dest RAM mode 9).

### This hop (`level6-stairs3a-71`) — 1 red (v1)

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | mode 9 **or** play dest ≠ `0x3A` — **not reached** |
| v1 | leftover LEFT+DOWN clip **live**; occupancy halt `(114,149)` tile **116** with misses still 1. Dest `0x3A`. Center 0x68 unpushed. |

Prefix: `--through level6-clear3a` 1/1 leftover play `0x3A`
`(144,141)`. Continuous session, seamed=false,
mid_run_state_load=false, progression_writes=0, capacity_writes=0,
deaths 0. stairs3a / west39-reband not composed. Did not enter `0x29`.
Did not walk east door.

Glance v1: mode 5, room `0x3A`, `(114,149)`, TF=`0x1F`, rod=1 keys=4
bombs=8 bow=0 arrows=0 map=`0x0A`, health `0x66` lo==hi, deaths 0,
tile 116. PNG:
`nes/zelda_i/recordings/l6_stairs3a_71_continuous_v1_final.png` — west
door open, center 0x68 still at `(112,144)` unpushed, stairs graphic
not revealed, east door closed, Link SW of the block, bubble residual.

Dated:

- Occupancy DOWN leftover `(144,141)` miss f2 tile **118**. **Clipped**
  LEFT+DOWN (not occupancy halt). Clip live: `(144,141)` tile 118 →
  `(136,141)` tile 117 → `(128,144)` tile 118 → `(120,149)` tile 116 →
  leftover `(114,149)` tile **116**. Hop **28f** tape 219,677f prefix
  219,649f. `l6_stairs3a_71_continuous_v1`.
- Halt was **not** a new occupancy miss (`misses` stayed 1). Clip
  geometry ended when x=114 reached dest x=112 ±2; y still 149 <
  south-face 160. Policy occupancy_halted instead of DOWN y-align.
- Did not push center 0x68. Did not find tile `0x71`. Did not idle on
  tile 119 (v3 stairs3a). Did not hold-UP past the hole (v2 stairs3a
  leftover `(112,133)` tile 179). Isolated BFS unused.

### Historical reds (not this checkbox)

`--through level6-west39-reband` 3 serial reds (halted; LEFT+DOWN at
`(125,133)` y-dead leftover `(124,133)` tile 118). `--through
level6-west39-upclip` 3 serial reds (halted; occupancy LEFT `(125,133)`
tile 118). `--through level6-clear39-west` 3 serial reds (halted;
occupancy LEFT `(136,141)` tile 117). `--through level6-west39` 3
serial reds (halted; v3 occupancy DOWN `(144,109)` tile 118).
`--through level6-aisle-west28` 3 serial reds. `--through
level6-west28` 3 serial reds. `--through level6-east28` 3 serial reds.
`--through level6-west38` 3 serial reds. `--through level6-east38-lane`
3 serial reds. `--through level6-east38` 3 serial reds. `--through
level6-bomb38-south` 3 serial reds. `--through level6-clear38-south` 3
serial reds. `--through level6-south38` 3 serial reds. `--through
level6-aisle28` 3 serial reds. `--through level6-south28` 3 serial
reds. `--through level6-exit-ow` 3 serial reds. stairs3a 3 serial reds
(push yes, warp no; v3 idle tile 119 still mode 5). stairs18 v1–v5 red
(north hole of 0x18 is decorative, not mode 9).

### Next action (not this worker)

- **1 red** on `level6-stairs3a-71` v1. One change: after leftover
  LEFT+DOWN clip x-aligns (x≈112±2), **DOWN** to south-face y=160. Do
  **not** occupancy_halt when clip geometry ends with misses still 1
  (dated v1 leftover `(114,149)` tile 116). Then reuse live stairs3a
  push (south-face UP, y-move 8px) and tile **0x71** still-stand
  (stairs09 analog). Do not idle on tile 119. Do not hold-UP past the
  hole. Occupancy halt at first **new** miss only.
- Do not start west39-reband v4 / west39-upclip v4 / stairs3a v4 /
  west39 v4 / clear39-west v4. Do not KEY-UP north from 0x39. Do not
  take 0x29. Do not walk east door unarmed. Do not invent Gohma. Do not
  poke bow/arrows/doors/keys. Isolated BFS banned.

```bash
QT_QPA_PLATFORM=offscreen UV_CACHE_DIR=/tmp/retro_rl_uv_cache \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-stairs3a-71 --no-video --trials 1 --tag l6_stairs3a_71_continuous_v2
```

### Non-claims

- Did not STATUS-promote
- Did not overwrite Clean M5
- Did not poke doors/keys/bow/arrows/undiscovered items
- Did not grant Map/Whistle
- Did not KEY-UP `0x09`
- Did not close `rr-tne2`
- Did not start west39-reband v4
- Did not start west39-upclip v4
- Did not start stairs3a v4
- Did not take 0x29
- Did not fight Gohma
- Did not walk east door unarmed
- Did not idle on tile 119
- Did not hold-UP past the hole
- Did not occupancy_halt leftover `(144,141)` (dated; clipped)
- Did not start a second trial

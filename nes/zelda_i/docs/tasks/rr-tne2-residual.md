## Residual — rr-tne2 L6 Survival compose (stairs / Gohma / TF `0x20`)

**Status:** `--through level6-stairs3a-71` **BLOCKED 3 serial reds**.
v1 occupancy_halt `(114,149)` tile 116 misses still 1. v2 push live,
TO_NE y-first UP 0px leftover `(72,165)` tile 116. v3 RIGHT to x=184
live, then tile **119** at `(184,147)` RIGHT 0px / timeout. Dest stayed
`0x3A`. Did not reach tile `0x71`. East door **open** (do not walk).
Bead `rr-tne2` still open. Predecessor `--through level6-clear3a` **1/1**
do not re-prove. west39-reband / west39-upclip / west39 / clear39-west /
stairs3a stay dedicated reds (no v4). Do **not** start stairs3a-71 v4.
Do not take 0x29. Do not fight Gohma. Do not poke ADDR_BOW/ADDR_ARROWS.
Do not KEY-UP `0x09`. Do not walk east door unarmed.

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
`level6_stairs3a_71.py` 571. Dedicated skipped: stairs3a, west39-reband,
west39-upclip, west39, clear39-west, south28, aisle28, south38,
clear38-south, bomb38-south, east38, east38-lane, west38, exit-ow,
clear28-south, east28, west28, aisle-west28. north39 / inland29 /
west19 / south18 stay green north-leave (skipped as this through). Unit
tests passed (hygiene + through-list after clear3a is stairs3a-71 +
leftover LEFT+DOWN clip not UP + clip x-align DOWN not occupancy_halt +
tile 0x71 still-stand not UP + tile 119 sidestep not idle + v2 leftover
`(72,165)` RIGHT not UP + x>=184 UP + east-door fail + dest 0x29 / 0x09
/ 0x39 fail + occupancy halt first new miss + dest RAM mode 9).

### This hop (`level6-stairs3a-71`) — BLOCKED 3 reds

| Field | Live |
|-------|------|
| Start | clear3a play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 TF=`0x1F` |
| Stop | mode 9 **or** play dest ≠ `0x3A` — **not reached** |
| v1 | leftover LEFT+DOWN clip **live**; occupancy halt `(114,149)` tile **116** with misses still 1. Dest `0x3A`. Center 0x68 unpushed. |
| v2 | clip + DOWN y-align **live**; south-face UP y-move 8px **live** (`pushed_112_144_to_112_136`); TO_NE y-first UP 0px leftover `(72,165)` tile 116 timeout. Stairs revealed. East door open. |
| v3 | RIGHT to x=184 **live**; leftover `(184,147)` tile **119** RIGHT 0px / timeout 4000f. Dest `0x3A`. Did not reach `0x71`. |

Prefix: `--through level6-clear3a` 1/1 leftover play `0x3A`
`(144,141)`. Continuous session, seamed=false,
mid_run_state_load=false, progression_writes=0, capacity_writes=0,
deaths 0. stairs3a / west39-reband not composed. Did not enter `0x29`.
Did not walk east door.

Glance v3: mode 5, room `0x3A`, `(184,147)`, TF=`0x1F`, rod=1 keys=4
bombs=8 bow=0 arrows=0 map=`0x0A`, health `0x66` lo==hi, deaths 0,
tile 119. PNG:
`nes/zelda_i/recordings/l6_stairs3a_71_continuous_v3_final.png` — west
door open, center stairs revealed, NE 0x68 at `(208,96)`, east door
**open**, Link on tile 119 at the NE column, bubble SE. Did not idle
still-stand (RIGHT on 119). Did not enter east door (x=184 < 200).

Dated:

- Occupancy DOWN leftover `(144,141)` miss f2 tile **118**. **Clipped**
  LEFT+DOWN (not occupancy halt). Clip live: `(144,141)` tile 118 →
  `(136,141)` tile 117 → `(128,144)` tile 118 → `(120,149)` tile 116.
- v1 halt `(114,149)` tile 116, misses still 1. Clip x-aligned; y=149 <
  south-face 160. `l6_stairs3a_71_continuous_v1` hop 28f tape 219,677f.
- v2 DOWN after clip x-align **live**. `stand_y` `(112,152)` then
  `at_push_112_158_block_112_144`, `pushed_112_144_to_112_136`. TO_NE
  y-first UP from `(121,149)` 0px at y=165 tile 116, knockback west.
  Leftover `(72,165)` tile 116 timeout 4000f tape 223,649f.
  `l6_stairs3a_71_continuous_v2`. Stairs graphic revealed. East door
  open. Hold-UP also hit `(120,133)` tile **179** (stairs3a v2 class).
- v3 RIGHT until x>=184 **live** (f88–f128 `(124,149)`→`(177,149)`).
  Then `ne_y` UP at `(185,149)`. Tile **119** at NE column `(184,147)`
  forces `ne_sidestep` RIGHT (do not idle 119). RIGHT 0px; knockback
  yo-yo y=149↔181 / x=184↔154. Timeout leftover `(184,147)` tile 119
  misses still 1. `l6_stairs3a_71_continuous_v3` hop 4000f tape
  223,649f. Dest `0x3A` mode 5. Did not find tile `0x71`. Isolated BFS
  unused.

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

- **BLOCKED 3/3** on `level6-stairs3a-71`. **No v4.** Occupancy halt at
  first **new** miss. Reuse live push of center 0x68. Still-stand on
  tile **`0x71`** (not 119). Dest RAM mode 9 or play ≠ `0x3A`.
- Dated v3: NE column x=184 y=147 **is tile 119**. Hole-tile RIGHT
  sidestep 0px there. Do not idle 119. Do not hold-UP past the hole.
  Do not walk east door (open after push). Isolated BFS banned.
- Do not start west39-reband v4 / west39-upclip v4 / stairs3a v4 /
  west39 v4 / clear39-west v4 / stairs3a-71 v4. Do not KEY-UP north
  from 0x39. Do not take 0x29. Do not invent Gohma. Do not poke
  bow/arrows/doors/keys.

```bash
# Do not run v4. Manager retarget.
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
- Did not start stairs3a-71 v4
- Did not take 0x29
- Did not fight Gohma
- Did not walk east door unarmed
- Did not idle on tile 119 (sidestep RIGHT; 0px at `(184,147)`)
- Did not hold-UP past the hole (v3 RIGHT-first; v2 dated `(120,133)` tile 179)
- Did not occupancy_halt leftover `(144,141)` (dated; clipped)
- Did not occupancy_halt clip x-align `(114,149)` (v2 DOWN live)

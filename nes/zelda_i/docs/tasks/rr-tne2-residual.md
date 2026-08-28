# Residual — rr-tne2 L6 Survival (L1 bow cellar stairs 1/1)

**Status:** `--through level6-north2c` is still **1/1** play `0x1C`
`(120,205)` Gohma unfought. `--through level1-bow` is **1/1** enter-stop
play `0x22` `(224,141)` keys 1→0. `--through level1-bow-cellar` is now
**1/1** mode 9 room `0x7F` `(128,141)` tile `0x71` (`l1_bow_cellar`, hop
270f). `ADDR_BOW` remains 0. Bead `rr-tne2` stays open. Do not fight
Gohma or poke `ADDR_BOW` / `ADDR_ARROWS`. `--through level2` and later
still skip this branch. Default `--through level1` still skips bow and
still finishes L1 TF.

## L6 dest (still the live leftover)

`--through level6-north2c` 1/1: `l6_north2c_continuous.json` 221,280f,
play `0x1C` `(120,205)`, keys 4→3, rod=1, bow=0, arrows=0, TF=`0x1F`,
HUD ~39R. Red Gohma needs one wooden arrow in the open eye. L6 has no
bow room. Arrow shop ~80R; leftover ~39R is short.

## L1 bow insert

Enter-stop `--through level1-bow` **1/1** (`l1_bow22_x112_v2`): play
`0x22` `(224,141)` keys 1→0 hop 345f. Live 0x68 west block `(96,144)`.
Q1 ROM `0x22` E=key N/S/W=wall secret=none item=`0x03` mon=`0x0A`.
`--through level2` and later still skip this branch. Bow stays a side
`--through` until walked `ADDR_BOW` (no poke) + backtrack to play
`0x23` exist; do not splice a red pickup hop into the L2–L6 tape.

`--through level1-bow-cellar` (`level1_bow_cellar.py`): dest mode 9 **1/1**.
Continuous power-on Survival, no state load, deaths 0, progression /
capacity / position writes 0. West `0x68` UP `(96,144)→(96,128)`. Glance
`BOW_CELLAR_LEAVE` is mode 9 `0x7F` `(128,141)`. `ADDR_BOW` still 0.

The staircase is visible at room center before the push. The west block
pushes **UP**, not DOWN. CheckWarps UW needs X a multiple of `$10` and
Y `($10k+$D)`; tile `$70–$73`. Idle at x=126 (tile `0x73`) never warps.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 `l1_bow_cellar_v1` | `0x22` `(208,93)` tile 119 | Occupancy straight to north face `(96,128)`. Climbed the north wall; boxed on the NE statue. Block `(96,144)` live. |
| v2 `l1_bow_cellar_v2` | `0x22` `(208,93)` tile 203 | Peel UP at inland x=208. UP 141→109 **live**; LEFT at `(208,109)` is the statue. Same y=93 box. |
| v3 `l1_bow_cellar_v3` | `0x22` `(176,141)` tile 118 | Door-band LEFT to x=160 through the diamond. LEFT at y=141 **live to `(176,141)`** (east diamond face). Dest x=160 is the block. |
| v4 `l1_bow_cellar_v4` | `0x22` `(144,109)` tile 118 | Peel UP at x=176 **live** (y 141→110). LEFT at y=109 176→144 **live**. LEFT at `(144,109)` is the **north diamond**. |
| northwall `l1_bow_cellar_northwall` | `0x22` `(112,109)` tile 178 | From leftover UP y=93 **live**; LEFT y=93 144→113 **live**; then tile 119 **bricked north door** at x≈120. Occupancy DOWN into diamond; boxed north vertex. |
| south189 `l1_bow_cellar_south189` | `0x22` `(176,189)` tile 117 | DOWN x=176 141→189 **live** (passed y=173). LEFT y=189 176→127 **live**; then tile 119 **bricked south door**. Oscillated 176↔208. |
| south173 `l1_bow_cellar_south173` | `0x22` `(144,173)` tile 118 | LEFT y=173 from live `(176,173)` 176→144 **live**. LEFT at `(144,173)` is the **south diamond** (SE point; v4 mirror). PNG: stairs still covered. |
| south157 `l1_bow_cellar_south157` | `0x22` `(160,157)` tile 118 | LEFT y=157 from live `(176,157)` 176→160 **live**; then tile 118 **SE diamond edge**. Occupancy stood (no path to x=64). PNG: stairs still covered. Wrong belief: y=157 is south of the east block. SE diagonal is `(176,141)→(160,157)→(144,173)`. |
| south128 `l1_bow_cellar_south128` | `0x22` `(128,181)` tile 179 | LEFT y=189 176→128 **live** (east of south door). Cardinal UP 189→181 then **south diamond face**. Occupancy LEFT 128→125 then stood (no path to north-face y). Pocket between south diamond and south-door column. |
| clip `l1_bow_cellar_south128_clip` | `0x22` `(176,149)` tile 116 | LEFT+UP at `(128,188)` **slid east** 128→130→150 y=181, then occupancy south_peel stood and bounced to east diamond `(176,149)`. Wrong belief: LEFT+UP at the south face clips west (L6 y=181). PNG: still on the east diamond; stairs covered. |
| north-lane occupancy | `0x22` `(176,125)` tile 119 | LEFT requested at y=128 completed the prior 8px UP grid line to y=125; OccupancyWalker called the legal steering a miss and stood. |
| north-lane direct | `0x22` `(160,125)` tile 179 | Holding LEFT after the grid completion reached the north/east block corner, then stayed solid. y=128 is not a full west lane to the block. |
| natural south `l1_bow_cellar_north_lane` | `0x22` `(96,149)` tile 176, block `(96,128)` | Cardinal UP first stabilized the south face, then LEFT+UP threaded to the block. UP push was live `(96,144)→(96,128)`. Wrong final order: RIGHT from `(96,149)` cannot enter the center; walk UP into the vacated slot first. |
| slot-then-RIGHT idle `l1_bow_cellar` trial 1 | `0x22` `(126,141)` tile 115 (`0x73`) | UP-through-slot then RIGHT **live** onto the stairs. Wrong idle: `PUSH_ALIGN_TOL` stopped at x=126. CheckWarps UW requires X multiple of `$10`; y=141 already matches `$10k+$D`. Tile `0x73` is a legal stairs tile (`$70–$73`). |
| **1/1** `l1_bow_cellar` | mode 9 `0x7F` `(128,141)` tile 113 (`0x71`) | Exact x=128 then idle. Hop 270f, tape 15187f. Glance `BOW_CELLAR_LEAVE` matched. `ADDR_BOW=0`. |

Live path: leftover `(224,141)` LEFT along y=141 to `(176,141)`; DOWN to
y=189; LEFT to x=128; cardinal UP first to the stable y=181 face; then
LEFT+UP threads southwest; UP push `(96,144)→(96,128)`; UP through
vacated slot y=144; RIGHT to exact x=128 y=141; idle for CheckWarps.

Do not LEFT past x=144 at y=109 or y=173 (diamond). Do not LEFT at y=157
past x=160 (SE edge). Do not LEFT at y=93 or y=189 past x≈120 (bricked
N/S door columns). Do not UP at x=208 (statue) or target the west aisle.
Do not idle at x=126. Do not push DOWN.

Next: walk onto the bow in mode 9 `0x7F` leftover `(128,141)` (no
`ADDR_BOW` poke). Then cellar exit, east `0x22` → play `0x23`. L1 TF
finish on this branch still waits on walked `ADDR_BOW` + backtrack to
play `0x23`.

## Splice (not this tape)

`--through level1` and `--through level2+` share `level1_triforce_stages`.
That table is currently `clear23_key` → `backtrack44` → TF. `backtrack44`
assumes play `0x23`. Insert the bow detour **between those two**, after
pickup + return are green:

1. `level1_bow_0x22` KEY-LEFT — already **1/1**
2. `level1_bow_cellar` stairs — dest mode 9 **1/1**
3. mode-9 walk onto the bow (no `ADDR_BOW` poke) — still open
4. cellar exit, east `0x22` → play `0x23`

Then existing `backtrack44`. That is how L6 Gohma gets a bow: `--through
level2+` already runs this L1 prefix. Do not splice a red pickup hop.
Do not change default `--through level1` until step 3–4 are 1/1.
`--through level1-bow` / `--through level1-bow-cellar` stay side-branch
early-returns until that insert lands.

## Non-claims

- Did not pick up `ADDR_BOW`; did not buy arrows.
- Did not kill Gohma; did not collect L6 Heart / TF `0x20`.
- Did not STATUS-promote or overwrite Clean M5.
- Did not poke doors, keys, bow, arrows, rupees, undiscovered items,
  progression, or capacity.
- Did not splice bow into `--through level1` / `--through level2` / L6.
  Did not close `rr-tne2`, start L7, or push.

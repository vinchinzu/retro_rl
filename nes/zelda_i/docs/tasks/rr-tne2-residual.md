# Residual — rr-tne2 L6 Survival (Gohma unarmed; L1 0x22 stairs)

**Status:** `--through level6-north2c` is still **1/1** play `0x1C`
`(120,205)` Gohma unfought. `--through level1-bow` is **1/1** enter-stop
play `0x22` `(224,141)` keys 1→0. `--through level1-bow-cellar` dest
mode 9 is **open**; LEFT at y=157 **stopped** at `(160,157)` on the SE
diamond edge. `ADDR_BOW` still 0. Bead `rr-tne2` stays open. Do not
fight Gohma. Do not poke `ADDR_BOW` / `ADDR_ARROWS`. `--through level2`
and later still skip this branch. Default `--through level1` still
skips bow and still finishes L1 TF.

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
`--through` until mode 9 + `ADDR_BOW` + backtrack exist; do not splice
a red stairs hop into the L2–L6 tape.

`--through level1-bow-cellar` (`level1_bow_cellar.py`): dest mode 9, no
`ADDR_BOW`. Wiki: push west block DOWN, stairs, cellar. Policy now
south-lane y=157 then west aisle (last trial).

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

Live so far (do not regress): leftover `(224,141)` LEFT along y=141 to
`(176,141)`; DOWN at x=176 through `(176,157)` to `(176,173)` and on to
`(176,189)`; UP at x=176 to y=109; LEFT y=109 to `(144,109)`; UP at
x=144 to y=93; LEFT y=93 to x=113; LEFT y=173 to `(144,173)`; LEFT y=157
to `(160,157)`. y=189 LEFT reached **x=127** live (west of diamond
x=144) then tile 119 south door; occupancy retraced 176↔208. West 0x68
stays `(96,144)`. deaths/progression/capacity 0.

Do not LEFT past x=144 at y=109 or y=173 (diamond). Do not LEFT at y=157
past x=160 (SE edge). Do not LEFT at y=93 or y=189 past x≈120 (bricked
N/S door columns). Do not UP at x=208 (statue).

Next retarget (offline): from live `(176,189)` LEFT only to x=128 (west
of south diamond x=144; east of south-door column x≈120), then UP.
south189 already proved y=189 LEFT to x=127. Do not dest west-aisle at
y=189. Do not go back to Gohma. L1 TF finish stays after mode 9 +
`ADDR_BOW` + backtrack; do not splice a red stairs hop into
`--through level1` / `--through level2`.

## Non-claims

- Did not enter mode 9; did not pick up `ADDR_BOW`; did not buy arrows.
- Did not kill Gohma; did not collect L6 Heart / TF `0x20`.
- Did not STATUS-promote or overwrite Clean M5.
- Did not poke doors, keys, bow, arrows, rupees, undiscovered items,
  progression, or capacity.
- Did not splice bow into `--through level2` / L6. Did not close
  `rr-tne2`, start L7, or push.

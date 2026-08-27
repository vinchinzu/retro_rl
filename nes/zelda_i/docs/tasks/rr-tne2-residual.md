# Residual — rr-tne2 L6 Survival (Gohma unarmed; L1 0x22 stairs peel)

**Status:** `--through level6-north2c` is still **1/1** play `0x1C`
`(120,205)` Gohma unfought. `--through level1-bow` is **1/1** enter-stop
play `0x22` `(224,141)` keys 1→0. `--through level1-bow-cellar` dest
mode 9 is **open**; north-peel LEFT at y=109 **stopped** at the diamond
`(144,109)`. `ADDR_BOW` still 0. Bead `rr-tne2` stays open. Do not fight
Gohma. Do not poke `ADDR_BOW` / `ADDR_ARROWS`. No peel v5.

## L6 dest (still the live leftover)

`--through level6-north2c` 1/1: `l6_north2c_continuous.json` 221,280f,
play `0x1C` `(120,205)`, keys 4→3, rod=1, bow=0, arrows=0, TF=`0x1F`,
HUD ~39R. Red Gohma needs one wooden arrow in the open eye. L6 has no
bow room. Arrow shop ~80R; leftover ~39R is short.

## L1 bow insert

Enter-stop `--through level1-bow` **1/1** (`l1_bow22_x112_v2`): play
`0x22` `(224,141)` keys 1→0 hop 345f. Live 0x68 west block `(96,144)`.
Q1 ROM `0x22` E=key N/S/W=wall secret=none item=`0x03` mon=`0x0A`.
`--through level2` and later still skip this branch.

`--through level1-bow-cellar` (new file `level1_bow_cellar.py`): dest
mode 9, no `ADDR_BOW`. Wiki: push west block DOWN, stairs, cellar.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 `l1_bow_cellar_v1` | `0x22` `(208,93)` tile 119 | Occupancy straight to north face `(96,128)`. Climbed the north wall; boxed on the NE statue. Block `(96,144)` live. |
| v2 `l1_bow_cellar_v2` | `0x22` `(208,93)` tile 203 | Peel UP at inland x=208. UP 141→109 **live**; LEFT at `(208,109)` is the statue. Same y=93 box. |
| v3 `l1_bow_cellar_v3` | `0x22` `(176,141)` tile 118 | Door-band LEFT to x=160 through the diamond. LEFT at y=141 **live to `(176,141)`** (east diamond face). Dest x=160 is the block. |
| v4 `l1_bow_cellar_v4` | `0x22` `(144,109)` tile 118 | Peel UP at x=176 **live** (y 141→110). LEFT at y=109 176→144 **live**. LEFT at `(144,109)` is the **north diamond** (not a corridor). No v5. |

Live so far (do not regress): leftover `(224,141)` LEFT along y=141 to
`(176,141)`; UP at x=176 to y=109; LEFT to `(144,109)`. West 0x68 stays
`(96,144)`. deaths/progression/capacity 0. PNG: Link on the NE diamond
point, stairs still covered.

Next retarget (offline, not a 5th peel): from `(144,109)` **UP to y=93**
(west of the NE statue at x=176–208), LEFT along the north wall to
x=96, DOWN onto north face `(96,128)`, then DOWN-push. South-around
y=173 from `(176,141)` is untested. Do not LEFT at y=109. Do not UP at
x=208.

## Non-claims

- Did not enter mode 9; did not pick up `ADDR_BOW`; did not buy arrows.
- Did not kill Gohma; did not collect L6 Heart / TF `0x20`.
- Did not STATUS-promote or overwrite Clean M5.
- Did not poke doors, keys, bow, arrows, rupees, undiscovered items,
  progression, or capacity.
- Did not splice bow into `--through level2` / L6. Did not close
  `rr-tne2`, start L7, or push.

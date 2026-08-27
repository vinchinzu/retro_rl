# Residual — rr-tne2 L6 Survival (Gohma 0x1C unarmed; L1 bow BLOCKED)

**Status:** `--through level6-north2c` is still **1/1** play `0x1C`
`(120,205)`. Gohma is on screen, not fought. `--through level1-bow` is
**BLOCKED 3/3** on the west-wall-around checkbox in L1 `0x23`. Bead
`rr-tne2` stays open. Do not fight Gohma. Do not poke `ADDR_BOW` /
`ADDR_ARROWS`. No westwall v4.

## L6 dest (still the live leftover)

`--through level6-north2c` 1/1: `l6_north2c_continuous.json` 221,280f,
hop 393f, play `0x1C` `(120,205)`, keys 4→3, rod=1, bow=0, arrows=0,
TF=`0x1F`, bombs=8, HUD ~39R. PNG: Gohma on screen, north shutter black.
`position_writes=1` (the 0x3A warp only). deaths/progression/capacity 0.

ROM: `0x1C` N=shutter S=key item=heart mon=`0x34`. `0x0C` north is TF
`0x20` after the kill. Do not walk `0x2C` south (`0x3C`).

Wiki (Zelda Dungeon / GameFAQs): red Gohma dies to **one arrow in the
open eye**. Magical Rod does not replace the bow. L6 has no bow room.
Wooden arrows are a shop buy (~80R); leftover ~39R is short. Do not
use candle shop `0x5E` (Shield/Key/Candle, no arrows). Gathering hyp
arrow shop `0x6B` is **not live**.

## L1 bow insert — BLOCKED 3/3 (west-wall around)

Prefix insert (spine re-runs from power-on) is the bow source: west of
verified `0x23`. Q1 ROM `$18700`: `0x23` W=**key**, dest `0x22` E=key
N/S/W=wall item=`0x03` secret=none. Walkthrough: 4 blade traps, westmost
block, stairs, cellar bow. **Enter-stop `0x22` is not on tape.**

`--through level1-bow` (Survival-only; Clean M5 unchanged) runs prefix
through `clear23_key` then occupancy KEY-LEFT. Live start after
`clear23_key` is play `0x23` `(136,117)` keys=1 bombs=0.

Clip checkbox (prior session) BLOCKED 3/3 — not this hop:

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 `l1_bow22_continuous` | `0x23` `(136,125)` tile 119 | y-align DOWN to 141. Documented `(136,125)` water pocket. |
| v2 `l1_bow22_continuous_v2` | `0x23` `(64,141)` tile 244 | west aisle then cardinal LEFT. LEFT is the moat. Upper channel LEFT to `(64,117)` **was live**. |
| v3 `l1_bow22_continuous_v3` | `0x23` `(64,136)` tile 118 | LEFT+UP clip at `(64,141)`. UP 141→136; LEFT 0px. |

West-wall around checkbox **BLOCKED 3/3** (this session; no v4):

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 `l1_bow22_westwall` | `0x23` `(69,117)` tile 116 | `DOOR_TOL=4` aisle fudge. Switched to north-band UP at x=66; stood east of x=64. |
| v2 `l1_bow22_westwall_v2` | `0x23` `(64,117)` tile 119 | UP at west aisle x=64 from y=117 reaches y=93. **Solid.** Channel LEFT to `(64,117)` still live. |
| v3 `l1_bow22_westwall_v3` | `0x23` `(80,117)` tile 119 | `ROOM_23_SPEC` `(80,93)` means UP at x=80 from y=117. LEFT to `(80,117)` **live**; UP **solid** (tile 119 ceiling). 18572f hop 4000f; deaths/progression/capacity 0. |

Upper channel y=117 is a 1-tile corridor: UP solid at both x=64 and x=80
(tile 119). `(80,93)` is walkable on the combat patrol, but not by
climbing the channel at x=80.

No v4 under session gate. Next retarget (offline, not a 4th westwall):
**UP at x=112** (`ROOM_23_SPEC` `(112,93)` / `(114,117)` / `(112,133)`),
then LEFT along north y=93 onto `(32,141)`. South band y=189 is still
untested; do not DOWN at x≈128 (v1 pocket). Do not LEFT+UP clip.

## Non-claims

- Did not kill Gohma; did not collect L6 Heart / TF `0x20`.
- Did not reach play `0x22` or `ADDR_BOW`.
- Did not STATUS-promote or overwrite Clean M5.
- Did not poke doors, keys, bow, arrows, rupees, undiscovered items,
  progression, or capacity.
- Did not close `rr-tne2`, start L7, or push.

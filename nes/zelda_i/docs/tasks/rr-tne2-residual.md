# Residual — rr-tne2 L6 Survival (Gohma 0x1C unarmed; L1 bow enter-stop 1/1)

**Status:** `--through level6-north2c` is still **1/1** play `0x1C`
`(120,205)` with Gohma on screen, unfought. `--through level1-bow` is
**1/1** enter-stop play `0x22` `(224,141)` keys 1→0. `ADDR_BOW` still 0.
Bead `rr-tne2` stays open. Do not fight Gohma. Do not poke `ADDR_BOW` /
`ADDR_ARROWS`. Do not splice the bow branch into `--through level2` yet.

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

## L1 bow insert — enter-stop 0x22 1/1

Prefix insert (spine re-runs from power-on) is the bow source: west of
verified `0x23`. Q1 ROM `$18700`: `0x23` W=**key**, dest `0x22` E=key
N/S/W=wall item=`0x03` secret=none. Walkthrough: 4 blade traps, westmost
block, stairs, cellar bow.

`--through level1-bow` (Survival-only; Clean M5 unchanged) runs prefix
through `clear23_key` then occupancy KEY-LEFT. Live start after
`clear23_key` is play `0x23` `(136,117)` keys=1 bombs=0.

West-wall around checkbox **BLOCKED 3/3** (dated; not this hop):

| tag | leftover | wrong belief |
|-----|----------|----------------|
| v1 `l1_bow22_westwall` | `0x23` `(69,117)` tile 116 | `DOOR_TOL=4` aisle fudge. Switched to north-band UP at x=66. |
| v2 `l1_bow22_westwall_v2` | `0x23` `(64,117)` tile 119 | UP at west aisle x=64 from y=117 reaches y=93. **Solid.** |
| v3 `l1_bow22_westwall_v3` | `0x23` `(80,117)` tile 119 | `ROOM_23_SPEC` `(80,93)` means UP at x=80 from y=117. LEFT to `(80,117)` **live**; UP **solid**. |

Plus-stem x=112 (this session):

| tag | leftover | result |
|-----|----------|--------|
| v1 `l1_bow22_x112` | `0x23` `(32,117)` tile 200 | Climb UP at x=112 **live** (y 117→94). North LEFT to `(33,93)` **live**. Door-drop DOWN to y=109 **live**. Wrong belief: `y>97` means still on the channel — that pull sent `(32,109)` east. |
| v2 `l1_bow22_x112_v2` | play `0x22` `(224,141)` | **1/1** hop 345f, 14,917f total. keys 1→0. bow=0 arrows=0. deaths/progression/capacity 0. Glance `BOW22_LEAVE` matched. PNG: 4 blade traps + center stairs. |

Policy that closed enter-stop: occupancy LEFT to `(112,117)`, UP y=93,
LEFT `(32,93)`, DOWN `(32,141)`, KEY-LEFT. West column y>97 is door-drop,
not channel recovery.

`--through level2` and later still skip this branch. Do not splice until
`ADDR_BOW` is owned.

## Next hop (one boundary)

Play `0x22` east mouth `(224,141)`: blade traps, push the westmost block,
stairs, cellar, pick up the bow. Do not claim `ADDR_BOW` on the enter-stop.
After bow: farm/buy ~80R wooden arrows, then return to L6 `0x1C` with
B = arrows. Do not fight Gohma unarmed.

## Non-claims

- Did not kill Gohma; did not collect L6 Heart / TF `0x20`.
- Did not pick up `ADDR_BOW` (enter-stop only). Did not buy arrows.
- Did not STATUS-promote or overwrite Clean M5.
- Did not poke doors, keys, bow, arrows, rupees, undiscovered items,
  progression, or capacity.
- Did not splice bow into `--through level2` / L6. Did not close
  `rr-tne2`, start L7, or push.

# Residual — rr-tne2 L6 power-on recompose to Gohma wing

**Status:** power-on Survival spine is **1/1** through `--through level6-gohma`
(play `0x1C` body gone). Bead `rr-tne2` stays open until TF `0x20`. Do not
STATUS-promote.

## Green from power-on this sitting (2026-09-01)

| through | leftover | notes |
|---------|----------|-------|
| level6-west2d | play `0x2C` `(224,141)` keys=3 | cardinal y-align then LEFT; hop 335f; 0 misses |
| level6-north2c | play `0x1C` `(120,205)` keys 3→2 | cardinal x-align then KEY-UP; hop 308f; 0 misses |
| level6-gohma | play `0x1C` `(120,189)` keys=2 | wooden arrows 0→1; body gone hop 54f; pulses=1 |

Prefix through cellar08/south1d still 1/1 (keys=3). Glance: TF=`0x1F`,
rod=1, Bow=1, bombs=8, health `0x66` lo==hi. Deaths / state loads /
progression / capacity writes 0. One 0x3A position write. Compose is
power-on spine (no `--from-state`). **Keys are 2 here, not historical 3 —
do not top up.**

## What was wrong with west2d / north2c occupancy

Power-on leftover for west2d is north-mouth `(120,77)`. Occupancy y-align
LEFT false-misses the waist (2px DOWN, then wizzrobe knockback), stands at
`(80,141)`, and drifts to the SW pocket `(32,189)` tile 221
(`survival_spine.json` through north2c, 66 misses).

Power-on leftover for north2c is east-mouth `(224,141)`. Occupancy LEFT
false-misses, BFS wants DOWN, `south_open_halt` stands, wizzrobes shuffle x
along y=141. Leftover `(71,141)` keys still 3 (`l6_north2c_recompose`).

**Fix:** `WEST2D_SPEC` / `NORTH2C_SPEC` now `cardinal_hold` + `align` y/x.
`Level6DoorHopController._hold` cardinal-aligns then holds the door
button. Same class as `EAST39_SPEC`. Tests:
`test_west2d_align_y_then_left`, `test_north2c_align_x_then_up`.

## Gohma kill (wired hop, 1/1)

`--through level6-gohma` 1/1 (`l6_gohma_recompose`, 212,260f hop 54f).
One wooden-arrow grant (`arrow_poke_writes=1_from=0`), B-slot 2, Bow
untouched. PNG: Gohma sparkle + heart on floor, north shutter black.
Enter-stop leftover was unarmed; this hop is the kill. TF still `0x1F`.

## Next sitting

Heart in `0x1C` then north shutter `0x0C` TF `0x20`. No spine hop exists
for that yet. Do not poke TF/doors. No STATUS/close/push.

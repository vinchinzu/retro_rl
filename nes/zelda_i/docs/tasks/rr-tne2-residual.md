# Residual — rr-tne2 L6 0x29 south-door leftover

**Status:** `--through level6-south29 --tag l6_south29_clip` **1/1** from
power-on. Bead `rr-tne2` stays open. Do not STATUS-promote.

Clear 0x29 leftover is the south door **(120, 189)**. After last kill the
controller clips RIGHT+DOWN to the y=141 waist (cardinals cannot round the
plus — RIGHT @ y=109 boxed at x=96), aligns x=120, then DOWN. SOUTH29 is
SOUTH19-shaped: hold DOWN into play `0x39` `(120,93)`.

| through | tag | leftover | hop |
|---------|-----|----------|-----|
| level6-clear29 | l6_south29_clip | play `0x29` south door, then SOUTH29 | 1,411f |
| level6-south29 | l6_south29_clip | play `0x39` `(120,93)` keys=3 | 199f |

Glance: play `0x39` `(120,93)` rod=1 keys=3 TF=`0x1F` bombs=8 Bow=1 health
`0x66` lo==hi. Deaths / progression / capacity writes 0. Mid-dungeon L6
pins dropped; compose is power-on spine (no `--from-state`).

`clear29.py` / `inland29.py` deleted. Do not leftover on tile 244 at y=133.
Do not retry east `(184,144)` or west y=157. Next sitting:
`--through level6-settle39`.

# Residual — rr-tne2 L6 power-on recompose to TF 0x20

**Status:** power-on Survival spine is **1/1** through `--through level6`
(fanfare `0x0C` TF `0x3F`). Bead `rr-tne2` stays open. Do not STATUS-promote.
Do not close. Do not push.

Sitting 2026-09-01: claimed `rr-17co` (south-band east-column CheckWarp).
One policy, one `--through level6-stairs3a-warp` trial, stop at first red.
`rr-wabn` first checkbox in parallel: leftover already has rupees; reports
still omit them; probe an 80R merchant. Do not splice arrows into default
`--through level6`. Wooden-arrow grant may stay. Then Phase 4 `rr-ibkf`.

## Green from power-on (2026-09-01)

| through | leftover | notes |
|---------|----------|-------|
| level6-heart | play `0x1C` `(120,149)` health `0x77` | occupancy UP; hop 31f; hc 7→8; item still `0x1A` at stop |
| level6-north0c | play `0x0C` `(120,205)` keys=2 | `cur_opened_doors` already had UP; cardinal UP; hop 209f |
| level6 | fanfare `0x0C` `(120,149)` TF `0x3F` | occupancy onto shard; hop 43f; mode 18 |

Prefix through Gohma still 1/1 (hop 54f, arrows 0→1). Glance: TF `0x1F→0x3F`,
rod=1, Bow=1, bombs=8, keys=2, health `0x77` lo==hi, accepted_containers=8.
Deaths 0 / claimed state-load 0 / progression / capacity writes 0. One 0x3A
position write. One wooden-arrow grant. Compose is power-on spine (no
`--from-state`). **Keys stay 2 — do not top up.** `status_claim=false`.
L6 reports omitted rupees; record them on the next leftover.

Heart / shutter / shard hops stay dedicated SpineHops after Gohma
(`level6-heart`, `level6-north0c`, canonical `level6`). `triforce_writes=0`.

## Position cheat removal (locked)

ne71 v1–v3 all pushed center 0x68 `(112,144)→(112,136)` and saw NE 0x68 at
`(208,96)`, then occupancy-halted on the y=149 waist:

| tag | leftover | wrong belief |
|-----|----------|----------------|
| ne71 v1 | `0x3A` `(158,149)` | LEFT around tile 119 at y=149 reaches the NE block |
| ne71 v2 | `0x3A` `(144,149)` | occupancy UP at y=149 after that LEFT miss |
| ne71 v3 | `0x3A` `(136,149)` | occupancy UP again at the same waist |

Center hole after the push is tile 119 / `0x77` (decorative, same class as
the 0x09/0x18 fake holes). Real CheckWarp is tile `0x71` at `(208,93)`,
the 0x09 analog. The poke teleports onto that cell. Walk there.

Replacement hop, **one module**, dest glance `STAIRS3A_DEST` (mode 9 cellar
`0x08`):

1. Same live center push.
2. Peel **south** of y=149 (`CLIP_CLEAR_Y=181` or south band y=189). Do not
   occupancy toward `(208,96)` along y=149.
3. Cardinal RIGHT on that south band to east column x=208. Fail if screen
   becomes `0x3B` (east door is y=141, still open).
4. UP the east column to south-face of NE 0x68 `(208,96)`.
5. UP onto tile `0x71` at `(208,93)`. Hold until mode 9.

Do not restore through-names `stairs3a` / `-71` / `-ne` / `-ne71`. Do not
idle the center hole. `position_writes` must be 0. Cellar08 B-side cross is
unchanged. Fail publishes leftover. Three serial reds on this checkbox
blocks.

## Arrow splice (plan)

`ADDR_ARROWS` is item type, not ammo. Enemy drops do not grant ownership.
Rupees are ammo after the 80R buy. L6 red Gohma is 3 wooden shots.

Bow is already on the L1 Survival splice. Do not exit L6 to fetch arrows.
Window: after `level1-bow-pickup`, before Gohma.

`0x5E` is live Shield 160 / Key 100 / Candle 60, **no arrows**. Gathering
hyp `0x6B` (start right×3, up, right) is not live. First checkbox: record
rupees on a leftover, farm Octoroks to ≥80 if short, probe a merchant until
`ADDR_ARROWS` 0→1. `CandleShop5E` buy geometry is the template (settle, UP
stairs, touch pedestal, DOWN exit). Do not poke rupees.

Wire like the bow detour. `--through level1-arrows` stop on arrows=1.
`poke_wooden_arrows` already skips when arrows are wooden. Clean M5 skips.

## Next sitting

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-stairs3a-warp --no-video --trials 1 \
  --tag l6_stairs3a_southband
```

One policy: south-band east-column, not occupancy at y=149. Stop at the
first red. No STATUS/push.

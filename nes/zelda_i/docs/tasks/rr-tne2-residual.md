# Residual — rr-tne2 L6 power-on recompose to TF 0x20

**Status:** CheckWarp walk is **1/1** (`position_writes=0`). Cellar08 is
**1/1**. `--through level6` is **red** at Gohma occupancy_stand. Bead
`rr-tne2` stays open. `rr-17co` stays open. Do not STATUS-promote. Do not
close. Do not push.

Sitting 2026-09-01 claimed `rr-17co`. South-band east-column greened the
warp. Compose stopped at Gohma. `rr-wabn` found live shop `0x4A`; dedicated
`--through level1-arrows` red on the 80R farm. Wooden-arrow grant still
on Gohma. Then Phase 4 `rr-ibkf`.

## Green from power-on (2026-09-01)

| through | leftover | notes |
|---------|----------|-------|
| level6-heart | play `0x1C` `(120,149)` health `0x77` | occupancy UP; hop 31f; hc 7→8; item still `0x1A` at stop |
| level6-north0c | play `0x0C` `(120,205)` keys=2 | `cur_opened_doors` already had UP; cardinal UP; hop 209f |
| level6 | fanfare `0x0C` `(120,149)` TF `0x3F` | occupancy onto shard; hop 43f; mode 18 |

Prefix through Gohma was 1/1 on the **poked** warp (hop 54f, arrows 0→1).
Glance then: TF `0x1F→0x3F`, rod=1, Bow=1, bombs=8, keys=2, health `0x77`
lo==hi, accepted_containers=8. Deaths 0 / claimed state-load 0 /
progression / capacity writes 0. One wooden-arrow grant. Compose is
power-on spine (no `--from-state`). **Keys stay 2 — do not top up.**
`status_claim=false`. Leftover and `spine_final_fields` now record rupees
(this tape 43R).

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
idle the center hole. Fail publishes leftover. Three serial reds on this
checkbox blocks.

Walk is **1/1** (`l6_stairs3a_southband`): hop 290f, mode 9 cellar `0x08`
`(208,93)` tile `0x71`, notes `center_pushed` / `south_band_112_181` /
`east_column_208_181` / `warped_9_08_208_93`. `position_writes=0`.
Cellar08 **1/1** (`l6_cellar08_southband`): hop 465f, play `0x1D`
`(96,157)` keys=3 bombs=8 rupees=43. `--through level6`
(`l6_southband_finish`) **red** at `level6_gohma_0x1c` 20000f timeout.
north2c leftover `(120,205)` keys 3→2 still greened. Gohma: poke arrows
0→1, inland to y=165 by f32 (9 misses), then knockback to `(120,189)`
tile 118 and `occupancy_stand` from f80 (27 misses, no `arrow_shot`).
Type `0x34` same as the 54f poked-warp kill. HUD B ended on bombs.
Do not retry south-band. Do not occupancy at y=149. Fold `stairs3a.py`
after `--through level6` is 1/1.

## Arrow splice (plan)

`ADDR_ARROWS` is item type, not ammo. Enemy drops do not grant ownership.
Rupees are ammo after the 80R buy. L6 red Gohma is 3 wooden shots.

Bow is already on the L1 Survival splice. Do not exit L6 to fetch arrows.
Window: after `level1-bow-pickup`, before Gohma.

`0x5E` is live Shield 160 / Key 100 / Candle 60, **no arrows**. Gathering
hyp `0x6B` is not live. Live 80R merchant is OW **`0x4A`** (K-5): Magical
Shield 130 / Bombs 20 / Arrows 80. Cave mouth mode 16 `(176,77)`. Spawn
mode 11 `(112,213)`. Buy: settle, UP stairs, RIGHT on y=165 to x=152, UP
touch `(152,157)`. y=149 walks through mid bombs. Scratch recon poke 200R
got `ADDR_ARROWS` 0→1. Do not poke rupees on the spine.

`--through level1-arrows` is dedicated (not in `level1_survival_tf_stages`).
**Red** `l1_arrows`: farm timeout leftover play `0x4A` `(63,173)` bow=1
arrows=0 rupees 9→10 in 37665f. L1 exit leftover was 6R. Do not splice
into default L6. `poke_wooden_arrows` still on Gohma. Clean M5 skips.

## Next sitting

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-gohma --no-video --trials 1 \
  --tag l6_gohma_after_walk
```

One policy: Gohma inland after the walked warp. Occupancy boxed
`(120,189)` tile 118 after reaching y=165. Do not retry south-band. Do
not occupancy at y=149. Stop at the first red. No STATUS/push.

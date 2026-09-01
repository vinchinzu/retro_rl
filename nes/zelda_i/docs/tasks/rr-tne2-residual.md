# Residual — rr-tne2 L6 power-on recompose to TF 0x20

**Status:** CheckWarp walk is **1/1** (`position_writes=0`). Cellar08 is
**1/1**. `--through level6-gohma` after the walk is **red** (2/2 this
sitting). Bead `rr-tne2` stays open. `rr-17co` stays open. Do not
STATUS-promote. Do not close. Do not push.

Sitting 2026-09-01 claimed `rr-17co`. South-band east-column greened the
warp. Gohma inland after that walk is the red. `rr-wabn` found live shop
`0x4A`; dedicated `--through level1-arrows` red on the 80R farm. Wooden
arrows still poked on Gohma; rupees are ammo. Then Phase 4 `rr-ibkf`.

## Green from power-on (2026-09-01)

| through | leftover | notes |
|---------|----------|-------|
| level6-heart | play `0x1C` `(120,149)` health `0x77` | occupancy UP; hop 31f; hc 7→8; item still `0x1A` at stop |
| level6-north0c | play `0x0C` `(120,205)` keys=2 | `cur_opened_doors` already had UP; cardinal UP; hop 209f |
| level6 | fanfare `0x0C` `(120,149)` TF `0x3F` | occupancy onto shard; hop 43f; mode 18 |

Prefix through Gohma was 1/1 on the **poked** warp (hop 54f, arrows 0→1,
**1 pulse**, leftover rupees 42). Glance then: TF `0x1F→0x3F`, rod=1,
Bow=1, bombs=8, keys=2, health `0x77` lo==hi, accepted_containers=8.
Deaths 0 / claimed state-load 0 / progression / capacity writes 0. One
wooden-arrow grant. Compose is power-on spine (no `--from-state`).
**Keys stay 2 — do not top up.** `status_claim=false`.

Heart / shutter / shard hops stay dedicated SpineHops after Gohma
(`level6-heart`, `level6-north0c`, canonical `level6`). `triforce_writes=0`.

## Position cheat removal (locked)

Walk is **1/1** (`l6_stairs3a_southband`): hop 290f, mode 9 cellar `0x08`
`(208,93)` tile `0x71`, `position_writes=0`. Cellar08 **1/1**
(`l6_cellar08_southband`): hop 465f, play `0x1D` `(96,157)` keys=3
bombs=8 rupees=43. north2c leftover `(120,205)` keys 3→2 still greened.
Do not retry south-band. Do not occupancy at y=149. Fold `stairs3a.py`
after `--through level6` is 1/1.

## Gohma after walked warp (red 2/2)

Live body is type `0x34` HP 32 `gst=0` (same as the 54f poked-warp kill).
Poke still writes `ADDR_ARROWS` 0→1 + B=2. Rupees start 43 and are ammo.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| occupancy (`l6_southband_finish`) | `(120,189)` tile 118, 0 pulses | occupancy inland; knockback boxes UP → `occupancy_stand` |
| v1 cardinal+hold UP (`l6_gohma_after_walk`) | `(115,93)` tile 117, 599 pulses, rupees 0 | hold UP on cooldown; walked through the body; shots hit the north shutter |
| v2 hold y=165 (`l6_gohma_stand165`) | `(136,164)` tile 119, 795 pulses, rupees 43→0 by ~f2000, ghp stayed 32 | spray from y=165 after first shot f144; spawn-open window already missed |

Poked-warp kill was **1 pulse at ~f32** from y=169 x=120 while Gohma was
still gx=128. Walked-warp v2 first shot is f144 (align_x delay). `gst=0`
on both tapes — not an eye-open flag. Do not occupancy. Do not hold UP
through the body. Do not spray 43R at closed eye. HUD B is arrows on v1/v2
leftovers (bomb icon on the occupancy leftover was HUD lag).

## Arrow splice (plan)

`ADDR_ARROWS` is item type, not ammo. Rupees are ammo after the 80R buy.
Live 80R merchant is OW **`0x4A`**. `--through level1-arrows` **red**
`l1_arrows`: farm leftover play `0x4A` `(63,173)` rupees 9→10.
`poke_wooden_arrows` still on Gohma. Do not splice into default L6. Do
not close `rr-wabn`. Walked-warp Gohma spent the incoming 43R with no HP
drop — the shop is ammo, not just ownership.

## Next sitting

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-gohma --no-video --trials 1 \
  --tag l6_gohma_spawn_shot
```

One policy: shoot on the first stand frames (54f spawn window), not
align_x delay to f144. Stop at the first red. Do not occupancy. Do not
hold UP. Do not spray rupees. No STATUS/push.

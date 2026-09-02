# Residual — rr-tne2 L6 power-on recompose to TF 0x20

**Status:** CheckWarp walk re-validated **1/1** on this tape (`position_writes=0`,
mode 9 cellar `0x08` `(208,93)` tile `0x71`). Cellar08 **1/1**. north2c
**1/1**. `--through level6-gohma` column shot is **red** 1/1. Bead
`rr-tne2` stays open. `rr-17co` stays in progress. Do not STATUS-promote.
Do not push.

Sitting 2026-09-02 claimed `rr-tne2` (Gohma after walked warp). Warp is
not cheating: `stairs3a.py` / `stairs3a_warp.py` / `gohma.py` do not call
`poke_link_position` or `mem_write`. Live warp hop 290f,
`position_assist.position_writes=0`, `set_state_count=0` measured.
`STAIRS3A_DEST` glance keys spec is still 4 vs live 3; dest xy/mode/room
are the walk. Do not retry south-band. Do not occupancy at y=149.

## L4 Gleeok continuous TF-exit (green 1/1, prior sitting)

`--through level4 --no-video --trials 1 --tag l4_gleeok_continuous`:
ok, 110926f, `set_state_count=0`, TF `0x07→0x0F`, fanfare `0x03`
`(120,149)`. Spine Gleeok is `continuous_mode=True`.

## Warp / cellar / north2c (green on this tape)

`--through level6-gohma --tag l6_gohma_column_shot` prefix:

| hop | leftover | notes |
|-----|----------|-------|
| stairs3a-warp | mode 9 cellar `0x08` `(208,93)` tile `0x71` | hop 290f, `position_writes=0` |
| cellar08 | play `0x1D` `(96,157)` | hop 465f, glance clean |
| north2c | play `0x1C` `(120,205)` keys 3→2 | hop 324f, glance clean |

deaths 0 / `set_state_count=0` / progression / capacity writes 0.
Keys stay 2 after KEY-UP. Rupees 43 into Gohma. One wooden-arrow grant
on the Gohma hop (`ADDR_ARROWS` 0→1, B=2).

## Gohma after walked warp (red 4/4; this sitting 1/1)

Live body is type `0x34` HP 32 `gst=0`. Poke still writes `ADDR_ARROWS`
0→1 + B=2. Rupees start 43 and are ammo.

| tag | leftover | wrong belief |
|-----|----------|----------------|
| occupancy (`l6_southband_finish`) | `(120,189)` tile 118, 0 pulses | occupancy inland; knockback boxes UP → `occupancy_stand` |
| v1 cardinal+hold UP (`l6_gohma_after_walk`) | `(115,93)` tile 117, 599 pulses, rupees 0 | hold UP on cooldown; walked through the body; shots hit the north shutter |
| v2 hold y=165 (`l6_gohma_stand165`) | `(136,164)` tile 119, 795 pulses, rupees 43→0 by ~f2000, ghp stayed 32 | spray from y=165 after first shot f144; spawn-open window already missed |
| v3 stand-shot (`l6_gohma_spawn_shot`) | `(120,189)` tile 118, **1 pulse** f29, rupees 43→42, ghp stayed 32, hop 120f | 54f is a time window, not a free column; first stand is too late |
| v4 column mouth (`l6_gohma_column_shot`) | `(120,204)` tile 118, **1 pulse** f2 gx=128, rupees **43**, ghp stayed 32, hop 120f | UP+B at south mouth while gx=128 fires an arrow |

### v4 diagnosis (stop here)

`--through level6-gohma --no-video --trials 1 --tag l6_gohma_column_shot`:
ok=false, end 212549f, failed `level6_gohma_0x1c`, hop 120f.

| field | value |
|-------|-------|
| leftover | play `0x1C` `(120,204)` tile 118 mode 5 |
| TF / keys / bombs / rupees | `0x1F` / 2 / 8 / **43** |
| health | `0x66` lo==hi |
| bow / arrows | 1 / 1 (poked) |
| pulses | 1 at f2 `(120,205)` gx=128 gy=112 ghp=32 |
| dest glance | miss `y=204 not in [173, 197]` (body still live) |

Per-frame:

- f1 `column_wait` ghost type at gx=80 gy=93 ghp=0; poke lands (`arrows` 0→1)
- f2 spawn `gx=128 gy=112 ghp=32` dx=8, reason `arrow_shot` UP+B
- f3 leftover y=204 (one pixel inland), rupees still 43, ghp 32
- Gohma strafes right ~0.5 px/f; f64 gx=158, then left; f112 gx=138
- timeout f120 still `(120,204)` body live

Mouth UP+B does not spend a rupee. Tile 118 at y=205 eats the shot.
v3 from y=169 did spend 1R. Do not retry mouth UP+B. Do not occupancy.
Do not hold UP. Do not spray. Do not walk to y=165. Do not retry
south-band or y=149.

Patrol return toward gx=128 after f112 is untested (cap was 120).

## Arrow splice (plan)

`ADDR_ARROWS` is item type, not ammo. Rupees are ammo after the 80R buy.
Live 80R merchant is OW **`0x4A`**. `--through level1-arrows` **red**
`l1_arrows`: farm leftover play `0x4A` `(63,173)` rupees 9→10.
`poke_wooden_arrows` still on Gohma. Do not splice into default L6. Do
not close `rr-wabn`.

## Next sitting

Warp is not the cheat. Next is still Gohma after the walked warp
(`rr-17co`). Get inland of tile 118 / y=205 before B. Spawn gx=128 is
only live at the door. Stop at the first red.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-gohma --no-video --trials 1 \
  --tag l6_gohma_inland_column
```

One policy: a few UP frames off the south mouth, then one UP+B. Do not
mouth-shot. Do not occupancy. Do not hold UP. Do not spray. No
STATUS/push.

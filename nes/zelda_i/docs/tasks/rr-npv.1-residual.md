# rr-npv.1 residual — Clean L3 Entrance→TF dest hops

Stopped at fixture-live. `route_eligible=false`. Do not STATUS. Do not close
the bead. **Blocked:** 3 serial reds on `manhandla_tf`.

## Glance (this trial leftover)

room **0x4d**, mode **17** (death), xy **(138,173)**, tf **0x03**, keys **4**,
bombs **2**, health **0x70** (lo 0, hi 7), deaths **1**. PNG
`recordings/l3_entrance_tf_t0_final.png` (death, south band).

## Green dest hops (Level3Entrance pin, `--no-infinite-life --no-video`)

| Stage | Frames | Leftover |
|-------|--------|----------|
| west_key | 507 | 0x7b key dest |
| north_chain | 2791 | 0x5b Darknuts cleared |
| bomb_5b | 364 | play 0x5c; bombs 8→7 |
| clear_5c | 1073 | 0x5c Darknuts cleared |
| right_5d | 311 | play 0x5d (32,141) |
| clear_5d | 2932 | play 0x5d (120,175) |
| up_4d | 480 | play **0x4d (120,189)** |

Total frames at fail: 8597.

## Serial reds on `manhandla_tf` (blocked)

| Sitting | Leftover | What failed |
|---------|----------|-------------|
| 1 | (104,142) death | retreat then approach UP into the flower at the waist |
| 2 | (184,173) death | y=MAX `away` RIGHT into the east wall; first retreat DOWN to y=189 |
| **3 (this)** | **(138,173) death** | east wall and south-door retreat **gone**; heads still kill at y=MAX |

Do not poke. Do not bump `max_frames`. No fourth ROM trial.

This trial samples (`manhandla_tf` 139f, bombs 4→2):

| f | reason | xy | bombs | health |
|---|--------|-----|-------|--------|
| 1 | climb | (120,189) | 4 | 0x71 |
| 16 | approach | (125,173) | 4 | 0x71 |
| 48 | place_bomb | (160,167) | 4 | 0x71 |
| 64 | retreat_bomb | (152,173) | 3 | 0x71 |
| 80 | retreat_bomb | (169,173) | 3 | 0x70 |
| 96 | combat_backstep | (158,173) | 3 | 0x70 |
| 112 | approach | (149,173) | 3 | 0x70 |
| 128 | retreat_bomb | (154,173) | 2 | 0x70 |
| 139 | link_death | (138,173) | 2 | 0x70 |

x range 120–169 (not 184). y>=167 in fight; y=189 is spawn climb only. No y<141.

## Manhandla grade (this leftover only)

| Policy | This trial |
|--------|------------|
| south-band y>=141 / no north chase | green |
| no waist re-enter | green vs sitting 1 |
| no east-wall / south-door retreat | **green vs sitting 2** |
| dest TF 0x04 | **red**: still dies at y=MAX while heads live |

`writes=0`. `route_eligible=false`. `boss_path.py` 964 LOC.

## Next sitting

Blocked on Manhandla contact at the south stand (y=173). Do not walk east wall
or south door. Do not chase north. Isolated runner:

`run_level3_complete.py --from-state Level3Entrance --no-infinite-life --no-video --trials 1`

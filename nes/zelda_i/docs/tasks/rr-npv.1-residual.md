> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-npv.1 residual — Clean L3 Entrance→TF dest hops

Clean fixture-live: Level3Entrance → TF 0x04 dest hops completed green.
`route_eligible=false`. Do not touch STATUS. Do not close the bead.

## Glance (this trial leftover)

room **0x3d**, mode **18** (Triforce fanfare), xy **(120,149)**, tf **0x07** (bit 0x04 collected),
keys **4**, bombs **2**, health **129** (0x81 / hearts 8/1), deaths **0**. PNG
`recordings/l3_entrance_tf_t0_final.png` (Triforce pedestal room 0x3d).

## Green dest hops (Level3Entrance pin, `--no-infinite-life --no-video`)

| Stage | Frames | Leftover |
|-------|--------|----------|
| west_key | 507 | 0x7b key dest |
| north_chain | 2791 | 0x5b Darknuts cleared |
| bomb_5b | 364 | play 0x5c; bombs 8→7 |
| clear_5c | 1073 | 0x5c Darknuts cleared |
| right_5d | 311 | play 0x5d (32,141) |
| clear_5d | 2932 | play 0x5d (120,175) |
| up_4d | 480 | play 0x4d (120,189) |
| **manhandla_tf** | **526** | **play 0x3d (120,149) TF bit 0x04** |

Total frames: 8984.

## Resolution of `manhandla_tf` Serial Reds

| Sitting | Leftover | Outcome | Root cause / resolution |
|---------|----------|---------|-------------------------|
| 1 | (104,142) death | RED | retreat then approach UP into the flower at the waist |
| 2 | (184,173) death | RED | y=MAX `away` RIGHT into east wall; first retreat DOWN into south door |
| 3 | (138,173) death | RED | heads killed Link at y=MAX; bombs dropped without lead / fireball contact |
| **4 (this)** | **(120,149) TF 0x04** | **GREEN** | **Predictive interception controller**: zero damage, wiped heads, picked up HC, collected TF 0x04 |

### Controller Class Transformation

1. **Active Fireball Avoidance**:
   Scans `snap.objects` for incoming fireballs (`type_id == 0x56`) within Link's corridor and evades horizontally (`dodge_fireball`) away from projectile trajectories while respecting room boundaries.

2. **Centroid Interception Lead Bombing**:
   Manhandla's core is slot 5 with movement velocity given by the facing byte (`$0098 + 5`). Bomb fuse is 49 frames. Controller computes trajectory lead (`pred_cx`) and places bomb when Manhandla approaches south band (`124 <= cy <= 136`, `dy > 0`), exploding directly inside Manhandla's center and destroying all heads in one blast.

3. **Fuse-Synchronized Safe Retreat**:
   Expanded `retreat_frames` to 55 (matching the 49-frame bomb detonation). Retreat holds south band (`y=173`) and moves toward center column (`x > 120` moves LEFT; `x <= 120` moves RIGHT), preventing east-wall pinning or south-door re-entry.

4. **Heart Container & Shutter Push**:
   Upon boss defeat, Link collects the spawned Heart Container, aligns to `NORTH_DOOR_X = 120`, steps through the northern door into room `0x3d`, and touches Triforce piece `0x04` at `(120, 149)`.

## Manhandla Grade (this trial)

| Policy | This trial |
|--------|------------|
| south-band y>=141 / no north chase | **green** |
| no waist re-enter | **green** |
| no east-wall / south-door retreat | **green** |
| active fireball evasion | **green** |
| predictive centroid bomb placement | **green** |
| zero damage during boss fight | **green** |
| Heart Container pickup | **green** |
| dest TF 0x04 | **green** |

## Code Hygiene & Tests

- `nes/zelda_i/level3/boss_path.py`: 979 LOC (under ~1000 LOC soft max).
- Unit tests: 64 passed in 0.82s (`pytest nes/zelda_i/tests/test_level3*.py`).
- Isolated runner:
  `QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level3_complete.py --from-state Level3Entrance --no-infinite-life --no-video --trials 1`
  exited 0 (`ok=True`, `deaths=0`, `tf04=True`).

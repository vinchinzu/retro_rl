# rr-4oz residual — Clean L2 heatmap hotspots 0x0e / 0x4f

Fixture-live. `route_eligible=false`. Do not STATUS. Do not close the bead.
Do not poke bombs/keys. Do not `--infinite-life`.

Hottest Survival `damage_by_location`: **0x0e=15**, **0x4f=4**.

## This sitting (2026-09-10)

Do not 1px-grade a moving Dodongo as a wall. Place only on stable
`in_front_of_mouth` + `mouth_path_clear`. Then ROM-prove 0x4f.

### Unit (no emu)

`Level2DodongoController`: greedy mouth approach (`goto_action`); stand off
the body; stand while face is unstable or `mouth_path_clear` is false; place
only when at-mouth + front + path-clear + stable face. `occupancy_misses=0`.
`poke=false`. `test_level2_tf_spine.py` + `test_level2_spine.py` **17/17**.

### ROM (Clean, `--no-infinite-life --no-video`)

| Room | Pin | Frames | Result | Notes |
|------|-----|--------|--------|-------|
| 0x0e | `Level2_0E` | 1309 | **green** | `dodongo_dead`, bombs 6 used, occupancy 0/0, deaths 0 |
| 0x4f | `Level2_4F` | 1506 | **green** | boom collect, combat 1174f, occupancy misses 1111 / 6 blocked, deaths 0 |

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes uv run python \
  nes/zelda_i/scripts/run_level2_hotspots.py \
  --room 0x0e --from-state Level2_0E --no-infinite-life --no-video --trials 1
QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes uv run python \
  nes/zelda_i/scripts/run_level2_hotspots.py \
  --room 0x4f --from-state Level2_4F --no-infinite-life --no-video --trials 1
```

PNGs: `recordings/l2_hotspot_0e_t0_final.png`,
`recordings/l2_hotspot_4f_t0_final.png`.

## Glance (last trial leftover — 0x4f)

room **0x4f**, mode **5**, xy **(126,133)**, tf **0x01**, keys **2**, bombs **10**,
health **0x37** (lo 7, hi 3, lo!=hi), deaths 0. Magical Boomerang owned
(`Level2Clear4fController` success). Pickup band ~(136,135).

0x0e leftover after kill: room **0x0e**, mode **5**, xy **(185,165)**, tf
**0x01**, keys **3**, bombs **1**, health **0x4F** (lo 15, hi 4, lo!=hi).

## Dead

- OccupancyWalker 1px-grade toward a moving Dodongo mouth (v1: 826 misses /
  7 wasted B-places / 0 HP drops).
- Place without stable face + `mouth_path_clear`.
- Bomb/key pokes. `--infinite-life`.

## Next sitting

Both heatmap rooms are fixture-live green. Integrator promotes. Do not STATUS
from this lane. Optional later: 0x4f still occupancy-grades Goriya motion
(1111 misses) — not blocking. TF collect from post-Dodongo leftover is a
later hop, not this bead's hotspots.

`route_eligible` stays false.

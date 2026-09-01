# Residual — rr-tne2 L6 power-on recompose to TF 0x20

**Status:** power-on Survival spine is **1/1** through `--through level6`
(fanfare `0x0C` TF `0x3F`). Bead `rr-tne2` stays open for Phase 4
(measured `AuditedEnv` post-reset state-load count + L4 Gleeok continuous
mode). Do not STATUS-promote. Do not close.

## Green from power-on this sitting (2026-09-01)

| through | leftover | notes |
|---------|----------|-------|
| level6-heart | play `0x1C` `(120,149)` health `0x77` | occupancy UP; hop 31f; hc 7→8; item still `0x1A` at stop |
| level6-north0c | play `0x0C` `(120,205)` keys=2 | `cur_opened_doors` already had UP; cardinal x-align UP; hop 209f |
| level6 | fanfare `0x0C` `(120,149)` TF `0x3F` | occupancy onto shard; hop 43f; mode 18 |

Prefix through Gohma still 1/1 (hop 54f, arrows 0→1). Glance: TF `0x1F→0x3F`,
rod=1, Bow=1, bombs=8, keys=2, health `0x77` lo==hi, accepted_containers=8.
Deaths 0 / claimed state-load 0 / progression / capacity writes 0. One 0x3A
position write. One wooden-arrow grant. Compose is power-on spine (no
`--from-state`). **Keys stay 2 — do not top up.** `status_claim=false`.

## Heart / shutter / shard (wired hops, 1/1)

`--through level6-heart` 1/1 (`l6_heart_recompose`, 212,291f hop 31f).
Occupancy from Gohma leftover `(120,189)` to center `(120,141)` collected
at `(120,149)`. 10 occupancy misses (2px UP). PNG still showed the sprite;
RAM hc 7→8 and assist accepted 8 with 0 clamps.

`--through level6-north0c` 1/1 (`l6_north0c_recompose`, 212,500f hop 209f).
North RAM bit was already set (`cur_opened_doors=0x0C`); visual shutter was
still black on the heart leftover. Cardinal UP entered play `0x0C` south
mouth. TF still `0x1F`.

`--through level6` 1/1 (`l6_tf_recompose`, 212,543f hop 43f). Occupancy UP
onto the center shard. Fanfare mode 18, TF `0x3F`, leftover `(120,149)`.
`triforce_writes=0`.

## Next sitting

Phase 4: wrap the spine env with `retro_harness.audit.AuditedEnv` so
`mid_run_state_load` is a measured count, and give L4 Gleeok TF-exit an
explicit continuous mode (no `set_state` restore loop). Then one
`--through level6` acceptance trial can close `rr-tne2`. No STATUS/push.

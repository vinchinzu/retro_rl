# Queue — Zelda I

**Source of truth:** `bd ready -l zelda_i`

Human snapshot only. Expand dungeon room beads when that dungeon is the tip.

## Live tip

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → L1 Triforce | **Clean green** | M5; see STATUS |
| L2 path prefix 0x4A | **verified** | 3/3 isolated |
| L2 door 0x3C | **Clean blocked** (hearts) | assist ready (`--infinite-life`) |
| L2 interior → TF bit 0x02 | open | epic `rr-ci7` |
| L3–L9 + Ganon | planned epics | expand rooms when tip arrives |

### Ready (snapshot)

```text
P0  rr-gfx   Z2.10 Promote 0x5C maze into hop controller
P1  rr-xbm   Z0.3 Adventure harness L1–L2 audit
P2  rr-hxs   Z2.8 Clean heart-safe door path (parallel)
```

Serial after maze: `rr-5v5` → `rr-cy1` → `rr-mcz` → keys → boomerang → Dodongo → natural → graph.

## Process

- [`PROCESS.md`](PROCESS.md) — dual track, pure-first, bead grain
- [`ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md) — infinite life
- [`STATUS.md`](../STATUS.md) — verified claims only
- Shared graph: `retro_harness.adventure`

## Residual — rr-20w.2.3 D2 CLEAR_PLOT (west pocket)

**Status:** OPEN. D2 CLEAR_PLOT: bushes/weeds first, west pocket inside
y=31, lift-only. Do **not** STATUS-promote Gate B.
**Pin in:** `Y1_After_Buy_Potato` (NOT `Y1_D2_Morning_After_D1`).
**Probe:** `harvest.scripts.pocket_clear_probe` (`HEADLESS=1`).

```bash
HEADLESS=1 uv run python -m harvest.scripts.pocket_clear_probe \
  --state Y1_After_Buy_Potato --out recordings/pocket_clear_probe.json
```

Glance with `harvest.clock_glance` — no MP4. Halt after 3 serial reds on
the same checkbox → BLOCKED, stop. Overwrite one JSON. Do not mint `_vN`
or `_window_*`.

### Already green (do not re-prove)

| Layer | Evidence |
|-------|----------|
| Grape + shop + 8-ring plant+water from this pin | `rr-m7mk` / `rr-bvam`, `recordings/d2_plant_water.json` |

### Next action

- **One change:** prove pocket CLEAR_PLOT lifts `0x03` before stones/stumps,
  toss RAM-open, `(13,28)` walkable, do not stand on remaining bush 1000f+.
  Not whole-farm. Not house→bin tape. No BFS-onto-weed.
- **Glance:** farm tilemap `0x00`, clock advancing, plot flags, no frozen
  clock, not still in house.

### Non-claims

- Did not STATUS-promote / Gate B promote
- Did not start from `Y1_D2_Morning_After_D1`
- Did not record a walk BFS can close

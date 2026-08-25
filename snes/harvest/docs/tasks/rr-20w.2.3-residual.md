## Residual — rr-20w.2.3 D2 whole-farm clear

**Status:** CLEAR_BUSHES is green from the 400-weed pin. CLEAR_FENCES is not
exhaustive: 18 house-paddock posts remain after a 200k-frame pond dump.
**Natural entry:** power-on. Pins may debug a skill but cannot green the rung.
**Probe:** `harvest.scripts.d2_leftover_probe` (`HEADLESS=1`).

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --state Y1_D2_After_Bushes --section fences --timeout 200000 \
  --out recordings/d2_leftover_smash.json
```

Glance with `harvest.clock_glance` — no MP4. Halt after 3 serial reds on
the same checkbox → BLOCKED, stop. Overwrite one JSON. Do not mint `_vN`
or `_window_*`.

### Already green (do not re-prove)

| Layer | Evidence |
|-------|----------|
| Grape + shop + 8-ring plant+water pin | `rr-m7mk` / `rr-bvam`, `recordings/d2_plant_water.json` |
| CLEAR_BUSHES from `Y1_D2_After_400_Weeds` | 103→0 weeds in 21,618f, farm `0x00` 18:05, Clean. Saved `Y1_D2_After_Bushes`. Watchdog rejected `(789,902)` oscillation and continued. Pathable-stand select + boxed-weed stone/fence opener landed the last seven. |

### Next action

- **Bushes (this pin):** `d2_leftover_probe --section bushes` from
  `Y1_D2_After_400_Weeds` succeeded. All 500 farm weeds are gone on that
  pin (400 already cleared + 103 this run). Do not re-prove bushes from
  this pin.
- **Fences (not green):** from `Y1_D2_After_Bushes`, pond dump 80→18 in
  200,001 frames / 18:09. Extra budget past ~90k dumped one more post.
  Remaining 18 are the house paddock: y=13 x=2–9 and west column x=2
  y=14–21, boxed by house `0xA6`. Do not spend another 200k on the same
  18; add a house-paddock stand (not `0xA6` toss). Restart from
  `Y1_D2_After_Bushes`.
- **Acceptance:** every debris count is zero, eight potatoes are wet,
  shipping occurred before 17:00, and Clean counters remain zero.
- **Glance:** farm tilemap `0x00`, clock/ClockTimeline, crop flags,
  shipping delta, and debris counts. The hour may remain at 18 while
  work continues.

### Non-claims

- Did not STATUS-promote from a pin
- Did not start from `Y1_D2_Morning_After_D1`
- Did not record a walk BFS can close
- Did not treat CrossMap origin-return as shop success
- Did not green CLEAR_FENCES; 18 house posts remain

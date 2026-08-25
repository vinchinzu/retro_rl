## Residual — rr-20w.2.3 D2 whole-farm clear

**Status:** CLEAR_BUSHES is green from the 400-weed pin. CLEAR_FENCES is not
green. Headed pond-dump from `Y1_D2_After_Bushes` still **rams 2×2
boulders** (`0x0D` quads) and burns frames on push-facing.
**Natural entry:** power-on. Pins may debug a skill but cannot green the rung.
**Probe:** `harvest.scripts.d2_leftover_probe`. Watch is `--headed`
(`retro_harness.headed`: `[` `]` speed, TAB turbo), not harvest
`--watch` / WatchDisplay.

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --state Y1_D2_After_Bushes --section fences \
  --out recordings/d2_leftover_smash.json
uv run python -m harvest.scripts.d2_leftover_probe --headed --section fences \
  --state Y1_D2_After_Bushes
```

Glance with `harvest.clock_glance` — no MP4. Halt after 3 serial reds on
the same checkbox → BLOCKED, stop. Overwrite one JSON. Do not mint `_vN`
or `_window_*`.

### Already green (do not re-prove)

| Layer | Evidence |
|-------|----------|
| Grape + shop + 8-ring plant+water pin | `rr-m7mk` / `rr-bvam`, `recordings/d2_plant_water.json` |
| CLEAR_BUSHES from `Y1_D2_After_400_Weeds` | 103→0 weeds in 21,618f, farm `0x00` 18:05, Clean. Saved `Y1_D2_After_Bushes`. Watchdog rejected `(789,902)` oscillation and continued. Pathable-stand select + boxed-weed stone/fence opener landed the last seven. |

### Landed this session (not a green rung)

- Leftover watch is `--headed` / `retro_harness.headed`.
- Travel occupies a 2×2 stump/rock from the TL even when siblings stay dirt/`0x00`.
- Pond-dump keeps the post until F0 `(32,34)`: no timeout local-drop, no
  south-charge from north farm (charge only y=30–31), hops even when not
  strictly closer.
- Scan hops to the nearest y=31 wall post; skip/retry clears `temp_blocked`.
- Headed pin runs: 80→80 `no reachable fence`; 80→78 then `too many fence
  failures`; pick-and-drop was recovery local-drop. Unattended 200k earlier:
  80→18. **Live still pounds 2×2 boulders** — occupancy is not enough
  (viewport `0x00` fringe and/or carry pixel clip).

Leftover JSON (`recordings/d2_leftover_smash.json`) now always carries
`leftover` (last RAM still: tilemap, clock, tile xy, carry, debris) and
`glance_misses` from `FENCE_STAND`. A dump fail is that stand, not a
journal-only blob. Next fences takeoff is the leftover still, not a
re-run of bushes from `Y1_D2_After_Bushes`.

### Next action

- **Do not re-prove bushes** from `Y1_D2_After_Bushes`.
- **Do not spend another 200k** on the boxed house paddock (y=13 x=2–9 and
  x=2 y=14–21, `0xA6`).
- **First:** carry-to-pond travel must go around live 2×2 rocks (viewport
  hop + pixel push), from `Y1_D2_After_Bushes`. Then house-paddock pond
  stand (not `0xA6` toss).
- **Acceptance:** every debris count is zero, eight potatoes are wet,
  shipping occurred before 17:00, and Clean counters remain zero.

### Non-claims

- Did not STATUS-promote from a pin
- Did not start from `Y1_D2_Morning_After_D1`
- Did not record a walk BFS can close
- Did not treat CrossMap origin-return as shop success
- Did not green CLEAR_FENCES; 2×2 ram remains; 18 house posts remain

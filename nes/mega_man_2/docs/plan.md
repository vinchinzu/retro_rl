# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4 gap after last solid land (prog ~949 / `AirFanPlatform`).**

Screen2 late recipe (approach 45/16 → fan hold 145–180 → late 40/16) clears
camera screen 4 mid-air but never lands past the fan-platform. Pure RIGHT jump
tuning maxes ~prog 1073 then pit death. Geometry at that platform:

- Fan body left of the striped platform (need updraft, not walk-off-left)
- Ladder further left on s4 entry screenshots
- Target cloud(s) require height + horizontal align the period recipe misses

## Suggested next experiments

1. Start from `AirFanPlatform` (grounded scr3 prog949) — shorter iteration.
2. Scripted fan ride: short LEFT into airstream while airborne, ride up, RIGHT at apex.
3. Ladder climb if UP on ladder is required (confirm NES ladder controls).
4. If a grounded s4/s5 state is found, promote checkpoint and freeze frame recipe.
5. Only then extend continuous AirScreen2→boss-door 3/3 trials.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- Probe screenshots: `recordings/air_post4_probe/`.

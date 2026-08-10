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

Screen2 late recipe clears camera screen 4 mid-air but never lands past the
Goblin platform. Pure RIGHT jump maxes ~prog 1073; Pipi bird-boost ~1085–1086,
still pit death on screen 4.

Geometry (verified 2026-08-09):

- Pink head is **Goblin / Air Tikki** (platform when spikes down / 5px corner),
  not an updraft fan
- `AirLeftPlatform` = grounded left of Goblin (prog~902)
- Ladder bar does not set `tile_feet==2` on this path
- Pipi to the right of platform: collision boost min_sy~23–26 with damage
- Decorative small clouds not solid in hop grids

## Suggested next experiments

1. Start from `AirFanPlatform` (not mid-air AirScreen4).
2. Pixel Goblin head land (RTA “5 pixel toe”) with spikes-down timing.
3. Pipi: shoot first then land cloud, or angle bird-boost onto solid chariot.
4. Lightning Lord section may be next after Goblins — kill and ride cloud.
5. Only then extend continuous AirScreen2→boss-door 3/3 trials.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.

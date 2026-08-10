# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4 gap: solid island prog 937–984 (`AirFanPlatform`); nothing solid past
984 or on Goblin top in tested windows.**

Screen2 late recipe clears camera screen 4 mid-air but never lands past the
island. Pure RIGHT ~prog 1064–1072; shoot+Pipi boost ~1086 min_sy~23; still pit
on screen 4. No camera ≥5.

Geometry (verified 2026-08-09 overnight):

- Pink head = **Goblin / Air Tikki** (obj slot14 @~39,49), not updraft fan
- Goblin top: dense spike-cycle hop grids → **0** feet=1 lands in gap zone
- `AirLeftPlatform` = short left ledge prog 902–905 only
- Ladder bar does not set `tile_feet==2`
- Type35 eggs walk platform y~84; Pipi bounce damages and lifts, no land
- Decorative clouds not solid under current trajectories

## Suggested next experiments

1. Confirm whether this Goblin is ever solid (RAM collision / known RTA setup).
2. Lightning Lord entry — may require different earlier path, not this island.
3. Pipi: kill then ride residual cloud if one exists; log object HP slots.
4. Only then freeze AirScreen2→target 5 (3/3) and boss door.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.

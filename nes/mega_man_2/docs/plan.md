# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

Air Man stage beyond screen 4 (open-sky cloud platforms, remaining fans, boss).

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2: approach 45/16,
  fan hold 145–180, late 40/16. Dies ~prog 1047 on screen 4 if pressed further.

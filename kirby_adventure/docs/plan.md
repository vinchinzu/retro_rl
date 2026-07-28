# Plan — Kirby's Adventure (NES)

## Goal

Advance from M1 (boot) toward a verified continuous clear of Kirby's Adventure.

## Next milestones

1. **M2 instrumentation** — map player position, mode, death, and stage/progress.
2. **M3 isolated segment** — clear one early segment from `Level1.state` with timeout.
3. **M4 natural-entry** — same segment from the real predecessor state (not a warp).

## Bottleneck

first stage/segment clear from Vegetable Valley hub.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.

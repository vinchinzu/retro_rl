# Plan — Zelda II (NES)

## Goal

Advance from M1 (boot) toward a verified continuous clear of Zelda II: The Adventure of Link.

## Next milestones

1. **M2 instrumentation** — map player position, mode, death, and stage/progress.
2. **M3 isolated segment** — clear one early segment from `Level1.state` with timeout.
3. **M4 natural-entry** — same segment from the real predecessor state (not a warp).

## Bottleneck

leave North Palace / first side-scroll segment.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.

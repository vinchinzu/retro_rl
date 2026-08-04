# Plan — Castlevania (NES)

## Goal

Advance from M1 (boot) toward a verified continuous clear of Castlevania.

## Next milestones

1. **M2 instrumentation** — map player position, mode, death, and stage/progress.
2. **M3 isolated segment** — clear one early segment from `Level1.state` with timeout.
3. **M4 natural-entry** — same segment from the real predecessor state (not a warp).

## Bottleneck

first Stage 1 segment clear.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.

# Plan — Kirby's Adventure (NES)

## Goal

Advance from boot toward a verified continuous clear of Kirby's Adventure.

## Next milestones

1. **M1 boot** — power-on to RAM-verified first controllable frame.
2. **M2 instrumentation** — map player position, mode, death, and stage/progress.
3. **M3 isolated segment** — clear one early segment from `Level1.state` with timeout.

## Bottleneck

first stage/segment clear.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.

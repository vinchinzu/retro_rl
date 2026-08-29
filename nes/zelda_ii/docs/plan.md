# Plan — Zelda II (NES)

## Goal

Advance from M1 (boot) toward a verified continuous clear of Zelda II: The Adventure of Link.

## Next milestones

1. **M3 leftover** — from `NorthPalaceExit` (overworld palace tile), walk to
   Rauru / first encounter and stop on the next `$0736` side-scroll latch.
2. **M4 natural-entry** — leave North Palace from power-on (not `Level1` warp).
3. Broader instrumentation only as later stops need it.

## Bottleneck

first overworld walk / first encounter side-scroll from `NorthPalaceExit`.

Leave-palace (Level1 LEFT → `$0736 == 5`) is an isolated segment; STATUS stays
M1 until that gate is promoted.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.

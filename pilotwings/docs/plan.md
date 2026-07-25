# Plan — Pilotwings

Ladder #8 (tier 2). See
`snes_oneshot/docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

mission-specific continuous control (skydiving, light plane, rocketbelt, etc.)

## Useful RAM (targets)

mission id, altitude, velocity/heading, position, score, landing/target flags

The initial map confirms HUD altitude at `0x0058`, pitch control at `0x005D`,
and raw heading at `0x0060`. Continue discovery via controlled
perturbations before writing a landing policy.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m snes_oneshot.setup_all_roms pilotwings`).
2. Use the verified `Lesson1Plane.state` checkpoint and directional RAM probe.
3. Clear **one segment at a time** from save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

Complete the Lesson 1 light-plane objective from `Lesson1Plane.state`.

## Notes

Multi-policy benchmark: one scripted policy per lesson type after the first mission clears.

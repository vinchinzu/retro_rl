# Plan — Rival Turf!

Ladder #5 (tier 1). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

linear beat-'em-up (walk, punch, special)

## Useful RAM (targets)

player X/Y, health, enemy presence/coords, stage, camera X

Confirmed: active run state `0x00AB`, player active `0x0200`, player X
`0x0202`, player Y `0x0205`. Next discover enemy slots/HP, camera/progress,
stage, and combat-lock clear flags, then expand
`custom_integrations/RivalTurf-Snes/data.json`.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m retro_harness.setup_all_roms rival_turf`).
2. Boot the integration and capture development save states at useful
   segment starts (stage open, mid-stage lock, boss door, etc.).
3. Clear **one segment at a time** from those save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

Boot and fight-ready `Stage1.state` are done. Clear the opening two-enemy
street lock and detect camera/progress advance.

## Notes

Generalization target after Final Fight / TMNT IV; same fight-or-walk-right loop.

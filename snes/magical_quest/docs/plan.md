# Plan — The Magical Quest Starring Mickey Mouse

Ladder #7 (tier 3). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

forgiving side-scrolling platformer (run, jump, transform abilities)

## Useful RAM (targets)

Confirmed: player/world X `0x0024`, screen Y `0x0027`, horizontal progress
`0x002A`, gameplay active `0x02C0`, current/max hearts `0x02B1`/`0x02B0`,
lives `0x0372`. First-door stop: `x >= 374` and `y >= 36` and `HP > 0`.
Still needed: velocity, grounded, room id after the house door, enemies.

The confirmed fields are in
`custom_integrations/MagicalQuest-Snes/data.json`; extend them with controlled
jump, damage, door, and room-transition probes.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m retro_harness.setup_all_roms magical_quest`).
2. Boot the integration and capture development save states at useful
   segment starts (stage open, mid-stage lock, boss door, etc.).
3. Clear **one segment at a time** from those save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

Reset-to-`Stage1.state` is done. Clear the first room/segment to the next door
or checkpoint.

## Notes

Strong first platformer candidate; map transforms after basic run/jump navigation works.

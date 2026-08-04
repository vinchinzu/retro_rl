# Plan — Battle Clash

Ladder #9 (tier 2). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

Super Scope aiming (cursor + trigger); peripheral emulation required

## Useful RAM (targets)

cursor X/Y, target/weak-point flags, boss phase, player energy, stage

Addresses remain undiscovered because gameplay input is blocked. Once Super
Scope injection exists, use `retro_harness.ram_state` + controlled perturbations
and fill `custom_integrations/BattleClash-Snes/data.json`.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m retro_harness.setup_all_roms battle_clash`).
2. Boot the integration and capture development save states at useful
   segment starts (stage open, mid-stage lock, boss door, etc.).
3. Clear **one segment at a time** from those save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

From a first-boss development save state, land enough hits to finish that boss segment (cursor track + fire).

## Current blocker

The ROM reaches its title, but the installed stable-retro interface exposes
only the standard 12-button SNES joypad. `RetroEmulator` has no cursor,
light-gun, or mouse injection method, so the Super Scope cannot aim or fire.
Resolve peripheral input before creating a gameplay checkpoint.

## Notes

Gameplay can be simpler than platforming once Scope/cursor input is wired through the emulator.

# Plan — F-Zero

Ladder #6 (tier 2). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

continuous racing (steer, accelerate, boost)

## Useful RAM (targets)

Confirmed: raw speed `0x0002`, live race/track state `0x0046/0x0047`,
lateral `0x007F`, fine lateral `0x00A6`. Still needed: track progress,
heading, lap, rank, energy, and collision state.

The confirmed fields are in `custom_integrations/FZero-Snes/data.json`; extend
them with controlled steering, wall-contact, and finish-line perturbations.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m retro_harness.setup_all_roms f_zero`).
2. Boot the integration and capture development save states at useful
   segment starts (stage open, mid-stage lock, boss door, etc.).
3. Clear **one segment at a time** from those save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

Reset-to-`MuteCity.state` is done. Record a centerline from that checkpoint and
complete one lap without crashing out (centerline follow + basic recovery).

## Notes

Record a centerline trajectory early; then add boost and collision recovery before cup runs.

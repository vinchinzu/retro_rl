# Plan — F-Zero

Ladder #6 (tier 2). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

continuous racing (steer, accelerate, boost)

## Useful RAM (targets)

Confirmed: raw speed `0x0002`, lateral `0x007F`, camera Y `0x00A6`,
finish-line HUD `0x00B8` bit 4, explosion `0x00C3` bit 6, machine power
`0x00C9`, heading `0x0BD1`, checkpoint facing `0x00C5`. Still needed: a
sticky lap index, rank, and a track-relative offset (crash vs finish-line
was the M3 risk).

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

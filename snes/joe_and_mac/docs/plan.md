# Plan — Joe & Mac

Ladder #10 (tier 3). See
`docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

action-platformer with ranged attacks

## Useful RAM (targets)

player X/Y, health, lives, projectile slots, enemy slots, stage/room, boss-active

The initial map confirms gameplay active at `0x0081`, actor state at `0x0082`,
and horizontal progress at `0x006C`. Continue discovery via controlled
perturbations while developing the first traversal policy.

## Development approach

1. Run `uv run python scripts/setup_rom.py` (or
   `uv run python -m retro_harness.setup_all_roms joe_and_mac`).
2. Use the verified `Stage1.state` checkpoint and movement RAM probe.
3. Clear **one segment at a time** from those save states; promote policies
   upward only after segments are stable.
4. Later: chain segments into a continuous run (optional full-game eval).

## First milestone

From a stage-1 development save state, clear the first traversable segment (move right, jump gaps, shoot nearest threat).

## Notes

More projectile chaos than Magical Quest; still linear enough for segment scripts.

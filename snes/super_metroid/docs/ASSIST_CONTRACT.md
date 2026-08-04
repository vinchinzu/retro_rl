# Assist contract — unlimited health/energy and ammo

This project intentionally uses disclosed survival/resource assists. The goal
is reliable navigation, item routing, room transitions, bosses, and the
endgame—not resource conservation.

## Allowed writes

### Unlimited energy

- During ordinary controllable gameplay, restore current energy to the
  naturally available maximum.
- Do not increase maximum energy or grant Energy Tanks.
- Observe and count natural damage before restoring it.
- Do not revive a completed death transition.
- Suspend or specialize the refill during cutscenes and scripted sequences
  whose progression depends on energy values.
- Suspend the refill throughout Ceres ordinary gameplay: Ridley's natural
  damage must reach the evacuation-countdown threshold. Energy refill begins
  only after the run reaches Zebes.

### Unlimited ammo

- Refill only ammo types that have been naturally unlocked.
- Refill current ammo only up to its naturally collected capacity.
- A zero capacity means the ammo type is still locked and must remain zero.
- Do not grant Missiles, Super Missiles, Power Bombs, or capacity upgrades.

The implementation should be a separate assist controller, not scattered
`set_value` calls in route policy.

## Forbidden writes

- item/equipment ownership
- ammo or energy capacity
- collected-item bits
- boss/event flags
- door, room, area, map, or elevator state
- player position, velocity, pose, or movement ability
- timers
- save-file completion state

If a new write is needed, update this contract before using it.

## Phase guard

The assist controller must distinguish at least:

- ordinary controllable gameplay
- room/door/elevator transition
- pause/inventory/menu
- cutscene/scripted sequence
- death/game over
- ending/credits

Default behavior outside ordinary gameplay is no write. Add a phase exception
only after a focused probe proves it preserves natural progression.

## Required telemetry

The full-run report records:

- total energy restored
- energy write count and affected frames
- ammo restored and write count per ammo type
- first frame each ammo type became naturally available
- any phase in which assists were suspended
- maximum single-frame damage
- deaths and game-over entries
- forbidden/progression writes (must be zero)

## Integrity assertions

A successful assisted clear must show:

- continuous emulator session from the declared start
- no state loads after power-on
- natural inventory/capacity acquisition
- no progression writes
- natural room and boss/event advancement
- natural endgame escape and ending/credits evidence

## Clean mode (parallel track)

**Clean** means both unlimited energy and unlimited ammo are **off**: zero
resource restores and zero resource writes. Observation may still be Bronze
(read-only RAM). Clean is a **parallel** privilege-reduction workstream; it
does not replace this assisted contract or the primary KPDR continuous tip.

Rules, artifact isolation (`*_clean` stems), tickets, and the Bomb Torizo
clean tip ladder: [`CLEAN_TRACK.md`](CLEAN_TRACK.md).

Hard constraints:

- Default continuous CLI remains resource-assisted.
- Clean runs must not overwrite assisted `recordings/<tip>.json` / videos.
- STATUS primary program gate stays assisted until an explicit program decision
  changes it; Clean results are documented as a secondary track.

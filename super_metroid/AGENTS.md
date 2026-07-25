# Agent Instructions — Super Metroid

Super Metroid scripted full-clear project. Shared process:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Evaluation contract

- Target: one continuous power-on-to-ending run.
- Allowed assists: unlimited health (in-game energy) and unlimited ammo,
  exactly as defined in
  [`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md).
- Resource assists remove attrition only. They must not grant uncollected ammo
  types, capacity, equipment, items, movement abilities, door state, map
  state, boss/event flags, rooms, or completion.
- Record every assist write in the full-run manifest.
- Completion requires the natural endgame escape and ending/credits evidence;
  defeating the final boss alone is not a clear.

## Organization

- Keep RAM addresses, route logic, maps, states, logs, recordings, and policy
  in `super_metroid/`.
- Save states belong under `custom_integrations/<GameId>/`.
- Use clean states for fast development and natural-entry states for
  acceptance.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Keep the last successful full-run baseline; candidates use separate reports.

## Immediate goal

The continuous power-on → both early Missile expansions → Bomb Torizo/Bombs
baseline is established. Extend the natural suffix from post-Torizo Parlor
through Terminator/Green Brinstar and the next required major upgrade. Reuse
the typed graph, checked replay segments, boundary fingerprints, independent
artifact verifier, and staged model candidates without weakening the
no-progression-write contract.

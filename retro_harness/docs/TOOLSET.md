# Shared Toolset Boundary

The toolset is layered by how widely behavior can be reused:

```text
game/ RAM + maps + policy
  ├── snes_oneshot/       scripted clear / combat / watchdog policy
  ├── platformer_common/  platformer route and optimizer policy
  └── retro_harness/      emulator, input, state, recording, task contracts
```

## Small public APIs

| Need | Import |
|---|---|
| New SNES game identity and environment | `retro_harness.snes.GameSpec` |
| Named controller action | `retro_harness.snes.snes_action` |
| Timed button/menu macro | `InputStep`, `StartupPlan`, `run_startup` |
| Old Gym / Gymnasium compatibility | `retro_harness.runtime` |
| State path/read/write | `retro_harness.env` or `GameSpec` |
| Human task JSON + trace analysis | `retro_harness.task_recording` (`RecordedTask`, `pressed_buttons`, stasis/run coalesce) |
| Route guide polyline overlay | `retro_harness.path_overlay` (+ `PlaySession.on_overlay`) |
| Scripted clear policy | `snes_oneshot` |
| Platformer optimization | `platformer_common` |

`retro_harness.snes` is the intentionally small new-game facade. The larger
`retro_harness` package root remains a compatibility barrel for existing code.

## Ownership rules

- `retro_harness.actions` owns controller layout and action construction.
- `retro_harness.input_script` owns fixed-duration inputs, compact script
  parsing, title/menu plans, and execution until a readiness predicate.
- `retro_harness.runtime` owns environment return-value normalization.
- `retro_harness.env` owns integration/state paths and `GameSpec`.
- `snes_oneshot.actions` and `snes_oneshot.primitives` are compatibility shims;
  new shared code imports their implementations from `retro_harness`.
- `platformer_common.auto_state.NavStep` and `parse_nav_string` remain supported
  names backed by `retro_harness.input_script`.
- Game-specific readiness detection stays local because “first playable frame”
  must be verified from that game's RAM, pixels, or both.

## Promotion test

Move code downward only when its inputs and outputs no longer mention a game's
RAM addresses, maps, entities, filenames, or policy. A second consumer is the
best evidence that the abstraction is real. Keep a compatibility import at the
old path when multiple games already depend on it.

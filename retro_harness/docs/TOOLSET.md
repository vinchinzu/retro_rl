# Shared Toolset Boundary

The toolset is layered by how widely behavior can be reused:

```text
game/ RAM + maps + policy
  └── retro_harness/
        ├── platformer/   platformer route and optimizer policy
        ├── fighters/     fighting-game env / training
        ├── adventure/    route graphs / waypoints / named routes
        └── (core)        emulator I/O + shared scripted-completion policy
```

## Small public APIs

| Need | Import |
|---|---|
| New SNES game identity and environment | `retro_harness.snes.GameSpec` |
| Named controller action | `retro_harness.snes.snes_action` / `retro_harness.actions` |
| Timed button/menu macro | `InputStep`, `StartupPlan`, `run_startup` (`input_script`) |
| Old Gym / Gymnasium compatibility | `retro_harness.runtime` |
| State path/read/write + ROM wiring | `retro_harness.env` (`GameSpec`, `setup_game_rom`) |
| Typed RAM schema / discovery / GameState | `retro_harness.ram_state` |
| Behavior trees + stuck watchdog | `retro_harness.bot_runner` |
| Segment stop heuristics | `retro_harness.segment_runner` |
| Beat-em-up combat helpers | `retro_harness.combat` |
| Point-and-click cursor | `retro_harness.cursor` |
| Video capture + showcase footer | `retro_harness.video` |
| Human task JSON + trace analysis | `retro_harness.task_recording` |
| Route guide polyline overlay | `retro_harness.path_overlay` |
| Platformer optimization | `retro_harness.platformer` |
| Fighting-game env / PPO | `retro_harness.fighters` |
| Adventure route graphs | `retro_harness.adventure` |
| Wire all ladder ROMs | `uv run python -m retro_harness.setup_all_roms` |

`retro_harness.snes` is the intentionally small new-game facade. The larger
package root remains a compatibility barrel for existing code — prefer the
specific submodule for new imports.

## Ownership rules

- `actions` owns controller layout and action construction.
- `input_script` owns fixed-duration inputs, compact script parsing,
  title/menu plans, and execution until a readiness predicate.
- `runtime` owns environment return-value normalization.
- `env` owns integration/state paths, `GameSpec`, and shared ROM zip setup.
- `ram_state` owns typed readers, schemas, watchers, normalized `GameState`,
  and differential RAM discovery.
- `bot_runner` owns Task autopilot wrappers **and** minimal behavior-tree
  nodes / stuck detection used by scripted clears.
- `video` owns ffmpeg capture, button footers, and footer-driven
  `RecordingSession` (distinct from `recorder.RecordingSession` labeled
  human saves).
- `retro_harness.platformer.auto_state.NavStep` remains supported names backed by
  `input_script`.
- Game-specific readiness detection stays local because “first playable frame”
  must be verified from that game's RAM, pixels, or both.

## Promotion test

Move code downward only when its inputs and outputs no longer mention a game's
RAM addresses, maps, entities, filenames, or policy. A second consumer is the
best evidence that the abstraction is real.

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
| Timed button/menu macro | `InputStep`, `StartupPlan`, `period_script`, `run_startup` (`input_script`) |
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
| One-game `scripts/setup_rom.py` | `retro_harness.setup_rom_cli` |
| Shared boot-to-checkpoint probe | `retro_harness.boot_probe` |
| Watch a headless probe (`--headed`) | `retro_harness.headed` (`add_headed_flag`, `attach_headed`) |
| Gymnasium vs classic `env.reset` | `retro_harness.env.reset_obs` |
| Fighting watch / validate CLIs | `retro_harness.fighters.{watch,validate_states}` |

`retro_harness.snes` is the intentionally small new-game facade. The larger
package root remains a compatibility barrel for existing code — prefer the
specific submodule for new imports.

## Ownership rules

- `actions` owns controller layout and action construction.
- `input_script` owns fixed-duration inputs, compact script parsing,
  title/menu plans, repeating period boot menus, and execution until a
  readiness predicate.
- `runtime` owns environment return-value normalization.
- `env` owns integration/state paths, `GameSpec`, and shared ROM zip setup.
- `ram_state` owns typed readers, schemas, watchers, normalized `GameState`,
  and differential RAM discovery.
- `bot_runner` owns Task autopilot wrappers **and** minimal behavior-tree
  nodes / stuck detection used by scripted clears.
- `video` owns ffmpeg capture, button footers, and footer-driven
  `CaptureSession` (distinct from `recorder.RecordingSession` labeled
  human saves).
- `retro_harness.platformer.auto_state.NavStep` remains supported names backed by
  `input_script`.
- Game-specific readiness detection stays local because “first playable frame”
  must be verified from that game's RAM, pixels, or both.
- Game-owned platformer `LevelConfig` packs (e.g. Super Metroid) live under
  `snes/<game>/` and register into `platformer.level_config` on import; do not
  re-embed game RAM maps under `retro_harness/platformer/levels/`.

## Per-game layout + Clean artifacts

| Need | Import |
|---|---|
| Standard `GAME_DIR` / `INTEGRATION_DIR` / `recordings/` | `retro_harness.game_layout.game_paths` |
| Clean-track stem rewrite (`foo` → `foo_clean`) | `retro_harness.artifacts.clean_artifact_stem` |
| `(video.mp4, report.json)` under recordings | `retro_harness.artifacts.recording_artifacts` |

Game `paths.py` should call `game_paths(__file__, "Integration-Id")` instead of
copying the six path constants. Super Metroid / TMNT re-export Clean helpers
for local imports but the canonical home is `retro_harness.artifacts`.

## Route lookup naming

Two different `get_route` helpers used to share a name:

| Domain | Prefer | Returns |
|---|---|---|
| Platformer speedrun catalogs | `platformer.route.get_platformer_route` | `RouteConfig` |
| Adventure named routes | `adventure.get_named_route` | `NamedRoute` |

Bare `get_route` remains a compatibility alias on both modules.

## Scripted-completion dual stack (intentional for now)

Two orchestration models coexist. They are **not** interchangeable — do not
mix without an explicit adapter.

| Stack | Core types | Used for |
|---|---|---|
| **Task / PlaySession autopilot** | `protocol.WorldState` + `Task` / `TaskSequencer` / `BotRunner` | Human PlaySession bots, multi-task missions, interactive autopilot |
| **Behavior tree / oneshot clear** | `ram_state.GameState` + `BehaviorNode` (`Sequence` / `Selector`) / `StuckDetector` | Oneshot segment clears, combat trees, beat-em-up policies |

Guidance:

- Prefer **BT + `GameState`** for new oneshot segment clears.
- Prefer **`Task` + `WorldState`** for interactive PlaySession bots and sequenced human-assist missions.
- `bot_runner` houses both surfaces on purpose: Task wrappers for PlaySession, BT nodes for scripted clears. That packaging is shared tooling, not a signal that the models are the same.
- Medium-term options (not required yet): `BehaviorNode` implements `Task`, or oneshot clears standardize on BT while Task stays PlaySession-only.

Recording naming (related dual; rename landed):

- **Video capture session** — `video.CaptureSession` (footer/ffmpeg showcase; formerly `video.RecordingSession`).
- **Labeled recorder session** — `recorder.RecordingSession` (human save points / labeled states).
- Do not treat these as one type; pass the module-qualified name when docs or APIs could collide.

## Promotion test

Move code downward only when its inputs and outputs no longer mention a game's
RAM addresses, maps, entities, filenames, or policy. A second consumer is the
best evidence that the abstraction is real.

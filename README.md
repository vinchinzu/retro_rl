# retro_rl

`retro_rl` is a multi-game **NES + SNES game solver**: reactive policies that
read state, plan, and act each frame across a prioritized canonical library —
beginning with RAM-aware skill scripts and gradually reducing privileged
information, including randomizers (flagship: Super Metroid + ALTTP + SMZ3).

The repository supports scripted skills, reinforcement learning, demonstrations,
replay, RAM discovery, editors, benchmark instrumentation, and planners as one
cumulative program. Input tapes are regression fixtures and skill demos, not the
solver backbone. See [`docs/SOLVER_ARCHITECTURE.md`](docs/SOLVER_ARCHITECTURE.md).

ROMs, save states, recordings, trained models, and ROM-derived assets are not
included. Bring legally obtained game dumps and keep generated artifacts in the
owning game directory.

Program docs:

| Document | Role |
|----------|------|
| [`docs/VISION.md`](docs/VISION.md) | Why the project exists |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Multi-horizon NES + SNES strategy |
| [`docs/SOLVER_ARCHITECTURE.md`](docs/SOLVER_ARCHITECTURE.md) | Layer stack, tapes demoted, flagship triangle |
| [`docs/DEVELOPMENT_LADDER.md`](docs/DEVELOPMENT_LADDER.md) | M0–M8 gates and capability phases |
| [`docs/PROGRAM_STATUS.md`](docs/PROGRAM_STATUS.md) | Live flagship results and bottlenecks |
| [`docs/GAME_MATRIX.md`](docs/GAME_MATRIX.md) | Generated game board |
| [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md) | Bronze/Silver/Gold, assists, seed-robustness |

## Quick Start

Requirements:

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- system support required by `stable-retro`, SDL, and Qt for the workflows you
  intend to run

```bash
git clone https://github.com/vinchinzu/retro_rl.git
cd retro_rl

# Create .venv and install the base dependencies.
./setup.sh

# Optional: install RL/vision and Cursor editor integrations too.
uv sync --all-extras
```

Useful smoke checks:

```bash
# Shared harness tests do not require a ROM.
uv run python -m pytest retro_harness/tests -q

# Documentation integrity (links, manifests, maturity fields).
uv run pytest tests/test_docs.py -q

# Discover registered game editors.
uv run python -m retro_harness.editor_launcher --list

# Inspect a platformer or fighter entry point.
uv run python -m SMW --help
uv run python retro_harness/fighters/train_ppo.py --help
```

## Architecture

```text
game workspace
  └── game-specific RAM maps, routes, policies, states, and evidence
retro_harness/
  ├── core I/O + scripted-completion helpers
  ├── platformer/   routes, replay, evaluation, optimizers
  ├── fighters/     fighting-game envs and PPO training
  └── adventure/    route graphs, waypoints, named routes
```

| Package | Responsibility |
|---|---|
| `retro_harness/` | Stable-retro env setup, SNES/NES actions, input scripts, save-state paths, runtime, play sessions, RAM/`GameState`, recording, combat/cursor/segment helpers, ROM setup, tasks, splits, benchmarks |
| `retro_harness/editor/` | Reusable Qt editor bridge, stdio JSON protocol, embedded emulator panel, map rendering helpers, recording, script segments, and optional Cursor agent panel |
| `retro_harness/platformer/` | Platformer level configuration, progress tracking, replay, route evaluation, hill climbing, and genetic/neuroevolution tools |
| `retro_harness/fighters/` | Fighting-game wrappers, RAM observations, reward shaping, menu navigation, model registry, PPO training, and evaluation |
| `retro_harness/adventure/` | Nonlinear route graphs, capability-aware planning, waypoints, and named route registries |

New SNES integrations should begin with the compact
`retro_harness.snes` API (`GameSpec`, named actions, `StartupPlan`, and input
scripts); NES workspaces use the same maturity ladder and
`retro_harness.env`. See [`docs/ADDING_GAMES.md`](./docs/ADDING_GAMES.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Development Ladder

Every game advances through the same maturity gates:

```text
M0 Contract
M1 Integration and boot
M2 Instrumentation
M3 Isolated segment
M4 Natural-entry segment
M5 Chained suffix
M6 Complete route graph
M7 Continuous dry run
M8 Verified capture
```

Central rule:

> A checkpoint clear is not route-ready until it also clears from the state
> produced by the real preceding route.

Genre work is organized as **parallel capability tracks** (linear combat,
platforming, continuous control, graph navigation, planning, …), not a single
numerical game ranking. See [`docs/DEVELOPMENT_LADDER.md`](./docs/DEVELOPMENT_LADDER.md)
and the engineering process in
[`docs/FULL_RUN_PROCESS.md`](./docs/FULL_RUN_PROCESS.md).

## Program documents

| Document | Role |
|---|---|
| [`docs/VISION.md`](./docs/VISION.md) | Why the project exists; scriptably beatable + solver |
| [`docs/ROADMAP.md`](./docs/ROADMAP.md) | Multi-horizon NES + SNES strategy |
| [`docs/SOLVER_ARCHITECTURE.md`](./docs/SOLVER_ARCHITECTURE.md) | Layer stack, tapes demoted, flagship triangle |
| [`docs/DEVELOPMENT_LADDER.md`](./docs/DEVELOPMENT_LADDER.md) | M0–M8 gates and capability phases |
| [`docs/BENCHMARK_SPEC.md`](./docs/BENCHMARK_SPEC.md) | Bronze/Silver/Gold, assists, seed-robustness |
| [`docs/PROGRAM_STATUS.md`](./docs/PROGRAM_STATUS.md) | Live clears, bottlenecks, priorities |
| [`docs/GAME_MATRIX.md`](./docs/GAME_MATRIX.md) | All games (generated from manifests) |
| [`docs/GLOSSARY.md`](./docs/GLOSSARY.md) | Shared vocabulary |
| [`AGENTS.md`](./AGENTS.md) | Repo-wide agent rules |
| [`docs/ADDING_GAMES.md`](./docs/ADDING_GAMES.md) | New game onboarding |

## Game Workspaces

Games live under `snes/<game>/` and `nes/<game>/` but keep package import names
(`import alttp`, `import super_metroid`). Run `./setup.sh` (or recreate the
venv `.pth`) so those folders are on `sys.path`; pytest also uses
`tool.pytest.ini_options.pythonpath` in `pyproject.toml`.

| Track | Directories |
|---|---|
| Fighting-game RL | `snes/mortal_kombat/`, `snes/mortal_kombat_ii/`, `snes/street_fighter_ii/`, `snes/super_street_fighter_ii/` |
| Platformers | `snes/SMW/`, `snes/donkey_kong_country/`, `snes/magical_quest/`, `snes/joe_and_mac/` |
| Scripted completion | `snes/alttp/`, `snes/battle_clash/`, `snes/f_zero/`, `snes/final_fight/`, `snes/great_waldo_search/`, `snes/pilotwings/`, `snes/rival_turf/`, `snes/star_fox/`, `snes/super_double_dragon/`, `snes/super_metroid/`, `snes/tmnt_iv/` |
| Planning / simulation | `snes/harvest/`, `snes/hals_golf/` |
| NES | `nes/smb/`, `nes/smb3/`, `nes/metroid/`, `nes/zelda_i/`, … |

Authoritative package names: `super_metroid` (not `super_metroid_rl`),
`SMW` (not `super_mario_bros`), `alttp`.

Treat each game’s local `docs/STATUS.md` as authoritative for that title. The
program-wide board is [`docs/GAME_MATRIX.md`](./docs/GAME_MATRIX.md).

## Common Workflows

### Scripted SNES agents

Game-local `scripts/` directories contain boot probes, RAM probes, segment
runners, and full-run recorders. Shared ROM setup is available for compatible
projects:

```bash
uv run python -m retro_harness.setup_all_roms <game-directory>
```

Develop from short, reproducible checkpoints, verify natural entry from the
preceding route, and only then chain segments.

### Fighting-game training

```bash
uv run python retro_harness/fighters/train_ppo.py \
  --game sf2 \
  --state Fight_StreetFighterIITurbo \
  --steps 500000
```

Model output belongs under the corresponding game directory. See
[`retro_harness/fighters/AGENTS.md`](./retro_harness/fighters/AGENTS.md) and the individual
fighting-game docs for state creation, evaluation, and current model lineage.

### Platformer tooling

`retro_harness.platformer` supplies the shared CLI used by platformer workspaces:

```bash
uv run python -m SMW --help
uv run python -m retro_harness.platformer --help
```

Level-specific RAM layouts and state registrations stay in
`retro_harness/platformer/levels/`; generated routes, optimizer runs, recordings, and
models stay in the owning game workspace.

### Game editors

```bash
uv run python -m retro_harness.editor_launcher --list
uv run python -m retro_harness.editor_launcher harvest
```

The shared editor package supports an embedded emulator and an optional Cursor
SDK agent dock. Install all extras and set `CURSOR_API_KEY` to enable the agent
panel.

## Benchmark Vocabulary

Report **two** independent labels on every result:

**Runtime observation**

- **Gold:** pixels only
- **Silver:** primarily visual, limited generic internals
- **Bronze:** game-specific read-only RAM permitted

**Intervention class**

- **Clean:** no writes or state mutation during the attempt
- **Survival- / Resource- / Protection-assisted:** disclosed and counted
- **Progression-assisted:** normally excluded

Example: `Bronze / Resource-assisted`, not merely “Bronze.”

A game is **scriptably beatable** when a disclosed policy starts from a
published reset, uses controller actions for progression, reaches a defined
ending, detects success, recovers without humans, and reports a success rate.
Large game-specific code is allowed.

See [`docs/BENCHMARK_SPEC.md`](./docs/BENCHMARK_SPEC.md).

## Testing

```bash
uv run python -m pytest retro_harness/tests -q
uv run python -m pytest retro_harness/platformer/tests -q
uv run python -m pytest retro_harness/fighters/tests -q
uv run pytest tests/test_docs.py -q
```

ROM-backed tests may skip or require local integration files. Running suites
separately is recommended because several game workspaces contain same-named
test modules.

Regenerate the game matrix after editing manifests:

```bash
uv run python docs/generate_game_matrix.py
```

## Repository Rules

- Keep game-specific code, docs, states, logs, screenshots, recordings, maps,
  and model output inside the owning game directory.
- Put save states under
  `<game>/custom_integrations/<GameId>/`; do not place them at the repository
  root.
- Keep ROMs under a gitignored `roms/` directory or use the game-local setup
  script documented by that project.
- Promote a helper into a shared package only after its inputs and outputs are
  no longer game-specific (at least a second consumer).

The package boundaries are described in
[`retro_harness/docs/TOOLSET.md`](./retro_harness/docs/TOOLSET.md).

## License

Source code is available under the [MIT License](./LICENSE). Game ROMs and
other copyrighted game assets are not distributed with this repository.

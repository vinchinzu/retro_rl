# retro_rl

`retro_rl` is a multi-game SNES automation platform. The goal is to produce
verified reset-to-ending clears across a broad canonical library, beginning with
bespoke RAM-aware scripts and gradually reducing privileged information.

The repository supports scripted policies, reinforcement learning,
demonstrations, replay, RAM discovery, editors, benchmark instrumentation, and
game-specific planners as one cumulative program — not separate experiments.

ROMs, save states, recordings, trained models, and ROM-derived assets are not
included. Bring legally obtained game dumps and keep generated artifacts in the
owning game directory.

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
uv run python fighters_common/train_ppo.py --help
```

## Architecture

```text
game workspace
  ├── game-specific RAM maps, routes, policies, states, and evidence
  ├── snes_oneshot       scripted completion helpers (historical package name)
  ├── platformer_common  platformer routes, replay, evaluation, optimizers
  ├── fighters_common    fighting-game environments and PPO training
  └── retro_harness      emulator, input, state, recording, task contracts
```

| Package | Responsibility |
|---|---|
| `retro_harness/` | Stable-retro environment setup, SNES actions, input scripts, save-state paths, runtime normalization, play sessions, RAM schemas, recording, tasks, splits, and benchmarks |
| `retro_harness/editor/` | Reusable Qt editor bridge, stdio JSON protocol, embedded emulator panel, map rendering helpers, recording, script segments, and optional Cursor agent panel |
| `snes_oneshot/` | Shared behavior trees, combat and cursor policies, segment runners, watchdogs, RAM discovery, and continuous-run practices |
| `platformer_common/` | Platformer level configuration, progress tracking, replay, route evaluation, hill climbing, and genetic/neuroevolution tools |
| `fighters_common/` | Fighting-game wrappers, RAM observations, reward shaping, menu navigation, model registry, PPO training, and evaluation |

New SNES integrations should begin with the compact
`retro_harness.snes` API (`GameSpec`, named actions, `StartupPlan`, and input
scripts), then add only the genre layer they need. See
[`ADDING_GAMES.md`](./ADDING_GAMES.md) for the recommended layout and first
verification seam.

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
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](./snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Program documents

| Document | Role |
|---|---|
| [`docs/VISION.md`](./docs/VISION.md) | Why the project exists; scriptably beatable |
| [`docs/DEVELOPMENT_LADDER.md`](./docs/DEVELOPMENT_LADDER.md) | M0–M8 gates and capability phases |
| [`docs/BENCHMARK_SPEC.md`](./docs/BENCHMARK_SPEC.md) | Bronze/Silver/Gold and Clean vs assisted |
| [`docs/PROGRAM_STATUS.md`](./docs/PROGRAM_STATUS.md) | Live clears, bottlenecks, priorities |
| [`docs/GAME_MATRIX.md`](./docs/GAME_MATRIX.md) | All games (generated from manifests) |
| [`docs/GLOSSARY.md`](./docs/GLOSSARY.md) | Shared vocabulary |
| [`AGENTS.md`](./AGENTS.md) | Repo-wide agent rules |
| [`ADDING_GAMES.md`](./ADDING_GAMES.md) | New game onboarding |

## Game Workspaces

| Track | Directories |
|---|---|
| Fighting-game RL | `mortal_kombat/`, `mortal_kombat_ii/`, `street_fighter_ii/`, `super_street_fighter_ii/` |
| Platformers | `SMW/`, `donkey_kong_country/`, `magical_quest/`, `joe_and_mac/` |
| Scripted completion | `alttp/`, `battle_clash/`, `f_zero/`, `final_fight/`, `great_waldo_search/`, `pilotwings/`, `rival_turf/`, `star_fox/`, `super_double_dragon/`, `super_metroid/`, `tmnt_iv/` |
| Planning / simulation | `harvest/`, `hals_golf/` |

Authoritative names: `super_metroid/` (not `super_metroid_rl/`), `SMW/` (not
`super_mario_bros/`), `alttp/`.

Treat each game’s local `docs/STATUS.md` as authoritative for that title. The
program-wide board is [`docs/GAME_MATRIX.md`](./docs/GAME_MATRIX.md).

## Common Workflows

### Scripted SNES agents

Game-local `scripts/` directories contain boot probes, RAM probes, segment
runners, and full-run recorders. Shared ROM setup is available for compatible
projects:

```bash
uv run python -m snes_oneshot.setup_all_roms <game-directory>
```

Develop from short, reproducible checkpoints, verify natural entry from the
preceding route, and only then chain segments.

### Fighting-game training

```bash
uv run python fighters_common/train_ppo.py \
  --game sf2 \
  --state Fight_StreetFighterIITurbo \
  --steps 500000
```

Model output belongs under the corresponding game directory. See
[`fighters_common/AGENTS.md`](./fighters_common/AGENTS.md) and the individual
fighting-game docs for state creation, evaluation, and current model lineage.

### Platformer tooling

`platformer_common` supplies the shared CLI used by platformer workspaces:

```bash
uv run python -m SMW --help
uv run python -m platformer_common --help
```

Level-specific RAM layouts and state registrations stay in
`platformer_common/levels/`; generated routes, optimizer runs, recordings, and
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
uv run python -m pytest snes_oneshot/tests -q
uv run python -m pytest platformer_common/tests -q
uv run python -m pytest fighters_common/tests -q
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

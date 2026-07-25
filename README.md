# retro_rl

`retro_rl` is a multi-game SNES emulator automation and reinforcement-learning
monorepo. It combines shared stable-retro tooling with game-local integrations,
scripted policies, RL training, replay and recording workflows, and ROM-first
editors.

The repository supports several ways to make progress:

- controller-driven scripted agents developed from save-state segments
- reinforcement-learning environments and training pipelines
- human play, input recording, replay, and human-to-agent handoff
- RAM discovery, deterministic startup scripts, and benchmark instrumentation
- game-specific editors backed by a shared Qt-to-emulator bridge

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

# Discover registered game editors.
uv run python -m retro_harness.editor_launcher --list

# Inspect a platformer or fighter entry point.
uv run python -m SMW --help
uv run python fighters_common/train_ppo.py --help
```

Game commands, expected ROM names, state requirements, and current milestones
live in the nearest game-local `README.md`, `AGENTS.md`, or `docs/` directory.

## Architecture

```text
game workspace
  ├── game-specific RAM maps, routes, policies, states, and evidence
  ├── snes_oneshot       scripted clears, behavior trees, combat, watchdogs
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

## Game Workspaces

The checkout currently contains these game-specific projects:

| Track | Directories |
|---|---|
| Fighting-game RL | `mortal_kombat/`, `mortal_kombat_ii/`, `street_fighter_ii/`, `super_street_fighter_ii/` |
| Platformers and route optimization | `SMW/`, `donkey_kong_country/` |
| Scripted and long-horizon agents | `battle_clash/`, `f_zero/`, `final_fight/`, `great_waldo_search/`, `joe_and_mac/`, `magical_quest/`, `pilotwings/`, `rival_turf/`, `star_fox/`, `super_double_dragon/`, `super_metroid/`, `tmnt_iv/` |
| Simulation and game tooling | `hals_golf/`, `harvest/` |

These projects are at different stages, from integration scaffolds and
RAM-discovery probes to trained policies and continuous clears. Treat each
game's local status document as authoritative. The shared one-shot overview is
in [`snes_oneshot/docs/STATUS.md`](./snes_oneshot/docs/STATUS.md).

## Common Workflows

### Scripted SNES agents

Game-local `scripts/` directories contain boot probes, RAM probes, segment
runners, and full-run recorders. Shared ROM setup is available for compatible
projects:

```bash
uv run python -m snes_oneshot.setup_all_roms <game-directory>
```

Develop from short, reproducible checkpoints, verify natural entry from the
preceding route, and only then chain segments. The complete process is
documented in
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](./snes_oneshot/docs/FULL_RUN_PROCESS.md).

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

List or launch editors through the shared registry:

```bash
uv run python -m retro_harness.editor_launcher --list
uv run python -m retro_harness.editor_launcher harvest
```

The shared editor package supports an embedded emulator and an optional Cursor
SDK agent dock. Install all extras and set `CURSOR_API_KEY` to enable the agent
panel.

## Benchmark Vocabulary

Runtime claims use three broad tiers:

- **Bronze:** autonomous controller input; RAM reads, scripts, shaped rewards,
  save-state curricula, and game-specific heuristics are allowed.
- **Silver:** controller-only runtime mutation with substantially less
  privileged information and mostly visual control.
- **Gold:** pixels and agent memory in, controller actions out.

Training method is separate from runtime tier. Imitation learning and
privileged training signals are valid when disclosed; the claimed tier
describes the live evaluation loop. Save-state segment clears are development
evidence, not continuous full runs.

See [`BENCHMARK_STATUS.md`](./BENCHMARK_STATUS.md) for the shared benchmark
rules and [`snes_oneshot/docs/FULL_RUN_PROCESS.md`](./snes_oneshot/docs/FULL_RUN_PROCESS.md)
for reset-to-ending integrity requirements.

## Testing

Run the narrowest suite that covers your change:

```bash
uv run python -m pytest retro_harness/tests -q
uv run python -m pytest snes_oneshot/tests -q
uv run python -m pytest platformer_common/tests -q
uv run python -m pytest fighters_common/tests -q
```

ROM-backed tests may skip or require local integration files. Running suites
separately is recommended because several game workspaces contain same-named
test modules.

## Repository Rules

- Keep game-specific code, docs, states, logs, screenshots, recordings, maps,
  and model output inside the owning game directory.
- Put save states under
  `<game>/custom_integrations/<GameId>/`; do not place them at the repository
  root.
- Keep ROMs under a gitignored `roms/` directory or use the game-local setup
  script documented by that project.
- Promote a helper into a shared package only after its inputs and outputs are
  no longer game-specific.

The package boundaries are described in
[`retro_harness/docs/TOOLSET.md`](./retro_harness/docs/TOOLSET.md).

## License

Source code is available under the [MIT License](./LICENSE). Game ROMs and
other copyrighted game assets are not distributed with this repository.

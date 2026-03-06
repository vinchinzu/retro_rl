# retro_rl

Build harnesses and agents that beat retro games without human intervention.

The repo is organized around one practical goal: turn emulator integrations, task runners, recorded demos, and RL policies into reproducible benchmark runs. The benchmark ladder is:

- `Bronze`: autonomous, but pragmatic. Read RAM, use heuristics, use room graphs, use reward shaping, use per-room specialists.
- `Silver`: controller-legal hybrid systems with much less privileged runtime information.
- `Gold`: pixels in, controller out at runtime.

Training method is separate from runtime tier. Imitation learning, behavioral cloning, demo seeding, and self-imitation are all valid as long as the runtime benchmark still obeys the claimed Bronze/Silver/Gold constraints.

The current priority is improving the shared harness for top-down adventure games such as ALTTP. That means room transitions, interaction, combat, dialogue, recovery, benchmark logging, and pixel-first evaluation should become shared primitives instead of one-off game code. Benchmark definitions and current status live in [`BENCHMARK_STATUS.md`](./BENCHMARK_STATUS.md).

## Current Focus

- Make `retro_harness/` the default runtime for play, recording, replay, benchmarking, and bot execution.
- Make rendered sessions support human takeover at any point, then resume autopilot on the same mission without resetting task state.
- Use `alttp/` as the proving ground for a generic top-down adventure harness.
- Keep the fighting, farming, platformer, and Metroid tracks as working benchmarks and sources of reusable tooling.

## Quick Start

```bash
# Shared environment
./setup.sh
source .venv/bin/activate
uv sync --extra ml

# Core tests
PYTHONPATH=. python -m pytest retro_harness/tests fighters_common/tests -v
uv run python alttp/tests/run_tests.py --fast

# ALTTP runtime / proving ground
PYTHONPATH=. uv run python -m alttp.asset_editor play --room 32 --engine yaze

# Fighting-game PPO benchmark track
python fighters_common/train_ppo.py \
  --game sf2 \
  --state Fight_StreetFighterIITurbo \
  --steps 500000
```

## Repository Map

```text
retro_rl/
├── retro_harness/         shared runtime: env setup, controls, play loop, RAM schema,
│                          recording helpers, split tracking, task runner
├── alttp/                 top-down adventure proving ground
├── fighters_common/       shared fighting-game wrappers, configs, PPO trainer
├── platformer_common/     shared route / evaluator / recording tools for side scrollers
├── street_fighter_ii/     SF2 integration, states, logs, models
├── super_street_fighter_ii/
├── mortal_kombat/
├── mortal_kombat_ii/
├── donkey_kong_country/
├── super_mario_bros/
├── harvest/
├── super_metroid_rl/
├── roms/                  shared ROM directory, git-ignored
├── setup.sh
└── pyproject.toml
```

## Shared Harness Status

Today the root harness already provides:

- stable-retro environment creation plus custom integration discovery
- keyboard/controller input handling with SNES button mapping
- a generic `PlaySession` with save/load, turbo, HUD hooks, and bot/human hot-swap
- task abstractions via `Task`, `TaskResult`, `WorldState`, and `BotRunner`
- declarative RAM readers via `RAMSchema` and `RAMWatcher`
- recording and split helpers used by multiple game tracks

The handoff requirement is explicit: when a run is rendered, a human should be able to take control immediately, make progress or recover from a bad state, and hand control back to autopilot with the current mission still intact.

The missing pieces for the next phase are mostly around normalization:

- a single benchmark runner that records tier, start state, attempts, success criteria, and artifacts
- shared demo recording and replay APIs instead of per-game implementations
- top-down adventure adapters for room state, interaction locks, dialogue, transitions, and combat
- pixel-first evaluation paths for Gold benchmarks

## Top-Down Adventure Plan

`alttp/` is the first-class target for the next shared-harness push because it exercises the hard parts that do not show up in fighters:

- room-to-room traversal instead of single-arena resets
- interaction and dialogue sequencing
- object and enemy handling in a top-down spatial layout
- long-horizon objectives that need both scripted pragmatism and vision-based upgrades

In practice, the next useful milestone is not "beat all of ALTTP." It is: make a shared harness that can benchmark fixed-state top-down tasks cleanly, then scale those tasks from single-room navigation to multi-room dungeon objectives.

The current Bronze proof path in `alttp/` is now concrete: cold boot `YazeSlot000`, inject blank SRAM, create a new slot, wake Link with controller buttons, exit Link's House into the rainy overworld, and route to Hyrule Castle. That gives the top-down harness a real "true start" menu/bootstrap path plus a shorter published-state traversal benchmark built from that same fresh-profile run.

## Project Map

### Shared Libraries

| Directory | Purpose |
|-----------|---------|
| `retro_harness/` | Shared runtime: env setup, controls, play loop, RAM schema, recording, split tracking, task runner |
| `platformer_common/` | Shared route optimizer, evaluator, recording tools for side-scrollers |
| `fighters_common/` | Shared fighting-game wrappers, configs, PPO trainer |

### Active Game Projects

| Directory | Game |
|-----------|------|
| `alttp/` | The Legend of Zelda: A Link to the Past -- top-down adventure proving ground |
| `super_metroid_rl/` | Super Metroid -- RL training (PPO + BC), navigation, recording |
| `donkey_kong_country/` | Donkey Kong Country -- route optimization, hill climbing |
| `mortal_kombat/` | Mortal Kombat (SNES) -- multi-character PPO speedrun |
| `mortal_kombat_ii/` | Mortal Kombat II (SNES) |
| `street_fighter_ii/` | Street Fighter II Turbo -- Hyper Fighting (SNES) |
| `super_street_fighter_ii/` | Super Street Fighter II -- The New Challengers (SNES) |
| `harvest/` | Harvest Moon -- farm automation bot |
| `super_mario_bros/` | Super Mario Bros |

### Supporting Tools

| Directory | Purpose |
|-----------|---------|
| `alttp/yaze/` | Yet Another Zelda Editor -- C++ ALTTP ROM/SRAM tooling |
| `alttp/asset_editor/` | ALTTP asset editor and interactive play |
| `super_metroid_rl/super_metroid_editor/` | Super Metroid ROM editor |
| `super_mario_editor/` | Super Mario level editor |

## Related Docs

- [`BENCHMARK_STATUS.md`](./BENCHMARK_STATUS.md): benchmark tiers, status board, and ALTTP-first roadmap
- [`ADDING_GAMES.md`](./ADDING_GAMES.md): add a new stable-retro integration and game runner
- [`ARCHITECTURE_AND_CLEANUP_PLAN.md`](./ARCHITECTURE_AND_CLEANUP_PLAN.md): lessons learned and cleanup priorities
- [`retro_harness/docs/EMULATOR_FEATURES.md`](./retro_harness/docs/EMULATOR_FEATURES.md): emulator/runtime features to consolidate into `retro_harness/`
- [`fighters_common/docs/`](./fighters_common/docs/): fighters training guides, multi-game setup, waypoint workflow
- [`alttp/README.md`](./alttp/README.md): ALTTP arena and asset-port workflows

## License

ROMs are not included. This repo is for research, tooling, and experimentation around legally obtained game dumps.

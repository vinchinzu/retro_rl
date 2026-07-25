# Vision

Build a reusable SNES automation platform capable of producing verified
reset-to-ending clears across a broad canonical game library, beginning with
bespoke RAM-aware scripts and gradually reducing privileged game-specific
information.

This is broader than reinforcement learning, broader than “one-shot”
evaluation, and broader than any fixed ten-game ladder. The repository already
supports scripted policies, RL, demonstrations, replay, RAM discovery, editors,
benchmark instrumentation, and game-specific planners. Those are tools inside
one cumulative program, not separate experiments.

## Why this project exists

1. **Automate major SNES games start to finish** — scriptably beatable clears
   from published reset or initial states to legitimate endings.
2. **Build reusable genre-specific systems** — combat, platforming, continuous
   control, graph navigation, RPG campaigns, planning — with shared packages
   promoted only after a second consumer exists.
3. **Compare approaches fairly** — bespoke scripting, RL, imitation, and vision
   under the same evaluation contracts.
4. **Gradually reduce privileged information** — Bronze/Silver/Gold runtime
   observation, independent of Clean versus assisted intervention class.

## What “done” means for a game

A game is **scriptably beatable** when the repository contains a policy that:

- starts from a published reset or initial state
- uses controller actions for gameplay progression
- reaches a defined legitimate ending or campaign objective
- detects success independently
- can recover or restart without human input
- has a documented success rate over repeated attempts
- discloses runtime observations and assists

That definition permits large game-specific code, route graphs, boss tables,
room scripts, grinding, planners, stage controllers, and RAM-aware recovery.
General-intelligence purity must not block straightforward engineering.

## Human-facing language

Prefer:

- **scripted completion**
- **full-game automation**
- **continuous clear**
- **reset-to-ending evaluation**

Use **one-shot** only for the final uninterrupted evaluation class, or as
historical project terminology. The package name `snes_oneshot` is retained for
compatibility; it is the historical home for shared scripted-completion
policies.

## Canonical documents

| Document | Role |
|----------|------|
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | Capability phases and M0–M8 maturity gates |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Stable Bronze/Silver/Gold and assist rules |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live facts and near-term priorities |
| [GAME_MATRIX.md](GAME_MATRIX.md) | All games by genre track (generated from manifests) |
| [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |
| [../snes_oneshot/docs/FULL_RUN_PROCESS.md](../snes_oneshot/docs/FULL_RUN_PROCESS.md) | Engineering process for each game |
| [../snes_oneshot/docs/GAME_SELECTION_NOTES.md](../snes_oneshot/docs/GAME_SELECTION_NOTES.md) | Candidate and hard-game research notes |

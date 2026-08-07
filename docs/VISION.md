# Vision

Build a reusable **NES + SNES game solver**: reactive policies that read state,
plan, and act each frame across a broad canonical library — starting with
RAM-aware skill scripts and gradually reducing privileged game-specific
information — including **randomizers, mods, and edited ROMs**.

This is broader than reinforcement learning alone, broader than “one-shot”
evaluation, and broader than any fixed ten-game ladder. The repository already
supports scripted skills, RL, demonstrations, replay, RAM discovery, editors,
benchmark instrumentation, and game-specific planners. Those are layers inside
one cumulative program, not separate experiments.

The multi-horizon plan lives in [ROADMAP.md](ROADMAP.md). Live board facts live
in [PROGRAM_STATUS.md](PROGRAM_STATUS.md). The solver reframe (layers, tapes,
flagships) lives in [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

## Why this project exists

1. **Automate major NES and SNES games start to finish** — scriptably beatable
   clears from published reset or initial states to legitimate endings.
2. **Solve randomized and modded play** — same stack across seeds and edits:
   skill library + online world model + item-logic planning (not fixed tapes).
3. **Build reusable genre-specific systems** — combat, platforming, continuous
   control, graph navigation, RPG campaigns, planning — with shared packages
   promoted only after a second consumer exists.
4. **Compare approaches fairly** — bespoke scripting, RL, imitation, and vision
   under the same evaluation contracts.
5. **Gradually reduce privileged information** — Bronze/Silver/Gold runtime
   observation, independent of Clean versus assisted intervention class.

## Solver vs skills vs tapes

| Concept | Role |
|---------|------|
| **Solver** | Observe → plan → act each frame. Backbone for randomizers/mods. |
| **Skill library** | Seed-invariant low-level controllers (what most of the repo is today). |
| **Input tape** | Degenerate precomputed plan; useful as CI regression and imitation demo, not the high-level route. |

In a randomizer, **room physics are seed-invariant**; only **item-logic order**
changes. Existing room/boss/menu scripts become planner-invoked skills. What
remains to build is largely shared: observation bootstrap, world-model
discovery, logic-graph planning, and an emulator pool for search. See
[SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

Honest positioning: the repo is a deep **skill substrate** in search of its
**planner and discovery core**. RL earns its place as skill synthesis and
generalization under mods — not as a replacement for planning.

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
room scripts, grinding, planners, and RAM-aware recovery.
General-intelligence purity must not block straightforward engineering.

For **randomizer titles**, “done” additionally means **seed-abstract**
evidence: clear **S of T** random seeds within budget (see
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md)), not a single fixed route.

## Human-facing language

Prefer:

- **game solver** / **scripted completion** (skills layer)
- **full-game automation**
- **continuous clear**
- **reset-to-ending evaluation**
- **seed-robust clear** (randomizer class)

Use **one-shot** only for the final uninterrupted evaluation class, or as
historical project terminology. Shared scripted-completion helpers live under
`retro_harness/`. Prefer “scripted completion” for Layer 1 prose; “solver” for
the full observe–plan–act stack.

## Flagships

| Role | Title |
|------|-------|
| Harness fixture | Great Waldo Search (pipeline M8) |
| Solver substrate | Super Metroid + A Link to the Past (vanilla skill graphs) |
| Solver proof | SMZ3 combined randomizer (seed-abstract clears) |

## Canonical documents

| Document | Role |
|----------|------|
| [ROADMAP.md](ROADMAP.md) | Multi-horizon strategy and success metrics |
| [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md) | Layer stack, tape demotion, solver priorities |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | Capability phases and M0–M8 maturity gates |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Bronze/Silver/Gold, assists, seed-robustness |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live facts and near-term priorities |
| [GAME_MATRIX.md](GAME_MATRIX.md) | All games by genre track (generated from manifests) |
| [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |
| [FULL_RUN_PROCESS.md](FULL_RUN_PROCESS.md) | Engineering process for each game |
| [GAME_SELECTION_NOTES.md](GAME_SELECTION_NOTES.md) | Candidate and hard-game research notes |
| [REPO_HYGIENE.md](REPO_HYGIENE.md) | Agent-context budget and cleanup backlog |

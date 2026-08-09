# Development Ladder

Two ladders must never be conflated:

1. **Completion maturity (M0–M8)** — how far a single game’s automation has
   been engineered.
2. **Capability phases (Phase 0–7)** — which genre systems the program is
   building next.

Runtime observation (Bronze/Silver/Gold) and intervention class
(Clean/assisted) are orthogonal; see [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

Multi-horizon sequencing and success metrics live in [ROADMAP.md](ROADMAP.md).

## Completion maturity (M0–M8)

Every game advances through the same gates. Local `docs/STATUS.md` reports
exactly one current maturity gate.

| Gate | Name | Required evidence |
|------|------|-------------------|
| M0 | Contract | Start, finish, assists, forbidden actions, and metrics documented |
| M1 | Integration and boot | Reset reaches a RAM-verified first controllable frame |
| M2 | Instrumentation | Player, mode, progress, transitions, death, and completion mapped |
| M3 | Isolated segment | One checkpoint clears repeatedly with a hard timeout |
| M4 | Natural-entry segment | The segment also clears from the real predecessor entry state |
| M5 | Chained suffix | Predecessor plus target segment clears without a state load |
| M6 | Complete route graph | Every required milestone and transition has an owner and stop predicate |
| M7 | Continuous dry run | One reset-to-ending session passes every integrity invariant |
| M8 | Verified capture | A previously dry-verified policy produces the final audiovisual artifact |

Central natural-entry rule:

> A checkpoint clear is not route-ready until it also clears from the state
> produced by the real preceding route.

For nonlinear games, the route is a graph of rooms, doors, inventory
requirements, bosses, and events — not a stage-number list.

Detailed engineering practice lives in
[FULL_RUN_PROCESS.md](FULL_RUN_PROCESS.md).
The process applies to **NES and SNES** integrations equally.

## Capability phases

These are program-level tracks, not a single ranked game order. NES and SNES
titles share the same phase model.

### Phase 0 — Harness validation

Goal: prove the emulator and evidence pipeline.

Games: Great Waldo Search; simple fighting-game matches; short fixed-state
tasks; NES boot fixtures as needed.

Exit criteria: deterministic reset/start, legal controller input, RAM and
screenshot capture, success detection, report and recording generation.

Great Waldo Search is the pipeline fixture: either keep a continuous
title-to-ending capture current, or formally treat it as a completed fixture.

### Phase 1 — Linear full-game clears

Goal: several verified continuous completions in the linear combat track.

SNES: TMNT IV (reference), Final Fight, Super Double Dragon, Rival Turf!,
Knights of the Round later.

NES: TMNT I / II / III; Contra as a combat-heavy linear foothold.

### Phase 2 — Deterministic continuous control

Games: F-Zero, Pilotwings, Star Fox; Battle Clash only after Super Scope
injection works (`blocked: infrastructure` until then).

Capabilities: trajectory following, control-loop stability, crash/stall
recovery, mission objective detection.

### Phase 3 — Reusable platforming framework

SNES: Magical Quest, Joe & Mac, Super Mario World (`SMW/`), Donkey Kong
Country; Mega Man X should be added when ready.

NES: Super Mario Bros. (`smb/`), Super Mario Bros. 3 (`smb3/`), Mega Man 2,
DuckTales, Kirby’s Adventure; Castlevania also feeds later graph work.

Capabilities: grounded/jump estimation, waypoint routing, moving platforms,
death/checkpoint recovery, natural-entry robustness, route stitching.

### Phase 4 — Graph-based exploration

SNES: Super Metroid, A Link to the Past (`alttp/`); Soul Blazer or Goof Troop
later.

NES: The Legend of Zelda (`zelda_i/`), Zelda II (`zelda_ii/`), Castlevania as a
stage/graph hybrid after platforming basics.

Capabilities: room/door graphs, inventory prerequisites, event flags, path
replanning, transition recovery, nonlinear route completion.

Promote into `retro_harness.adventure` only after two concrete game
implementations prove the interface.

### Phase 5 — Long structured campaigns

Games: Chrono Trigger, Final Fantasy IV, Super Mario RPG, EarthBound; NES
Dragon Quest / Final Fantasy equivalents when the campaign stack is ready.

Capabilities: dialogue macros, quest state machines, equipment and combat
policies, grinding, campaign progression.

### Phase 6 — Planning-heavy games

Games: Harvest Moon (`harvest/`), Tactics Ogre, Ogre Battle, Uncharted Waters,
Civilization.

Capabilities: daily or turn planners, resource budgets, delayed consequences,
campaign-level objective selection.

### Phase 7 — Adaptive, procedural, and randomizer-robust play

Games: **`sm_rando` / `alttp_rando`** (single-game rungs, M0 scaffolded);
**SMZ3** (combined proof); Shiren the Wanderer; unseen-game adaptation later.

Capabilities: online map discovery, item-logic graph planning over inventory,
seed-agnostic world models, risk-sensitive planning, procedural inventory
decisions, policy adaptation without fixed high-level routes, skill synthesis
when mods change physics.

**Not deferred research only.** SMZ3 is the forcing function that pulls solver
Layers 2–4 forward while vanilla SM and ALTTP deepen the Layer 1 skill
substrate. Full architecture: [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

Seed-abstract evidence (S/T seeds within budget) is defined in
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md). Fixed-ROM M0–M8 still applies to vanilla
titles and skill quality.

## Solver layers (orthogonal to M-gates)

| Layer | Name | Maturity signal |
|-------|------|-----------------|
| L0 | Parallel emulator pool | Multi-env deterministic rollouts for search |
| L1 | Skill library | Per-game M3–M8 segments (current bulk of work) |
| L2 | Runtime observation bootstrap | Few-shot RAM / vision semantics per run |
| L3 | Online world-model discovery | Seed-agnostic room/door/item graph |
| L4 | Item-logic planner | Inventory-aware search; routes skills |

M0–M8 measure **one title’s completion engineering**. L0–L4 measure **shared
solver capability**. Both ladders matter; do not collapse them.

Shared infrastructure uses a separate evidence ladder: scaffolded →
fake-tested → real-ROM tested → first real-game consumer → second independent
consumer → publication-ready. Test-tier definitions and closure rules live in
[TEST_TIERS.md](TEST_TIERS.md); a green unit suite alone never implies a real
consumer or publication-ready capability.

## Active near-term focus

Concentrate implementation on these trunks (detail and horizon in
[ROADMAP.md](ROADMAP.md)):

1. **Solver flagship triangle** — Super Metroid + ALTTP (L1) → logic-graph
   planner (L4) → SMZ3 seed-abstract proof
2. **Final Fight** — generalize the proven TMNT combat stack toward continuous clear
3. **Magical Quest / Joe & Mac** — establish the platformer stack (`retro_harness.platformer`)
4. **NES parallel track** — Zelda I/II, TMNT I–III, SMB family skill growth

Also advance Super Double Dragon and Rival Turf in parallel with Final Fight.
Keep Great Waldo Search current as the harness fixture.

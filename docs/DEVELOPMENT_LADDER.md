# Development Ladder

Two ladders must never be conflated:

1. **Completion maturity (M0–M8)** — how far a single game’s automation has
   been engineered.
2. **Capability phases (Phase 0–7)** — which genre systems the program is
   building next.

Runtime observation (Bronze/Silver/Gold) and intervention class
(Clean/assisted) are orthogonal; see [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

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
[../snes_oneshot/docs/FULL_RUN_PROCESS.md](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Capability phases

These are program-level tracks, not a single ranked game order.

### Phase 0 — Harness validation

Goal: prove the emulator and evidence pipeline.

Games: Great Waldo Search; simple fighting-game matches; short fixed-state
tasks.

Exit criteria: deterministic reset/start, legal controller input, RAM and
screenshot capture, success detection, report and recording generation.

Great Waldo Search is the pipeline fixture: either keep a continuous
title-to-ending capture current, or formally treat it as a completed fixture.

### Phase 1 — Linear full-game clears

Goal: several verified continuous completions in the linear combat track.

Games: TMNT IV (reference), Final Fight, Super Double Dragon, Rival Turf!,
Knights of the Round later.

### Phase 2 — Deterministic continuous control

Games: F-Zero, Pilotwings, Star Fox; Battle Clash only after Super Scope
injection works (`blocked: infrastructure` until then).

Capabilities: trajectory following, control-loop stability, crash/stall
recovery, mission objective detection.

### Phase 3 — Reusable platforming framework

Games: Magical Quest, Joe & Mac, Super Mario World (`SMW/`), Donkey Kong
Country; Mega Man X should be added.

Capabilities: grounded/jump estimation, waypoint routing, moving platforms,
death/checkpoint recovery, natural-entry robustness, route stitching.

### Phase 4 — Graph-based exploration

Games: Super Metroid, A Link to the Past (planned / external ALTTP work),
Soul Blazer or Goof Troop later.

Capabilities: room/door graphs, inventory prerequisites, event flags, path
replanning, transition recovery, nonlinear route completion.

Promote a shared `adventure_common` abstraction only after two concrete game
implementations prove the interface.

### Phase 5 — Long structured campaigns

Games: Chrono Trigger, Final Fantasy IV, Super Mario RPG, EarthBound.

Capabilities: dialogue macros, quest state machines, equipment and combat
policies, grinding, campaign progression.

### Phase 6 — Planning-heavy games

Games: Harvest Moon (`harvest/`), Tactics Ogre, Ogre Battle, Uncharted Waters,
Civilization.

Capabilities: daily or turn planners, resource budgets, delayed consequences,
campaign-level objective selection.

### Phase 7 — Adaptive and procedural play

Games: Shiren the Wanderer; randomized or unseen scenarios; unseen-game
adaptation.

Capabilities: online map discovery, risk-sensitive planning, procedural
inventory decisions, policy adaptation without fixed room scripts.

This is a later research track, not the near-term completion board.

## Active near-term trio

Concentrate implementation on three trunks:

1. **Final Fight** — generalize the proven TMNT combat stack
2. **Magical Quest or Mega Man X** — establish the platformer stack
3. **Super Metroid** — establish route-graph and inventory-aware navigation

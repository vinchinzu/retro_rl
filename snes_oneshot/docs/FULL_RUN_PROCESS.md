# Scripted full-run development process

This is the shared workflow for turning a bootable SNES integration into a
verified reset-to-ending policy. Game selection lives in
[GAME_SELECTION_NOTES.md](GAME_SELECTION_NOTES.md) and the live board in
[../../docs/GAME_MATRIX.md](../../docs/GAME_MATRIX.md). Maturity gates M0–M8 are
defined in [../../docs/DEVELOPMENT_LADDER.md](../../docs/DEVELOPMENT_LADDER.md).

The central rule is:

> A checkpoint clear is not route-ready until it also clears from the state
> produced by the real preceding route.

Equivalently: a segment is not ready because it clears from one clean
checkpoint. It is ready when it also clears from the entry state produced by
the real route.

TMNT IV made this distinction concrete: the boss checkpoint cleared, while the
continuous route entered the same fight pinned behind a wall. Natural-entry
coverage is therefore a separate acceptance gate, not optional polish.

## 1. Freeze the evaluation contract first

Before RAM discovery or policy work, write down:

- ROM filename and SHA-256
- start condition: power-on, blank SRAM, existing file, or named state
- difficulty and menu settings
- exact completion evidence
- allowed assists and their write semantics
- forbidden writes and inputs
- timeout and failure conditions
- primary optimization metric and regression budgets

Do not use a vague endpoint such as “boss defeated.” Prefer a conjunction of
observable evidence: ending event, credits phase, final transition, and a
settle period. The full-run manifest must make the contract auditable.

Classify assists explicitly:

| Class | Examples | Default rule |
|-------|----------|--------------|
| Survival | health refill, lives | Allowed only when disclosed and counted |
| Resource | ammo, fuel, currency | Refill unlocked capacity; do not grant progression |
| Protection | iframes, hazard immunity | Phase-scoped and counted per frame |
| Information | RAM reads, route labels | Record what the policy can observe |
| Progression | item flags, rooms, bosses, stage writes | Forbidden unless the benchmark explicitly allows it |

An “unlimited” resource means attrition is removed. It does not implicitly
mean all resource types, upgrades, capacities, keys, or traversal abilities
are unlocked.

## 2. Keep a standard game-local evidence set

Each game should converge on:

```text
<game>/
├── AGENTS.md
├── docs/
│   ├── STATUS.md
│   ├── plan.md
│   ├── ram_map.md
│   └── ASSIST_CONTRACT.md       # when RAM-writing assists exist
├── scripts/
│   ├── boot_probe.py
│   ├── run_<segment>.py
│   ├── probe_<bottleneck>.py
│   └── record_full_run.py
├── tests/
├── recordings/
└── custom_integrations/<GameId>/
```

`STATUS.md` states only proven results (maturity gate, best verified result,
last verification, runtime class, intervention class, regressions, evidence).
`plan.md` owns future work only (bottleneck, next acceptance test, next three
milestones, deferred ideas, infrastructure blockers). `AGENTS.md` stays
operational (commands, constraints, traps). `ram_map.md` records each address
with width, meaning, confidence, read/write, evidence, and consumers.
Generated states, logs, reports, screenshots, and video stay in the game
directory.

## 3. Build in gates, not one long leap

| Gate | Name | Required evidence |
|------|------|-------------------|
| M0 | Contract | Start, finish, assists, forbidden actions, and metrics documented |
| M1 | Integration and boot | Reset reaches a RAM-verified first controllable frame |
| M2 | Instrumentation | Player, mode, progress, transitions, death, and completion mapped |
| M3 | Isolated segment | One checkpoint clears repeatedly with a hard timeout |
| M4 | Natural-entry segment | The segment clears from a state captured from its real predecessor |
| M5 | Chained suffix | The predecessor plus target segment clears without a state load |
| M6 | Complete route graph | Every required milestone and transition has an owner and stop predicate |
| M7 | Continuous dry run | One reset-to-ending session passes every integrity invariant |
| M8 | Verified capture | A previously dry-verified policy produces the final audiovisual artifact |

Local `docs/STATUS.md` reports exactly one current maturity gate. Runtime
observation class and intervention class are independent labels; see
[../../docs/BENCHMARK_SPEC.md](../../docs/BENCHMARK_SPEC.md).

For nonlinear games, the route is a graph of rooms, doors, inventory
requirements, bosses, and events. Do not force it into a stage-number list.

## 4. Treat entry-state coverage as a test matrix

For each important segment, keep at least these cases:

| Entry class | Purpose |
|-------------|---------|
| Clean | Fast deterministic development |
| Natural | Actual position, health, inventory, camera, and policy history |
| Boundary | Wall, door, platform, low resource, or awkward enemy layout |
| Recovery | Previously observed stall or failed full-run state |

Capture a compact fingerprint with every state:

- room/stage/event and progress
- player position, pose, velocity, health, and resources
- relevant inventory/boss/event flags
- active targets and their positions/health
- policy phase, if stateful

Synthetic state edits are useful for discovery but are not substitutes for a
natural-entry state. Name them so provenance is obvious.

## 5. Separate progress, survival, and efficiency

Every runner should report:

- outcome and final milestone
- frames and wall-clock equivalent
- damage, deaths, continues, and assist interventions
- resource writes by type
- milestone/split times
- action-reason counts per segment
- maximum no-progress interval
- forbidden inputs or writes

Use a game-specific progress vector in watchdogs. Player coordinates alone are
not enough. A useful vector may include room, door transition, camera,
inventory bits, boss/event bits, target HP, and objective counters.

Long waits can be legitimate transitions, cutscenes, spawn delays, or credits.
Label them before optimizing them. “Silly repeat” means an action loop with no
relevant progress, not merely a large reason count.

## 6. Use a candidate-promotion loop

Never overwrite the last known successful baseline while testing a candidate.

1. State one hypothesis and the expected metric.
2. Run the narrowest checkpoint from an identical start fingerprint.
3. Reject or revert candidates that do not beat the real outcome.
4. Run the natural-entry suffix.
5. Run the full dry evaluation.
6. Promote the candidate report to `latest` only after integrity checks pass.
7. Update `STATUS.md`, the baseline table, and the next bottleneck.

Recommended artifact names:

```text
recordings/baseline_success.json
recordings/candidates/<timestamp>_<change>.json
recordings/candidates/<timestamp>_<change>.log
recordings/full_run_latest.json       # promoted success only
```

An isolated speedup is evidence, not proof. Different entry position, global
RNG, prior inputs, or leaked policy state can reverse it later in the route.

## 7. Make full runs fail fast and leave evidence

A full runner should abort on:

- death/life loss when forbidden
- difficulty or character drift
- stage/room regression that the route does not allow
- forbidden input or RAM write
- invalid assist write
- milestone timeout
- no progress beyond a segment-specific budget

On failure, save:

- the last safe natural-entry state
- a failure state when recoverable
- the progress fingerprint
- the recent action-reason window
- the last relevant RAM values
- a screenshot

The failure state becomes a regression fixture. Do not spend another complete
run reproducing a state that can be tested in seconds.

## 8. Optimize in the right order

1. Finish every required transition.
2. Eliminate hard locks and infinite loops.
3. Cover natural and boundary entries.
4. Reduce deaths and assist volume.
5. Reduce high-cost repeat loops.
6. Shorten already reliable segments.
7. Remove assists, if desired.

Use action-reason counts to find candidates, then trace actual progress edges
to find causes. Preserve working timing-sensitive sequences until a
checkpoint comparison proves a replacement is better.

## 9. Close the loop after every successful full run

- verify the manifest with a machine-readable assertion
- compare against the previous successful baseline
- update damage/resource/split tables
- record regressions as well as wins
- update the closest `AGENTS.md` immediate goal
- keep the previous success for comparison
- encode video only after the dry policy is stable

The result should answer three questions without reading source code:

1. What exactly counts as a clear?
2. What assistance was used?
3. Can the run be reproduced and compared with its predecessor?

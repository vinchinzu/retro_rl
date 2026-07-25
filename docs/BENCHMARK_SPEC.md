# Benchmark Specification

Stable evaluation rules. Changing program facts live in
[PROGRAM_STATUS.md](PROGRAM_STATUS.md).

Do not use the bare word **tier** without qualification. Prefer the labels
below.

## Runtime observation class

What the live agent may observe during an attempt.

### Gold

- Observes only pixels, action history, and its own memory
- Outputs only controller input
- No RAM or emulator-internal state in the decision loop

### Silver

- Primarily visual control with limited generic internals
- No RAM writes, teleports, or mid-run emulator-state mutation for control
- RAM may still be used for training rewards, offline labeling, resets,
  bookkeeping, and limited generic safety checks
- A full hardcoded route should not be the entire solution

### Bronze

- Autonomous controller input after the attempt starts
- Game-specific **read-only** RAM is permitted
- Game-specific heuristics, room graphs, reward shaping, specialist models,
  scripted menus, and save-state curricula are allowed during development

Training method is separate from runtime observation class. Imitation learning
and privileged training signals are valid when disclosed; the claimed class
describes the live evaluation loop.

## Intervention class

What the attempt may mutate mid-run. Independent of runtime observation.

| Class | Meaning | Default rule |
|-------|---------|--------------|
| Clean | No RAM writes or emulator-state mutation during the attempt | Preferred publication class |
| Survival-assisted | Health, lives, or similar attrition relief | Disclosed and counted |
| Resource-assisted | Ammo, fuel, currency refill of unlocked capacity | Disclosed; no progression grants |
| Protection-assisted | Iframes, hazard immunity | Phase-scoped and counted |
| Progression-assisted | Item flags, rooms, bosses, stage writes | Normally excluded |

Results must use **both** labels, for example:

```text
Bronze / Clean
Bronze / Resource-assisted
Bronze / Resource-assisted + Protection-assisted
Silver / Clean
Gold / Clean
```

Do not call a write-assisted run simply “Bronze.” Bronze implies read-only RAM
observation, not permission to write.

Loading a save state or mutating emulator memory during an active attempt
invalidates a Clean attempt. Disclosed assists require a game-local
`docs/ASSIST_CONTRACT.md` and counted interventions in the manifest.

## Scriptably beatable

A game is scriptably beatable when the repository contains a policy that:

- starts from a published reset or initial state
- uses controller actions for gameplay progression
- reaches a defined legitimate ending or campaign objective
- detects success independently
- can recover or restart without human input
- has a documented success rate over repeated attempts
- discloses runtime observations and assists

Large game-specific codebases, route graphs, boss tables, room scripts,
grinding, planners, and RAM-aware recovery are allowed when disclosed.

## Valid attempt

- Publish the start state or deterministic reset sequence.
- Publish the success condition with observable evidence (not vague prose).
- Log attempt count, success rate, and completion time or frame count.
- Report runtime observation class and intervention class next to every result.
- If training used privileged signals, say so separately from the runtime class.
- Segment clears from development save states are development evidence, not
  continuous full runs.

## Completion evidence

Prefer a conjunction of observable signals: ending event, credits phase, final
transition, and a settle period. The full-run manifest must make the contract
auditable.

Every claimed continuous clear must link to a machine-readable manifest.
Every assisted result must link to an assist contract.
Every benchmark claim must carry runtime and intervention labels.

## Metrics

Runners should report at least:

- outcome and final milestone
- frames and wall-clock equivalent
- damage, deaths, continues, and assist interventions
- resource writes by type
- milestone/split times
- maximum no-progress interval
- forbidden inputs or writes

## Publication requirements

- Preserve the previous successful baseline while testing candidates.
- Promote candidate reports to `latest` only after integrity checks pass.
- Encode video only after the dry policy is stable.
- Keep ROMs and copyrighted assets out of the repository.

## Related ladders (not this document)

| Ladder | Labels | Document |
|--------|--------|----------|
| Completion maturity | M0–M8 | [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) |
| Capability phase | Phase 0–7 / genre tracks | [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) |
| Automation class | Replay / RAM script / hybrid vision / autonomous discovery / unseen-game generalization | [GLOSSARY.md](GLOSSARY.md) |

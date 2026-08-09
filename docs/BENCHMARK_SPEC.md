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

Offline route research is also separate from runtime class. Walkthroughs,
maps, wikis, speedrun notes, and other external references are permitted
development inputs for all games. Record source-informed route facts in the
closest game-local documentation and validate them in the emulator. Using
external knowledge does not count as an intervention; live observations,
state loads, memory writes, and controller inputs determine the benchmark
labels.

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

## Typed evaluation contract

The shared `retro_harness.benchmark` module represents these labels as typed
`RuntimeObservationClass` and `InterventionClass` values. A published claim
also binds a `StartIdentity` and `PolicyIdentity`; each identity carries a
stable (SHA-256 by default) digest. `EvaluationContract` combines the labels, identities, goal,
and (for assisted classes) both `assist_contract_path` and
`assist_contract_digest`. A case's intervention, assist-contract fields, and
`assist_mode` are bound to its `BenchmarkCase.contract` and cannot be replaced
by an explicit run contract.

`PolicyIdentity.name` is a display label, not the identity of an executed
policy. For scripted policies, `policy_identity_for(policy)` (also available
as `PolicyIdentity.from_policy`) derives the digest from the implementation's
module and qualified name plus an inspectable source or bytecode digest. An
opaque module/qualified-name fallback remains usable only for non-publication
compatibility runs.

Learned policies require a `PolicyArtifact`; implementation source does not
identify model weights. The manifest binds the checkpoint SHA-256, algorithm,
hyperparameters, training seed, observation/action/reward/wrapper schema
digests, ROM/start/core identities, dependency-lock SHA-256, and source
commit. Loading verifies the manifest digest, checkpoint bytes, and expected
schema digests. A model-like policy without this artifact is rejected before
evaluation.

An `AttemptAudit` records observed `ram_writes`, `mid_run_loads`, and counted
`assists`, alongside the identity digests and an `AuditCapabilities` proof.
Missing counters remain unknown (`null`), never zero. `AuditedEnv` is the
shared capability-bearing adapter; a game-owned environment may emit the same
typed fields when it owns all intervention paths (the Super Metroid structured
combat environment is the first consumer). `validate_claim(contract, audit)`
returns true only for matching identities and complete instrumentation. It
rejects uninstrumented environments, all three intervention types for Clean,
and invalid class strings. Assisted contracts cannot be constructed without
both assist-contract fields. Attempt and seed records expose these fields
directly as `runtime_observation_class`, `intervention_class`,
`start_identity_digest`, and `policy_identity_digest`, with the full audit
under `attempt_audit`.

The old `BenchmarkTier` enum remains only as a deprecated adapter to
`RuntimeObservationClass`; new contracts should use the typed classes and
should not publish a bare tier label.

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

## Seed-robustness (randomizer / solver class)

Fixed-ROM continuous clears (M7/M8) remain the default evidence class for
vanilla titles. Randomizers and the solver stack need an **additional** class
that measures generalization across seeds.

### Seed-robust clear

A policy is **seed-robust** for a published contract when:

1. **T** seeds are drawn independently under a documented randomizer config
   (generator, version, logic settings, goal).
2. Each attempt starts from that seed’s published power-on / ROM image with
   **0 human mid-run supervision**.
3. Success is the same legitimate ending / race goal as the game contract.
4. Each attempt has a published **frame or wall-clock budget**.
5. The policy clears at least **S of T** seeds (S and T published; e.g. 3/5,
   8/10). Report failures with terminal milestone and failure mode.
6. Runtime observation class and intervention class are labeled as usual
   (Bronze/Silver/Gold × Clean/assisted). Assists, if any, require
   `ASSIST_CONTRACT.md` and counts **per seed**.

Do not claim seed-robustness from a single cherry-picked seed or from a
spoiler-oracle policy unless the contract explicitly allows a development
oracle and labels it separately from the runtime solver.

### Machine-readable seed report

The shared [`retro_harness.benchmark` seed-report API](../retro_harness/benchmark.py)
provides a deterministic dry-run adapter for existing `BenchmarkCase` policies.
`SeedRobustnessConfig` publishes the generator, generator version, logic, goal,
ordered unique seed IDs (**T**), frame budget, success threshold (**S**), the
runtime/intervention labels, and contract-level identity digests.
When config-level start or policy identities are omitted, the report uses an
explicit per-seed identity scope and validates each seed's typed contract. An
explicit config identity or contract uses shared scope and must match every
per-seed result; the serialized scope fields make that distinction auditable.
`run_seed_robustness` calls a seed-to-case factory once per published seed,
resets the policy for each case, and never samples, shuffles, or replaces the
seed list.

Every seed case's `BenchmarkCase.max_steps` must equal the published frame
budget. Every per-seed `frames` result must be at most that budget; both direct
report construction and the runner reject over-budget results. Seed-aware
environments can put these fields in their final `info` mapping for automatic
extraction:

- `terminal_milestone`: the furthest durable milestone reached
- `failure_mode`: a stable failure category; the benchmark timeout reason is
  used when this is absent on a failed attempt
- `assists`: a mapping of assist name to per-seed intervention count

`write_seed_robustness_report` emits canonical JSON (`sort_keys=True`) with no
timestamps or wall-time measurements, so the same ordered inputs produce the
same artifact. Config metadata must be a JSON object with string keys and only
JSON values (null, booleans, strings, finite numbers, arrays, and nested
objects); unsupported values, non-string keys, and `NaN`/infinite numbers are
rejected before writing. Its schema is:

```json
{
  "event": "seed_robustness_report",
  "schema_version": 1,
  "policy": "policy-name",
  "config": {
    "generator": "generator-name",
    "generator_version": "1.2.3",
    "logic": "standard",
    "goal": "legitimate ending",
    "seeds": ["1337", "1338", "1339"],
    "seed_count": 3,
    "budget": 9000,
    "budget_unit": "frames",
    "success_threshold": 2,
    "runtime_observation_class": "Silver",
    "intervention_class": "Clean",
    "start_identity_digest": "sha256-start-set-digest",
    "policy_identity_digest": "sha256-policy-digest",
    "assist_contract_path": null,
    "assist_contract_digest": null,
    "assist_mode": null,
    "metadata": {}
  },
  "summary": {
    "seeds_total": 3,
    "seeds_successful": 2,
    "success_rate": 0.6666666666666666,
    "required_successes": 2,
    "threshold_met": true
  },
  "seed_results": [
    {
      "seed": "1337",
      "outcome": "success",
      "success": true,
      "frames": 8120,
      "terminal_milestone": "ending",
      "failure_mode": null,
      "assists": {},
      "runtime_observation_class": "Silver",
      "intervention_class": "Clean",
      "start_identity_digest": "sha256-start-digest",
      "policy_identity_digest": "sha256-policy-digest",
      "assist_mode": null,
      "ram_writes": 0,
      "mid_run_loads": 0,
      "attempt_audit": {
        "ram_writes": 0,
        "mid_run_loads": 0,
        "assists": {},
        "runtime_observation_class": "Silver",
        "intervention_class": "Clean",
        "start_identity_digest": "sha256-start-digest",
        "policy_identity_digest": "sha256-policy-digest"
      }
    }
  ]
}
```

The complete report contains one `seed_results` entry per published seed;
failures retain their terminal milestone and failure mode. A report is an
artifact of the stated contract, not permission to omit the per-seed ROM/start
state or assist contract required by the rules above.

### Mod-robust clear (later)

Same structure over a published set of **edited ROMs** (physics, rooms, or
item placement changes). Requires skill synthesis / rediscovery paths from
[SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md); not required for near-term
SMZ3 seed-robust proofs.

### Relationship to M-gates

| Evidence | Use |
|----------|-----|
| M0–M8 on one fixed ROM or one seed | Skill quality, continuous engineering, vanilla titles |
| Seed-robust S/T | Solver / randomizer flagship claims |
| Mod-robust set | Edit / mod robustness claims |

A title may hold both: e.g. SMZ3 M3 on seed 1337 **and** seed-robust 3/5 on
early portal→house once the multi-seed harness exists.

### Input tapes and solver evaluation

- **Tapes** may back fixed-game regression CI and imitation demos.
- **Seed-robust evaluation must not** be a single tape replay across seeds.
- Reactive observe → plan → skill invocation is the intended runtime loop;
  see [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

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

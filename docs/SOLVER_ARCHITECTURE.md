# Solver Architecture

Last updated: 2026-08-09.

Strategic reframe for `retro_rl`: a **game solver** that reads state, plans, and
acts each frame — robust to randomizers, mods, and edited ROMs — not a library
of fixed input tapes.

Live facts stay in [PROGRAM_STATUS.md](PROGRAM_STATUS.md). Multi-horizon
sequencing stays in [ROADMAP.md](ROADMAP.md). Evaluation contracts stay in
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

## Positioning

| Term | Role |
|------|------|
| **Solver** | Reactive loop: observe → plan → invoke skill → replan. Backbone of the product. |
| **Skill library** | Seed-invariant low-level controllers (rooms, bosses, menus, traversal). Layer 1 substrate. |
| **Input tape** | Fully precomputed plan. Degenerate case of a solver when the world is deterministic. |
| **Scripted completion** | Honest name for Layer 1 today; feeds the planner, is not the planner. |

**Input tapes remain useful as:**

1. **Regression fixtures** for fixed (non-randomized) games and CI.
2. **Low-level skill demonstrations** to bootstrap imitation.

They are **not** the solver backbone. Randomized or moddable play needs reactive
policy. A tape only survives as a physical how-to (traverse room X), never as a
high-level route across seeds.

## The randomizer epiphany

In a randomizer, **low-level physical execution of a room/transition is
seed-invariant**. Only the **high-level logical ordering** varies.

Room X, a morph-ball maze, a boss arena, a menu screen — same layout and
physics every seed. What changes is which transitions are reachable and in what
order (items shuffled).

That means:

- Existing per-room / per-boss / per-menu scripts are **not wasted**. They become
  the skill library a planner invokes.
- What is missing is the **planner** (item-logic graph search over inventory)
  and **online discovery** (so the planner is not hand-fed a per-seed map).

Fixed games will always let you cheat by hand-authoring routes. **One
randomizer title end-to-end forces Layers 2–4 into existence.**

## Layer stack

```text
Layer 4  Planner (item-logic graph search + inventory)
Layer 3  Online world-model discovery (rooms/doors/items; seed-agnostic)
Layer 2  Runtime observation bootstrapping (RAM / screenshot semantics)
Layer 1  Low-level skill library (seed-invariant physical controllers)
Layer 0  Deterministic parallel emulator pool (rollouts, save/load, speed)
```

| Layer | Status in this repo | Ownership |
|-------|---------------------|-----------|
| **L1** Skills | ~90% of current work; strong across ~20 games | Per-game (`snes/*`, `nes/*`) |
| **L4** Planner | Bounded capability planner has one real-game consumer; resource/risk extension is fake-tested | Shared — first consumer / extension fake-tested |
| **L3** Discovery | Super Metroid room-graph lessons; SMZ3 portal/world detect | Shared — incomplete |
| **L2** Observation | Dev-time RAM maps + miner tooling; not runtime bootstrap | Shared — incomplete |
| **L0** Emulator pool | Emulator-state-only pool; wrapper/RNG/episode snapshots remain open | Shared — fake-tested only |

**Strategic conclusion:** the hard bespoke work (L1) is largely done for many
titles. Transfer value lives in **L0 + L2–L4**, which are mostly unbuilt and
game-agnostic.

### Full stack under mods

Mods/edits break physics and room layout, so the skill library is fragile. The
solver must eventually **synthesize** low-level skills when the library lacks
one: RL, neuroevolution, hill-climbing, genetic routes
(`retro_harness.platformer.genetic`, roadmap optimizers). Honest home for the
"RL" in `retro_rl`:

```text
learn / synthesize skills (RL · neuroevo · optimizers)
  + discover world model (L3)
  + plan with item logic (L4)
```

RL is the **skill-generator and contractor**, not the whole solver.

## Flagships

| Role | Title | Directory | Why |
|------|-------|-----------|-----|
| **Harness fixture** | Great Waldo Search | `snes/great_waldo_search/` | Pipeline M8; deterministic clear evidence |
| **Solver flagship (SM)** | Super Metroid | `snes/super_metroid/` | Deep room graph, inventory, pure-first skills (L1 substrate) |
| **Solver flagship (Z3)** | A Link to the Past | `snes/alttp/` | Dungeon/overworld graphs, item logic, SMZ3 co-world |
| **Single-game rando (SM)** | Super Metroid Randomizer | `snes/sm_rando/` | Simpler logic/rooms than SMZ3; prove L4 + S/T first |
| **Single-game rando (Z3)** | ALTTP Randomizer | `snes/alttp_rando/` | Item-logic + dungeon/OW without portals |
| **Solver proof (combined)** | SMZ3 randomizer | `snes/smz3/` | Dual-world + portals; full stack stress test |

**Proof order (forcing function):**

1. Keep deepening **vanilla SM** and **vanilla ALTTP** skill libraries and
   natural-entry chains (L1 quality gates the planner).
2. Scaffold **single-game randos** (`sm_rando`, `alttp_rando`): seed packages,
   coarse item-logic graphs, play/record spine — less complicated than SMZ3.
3. Generalize item-logic planning on top of those graphs
   (`adventure.RouteGraph` → learnable logic format / shared L4).
4. Prove **seed-abstract** clears on **single-game rando** first (S/T within
   budget), then extend the same harness to **SMZ3**.

```text
vanilla SM / ALTTP skills
        → sm_rando / alttp_rando (logic + seed spine)
        → SMZ3 (portals + combined pool)
```

**Play spine:** `retro_harness.play_spine` + per-package
`scripts/play.py --vanilla` — every session writes a run manifest under
`recordings/` for demos, imitation, and multi-seed aggregation (fun + fast).

Zelda I/II and NES Metroid remain valuable **stepping stones** and skill
donors.

## Build priorities (solver-ordered)

| # | Deliverable | Layer | Notes |
|---|-------------|-------|-------|
| 1 | **Logic-graph solver** | L4 | Heart. Item-set prerequisites on transitions; Dijkstra/DFS over inventory. Probe transitions under controlled inventories to **discover** logic. Extend `adventure` graph. |
| 2 | **Online world-model + transition discovery** | L3 | Seed-agnostic room/door/item graph from runtime observation. SM room-graph + SMZ3 world detect generalize here. |
| 3 | **Runtime observation bootstrapping** | L2 | Promote RAM miner to runtime: few-shot address discovery or screenshot features. Gold / mod survival path. |
| 4 | **Imitation → skill generalization** | L1+ | Demos → condition-robust skills (not fixed tapes). |
| 5 | **Parallel emulator pool** | L0 | Deterministic rollouts for search/simulation/planning. |
| 6 | **Low-level skill synthesis** | L1+ | Optimizers / neuroevo / RL when library or physics break under mods. |

### L4 capability-edge planner surface

`retro_harness.adventure` provides the small, game-agnostic item-logic surface
used before any emulator integration:

- `GraphEdge.requires` is the normalized prerequisite capability set.
- `GraphEdge.acquires` records monotonic items, events, or defeated-boss flags
  gained after the transition; `RoutePatch` carries the same annotations.
- `PlanRequest` + `PlanBudget` run deterministic Dijkstra over monotonic
  progression states with same-node dominance pruning. `PlanResult` reports an
  explicit `FOUND`, `UNREACHABLE`, or `BUDGET_EXHAUSTED` status, path/cost,
  final progression, expansion/pruning counts, and stable frontier blockers.
  The default hard gate is 500 expansions.
- `plan()` and `resource_plan()` share that one search frontier, budget,
  blocker, reconstruction, and dominance implementation. Resource/risk search
  supplies only an extra state transition/cost adapter; a same-node state with
  at least as many resources participates in the same dominance pruning.
- `RouteGraph.inventory_aware_path` remains the compatibility adapter that
  returns only the least-cost edge sequence. `shortest_path` remains
  fixed-inventory BFS for callers that already supply a complete inventory.
- `SkillBinding` binds a versioned dispatch key, entry requirement digest, and
  progression-delta digest. These are solver skill identities, not ML/env
  `ContractBundle` digests. Each binding belongs to one explicit `edge_id`.
  `EdgeEvidence` promotes only through typed
  `ExecutionReadiness` values; natural-entry and higher evidence must link the
  predecessor exit observation digest to the target entry digest.
- `BindingCatalog.publication_edges` excludes unbound and below-natural-entry
  transitions. Parallel graph edges are independent because both bindings and
  evidence are keyed by edge ID, never only by source/target pair.
- `retro_harness.solver.SolverSession` is the execution kernel: it checks a
  `SkillSpec` observation requirement, dispatches the bound `SkillInstance`, emits
  actions until success/failure/timeout, validates observed progression and
  resource deltas, and replans after retryable failures. Its deterministic
  trace carries lifecycle transitions, observations, outcomes, actions, and
  policy identity digests. Legacy `protocol.py` task interfaces remain a thin
  facade while game consumers migrate edge by edge.
- `ResourcePlanRequest` extends the bounded state with typed consumable,
  renewable, and safety resources. Edge profiles declare consume/produce and
  minimum bounds; `PlanResult` carries the selected resource trajectory and
  typed resource blockers. Risk cost uses smoothed success and duration
  statistics aggregated from retained `SkillOutcome` values. This extension
  is still **fake-tested** until a real game planner consumes it.

### Versioned environment/model contracts

`retro_harness.contracts` makes compatibility semantic rather than a tensor-
dimension guess. `ContractBundle` binds five independently digestible records:

- ordered observation fields and preprocessing;
- ordered action rows and controller-button order;
- named reward components and weights;
- exact wrapper stack order and wrapper configuration (including frame skip);
- game/start/ROM/emulator-core identity.

`GameSpec.contract` carries this expanded environment identity when a consumer
has one. Learned checkpoints use a `PolicyArtifact` sidecar containing the four
schema digests plus ROM, state, and core identities; loading fails if the
checkpoint bytes, any schema, wrapper order, or environment identity differs.
`retro_harness.identity` is the sole canonical-JSON/file-hash primitive;
`retro_harness.model_artifacts` owns `PolicyArtifact` and its read/write
helpers. `retro_harness.audit.AuditedEnv` owns intervention counters at the
backend `data.set_value` and emulator `set_state` boundaries, so evidence
consumers do not author zero counts themselves.
The fighter PPO final-save/resume/eval path and platformer neuro checkpoint path
are the first consumers. Neuro checkpoints embed the complete bundle and no
longer silently resume a same-shaped but semantically different feature vector.

`retro_harness.entry_states.EntryStateCorpus` is the distribution layer above
those contracts. Each record binds emulator-state and RAM hashes to its source
skill/segment/trajectory, frame/parity, game metadata, observation schema, and
contract bundle. Splits are deterministic by state hash or source trajectory;
the latter forbids trajectory leakage. Platformer neuro training accepts a
corpus manifest but exposes only its train partition. The first real corpus is
the 64-state SM-rando Ceres→Landing distribution documented in
`snes/sm_rando/docs/STATUS.md`.

`retro_harness.trajectory` is the time-series layer beside the entry-state
distribution. It binds exact structured actions, observation boundaries,
reward components, milestones, terminal reasons, optional state digests, and
source provenance to the same contract/policy identities. `SolverSession`
exports directly to it. A content-addressed `CounterexampleLibrary` retains
failed episodes and imports a failure cluster for offline replay or BC instead
of preserving successes alone. The SM-rando vertical slice is its first real
consumer.

Game-local stop predicates remain in game adapters. A useful probe-driven
discovery loop is:

1. Hold the source and target transition constant while probing controlled
   inventories (one item/event variant per run).
2. Record whether the transition opens, the observed state change, and the
   smallest capability set that changes the result.
3. Encode that difference as `requires`; encode a collected item or cleared
   event on the successful transition as `acquires`.
4. Keep the edge `verification` at `planned` until emulator evidence promotes
   it. The planner consumes the graph; it does not claim that a low-level skill
   can execute an unverified edge.

## Shared subsystem evidence ladder

Issue closure and subsystem maturity are different facts. Shared work reports
the highest rung actually evidenced:

| Rung | Required evidence |
|------|-------------------|
| **Scaffolded** | API or implementation exists; no behavioral claim yet. |
| **Fake-tested** | Deterministic unit/adversarial tests pass with fixtures or doubles. |
| **Real-ROM tested** | A bounded stable-retro run exercises the subsystem on a real ROM. |
| **First real-game consumer** | A game-owned workflow retains evidence and depends on the shared interface. |
| **Second independent consumer** | A different game/package proves the abstraction without copying the first adapter. |
| **Publication-ready** | At least two independent consumers, fail-closed identities/audits, reproducible commands, retained artifacts, and no unresolved claim-critical child issue. |

Closure checklist for a shared-infrastructure bead:

1. Put the evidenced rung in the close reason or notes; never use “complete”
   to imply a higher rung.
2. Name tests, real-run artifacts, and game consumers separately.
3. Link open children for missing restoration, audit, integration, or campaign
   work.
4. A first-consumer-only interface may close its implementation task, but it
   is not called stable or publication-ready.
5. Publication-ready requires a second independent consumer. Any exception
   must be an explicitly narrower, first-consumer-only claim.

Current audit (2026-08-09):

| Subsystem | Highest rung | Evidence / remaining gate |
|-----------|--------------|---------------------------|
| Emulator pool (`rr-gbd.16`) | Fake-tested | Snapshot is emulator-only; full wrapper/RNG/episode restore remains `rr-gbd.32` / `.34`. |
| Capability planner + bindings + SolverSession | First real-game consumer | SM-rando real-ROM Landing→Pit vertical slice with recovery/replan. |
| Resource/risk planner | Fake-tested | Key/missile/reliability golden fixtures; no game-owned planning consumer yet. |
| Contracts + PolicyArtifact + benchmark audits | First real-game consumer | SM-rando vertical slice and audited Landing BC experiment; resumable multi-seed campaign remains `rr-gbd.33`. |
| EntryStateCorpus | First real-game consumer | 64-state SM-rando corpus; second predecessor trajectory is still required for stronger generalization evidence. |
| Trajectory + counterexamples | First real-game consumer | SM-rando vertical and held-out BC trajectories; no second game consumer. |

No shared solver subsystem is currently publication-ready.

## Workflow changes

| Old default | Solver default |
|-------------|----------------|
| Hand-author per-game **routes** as the product | Hand-author **skill policies**; planner sequences them |
| M-gates on a single route / seed | **Seed-abstract** M-gates: S clears out of T random seeds within budget |
| Each game a self-contained project | L1 per-game; **L0+L2–L4 shared once** |
| Input-tape CI as core | Tapes = regression + demos; core = reactive plan + skills |
| Breadth = new full games | Breadth = skills/assets; new full projects only after shared solver layer is proven |

## Benchmark class: seed robustness

Defined formally in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md). Summary:

- **Seed-robust clear:** autonomous policy clears **S of T** independently
  drawn seeds within a published frame/time budget, 0 human mid-run
  supervision, labels as usual (Bronze/Silver/Gold × intervention).
- **Mod-robust clear (later):** same idea over a published set of edited ROMs.
- Report the **distribution** (success count, failure modes, budget headroom),
  not a single cherry-picked seed.

Vanilla fixed-ROM M0–M8 remains valid for non-random titles and for L1 skill
quality. Seed-robustness is an **additional** class for randomizer / solver
proofs.

## Concrete first experiment

1. **Scaffold** a game-agnostic item-logic solver on
   `retro_harness.adventure` (inventory-aware search + probe harness).
2. **Ground** it on Super Metroid and/or ALTTP capability edges already in-repo.
3. **Wire** SMZ3 seed package / spoiler as optional oracle for development, with
   a seed-agnostic path that does not require spoilers at runtime.
4. **Define** a seed-robustness dry-run entrypoint: N seeds, shared budget,
   machine-readable report.
5. **Keep** Great Waldo as harness fixture; do not replace it — add the solver
   fixture beside it.

## What this is not

- Not a stop-work on Final Fight, platformers, NES breadth, or Harvest planning.
  Those trunks still build L1 skills and genre stacks.
- Not "RL replaces scripts." Scripts stay the reliable skill layer; RL
  synthesizes and generalizes skills.
- Not multiworld multiplayer SMZ3 or full tracker integration as day-one scope
  (see `snes/smz3/docs/plan.md`).

## Related documents

| Document | Role |
|----------|------|
| [VISION.md](VISION.md) | Program purpose (solver + library) |
| [ROADMAP.md](ROADMAP.md) | Horizons and ordered work |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | M0–M8 + Phase 0–7 (Phase 7 elevated) |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Seed-robustness class |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live flagship claims |
| `snes/smz3/docs/` | Combined randomizer ground truth |
| `retro_harness/adventure/` | Shared capability graph + inventory-aware path scaffold |
| `retro_harness/platformer/genetic.py` | Skill-synthesis optimizer foothold |

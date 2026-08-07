# Solver Architecture

Last updated: 2026-08-06.

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
| **L4** Planner sketch | `retro_harness.adventure` (`RouteGraph`, `shortest_path`, edge `requires`) | Shared — incomplete |
| **L3** Discovery | Super Metroid room-graph lessons; SMZ3 portal/world detect | Shared — incomplete |
| **L2** Observation | Dev-time RAM maps + miner tooling; not runtime bootstrap | Shared — incomplete |
| **L0** Emulator pool | Single-env harness + some dual-bot SMZ3 scaffold | Shared — incomplete |

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
| `retro_harness/adventure/` | Shared graph + shortest_path sketch |
| `retro_harness/platformer/genetic.py` | Skill-synthesis optimizer foothold |

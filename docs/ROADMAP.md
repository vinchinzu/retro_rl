# Strategic Roadmap — NES + SNES Canonical Library

Last updated: 2026-08-06.

This is the multi-horizon plan for `retro_rl`. Live facts and weekly priorities
belong in [PROGRAM_STATUS.md](PROGRAM_STATUS.md). Maturity and phase definitions
belong in [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md). Evaluation labels
belong in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md). The game board is generated in
[GAME_MATRIX.md](GAME_MATRIX.md). Solver layer stack and flagship triangle live
in [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

## Goal

Produce **verified reset-to-ending clears (M8)** for a large, prioritized
canonical library of major **NES and SNES** titles — while building a
**reactive game solver** (skills + planning + discovery) that generalizes to
**randomizers and mods**, and while progressively reducing privileged
information.

This is **not** a project to brute-force every ROM ever made. Target scale is a
prioritized set of roughly **50–100 high-impact titles**, not the long tail of
obscure cartridges.

### What “solved” means for a game

- A policy starts from a published reset / power-on (or documented initial state).
- It uses only controller actions for progression.
- It reaches a legitimate, independently detectable ending.
- Success is reproducible with documented rates.
- Runtime observation class (Bronze / Silver / Gold) and intervention class
  (Clean preferred; Survival- / Resource- / Protection-assisted disclosed) are
  explicitly labeled.
- Evidence (dry-run manifests, recordings, `STATUS.md`) lives under the game
  directory.
- **Randomizer titles:** additionally report **seed-abstract** success
  (S/T seeds within budget) per [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

This matches [VISION.md](VISION.md), the M0–M8 ladder, the natural-entry rule,
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md), [FULL_RUN_PROCESS.md](FULL_RUN_PROCESS.md),
and [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md).

## Guiding principles (do not violate)

1. **Natural-entry is sacred** — A segment is not ready until it also clears from
   the real predecessor state, not only a clean checkpoint.
2. **Skills are the substrate; the planner sequences them** — Prefer hand-authored
   skill policies over hand-authored full-game routes as the product. Fixed
   routes remain valid for deterministic games and CI; they are not the solver
   backbone.
3. **Input tapes are demoted** — Regression fixtures and imitation demos only;
   randomized play requires reactive observe → plan → act.
4. **Reusable stacks first** — Promote into `retro_harness` subdomains
   (`platformer`, `fighters`, `adventure`, …) only after a second consumer
   proves the abstraction. Solver Layers 2–4 are shared once, not reimplemented
   per game.
5. **Phases over flat ranking** — Games live in parallel capability tracks
   (linear combat, platforming, continuous control, graph navigation, campaigns,
   planning, adaptive/randomizer).
6. **Privilege reduction is first-class** — Start Bronze (game-specific
   read-only RAM); deliberately move toward Silver then Gold on mature titles.
7. **Prefer Clean** — Assists are allowed only when disclosed, counted, and
   contractually limited (`ASSIST_CONTRACT.md` pattern).
8. **NES is first-class** — Parallel library under the same ladder, not an
   afterthought.

## Flagship triangle + single-game rando rungs (solver)

| Role | Title | Purpose |
|------|-------|---------|
| Harness fixture | Great Waldo Search | Deterministic pipeline M8 |
| Substrate A | Super Metroid | L1 skills + room/inventory graph |
| Substrate B | A Link to the Past (Zelda 3) | L1 skills + dungeon/OW item logic |
| **Rando rung (SM)** | `sm_rando/` | Single-game logic + seed spine (simpler than SMZ3) |
| **Rando rung (Z3)** | `alttp_rando/` | Single-game ALTTPR-style logic + seed spine |
| Solver proof | SMZ3 | Combined worlds; forces portals + full stack |

Vanilla SM/ALTTP **gate** single-game randos; single-game randos **gate** SMZ3
complexity. Prefer proving multi-seed S/T on `sm_rando` / `alttp_rando` before
claiming SMZ3 seed-robustness.

## Logical stepping stones

### Phase 0 / Foundation (immediate, ongoing)

- Keep Great Waldo Search as the living pipeline fixture (already M8 Clean).
- Harden `retro_harness` for clean **NES + SNES** parity (GameSpec, startup
  plans, RAM schemas, recording, editor).
- Strengthen RAM discovery tools, editor workflows, and continuous documentation
  checks (`docs/generate_game_matrix.py`, `tests/test_docs.py`).
- Standardize every game on `STATUS.md` / `plan.md` / `ram_map.md` /
  `AGENTS.md` + optional `ASSIST_CONTRACT.md`.
- Scaffold **solver core** docs and packages (logic graph, seed-robust report).

### Near-term trunks (next 3–6 months)

Highest leverage — **solver flagship triangle first**, then parallel genre trunks:

1. **Solver stack (new flagship effort)** — See
   [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md) priority table:
   - **L4** Logic-graph solver on `retro_harness.adventure` (item-set
     prerequisites; inventory search; probe-driven logic discovery).
   - **L3** Online world-model / transition discovery (seed-agnostic).
   - **Seed-robustness harness** — report S/T seeds within budget.
   - Ground on **`sm_rando` / `alttp_rando`** graphs first; then SMZ3.
   - **Play spine** (`retro_harness.play_spine`) for fast human demos + run
     manifests every session.
2. **Super Metroid → M6 → continuous toward ending** — Verified continuous tip
   is power-on → Varia (M5). Pure reverse + K4 + boss pipeline. L1 substrate for
   `sm_rando` / SMZ3 and the planner.
3. **ALTTP (Zelda 3) → solid route graph (M3–M5+)** — Opening route exists;
   deepen dungeon/overworld skills and capability edges for `alttp_rando` /
   SMZ3 Z3 side.
4. **`sm_rando` / `alttp_rando` → M1 boot + early multi-seed** — Seed packages
   and coarse graphs scaffolded (M0). Next: patched ROM boot, skill-bound
   edges, S/T early tips.
5. **SMZ3 → seed-abstract segments** — Beyond single-seed parlor→house: longer
   segments + multi-seed dry-run once single-game patterns exist.
6. **Final Fight → M8** — Generalize the proven TMNT IV combat stack (behavior
   trees, combat policies, watchdogs, natural-entry, continuous dry-run). Phase 1
   reference.
7. **Magical Quest (and Joe & Mac) → solid segments (M3–M5)** — Establish and
   harden `retro_harness.platformer` route tooling, recovery, and evaluation
   (feeds L1 skill synthesis / optimizers later).
8. **Parallel NES advance**
   - Top-10 automation targets are **boot-verified (M1)**: SMB, Mega Man 2,
     Punch-Out, Contra, Kirby, TMNT II, DuckTales, Castlevania, Zelda I, SMB3
     (plus TMNT I/III and Zelda II foothold).
   - Zelda I / Metroid NES remain graph stepping stones and skill donors.
   - Prefer adding **skills**, not new full-game projects, until the shared
     solver layer is proven.

Also push Super Double Dragon and Rival Turf upward in parallel with Final Fight.

**Planning trunk (pull Harvest forward):** Harvest Moon is already M3 with a
continuous spring calendar, day planner, editor, and recording pipeline. Treat
it as the **pioneer planning/simulation stack** rather than parking all work in
long-horizon Phase 6. Near-term: close crop income (plant → water → harvest →
ship), natural-entry summer, then domain depth (animals, festivals). Structure
work lives under `harvest/docs/PLANNING_STACK.md` and should later promote to
shared planning only after a second consumer (solver L4 is a natural second
consumer of plan/search primitives).

### Medium-term (6–18 months) — genre trunks + solver layers mature

| Focus | Target |
|-------|--------|
| **Solver L4–L3** | Item-logic planner + online discovery proven on SMZ3 S/T seeds; second consumer (SM rando or ALTTP rando alone). |
| **Solver L2 / L0** | Runtime observation bootstrap foothold; deterministic parallel emulator pool for rollouts. |
| **Phase 1 complete** | 4–6 linear combat / beat-em-up M8s (SNES + NES). Combat framework highly reusable. |
| **Phase 3 mature** | 3–5 platformers to high maturity / M8. Promote optimizers (hill-climbing, neuroevolution, genetic routes) as **skill synthesis**. |
| **Phase 4** | Super Metroid to M8, ALTTP deep route graph, full Zelda I, Castlevania. `retro_harness.adventure` is the shared graph home. |
| **Phase 2** | Continuous control (F-Zero, Star Fox, Pilotwings) to mission clears. |
| **Fighters** | Elevate SF2 / MK / Super SF2 from match wins to full arcade-mode continuous clears. |
| **Privilege** | Systematic observation-class improvement on 2–3 mature games (Bronze → Silver). |

### Longer-term (18 months – multi-year)

- **Phase 5** — Structured RPG / campaign games (Chrono Trigger, Final Fantasy
  series, EarthBound, Super Mario RPG; NES Dragon Quest / Final Fantasy
  equivalents).
- **Phase 6** — Planning / simulation (Harvest Moon early trunk; later strategy).
- **Phase 7** — Adaptive / procedural / **randomizer-robust** play is no longer
  “later research only”: SMZ3 is the near-term forcing function; Shiren and
  unseen-game adaptation remain research depth.
- **Mod robustness** — Skill synthesis (RL / neuroevo / optimizers) when physics
  or rooms change under edits.
- **Library breadth** — Add high-value titles only via [ADDING_GAMES.md](ADDING_GAMES.md)
  once the relevant genre stack is mature enough that a new game reaches M3
  relatively quickly; prefer skill catalog growth over project sprawl.
- **Tooling scale** — Hierarchical planner + low-level controllers, vision for
  Gold, continuous multi-seed benchmarking.

## Prioritization heuristic for any new game

1. Closes a current bottleneck or completes an active trunk (especially the
   solver flagship triangle).
2. Extends or proves a shared package (high transfer value — L2–L4 preferred).
3. Clear, detectable ending + good observability.
4. Cultural / popularity weight.
5. Engineering difficulty matches current tooling maturity.

## RL, tapes, and privilege-reduction path

- Keep scripted policies + optimizers as the reliable **skill** baseline.
- Input tapes: **CI regression** for fixed games + **demos** for imitation —
  never the high-level randomizer route.
- Imitation → condition-robust skills (BC / policies conditioned on observation).
- RL / neuroevolution / genetic routes: **synthesize** skills when the library
  lacks one or mods break physics (`platformer/genetic.py` and roadmap
  optimizers).
- Hierarchical approaches once graphs and combat are solid (planner + skills).
- Observation-class migration (Bronze → Silver → Gold) as an **explicit**
  workstream, independent of Clean vs assisted — L2 runtime bootstrap is the
  engineering path.

## Immediate concrete next actions

Aligned with [PROGRAM_STATUS.md](PROGRAM_STATUS.md); update that file when facts
change, not this horizon plan.

1. **Solver** — Scaffold item-logic graph solver + seed-robustness report format;
   wire first consumer on SM/ALTTP/SMZ3 edges.
2. **Super Metroid** — Close remaining critical path rooms and inventory bridges
   (L1 substrate).
3. **ALTTP** — Advance beyond opening route; dungeon/item capability edges.
4. **SMZ3** — Longer one-bot segments; multi-seed dry-run once harness exists.
5. **Final Fight** — Natural-entry hardening + Stage 3 continuity.
6. **Magical Quest / Joe & Mac** — First reliable room/segment clears with natural
   entry.
7. **NES** — Zelda I Level 2 graph; TMNT / SMB / MM2 continuations as skill work.
8. **Harvest Moon** — Close crop loop; planning-stack hardening.
9. Keep regenerating the game matrix and updating local `STATUS.md` after every
   verified advance.
10. Ensure every assist has an explicit contract before it is used in published
    results.

## Success metrics

- Number of games at each M-gate (especially M5+ and M8).
- Number of full continuous verified clears (currently TMNT IV + Great Waldo
  Search).
- **Seed-robust** results: S/T seeds cleared within budget (SMZ3 and future
  randos).
- Reusable packages promoted and second consumers (especially adventure /
  solver layers).
- Distribution of observation / intervention classes.
- Decreasing time-to-M3 and time-to-M8 for new games in mature genres.
- Documented success rates and audiovisual evidence under each game directory.

## How this document relates to the rest

| Document | Role |
|----------|------|
| This file | Multi-horizon strategy; do not put daily gate claims here |
| [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md) | Layer stack, tape demotion, build order |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Verified flagship results + near-term bottlenecks |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | M0–M8 and Phase 0–7 definitions |
| [GAME_MATRIX.md](GAME_MATRIX.md) | Generated board from `docs/manifests/*.yaml` |
| Local `STATUS.md` | Exactly one maturity gate and evidence for that game |

This roadmap is deliberately conservative and cumulative. It keeps genre trunks
alive, elevates Super Metroid + ALTTP + SMZ3 as the **solver flagship
triangle**, and only expands the library once each genre stack (and the shared
solver core) is proven. Velocity increases as reusable components mature.

Follow the natural-entry rule religiously, keep `STATUS.md` authoritative and
sparse, and regenerate the matrix. That discipline is what will actually let the
project scale to a large NES + SNES library — and to seed-robust randomizer
play.

# Strategic Roadmap — NES + SNES Canonical Library

Last updated: 2026-07-31.

This is the multi-horizon plan for `retro_rl`. Live facts and weekly priorities
belong in [PROGRAM_STATUS.md](PROGRAM_STATUS.md). Maturity and phase definitions
belong in [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md). Evaluation labels
belong in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md). The game board is generated in
[GAME_MATRIX.md](GAME_MATRIX.md).

## Goal

Produce **verified reset-to-ending clears (M8)** for a large, prioritized
canonical library of major **NES and SNES** titles — while building reusable
genre systems and progressively reducing privileged information.

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

This matches [VISION.md](VISION.md), the M0–M8 ladder, the natural-entry rule,
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md), and
[../snes_oneshot/docs/FULL_RUN_PROCESS.md](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Guiding principles (do not violate)

1. **Natural-entry is sacred** — A segment is not ready until it also clears from
   the real predecessor state, not only a clean checkpoint.
2. **Reusable stacks first** — Promote to shared packages
   (`platformer_common`, future `adventure_common`, etc.) only after a second
   consumer proves the abstraction.
3. **Phases over flat ranking** — Games live in parallel capability tracks
   (linear combat, platforming, continuous control, graph navigation, campaigns,
   planning).
4. **Privilege reduction is first-class** — Start Bronze (game-specific
   read-only RAM); deliberately move toward Silver then Gold on mature titles.
5. **Prefer Clean** — Assists are allowed only when disclosed, counted, and
   contractually limited (`ASSIST_CONTRACT.md` pattern).
6. **NES is first-class** — Parallel library under the same ladder, not an
   afterthought. The harness already uses stable-retro (both platforms);
   normalize GameSpec / startup / RAM schemas where needed.

## Logical stepping stones

### Phase 0 / Foundation (immediate, ongoing)

- Keep Great Waldo Search as the living pipeline fixture (already M8 Clean).
- Harden `retro_harness` for clean **NES + SNES** parity (GameSpec, startup
  plans, RAM schemas, recording, editor).
- Strengthen RAM discovery tools, editor workflows, and continuous documentation
  checks (`docs/generate_game_matrix.py`, `tests/test_docs.py`).
- Standardize every game on `STATUS.md` / `plan.md` / `ram_map.md` /
  `AGENTS.md` + optional `ASSIST_CONTRACT.md`.

### Near-term trunks (next 3–6 months)

Highest leverage from current program status:

1. **Final Fight → M8** — Generalize the proven TMNT IV combat stack (behavior
   trees, combat policies, watchdogs, natural-entry, continuous dry-run). Phase 1
   reference.
2. **Magical Quest (and Joe & Mac) → solid segments (M3–M5)** — Establish and
   harden `platformer_common` route tooling, recovery, and evaluation.
3. **Super Metroid → M6 → continuous toward ending** — Verified continuous tip
   is power-on → Varia (M5). Next: pure reverse → Business, continuous K4
   (Bubble→Speed→Wave→Ice→Alpha PB), then Phantoon→…→MB per boss pipeline.
   Structure wins: selective RAM, declarative tip composition, graph-first
   ranking. Phase 4 flagship.
4. **Parallel NES advance**
   - Top-10 automation targets are **boot-verified (M1)**: SMB, Mega Man 2,
     Punch-Out, Contra, Kirby, TMNT II, DuckTales, Castlevania, Zelda I, SMB3
     (plus TMNT I/III and Zelda II foothold).
   - Next: first isolated segments (M3) on capability-diverse titles — SMB 1-1,
     MM2 Air Man, Glass Joe, TMNT II wave, Zelda cave — then instrumentation M2
     and natural-entry chains.
   - NES titles are often simpler and excellent for rapid iteration of shared
     policies; pair with SNES analogues (SMB↔Magical Quest, MM2↔MMX,
     Punch-Out↔Super Punch-Out, TMNT II↔TMNT IV, Zelda↔ALTTP, Metroid↔SM).

Also push Super Double Dragon and Rival Turf upward in parallel with Final Fight.

**Planning trunk (pull Harvest forward):** Harvest Moon is already M3 with a
continuous spring calendar, day planner, editor, and recording pipeline. Treat
it as the **pioneer planning/simulation stack** rather than parking all work in
long-horizon Phase 6. Near-term: close crop income (plant → water → harvest →
ship), natural-entry summer, then domain depth (animals, festivals). Structure
work (skill composition, phase contracts, observation cache, gated plan-advisor
apply) lives under `harvest/docs/PLANNING_STACK.md` and should later promote to
a shared `planning_common` only after a second consumer.

### Medium-term (6–18 months) — genre trunks mature

| Focus | Target |
|-------|--------|
| **Phase 1 complete** | 4–6 linear combat / beat-em-up M8s (SNES + NES). Combat framework highly reusable. |
| **Phase 3 mature** | 3–5 platformers to high maturity / M8 (Magical Quest, SMW, DKC + NES classics SMB / SMB3 / Mega Man). Promote optimizers (hill-climbing, neuroevolution, genetic routes). |
| **Phase 4** | Super Metroid to M8, then A Link to the Past, full Zelda I, Castlevania. After two solid implementations, promote `adventure_common` (room/door graphs, inventory prerequisites, event flags, path replanning). |
| **Phase 2** | Continuous control (F-Zero, Star Fox, Pilotwings) to mission clears. |
| **Fighters** | Elevate SF2 / MK / Super SF2 from match wins to full arcade-mode continuous clears (hybrid scripted + improved PPO). |
| **Privilege** | Systematic observation-class improvement on 2–3 mature games (Bronze → Silver). |

### Longer-term (18 months – multi-year)

- **Phase 5** — Structured RPG / campaign games (Chrono Trigger, Final Fantasy
  series, EarthBound, Super Mario RPG; NES Dragon Quest / Final Fantasy
  equivalents). Dialogue macros, quest state machines, equipment policies,
  grinding.
- **Phase 6** — Planning / simulation (Harvest Moon is the early trunk: close
  crop economy and multi-season natural entry first, then multi-year campaign
  benchmarks; later strategy titles).
- **Phase 7** — Adaptive / procedural (later research).
- **Library breadth** — Add high-value titles only via [ADDING_GAMES.md](../ADDING_GAMES.md)
  once the relevant genre stack is mature enough that a new game reaches M3
  relatively quickly.
- **Tooling scale** — Automated route discovery, curricula, hierarchical RL
  (high-level planner + low-level controllers), vision models for Gold-level
  agents, comprehensive continuous benchmarking.

## Prioritization heuristic for any new game

1. Closes a current bottleneck or completes an active trunk.
2. Extends or proves a shared package (high transfer value).
3. Clear, detectable ending + good observability.
4. Cultural / popularity weight.
5. Engineering difficulty matches current tooling maturity.

## RL and privilege-reduction path

- Keep scripted policies + optimizers as the reliable baseline (especially for
  continuous verified clears).
- Expand the existing fighters PPO template.
- For platformers and hard segments: recorded successful runs → imitation → RL
  fine-tuning.
- Hierarchical approaches once graphs and combat are solid.
- Treat observation-class migration (Bronze → Silver → Gold) as an **explicit
  workstream**, independent of Clean vs assisted.

## Immediate concrete next actions

Aligned with [PROGRAM_STATUS.md](PROGRAM_STATUS.md); update that file when facts
change, not this horizon plan.

1. **Final Fight** — natural-entry hardening + Stage 3 continuity → chain toward
   continuous dry-run.
2. **Magical Quest / Joe & Mac** — first reliable room/segment clears with natural
   entry.
3. **Super Metroid** — close remaining critical path rooms and inventory bridges.
4. **NES** — Zelda I Level 1 is complete; route its completion warp to Level
   2 and continue the dungeon graph; TMNT series first building/stage clears;
   harden SMB / SMB3 continuations.
5. **Harvest Moon** — close crop loop (money > $100); skill composition +
   planning-stack hardening; natural-entry summer (see
   `harvest/docs/PLANNING_STACK.md`).
6. Keep regenerating the game matrix and updating local `STATUS.md` after every
   verified advance.
7. Ensure every assist has an explicit contract before it is used in published
   results.

## Success metrics

- Number of games at each M-gate (especially M5+ and M8).
- Number of full continuous verified clears (currently TMNT IV + Great Waldo
  Search).
- Reusable packages promoted and second consumers.
- Distribution of observation / intervention classes.
- Decreasing time-to-M3 and time-to-M8 for new games in mature genres.
- Documented success rates and audiovisual evidence under each game directory.

## How this document relates to the rest

| Document | Role |
|----------|------|
| This file | Multi-horizon strategy; do not put daily gate claims here |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Verified flagship results + near-term bottlenecks |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | M0–M8 and Phase 0–7 definitions |
| [GAME_MATRIX.md](GAME_MATRIX.md) | Generated board from `docs/manifests/*.yaml` |
| Local `STATUS.md` | Exactly one maturity gate and evidence for that game |

This roadmap is deliberately conservative and cumulative. It follows the process
already defined in the repository, turns the current near-term trio into durable
shared infrastructure, treats NES as a first-class parallel track, and only
expands the library once each genre stack is proven. Velocity increases as
reusable components mature.

Follow the natural-entry rule religiously, keep `STATUS.md` authoritative and
sparse, and regenerate the matrix. That discipline is what will actually let the
project scale to a large NES + SNES library.

# Game structure research (2026-08-28)

Condensed notes on how this repo already tries to grow games without
sprawl, and where Super Metroid / Harvest Moon show the failure modes
anyway. Not a new law. Live board: [GAME_MATRIX.md](GAME_MATRIX.md).
Hygiene: [REPO_HYGIENE.md](REPO_HYGIENE.md). Ladder:
[DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md).

## Answer

Agents progress a game when the **unit of work is one hop/skill behind a
small interface** (SpineHop / Task / StageSpec), **STATUS vs plan vs
AGENTS are split**, session process lives in a **skill not AGENTS.md**,
and new files merge into an owner instead of cloning a runner. The repo
already writes this down. SM and Harvest still sprawl because the
documented unit is smaller than the trees agents actually leave: probe
CLIs, per-room hops, mixin extracts, dual graphs, leftover `utils`,
stacked residuals. Slimmer games that still ship (TMNT IV M8, Hal’s
golf M2 clear, SMB M8) keep one composer and grow **rows or skills**,
not sibling packages. Transfer the interfaces; do not rewrite SM or
Harvest.

## 1. Numbered findings

1. **Doc split is the agent-progress contract.** `STATUS.md` = verified +
   one maturity gate; `plan.md` = future; `AGENTS.md` = commands/traps
   only ([AGENTS.md](../AGENTS.md) Organization Rules;
   [REPO_HYGIENE.md](REPO_HYGIENE.md) Doc layout; ADDING_GAMES §5).
   Planner owns STATUS; executors must not promote from a pin
   (`.grok/skills/{sm,harvest}-session/SKILL.md`).

2. **Soft max ~1000 LOC, merge not sibling.** Root AGENTS: extract before
   1k, “prefer deleting complexity over rearranging it”
   ([AGENTS.md](../AGENTS.md) Working Norms). SM: “merge into the owner
   rather than a new sibling” plus “Where a new file goes”
   ([snes/super_metroid/AGENTS.md](../snes/super_metroid/AGENTS.md)
   Layout; ARCHITECTURE). Harvest: split-before-500 / refuse-a-knob-at-≥800
   ([snes/harvest/AGENTS.md](../snes/harvest/AGENTS.md) Layout).

3. **Second-consumer is the promotion seam.** Shared code and
   `retro_harness` subdomains promote only after a second consumer
   (root AGENTS; VISION why-this-exists #3; ROADMAP principle 4; TOOLSET
   Promotion test; SOLVER_ARCHITECTURE evidence ladder). Harvest
   Pathfinder stays local until then (PLANNING_STACK principle 3).

4. **Natural-entry is the product gate.** M3 isolated ≠ M4
   (DEVELOPMENT_LADDER). “Segment not route-ready until it clears from
   the real predecessor” (root AGENTS; ROADMAP principle 1). Harvest:
   “Natural entry is power-on” (harvest AGENTS; ADR 0001). SM:
   practice/door-warp greens are not continuous evidence (SM AGENTS
   Evaluation contract).

5. **Skills are the substrate; routes/tapes are not the product.**
   SOLVER_ARCHITECTURE Positioning; GLOSSARY Solver vocabulary; SM ADR
   0003; Harvest ADR 0004. ROADMAP: “prefer skill catalog growth over
   project sprawl.” ADDING_GAMES §1: do not copy input loops /
   `boot_probe` / `paths.py`.

6. **Process essays left AGENTS.md and became skills.** Hygiene backlog
   marks SM/Harvest AGENTS “done” because session loops live in
   `.grok/skills/{sm,harvest}-session/` (REPO_HYGIENE Slim backlog).
   Matches writing-for-agents always-loaded pointer vs disclosed
   reference.

7. **One living tip / one composer beats a zoo of runners.** SM ADR
   0002; `run_to` dispatches `TipSpec` only — “Do not add a new
   `start_to_*.py`” (ARCHITECTURE Continuous tip extension). TMNT: “Add
   a spec; do not clone a probe or segment CLI.” SMZ3 STATUS: “avoid
   clone `*_route.py` sprawl.”

8. **Nested vs flat is already two live layouts.** Harvest nested
   `harvest.*` with `core/maps/planner/runtime/tasks/tools` (harvest
   AGENTS; PLANNING_STACK Repo-level structure). Hal’s golf copies that
   nest in miniature. SM is a flat package with `routes`, `combat`,
   `rooms`, `human_tape` (SM AGENTS; ARCHITECTURE Package boundaries).
   Root AGENTS lists both trees and nested-import discovery.

9. **VISION explicitly permits large game-local code** if the run is
   scriptably beatable (VISION What “done” means). Written license for
   SM/Harvest size — and the tension with hygiene’s file/token budget.

## 2. Sprawl evidence (measured)

Method: `os.walk` of `.py` / `.md` under each game, excluding
`__pycache__`, `.venv`, `recordings`, `models`, `debug*`, `scratch`,
`custom_integrations`, `refs`, `HM-Decomp`, `logs`, `roms`,
`debug_alignment`, `state_presets`, `saves`. Not `git ls-files`. SM
`legacy` (3 py) included. Harvest `utils` included.

| Tree | py files | py LOC | md files | py ≥1000 | AGENTS.md lines |
|------|---------:|-------:|---------:|---------:|----------------:|
| `snes/super_metroid` | 527 | 149028 | 64 | 7 | 71 |
| `snes/harvest` | 286 | 93809 | 24 | 3 | 76 |
| `nes/zelda_i` | 298 | 87974 | 23 | 1 | 83 |
| `nes/smb` | 106 | 33588 | 8 | 4 | **198** |
| `retro_harness` | 230 | 52786 | 6 | 1 (test) | fighters 79 |
| `snes/tmnt_iv` | 63 | 12228 | 23 | 0 | 60 |
| `snes/alttp` | 60 | 14702 | 11 | 0 | 47 |
| `snes/smz3` | 46 | 7910 | 6 | 0 | 42 |
| `snes/hals_golf` | 38 | 7682 | 4 | 1 (test) | 56 |

Hygiene targets: root AGENTS ~45, game AGENTS ~50–60, dual `CLAUDE.md` =
0 ([REPO_HYGIENE.md](REPO_HYGIENE.md) Hard targets). Observed: root
AGENTS **89** (Cloud block from line 61); SM 71; harvest 76; smb 198.
No `CLAUDE.md` found in-repo (hygiene “Deleted dual CLAUDE files”
holds).

### Super Metroid

- **Shape:** flat package + domain folders. Continuous product is
  `routes` (kpdr area subdirs `ceres/gauntlet/ice/norfair/red_tower/spazer/wave/wrecked_ship`). Combat ~22 py
  / 7588 after Phantoon leftover recipes were deleted. Scripts probes
  dropped from 36 py / ~20k LOC to 21 py / ~10k (A/B loop `kpdr.py` plus
  residual/boss benches).
- **Fat files (≥1000):** remaining after Gut 2026-08-28: `source_states.py`
  1306, `scripts/record/guided_human.py` 1242. Gym/clone probes
  (`moat_spark_watch`, shine/landing/bubble gym, `spore_spawn_route`,
  Phantoon research CLIs) deleted; `combat/phantoon.py` is 408 shared
  helpers; A/B loader `scripts/probe/kpdr.py` folded to 713.
  `continuous.py` is 247, `early_continuous.py` 496.
- **Parallel trees (product-ish, not gitignored artifacts):**
  `human_tape` 15 py / 6741; `tas` 17 / 6263; `generalist` 13 / 2992;
  `practice_repertoire` 10 / 2246; `dev` 7 / 2312; `map_viewer` 5 /
  1431; `rooms` practice graph kept **intentionally dual** with
  continuous `progression`
  ([ARCHITECTURE.md](../snes/super_metroid/docs/ARCHITECTURE.md) Dual
  graph). `legacy` frozen (3 py) — documented, not deleted.
- **Sediment:** `docs/tasks/` 22 md + **236 log files** under
  `docs/tasks/logs/`; game-root `tasks` holds **5145** json/state files
  (observed on disk; not counted as py). Session skill says overwrite
  one residual and delete closed ones
  ([sm-session](../.grok/skills/sm-session/SKILL.md)); the tree still
  stacks recon/human-tape essays beside `rr-kw8t-residual.md`.
- **Clone pattern the architecture already named:** “Do not add a new
  `start_to_*.py`” / TipSpec table was a fix for “`continuous.py` clone
  tip runners” (debt item 1 **landed**). Remaining clones are **probe
  CLIs** (gym shine/moat/bubble, `record_pure_chain`, `post_spore_pb`,
  `probe/route.py`, Phantoon research benches) **deleted 2026-08-28**.
  Remaining probes are the A/B loop (`kpdr.py`) plus residual/boss
  benches. `routes/segment.py` is explicitly practice-only (146 lines)
  beside live `tips.play_hops`.

### Harvest Moon

- **Shape:** nested `snes/harvest/harvest/{core,maps,planner,runtime,tasks,tools,scripts}`.
  Domain `tasks` 62 py / 23757 LOC; `planner` 36 / 15569; tests 95 /
  26126; `core` 17 / 5424. Two task packages:
  `harvest/tasks/` and `harvest/planner/tasks/` (navigation 539 +
  `multi_nav.py` 1062; plan says MultNav was extracted because
  `navigation.py` was 1.3k — the source file remains).
- **Fat files:** `harvest/maps/map_routes.py` 1133, `multi_nav.py` 1062,
  `runtime/play_session.py` 1028. Fourteen more 800–999
  (`fence_flow`, `inventory_shed`, `day_plan_phases`, `run_to_day2`,
  `farm_clearer`, …). `crop_planter.py` is the CropWaterTask owner (~923
  after folding detect/step/act-verify); **9 `crop_*.py` remain**.
  Same extract pattern, reduced: **8 `cow_*.py`**, **3 `pond_*.py`**,
  **6 `farm_*.py`**. Line bar still met by mixins; composer depth did
  not fully follow
  ([PLANNING_STACK.md](../snes/harvest/docs/PLANNING_STACK.md)
  “cow/crop still under-consumed”).
- **Leftover trees:** Harvest utils graveyard and `docs/MILESTONES.md`
  deleted (plan.md Doc consolidation). Root `scripts` still has
  `record_town_day1_recon.sh` beside `harvest/scripts/`. `HM-Decomp`
  present (excluded from product counts; 7 py / 1533 as vendor sprawl).

## 3. Conventions that already fight sprawl

From AGENTS, REPO_HYGIENE, ADDING_GAMES, SOLVER_ARCHITECTURE,
DEVELOPMENT_LADDER, game AGENTS/ADRs, ARCHITECTURE, TOOLSET.

- Game code/docs/states under `<console>/<game>/`; artifacts in-game;
  states in `custom_integrations` ([AGENTS.md](../AGENTS.md)
  Organization Rules). Do not grow root AGENTS with game workflow
  ([docs/agents/domain.md](agents/domain.md)).
- No `docs/archive/`, `tasks/archive/`, dual `CLAUDE.md`
  ([REPO_HYGIENE.md](REPO_HYGIENE.md)). SM ARCHITECTURE: delete completed
  cards/residuals; do not keep archive trees.
- Add only the layer the game needs; RAM/maps stay local until two games
  share them ([ADDING_GAMES.md](ADDING_GAMES.md) §3). Shared CLI not
  clones: `boot_probe`, `setup_rom_cli`, `game_paths`,
  `clean_artifact_stem` (REPO_HYGIENE backlog **done**; TOOLSET).
- Prefer submodule imports, not the package-root barrel (REPO_HYGIENE
  cheat sheet).
- One hop runner / one tick / one table: SM `tips.play_hops`; TMNT
  `Stage1Policy` + `StageSpec`; Harvest `DayPlanTask` → skills
  (PLANNING_STACK Hierarchical composition). Hop ≠ fight (SM); scripts
  hold no path logic (zelda HYGIENE rule 4).
- Promote primitives after a second in-game consumer, then a second game
  (SM ARCHITECTURE Primitive library).
- Dual-track is labeled, not a second product: SM Survival vs Clean
  ([adr/0001](../snes/super_metroid/docs/adr/0001-survival-first-pass.md));
  practice vs continuous; Harvest Clean only ([adr/0006](../snes/harvest/docs/adr/0006-clean.md)).
- Harness already extracted `platformer` `fighters` `adventure` and
  killed `snes_oneshot` / `super_metroid_rl` names (REPO_HYGIENE Done
  recently; CONSOLIDATION_LEFTOVERS; GLOSSARY Directory names).

## 4. What agents actually need to progress a game

| Need | Where it is written | What goes wrong without it |
|------|---------------------|----------------------------|
| **One hop / segment / skill** | SM AGENTS Layout; `sm-pure-hop`; harvest-route “one hop”; DEVELOPMENT_LADDER M3 | Agents author a new runner or a day-long FSM |
| **STATUS / plan / AGENTS split** | Root + ADDING_GAMES §5 | Frame counts and ticket boards eat the always-loaded window |
| **Planner-owned STATUS** | sm-session / harvest-session Non-claims | Pin duals get published as the tip |
| **Dual-track** | SM Evaluation contract; zelda Dual track | Practice/assisted greens overwrite Clean evidence |
| **Natural-entry** | AGENTS Working Norms; M4; harvest traps | “Y1_D2_Morning_After_D1” / door-warp claimed as route |
| **Second-consumer** | VISION / ROADMAP / TOOLSET | Premature `retro_harness` modules and copy-paste twins |
| **Skills vs AGENTS.md** | REPO_HYGIENE Never put in AGENTS; session skills | Process essays on every turn; or no process at all |
| **Living residual, overwrite not mint `_vN`** | harvest-session loop; sm-session loop | Stacked cards, `_window_*` JSON |
| **Halt-3 / replace trajectory** | both session skills; HARD_ROOM_SPLITS | Same dual repeated; source file grows `if`s |
| **CONTEXT.md language** | SM/Harvest CONTEXT; domain.md “proceed silently” if absent | Agents invent “Gate B” / “Ice as living tip” |

Ticket size SM already names: “one hop, or both rooms of a failed seam;
prefer 30–90 min” ([snes/super_metroid/docs/plan.md](../snes/super_metroid/docs/plan.md)
Strategy). Harvest equivalent: one spine bead, one living residual.

## 5. Do not generate tons of files / rewrite chunks

Already forbidden: merge into the owner, not a file per door (SM AGENTS
/ ARCHITECTURE); do not recopy `boot_probe` / period menus / `paths.py`
(ADDING_GAMES); do not clone `run_stageN_segment.py` (TMNT AGENTS +
ARCHITECTURE); do not fork `alttp` + `super_metroid` into SMZ3
([snes/smz3/AGENTS.md](../snes/smz3/AGENTS.md)); no second hop runner
(`segment.py` is practice-only); delete stale one-offs / no archive
trees (root AGENTS; SMW AGENTS; REPO_HYGIENE); dual `CLAUDE.md` = 0;
“Do not rewrite the route or claim a new tip from a pin bench” (SM
plan.md); Harvest skills not a frozen tape (ADR 0004); prefer deleting
complexity over rearranging it (root AGENTS); “Do not grow new 50–100
KB task files; compose skills instead” (PLANNING_STACK). Transferable
interfaces when starting a game: composer table (`TipSpec` / `SpineHop`
/ `StageSpec` / `PhaseSpec`), thin hop adapter, `TaskSequence` factories,
`game_paths` + `boot_probe`, STATUS/plan/AGENTS + one session skill +
one living residual, natural-entry pin, in-game second consumer before
`retro_harness`.

**Violations (observed, often against the game’s own docs):** SM — 36
probe scripts; kpdr 144 files despite “merge if under 1k”;
`human_tape` + `tas` + `generalist` + `practice_repertoire` beside
the spine; residual/recon md sediment; `source_states.py` over 1k.
Harvest — 13-way `crop_*` split; `navigation.py` kept after extract;
stale LOC claims;
AGENTS over ~50. SMZ3 still has `portal_route.py` /
`outdoor_route.py` / `house_route.py` / `early_route.py` beside a
STATUS line that says not to.

## 6. Contrast: slimmer trees that still progress

**TMNT IV** (`snes/tmnt_iv/`) — M8 continuous hard clear
([docs/STATUS.md](../snes/tmnt_iv/docs/STATUS.md)). 63 py / 12k, **0**
files ≥1000. AGENTS + ARCHITECTURE: `ram.py` / `policy.py` / `tactics` /
`run` / `lab` / thin `scripts`. One production tick; new behavior is a
`next(state)` tactic or a spec row. AGENTS 60 lines. Lab fenced
(“KEEP ≠ production”).

**Hal’s golf** (`snes/hals_golf/`) — M2 + verified course clear and VS
HAL win ([docs/STATUS.md](../snes/hals_golf/docs/STATUS.md)). 38 py /
7.7k. Nested `hals_golf/{core,tasks,runtime}` — Harvest’s nest without
the mixin explosion. AGENTS 56 lines. HIO search is a separate CLI, “do
not wire it into the mission clear path.”

**SMB** (`nes/smb/`) — M8 Clean power-on
([docs/STATUS.md](../nes/smb/docs/STATUS.md)). 106 py / 34k. Flat package
(~24 top-level py) + `tas` + `scripts` + `retro_harness.platformer`.
Progress is the ending. Failure mode is **AGENTS.md 198 lines** — hygiene
listed smb as “done” at ~50 (Slim backlog 2b) and the file grew back.
Fat files are polish/oracle scripts, not a second framework.

**Zelda I is not slim.** 298 py / 88k (Harvest-scale) with **94**
`level*.py` at package root. [HYGIENE.md](../nes/zelda_i/docs/HYGIENE.md)
exists because of “L2/L3 copy-expand debt”; it now says add `SpineHop`
rows, not `*_stages`/`*_success` pairs. Graph-game warning, not a
target. AGENTS 83; prefer ~600 lines; never grow a dungeon table past
1k with controllers.

**`retro_harness`** is the successful extract: 230 py / 53k, genre
subpackages, TOOLSET ownership, dual Task vs BT stacks documented
rather than silently mixed (TOOLSET Scripted-completion dual stack).
No L0–L4 subsystem is publication-ready (still needs a second
independent consumer; SOLVER_ARCHITECTURE evidence ladder).

## 7. Deep-module vocabulary (applied)

Terms from [codebase-design/SKILL.md](/home/v/.agents/skills/codebase-design/SKILL.md).
Do not substitute service/component/API/boundary.

- **Module** (scale-agnostic): `tips.play_hops`, `StageSpec`,
  `DayPlanTask`, `Pathfinder`, `game_paths`. A 13-file `crop_*` cluster
  is many modules, not one deep one.
- **Interface** (everything a caller must know): SM hop = entry pin +
  leave spec + `play_*`. Harvest skill = `NavSkill` / `TaskSequence` /
  `TaskContract`. Per-probe CLI flags are large interfaces around little
  extra behavior (**shallow**).
- **Depth** (leverage at the interface): TMNT `StageSpec` is deep (one
  table drives segment/bridge/clean). Harvest `crop_planter` after extract
  is shallower (callers still learn mixin FSMs). SM `TipSpec` is the
  deepening that retired clone tip runners.
- **Seam** (where the interface lives): hop vs fight; practice `Segment`
  vs continuous `SpineHop`; Task/`WorldState` vs BT/`GameState` (needs an
  **adapter**, TOOLSET); game RAM vs `retro_harness`. SM dual graph is an
  extra seam — safety vs navigation cost.
- **Adapter:** gold-standard SM boss hop is `require_room` → `combat.*`
  → leave check. TMNT `lab` adapters are not production. Harvest
  `tasks/skills.py` factories are intended adapters; coop/crop “still
  mono” means the adapter is hypothetical (**one adapter = hypothetical
  seam; two = real** — the second-consumer rule). Harvest Pathfinder has
  one adapter; SM `RoomProgressionGraph` vs `adventure` is the same
  unfinished promotion.
- **Leverage / locality:** `boot_probe` / `game_paths` / `play_hops` /
  `ow_path.py` pay back across call sites; a new
  `scripts/probe/<room>.py` does not. SM combat stays in `combat`;
  Harvest nav is split across `tasks/nav.py`,
  `planner/tasks/navigation.py`, `multi_nav.py` (utils map probes deleted).
- **Deletion test:** drop TMNT `StageSpec` and N CLIs reappear (keep).
  Drop harvest `utils` and bot complexity does not reappear (graveyard;
  README already says so). Drop SM `rooms` and the continuous spine
  remains (second product).

## 8. Writing-for-agents / retro: review docs vs AGENTS.md

From [writing-for-agents/SKILL.md](/home/v/.agents/skills/writing-for-agents/SKILL.md)
and [retro/SKILL.md](/home/v/.agents/skills/retro/SKILL.md).

- **Always-loaded AGENTS (context load):** 3–8 daily commands, burned
  traps, one-line pointers, the 1k-line bar — REPO_HYGIENE “Keep in
  AGENTS.” Retro: AGENTS are **navigation pointers**; implementation
  agents have context pressure.
- **Disclosed (skills, ARCHITECTURE, HYGIENE, PLANNING_STACK, ADRs,
  CONTEXT):** session loops, “where a new file goes,” anti-pattern
  catalogs, RAM tables, scorecards. Skills already moved process out of
  AGENTS. Router pattern is live: `sm-session` / `harvest-session` point
  at `sm-pure-hop`, `harvest-route`, etc.
- **Review-only (retro: coding standards / reviewer agent):** extract
  rules, clone-CLI bans, “scripts contain no path logic,” mixin vs
  composer. Those live in ARCHITECTURE / HYGIENE / PLANNING_STACK, but
  SM/Harvest also inlined the 1k/500 bar into AGENTS (hygiene allows a
  “high-level structure bar only, not full review essays”).
- **Invocation:** `retro` is `disable-model-invocation: true`. Game
  session skills are model-invoked (descriptions always loaded) so hop
  work self-triggers — at a per-turn description cost.

## 9. Open questions / contradictions (unresolved)

- VISION permits “large game-specific code”; hygiene budgets files and
  AGENTS lines. Which wins when a flagship is M5 with 150k LOC?
- Line bars disagree: root/SM 1000; harvest 500/800; zelda ~600.
- REPO_HYGIENE Slim backlog marks fat AGENTS “done”; smb is 198, root
  89, SM/harvest over 50–60.
- Harvest crop_planter LOC claims (4.9k / 1.16k / 423) disagree across
  three docs and disk.
- SOLVER_ARCHITECTURE: L1 “~90% of current work” vs SM still short of
  credits and Harvest still on D2 farm clear.
- Dual orchestration stacks are “intentional for now” (TOOLSET) vs
  “do not mix without an adapter.” Harvest is on Task/`WorldState`;
  TMNT/combat on BT/`GameState`. No shared adapter module is named as
  required for new games.
- SM dual graph is documented-intentional; deletion test treats
  `rooms` as a second product. Is dual-track a seam to keep or
  sediment?
- Nested Harvest vs flat SM: both blessed. New graph/planning games
  have no written pick.
- Extract-before-1k produced Harvest mixin sprawl; “prefer deleting”
  would have merged. The two norms collide on a 1.2k FSM.
- SMZ3 STATUS forbids clone `*_route.py`; four `*_route.py` files exist.
- Generalist contractor (ADR 0009) is a third control path beside
  skills and tapes — extra seam, one game.
- ROADMAP pulled Harvest forward from Phase 6 as planning pioneer;
  DEVELOPMENT_LADDER still lists it under Phase 6.
- domain.md: do not flag missing CONTEXT/ADRs. Flagships have them;
  slimmer M8 TMNT has CONTEXT + ARCHITECTURE; Hal’s golf does not.

## Sources

- `/home/v/01_projects/11_games/retro_rl/AGENTS.md`
- `docs/REPO_HYGIENE.md`, `ADDING_GAMES.md`, `SOLVER_ARCHITECTURE.md`,
  `DEVELOPMENT_LADDER.md`, `VISION.md`, `ROADMAP.md`, `GLOSSARY.md`,
  `GAME_SELECTION_NOTES.md`, `docs/agents/domain.md`
- `retro_harness/docs/TOOLSET.md`, `CONSOLIDATION_LEFTOVERS.md`
- `snes/super_metroid/{AGENTS.md,CONTEXT.md,docs/ARCHITECTURE.md,docs/STATUS.md,docs/plan.md,docs/adr/0001,0002,0003,0009}`
- `snes/harvest/{AGENTS.md,CONTEXT.md,docs/STATUS.md,docs/plan.md,docs/PLANNING_STACK.md,docs/adr/0001,0004}`
- `snes/{tmnt_iv,hals_golf,smz3}/{AGENTS.md,docs/STATUS.md}`;
  `snes/tmnt_iv/docs/ARCHITECTURE.md`; `nes/zelda_i/{AGENTS.md,docs/HYGIENE.md}`;
  `nes/smb/{AGENTS.md,docs/STATUS.md}`
- `.grok/skills/{sm-session,sm-pure-hop,sm-compose,harvest-session,harvest-route}/SKILL.md`
- `/home/v/.agents/skills/{codebase-design,writing-for-agents,writing-for-agents/SKILL-MECHANICS,retro}/SKILL.md`
- Tree measurements, 2026-08-28, as in §2

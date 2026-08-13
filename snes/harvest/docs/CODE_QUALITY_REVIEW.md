# Code Quality Review — Harvest Moon (SNES)

**Date:** 2026-08-10  
**Scope:** Full `snes/harvest/` tree (package, docs, beads, milestones) — not a single PR.  
**Bar:** Strict maintainability review (structure, spaghetti growth, wrong-layer ownership, tracking completeness).  
**Verdict:** **Not approved** as a healthy codebase state. Product greens (Gate A, empty-can, power-on partial) are real; hot-path structure is regressing under residual patches.

Related boards:

- Verified product facts: [STATUS.md](STATUS.md)
- Future work: [plan.md](plan.md)
- Gate map: [MILESTONES.md](MILESTONES.md)
- Structure direction: [PLANNING_STACK.md](PLANNING_STACK.md)
- Layer ownership: [bot_architecture_plan.md](bot_architecture_plan.md)

---

## Executive summary

Harvest is a **strong planning trunk with a collapsing hot-path implementation**. Gate A and empty-can greens prove the ROM model is good enough. Nightly residual work is encoding playthrough geometry as nested conditionals inside multi-thousand-line monofiles while the skill/contract layer sits mostly unused on the critical path.

| Area | Assessment |
|------|------------|
| Package spine (`core` / `maps` / `planner` / `runtime` / `tasks`) | Solid |
| RAM/tile catalogs, multi-day shell, Clean power-on | Solid |
| `CropWaterTask` / `ReturnHomeTask` monofiles | **Structural crisis** |
| Skills / primitives adoption | Sketch only; production still mono FSMs |
| Beads (Gates A/B/C) | High-level OK; residuals and arch tax incomplete |
| Milestones / program matrix | Stale or missing until this review landed |

---

## File-size inventory (production hot path)

**Updated 2026-08-10 structure-lane pass** (six parallel lanes). Prior crisis numbers struck through in notes.

| Path | ~Lines | Notes |
|------|------:|-------|
| `harvest/planner/tasks/home.py` | **63** | Facade; `home_return` / `home_sleep` / approach / recover |
| `harvest/tasks/coop_task.py` | **506** | + `coop_layout` / `coop_feed_ops` / `coop_egg_ops` |
| `harvest/tasks/cow_task.py` | **374** | + `cow_slots` / `cow_target` / `cow_nav_ops` + prior `cow_*_ops` |
| `harvest/maps/map_config.py` | **501** | Facade; `map_types` / `farm_pond` / `map_routes` |
| `harvest/tasks/crop_planter.py` | **423** | + `crop_detect` / `crop_act_verify` / `crop_step` |
| `harvest/tasks/crop_refill.py` | **432** | + `crop_refill_pond` / `crop_refill_verify` |
| `harvest/planner/day_plan_orchestrator.py` | **584** | `MultiDayPlannerTask` → `multi_day_planner.py` |
| `harvest/planner/day_phase_catalog.py` | **654** | + `day_phase_{berry,chicken,cow}` |
| `harvest/tasks/town_day1_handoff.py` | **139** | + `town_day1_tasks` / `town_day1_build` |
| `harvest/planner/tasks/multi_nav.py` | **~941** | Soft solids / lift_throw / fail-closed seal |
| `harvest/planner/tasks/navigation.py` | **~453** | NavTask + recorded transitions; MultNav re-export |
| `harvest/runtime/rom_tools.py` | **389** | Facade; `rom_model` / `rom_parse` / `save_state_io` / `map_render` |
| `harvest/tools/editor_app.py` | **610** | + `editor_canvas` / `editor_panels` / `editor_farm_twin` |
| `harvest/scripts/town_day1_recon.py` | **213** | + `town_day1_recon_{lib,cmds}` |
| `harvest/tasks/farm_clearer.py` | **955** | under bar; nav in `nav.py` |
| `harvest/runtime/play_session.py` | **997** | under bar (headroom tight) |

**2026-08-11 structure pass:** all production **and test** modules previously
≥1000 LOC extracted under soft max. Largest remaining ~997
(`play_session.py`). Full suite green after extract (~1233 unittest cases).

**Rule (AGENTS + PLANNING_STACK):** soft max **~1000 LOC / file**. Do not grow
monofile thrash arms; land residuals as modules or data rules. MultNav travel
policy residuals → helpers under `multi_nav*` (not back into `navigation.py`).

---

## Structural findings (priority order)

### S1 — `crop_planter.py` / `CropWaterTask` mono (improved; residual)

- **~1.16k lines** after navigate extract + typed FSM + pond thrash extract. Line bar met; still dual-FSM composer, not full skill composition.
- Structure-lane follow-ups (2026-08-10 evening):
  - **`rr-7f54`:** `_state`/`_plot_phase` typed as `CropState`/`PlotPhase`; dispatch tables in `step`/`_handle_act`
  - **`rr-e6fw`:** densify thrash nested ifs → `pond_thrash.CORRIDOR_THRASH_RULES` evaluator
- Largest remaining methods **in mono**: `step` (~234), `_handle_verify` (~105).
- **New thrash must land as `pond_thrash` rules or skills, not mono `if`s.**

**Code-judo target shape:**

```text
CropWaterTask (thin composer)
  → DetectPlots
  → HoeSkill | PlantSkill | WaterSkill   # partial: crop_establish / crop_water_ops
  → RefillSkill                           # partial: crop_refill
       → FenceOpenSkill
       → PondCorridorPolicy                 # pond_policy
       → NavSkill + tool press              # partial: crop_navigate
  → ResidualCropWalk recovery as RetrySkill
```

Seams: `water_refill.py`, `fence_flow.py`, **`pond_*`**, **`crop_{fsm,refill,water_ops,establish,navigate}`**, `skills.py`, `primitives.py`.

**Tracking:** `rr-ds3` (line bar met; remaining for close: skill-composer rewrite of step/verify dual FSM — or accept modular mixins under strict line bar).

### S2 — `ReturnHomeTask` spaghetti (active product tip)

- File still large; string `_phase`; counters per thrash class.
- **2026-08-10 (rr-ws8h):** house-arrival short-circuit on every `step` + timeout;
  approach geometry/zones moved to `planner/tasks/home_approach.py`
  (`ApproachZone`, free-lane constants, `build_house_approach_waypoints`).
  Residual: further thin `step` failure arms onto zone dispatch; full soak
  under **rr-5in**.

### S3 — Skills layer is theater on the hot path

- Docs claim skill composition; production coop/cow/crop/harvest remain mono FSMs.
- `skills.py` is a good API sketch that does not earn keep until something critical is rewritten onto it.
- Coop has `rr-rbk`; **cow (2.4k) has no bead**; harvest skill split only partially covered by `rr-ds3`.

### S4 — Wrong-layer gravity

- `Pathfinder` / `Navigator` / `Point` live in `farm_clearer.py`; crop/home/inventory import farm-clearing for BFS.
- **Promote** to `harvest.tasks.nav` (or `core/pathfind.py`) so clear is not the monorepo gravity well.
- `map_config.py` candidate split: routes / landmarks / pond.
- `inventory.py` candidate split by concern (shed, ensure-tools, shipping wait, outdoor intro).

### S5 — Test monofiles

- `test_day_plan_sequences.py` at 4k lines blocks extraction: when code splits, tests must split with it.

### S6 — Residual implementation pattern (process smell)

1. Soak fails at a coordinate band  
2. Add named charge / thrash arm / densify exception  
3. Green on fixture + power-on  
4. Close bead  
5. Mono grows; next band fails  

**Required process change:** next residual lands as a **new module or data-driven corridor list**, not a 50-line branch inside `_handle_navigate` / `ReturnHomeTask.step`.

---

## Abstraction / type issues

| Issue | Remedy |
|-------|--------|
| Stringly `_state` / `_plot_phase` | `Enum`; exhaustive handlers |
| Silent `work_mode` fallback to `"full"` | `Literal` / enum; fail in builders |
| `_fence_subtask: Optional[object]` | Typed task protocol |
| Soft contracts only | OK as design; track soft-fail residuals as beads |
| Contracts + skills unused in production mono | Wire or stop advertising as done |

---

## What is solid (preserve)

- Package layout spine
- `ram_catalog`, `tile_catalog`, `world_snapshot` / `world_context`
- Phase catalog, registry, contracts API, multi-day orchestrator
- Clean power-on + Gate A evidence discipline
- Partial `water_refill.py` extraction (correct direction; incomplete)

---

## Beads audit (2026-08-10)

### Present (useful)

| ID | Role |
|----|------|
| `rr-20w` | Epic: full clean Spring 1 |
| `rr-5in` | Gate B: power-on → Summer + income |
| `rr-ws8h` | Tip residual: return_home late spring |
| `rr-1vc` | Gate C festival/Sunday/rain |
| `rr-pzw` | Hot-spring stamina gate |
| `rr-rbk` | Coop A4 |
| `rr-ds3` | Crop mono extract (under-scoped) |
| Closed kids | Gate A, empty-can, thrash bugs, D1→D2, sleep, shed hang, … |

### Graph hygiene problems

1. `rr-ws8h` was only `discovered-from` `rr-5in`, not a formal **blocks** edge → Gate B looked ready while residual was the tip.
2. Two `in_progress` claims (`rr-ws8h` + `rr-5in`) — tip should be one claim.
3. Ship verify timeouts **folded into** `rr-ws8h` description — two failure modes, one bead.
4. `rr-ds3` still says ~2.7k lines (actual ~4.9k); P3 while product depends on the mono.
5. Epic notes lag product ($400 power-on money, Gate A closed).

### Missing beads (create / created with this review)

| Topic | Why |
|-------|-----|
| HARVEST_ROUTE ship verify timeouts on continuous | Split from `rr-ws8h` |
| Day-plan post-plant soft fails ENSURE_CAN / CROP_WATER | STATUS residual, untracked |
| Re-scope crop extract (size truth + priority) | Via `rr-ds3` update |
| Cow/barn skill extraction | 2.4k mono, M5 |
| A6 D1 skill routes (thin handoff) | Product path closed; structure open |
| Gift delivery | plan backlog |
| Summer/fall crop rotations | plan backlog |
| M4 natural-entry summer | plan milestone, no bead |
| Manifest / GAME_MATRIX sync | program board stale |
| Pathfinder promote out of farm_clearer | arch |

See [MILESTONES.md](MILESTONES.md) for gate ↔ bead map after sync.

---

## Docs drift found at review time

| Doc | Problem |
|-----|---------|
| `plan.md` Bottleneck | Claimed pure D1→D2 and water/ship open — false vs STATUS |
| `PLANNING_STACK` SOTA | Money ~$100, water flaky, D1 shed open — stale |
| `docs/manifests/harvest.yaml` | Blocker money floor $100 — false; `last_verified` 2026-08-01 |
| `GAME_MATRIX` (generated) | Same stale blocker |
| `STATUS.md` header | Last verification 2026-08-09 while 08-10 body dominates; next-acceptance is a changelog |
| Architecture plans | crop_planter size frozen ~2.3–2.7k |

---

## Ordered remediation (from review)

### P0 / tip

1. Claim only `rr-ws8h`; formal **blocks** → `rr-5in`.
2. Split ship-verify timeout bead under Gate B.
3. Fix home timeout with approach/arrival extract — not another nested `if` only.

### P1 tracking + arch

4. Create missing beads; re-scope `rr-ds3`.
5. Sync plan / PLANNING_STACK / manifest / matrix / STATUS header / milestones table.
6. Next water residual **must** land as module boundary (`pond_corridor` / `refill_nav`).
7. Promote Pathfinder/Navigator out of `farm_clearer`.

### P2 product after Gate B

8. Coop A4 (`rr-rbk`), hot-spring gate (`rr-pzw`), Gate C (`rr-1vc`).
9. Cow extract, gift, rotations, M4 natural summer.

### Explicit non-goals right now

- LLM advisor polish  
- Editor decomposition  
- Multi-year campaign  

---

## Approval checklist (skill bar)

| Criterion | Result |
|-----------|--------|
| No clear structural regression | **Fail** |
| No obvious missed code-judo | **Fail** |
| No unjustified multi-k hot-path files | **Fail** |
| No spaghetti branching growth | **Fail** |
| No wrong-layer / unused abstraction | **Fail** |
| Tracking complete | **Fail** (pre-review); remediate via MILESTONES + beads |

**Bottom line:** Next one-line product action remains close Gate B residuals — but only if patches **delete** navigate complexity rather than extend it. Tracking and architecture debt must not wait for “after Spring.”

---

## Structure pass progress (same day)

| Change | Status |
|--------|--------|
| `Pathfinder`/`Navigator` → `tasks/nav.py`; production imports flipped | Done |
| `pond_corridor` → `pond_policy` / `pond_charges` / `pond_hop` + thin barrel | Done |
| `crop_geometry` pure helpers out of mono | Done |
| `crop_refill` / `crop_water_ops` / `crop_establish` mixins + `crop_fsm` | Done |
| `crop_navigate` + mono &lt;1200 | Done (~1.16k) |
| Crop thrash property bag → direct `_corridor` | Done |
| `work_mode` silent fallback → ValueError | Done |
| `home_recover` policy extract | Done |
| `inventory_*` split + barrel | Done |
| Skills wired into production crop/coop/cow | **Not done** |
| Thin `CropWaterTask` composer / dual-FSM delete | **Partial** — line bar met (~1.16k mixins); dual-FSM skill rewrite optional residual (`rr-ds3`) |
| Cow mono skill rewrite | Scaffold only / open (`rr-y80y`) |


# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).
Structure / API direction: [PLANNING_STACK.md](PLANNING_STACK.md).
Layer ownership: [bot_architecture_plan.md](bot_architecture_plan.md).

## Bottleneck

**Spring calendar multi-day is ROM-verified** (D2 → Summer D1), but it is a
separate D2 fixture run. Clean power-on reaches D1 town gate; D1 handoff to D2
works via Ann|Eve + rest recording (peak mask `0x3F`), not yet as a pure
power-on→D2 continuous claim. After that (or in parallel), same-day water
reliability and **grow → harvest → ship** remain open so money rises above the
post-seed $100 floor. Details: [town_day1_recon.md](town_day1_recon.md).

## Product backlog (from AGENTS slim)

- Power-on → full D1 → D2 with shed grass+can on `house_size=0` (no AnnEve).
- Extract coop feed/collect/ship into `tasks/skills.py`; Spring 22 multi-adult /
  dynamic egg tiles before daily plan restore.
- Gift delivery (carry egg to NPC) once town nav is solid.
- Expand `build_day_phases()` for summer/fall crop rotations.

## Next acceptance tests

1. From clean power-on, complete six D1 talks + truck + shed grass/can
   (`house_size=0`) + sleep → D2 without the AnnEve fixture.
2. From `Y1_After_Buy_Potato` (or live day plan): plant + **natural** water same day
   (no RAM can poke).
3. From `Y1_Test_Crops_Planted_Watered` (or live plant): ~6 days growth → harvest
   → ship; assert **money rises after 5pm**.
4. From `Y1_Inside_House`, 10-day or `--end-of-spring` soak with money > 100
   after first potato harvest window; no mid-run state load.

## Architecture track (planning trunk)

Ordered structural work — detail in PLANNING_STACK workstreams A1–A8.

| Priority | Item | Notes |
|----------|------|-------|
| Done | A1 Phase contracts on crop/coop/sleep/hot-spring | `evaluate_task_contract` + catalog wiring |
| Done | A2 Skill boundary factories | feed/ship/talk/farm bin in `tasks/skills.py` |
| Partial | A3 Crop close-loop acceptance | See A3 progress below |
| Next | A4 Coop skill composition + multi-adult fix | Stop growing `coop_task.py` |
| Done | A5 Contract preflight in day-plan probe | Soft notes when map/tool mismatch |
| Later | A6 D1 skill routes from power-on | Thin handoff over Nav+Talk skills |
| Later | A7 Festival + rainy-day distillation | Phase ordering from recordings |
| Later | A8 Promote Pathfinder / primitives | After second consumer |

### A3 progress (2026-08-01 subagent push)

| Piece | Status |
|-------|--------|
| Day-plan establish → ensure can → water order | **Verified** (unit tests lock sequence + refill_bounds) |
| Same-day water with charged can | **ROM OK**: `Y1_Test_Crops_Planted_Dry` + can=20 → 3/3 wet `0x55` |
| Ship verify without instant money | **Fixed**: bin drop counts ship; money may settle at 5pm |
| Empty-can natural refill | **ROM GREEN closed (rr-3q27)** — dry fixture `can_peak=20` + watered=3 `dry_end=[]` Clean via east→south + F0 + residual crop-walk. return_home re-nav cap. Full Inside_House→Summer income still **rr-20w**. |
| Multi-day growth → harvest → money > $100 | **Gate A CLOSED (rr-y8n)**: multi-day Day09 successor final_money=$3180, HARVEST+CROP_ESTABLISH, mid_run loads=0; farm 5pm wait wired. Full Inside_House→Summer still open (empty-can water / rr-20w). |

**Empty-can refill traps (ROM-mapped 2026-08-01):**

1. `CheckToolSuccess` farm fill only when tile-in-front property is `F0, F9–FD`
   (`ToolAnimationWateringCan` sets can=`0x14`). Prefer
   `REFILL_PREFERRED_WATER_TILES`; F1/F2/F7/F8 do **not** refill — selection
   now preferred-only (no F8 full-path trap).
2. **Main pond F0** ~(31–34,31–33); stands `(32,34)` face up (human recording)
   and `(33,30)` face down (ROM fill 0→20). Band order: pond → south → north.
3. Early west plant pocket is cut off by **y=31 fence wall (0x05, x=11–29)**.
   Clearing one fence opens full BFS to F0. `_start_refill` starts a limited
   `FenceClearLoopTask` when preferred water is unreachable and the wall is up.
4. Shipping F2 pocket remains blacklisted (`BAD_REFILL_STAND_BOUNDS`).

## Domain milestones

### Immediate (close the bottleneck)

- Natural empty-can refill (north stream + pond, not south-only); same-day water
  without RAM can poke.
- ~~Multi-day growth from `Y1_Test_Crops_Planted_Watered` → mature potatoes.~~ **Done** (rr-3v9).
- ~~Harvest + ship + post-5pm money~~ — done (rr-53g).
- ~~Wire Day09 ship path + Gate A multi-day money>$100~~ — done (rr-y8n); successor `run_spring_gate_a_day09.json`.
- Full `Y1_Inside_House`→Summer soak with income (empty-can refill + stable return_home; parent rr-20w / Gate B).
- Power-on → full D1 → D2 with shed pickups on real `house_size=0`.
- Fix `CoopChoresTask` for multi-adult / multi-egg dynamic tiles (Spring 22 case).

### M4 (natural-entry summer)

- Continue from live `Y1_Summer_D1_Morning`.
- Sunday / festival handling (extract from `spring_festival.json`).
- Hot-spring stamina gate already ROM-verified; wire into day-plan when low.

### M5 (domain depth)

- Cow / barn chores (milking, brushing, feeding) distilled from rainy-day + other recordings.
- Rainy-day ordering (barn → coop → crops → town social).
- Multi-seed bags, gift delivery, town NPC loops.
- Stamina / tool management fully closed.

### Long-horizon (campaign)

- Multi-year planner (seasonal crop rotations, animal expansion, marriage flags, ending path).
- Hierarchical planner: day → week/season goals → recovery.
- Power-on → new game → continuous spring month (promote D2 soak).
- Observation class improvement (Bronze → Silver) once the route is stable.

## Active workstreams

| Stream | Focus |
|--------|--------|
| Seed equip | **Done (ROM)**: shed shelf + hoe |
| Hoe/plant | **Done (ROM)**: virgin plan + hoe + plant; near-player fallback |
| Water after plant | `work_mode=establish` then water + can re-fetch; natural refill flaky |
| Hot spring stamina | **ROM natural-entry OK**. Next: day-plan gate when stam low |
| Harvest/ship | `HARVEST_ROUTE` when mature tiles present; money @ 5pm |
| Multi-day soak | `--end-of-spring` with crop income |
| D1 town handoff | Rest auto green; pure power-on path + shed open |
| Skill composition | Factories exist; extract production coop/cow/harvest |
| Phase contracts | **Wired** + probe preflight (A5 done 2026-08-03) |
| Plan advisor | Advisory default; gated apply (`HARVEST_PLAN_LLM_APPLY`) |
| Navigation | `densify_waypoints`; promote Pathfinder later |

## Deferred

- Full multi-year campaign objective contract
- Ending credits path (scene classifier already blocks)
- Unrestricted LLM plan rewrites (only schema-validated optional changes)
- Hard-abort builders on failed contracts (soft evaluate first)

## Infrastructure blockers

- ROM not in git; `retro_setup` SHA1 gate
- Long soaks manual under `logs/long_runs/`

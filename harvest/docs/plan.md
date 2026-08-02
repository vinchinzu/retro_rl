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
| Next | A3 Crop close-loop acceptance | Domain bottleneck; not pure architecture |
| Next | A4 Coop skill composition + multi-adult fix | Stop growing `coop_task.py` |
| Next | A5 Contract preflight in day-plan probe | Soft notes when map/tool mismatch |
| Later | A6 D1 skill routes from power-on | Thin handoff over Nav+Talk skills |
| Later | A7 Festival + rainy-day distillation | Phase ordering from recordings |
| Later | A8 Promote Pathfinder / primitives | After second consumer |

## Domain milestones

### Immediate (close the bottleneck)

- Natural empty-can refill (north stream + pond, not south-only); same-day water
  without RAM can poke.
- Multi-day growth from `Y1_Test_Crops_Planted_Watered` → mature potatoes.
- Harvest + ship route; **shipping money settles at 5pm** (save pre/post-5pm points).
- Full spring soak with income growth, no mid-run loads.
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
| Phase contracts | **Wired** on critical phases; probe preflight next |
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

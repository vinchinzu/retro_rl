# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).
Structure / API direction: [PLANNING_STACK.md](PLANNING_STACK.md).

## Bottleneck

**Spring calendar multi-day is ROM-verified** (D2 → Summer D1). Plant path is
now ROM-verified (shed equip → hoe → plant → `0x54` tiles). Remaining gap:
**same-day water reliability**, then **grow → harvest → ship** so money rises
above the post-seed $100 floor.

## Next acceptance test

1. From clean power-on, return from the D1 town-gate handoff to farm and sleep
   naturally to D2. Then repeat the spring soak without a state load.
2. From `Y1_After_Buy_Potato` (or live day plan): plant + water same day.
3. From `Y1_Inside_House`, 10-day or `--end-of-spring` soak with money > 100
   after first potato harvest (~6 days post-plant).
4. No mid-run state load.

## Milestones

### Immediate (close the bottleneck)

- Natural empty-can refill (pond/stream); same-day water without RAM can poke.
- Multi-day growth from `Y1_Test_Crops_Planted_Watered` → mature potatoes.
- Harvest + ship route; **shipping money settles at 5pm** (save pre/post-5pm points).
- Full spring soak with income growth, no mid-run loads.
- Correct `town_to_farm` for the real opening gate `(712,424)`; only then
  promote the current D2→Summer soak to a continuous power-on result.
- Fix `CoopChoresTask` for multi-adult / multi-egg dynamic tiles (Spring 22 case).

### M4 (natural-entry summer)

- Continue from live `Y1_Summer_D1_Morning`.
- Sunday / festival handling (extract from `spring_festival.json`).
- Hot-spring stamina gate already verified; wire into day-plan when low.

### M5 (domain depth)

- Cow / barn chores (milking, brushing, feeding) distilled from rainy-day + other recordings.
- Rainy-day ordering (barn → coop → crops → town social).
- Multi-seed bags, gift delivery, town NPC loops.
- Stamina / tool management fully closed.

### Long-horizon (campaign)

- Multi-year planner (seasonal crop rotations, animal expansion, marriage flags, ending path).
- Hierarchical planner: day → week/season goals → recovery.
- Power-on → new game → Spring 1 (deferred today).
- Observation class improvement (Bronze → Silver) once the route is stable.

## Active workstreams

| Stream | Focus |
|--------|--------|
| Seed equip | **Done (ROM)**: shed shelf `(190,118)` + hoe `(168,166)` |
| Hoe/plant | **Done (ROM)**: virgin plan + hoe + plant; reject failed centers |
| Water after plant | `work_mode=establish` then `work_mode=water` + can re-fetch; south stream refill |
| Hot spring stamina | **ROM natural-entry OK** (farm→bath→farm, stam 30→120). Next: day-plan gate when stam low; optional summer/fall/winter lip smoke |
| Harvest/ship | `HARVEST_ROUTE` when mature tiles present |
| Multi-day soak | `--end-of-spring` with crop income |
| Skill composition | Extract Nav/Feed/CollectEgg/Ship skills from coop; avoid new mono task files — see [PLANNING_STACK.md](PLANNING_STACK.md) |
| Plan advisor | Advisory default; gated validated apply for optional reorder/append (`HARVEST_PLAN_LLM_APPLY`) |
| Navigation | `densify_waypoints` for viewport BFS; promote Pathfinder when a second consumer appears |

## Planning-stack tasks (structure / efficiency)

1. Extract 3–4 reusable skills (Nav, Feed, CollectEgg, Ship) from `CoopChoresTask` + tests.
2. Enrich `TaskBuildContext` + contract fields on `PhaseSpec` (scaffolded; wire into builders).
3. Viewport-aware auto-waypoint helper (`densify_waypoints`) + shared Pathfinder later.
4. Gated apply path for `local_llm` / future advisors with tight JSON schema.
5. Close the crop loop acceptance test (above).
6. Distill high-value recordings: `tasks/spring_festival.json`, `tasks/fix_rainy_day.json`.

## Deferred

- Full multi-year campaign objective contract
- Ending credits path
- Unrestricted LLM plan rewrites (only schema-validated optional changes)

## Infrastructure blockers

- ROM not in git; `retro_setup` SHA1 gate
- Long soaks manual under `logs/long_runs/`

# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).

## Bottleneck

**Spring calendar multi-day is ROM-verified** (D2 → Summer D1). Plant path is
now ROM-verified (shed equip → hoe → plant → `0x54` tiles). Remaining gap:
**same-day water reliability**, then **grow → harvest → ship** so money rises
above the post-seed $100 floor.

## Next acceptance test

1. From `Y1_After_Buy_Potato` (or live day plan): plant + water same day.
2. From `Y1_Inside_House`, 10-day or `--end-of-spring` soak with money > 100
   after first potato harvest (~6 days post-plant).
3. No mid-run state load.

## Next three milestones

1. **Crop close-loop finish** — water after plant; harvest/ship; money mid-spring.
2. **M4 natural-entry** — continue from live `Y1_Summer_D1_Morning` into summer
   without reload; festival/Sunday handling.
3. **M5 domain depth** — animals, rainy-day ordering, multi-seed bags.

## Active workstreams

| Stream | Focus |
|--------|--------|
| Seed equip | **Done (ROM)**: shed shelf `(190,118)` + hoe `(168,166)` |
| Hoe/plant | **Done (ROM)**: virgin plan + hoe + plant; reject failed centers |
| Water after plant | `work_mode=establish` then `work_mode=water` + can re-fetch; south stream refill |
| Hot spring stamina | **ROM natural-entry OK** (farm→bath→farm, stam 30→120). Next: day-plan gate when stam low; optional summer/fall/winter lip smoke |
| Harvest/ship | `HARVEST_ROUTE` when mature tiles present |
| Multi-day soak | `--end-of-spring` with crop income |

## Deferred

- Power-on title → new game → Spring 1
- Full multi-year campaign objective contract
- Ending credits path
- Local LLM plan advisor rewrites

## Infrastructure blockers

- ROM not in git; `retro_setup` SHA1 gate
- Long soaks manual under `logs/long_runs/`

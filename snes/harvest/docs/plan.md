# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).
Structure / API direction: [PLANNING_STACK.md](PLANNING_STACK.md).
Tracker: `bd ready -l harvest`.

**Doc consolidation (2026-08-18):** deleted `CODE_QUALITY_REVIEW.md`
(review essay), `bot_architecture_plan.md` (layer ownership folded here
and in PLANNING_STACK), and `MILESTONES.md` (gate table folded below).
Kept STATUS, plan, PLANNING_STACK, INTERACT, ram_map, and recon notes.
Ready work stays in beads — do not recreate a gate board.

## Bottleneck

**Tip (2026-08-14):** Gate B continuous — power-on → Summer D1 with income
(`rr-5in`). D2 grape+shop compose **closed** (`rr-zmss`). Return-home /
ExitToFarm `0x08` / ship kids **closed**. Remaining soak residual:
`rr-3ae8` CROP_WATER refill exhausted + `rr-yuel` NAV_CROP hang (D23) +
`rr-rzpd` sparse one-cell CROP_WATER skip after the now-GREEN D2 same-day
clear/hoe/plant (`rr-20w.1`).

**Already closed (do not re-open as bottleneck):**

- Gate A multi-day economy (Day09 successor money $1260→$3180) — `rr-y8n`
- Ship debris → bin thrash (`rr-9xyy`) — clear_hands before pick
- Natural empty-can fill + thrash stabilizations — `rr-3q27` + kids
- Power-on full D1→D2 shed on `house_size=0` — `rr-bhr`
- Same-day D2 grape→shop→clear→hoe→plant — `rr-20w.1`
- Spring calendar shell D2→Summer (fixture, no income) — historical soak

**Architecture tax on the tip path:** keep monofiles under ~1000 LOC (AGENTS).
`MultiMapNavTask` extracted to `multi_nav.py` (was `navigation.py` 1.3k mono).
Further MultNav residuals (forage interact, corridor densify) land as helpers,
not thrash `if`s in the MultNav step machine. Layer ownership: RAM catalog
and tile/map model stay in `core` / `maps`; domain tasks compose
`tasks/skills.py` instead of growing monofile FSMs.

## Gate table (from retired MILESTONES.md)

| Gate | Status | Beads |
|------|--------|-------|
| M3 calendar | Met (calendar-only) | historical soak |
| Gate A economy | Closed (Day09 $1260→$3180) | `rr-y8n` |
| Empty-can natural | Mostly closed | `rr-3q27` |
| Gate B continuous | Open (21 ovn partial) | `rr-5in` / `rr-20w` |
| Gate C calendar richness | Open | `rr-1vc` |
| M4 natural summer | Open after Gate B | `rr-hheu` |

**Farm-bush residual (P3, not D2):** that legacy `SHIP_BERRY` route still
leaves `shipping_money=0` (debris field north of the bush seals the interact).
Spring D2 no longer depends on it: mountain grape + `BuySeedsTask` compose is
Clean (`rr-zmss`). Keep the bush issue scoped to later repeat-forage days.

## Product backlog

- Close Gate B (CROP_WATER refill `rr-3ae8` → power-on soak `rr-5in` / epic `rr-20w`).
- Day-plan soft fails ENSURE_CAN / CROP_WATER after plant (if still red post-B).
- Extract coop feed/collect/ship (`rr-rbk`); cow mono extract (bead).
- Hot-spring stamina gate in day plan (`rr-pzw`).
- Festival/Sunday/rain ordering (`rr-1vc` / Gate C).
- Gift delivery (carry egg to NPC); summer/fall crop rotations.
- M4 natural-entry summer from `Y1_Summer_D1_Morning`.
- Thin D1 handoff (A6) over Nav+Talk skills — product path green, structure open.

## Next acceptance tests

1. ~~Power-on D1→D2 shed `house_size=0`~~ — closed (`rr-bhr`).
2. ~~Natural empty-can + same-day water fixture~~ — closed (`rr-3q27` / thrash kids).
3. ~~Gate A multi-day money > 100 + harvest~~ — closed (`rr-y8n`).
4. **Gate B:** `run_to_day2 --power-on --end-of-spring` → Summer D1, money > 100,
   Clean, no mid-run load (`rr-5in`).
5. Optional after B: rainy/festival phase order (`rr-1vc`); spa when stam low (`rr-pzw`).

## Architecture track (planning trunk)

Ordered structural work — detail in PLANNING_STACK workstreams A1–A8.

| Priority | Item | Notes |
|----------|------|-------|
| Done | A1 Phase contracts on crop/coop/sleep/hot-spring | `evaluate_task_contract` + catalog wiring |
| Done | A2 Skill boundary factories | feed/ship/talk/farm bin in `tasks/skills.py` |
| Partial | A3 Crop close-loop acceptance | Gate A closed; Gate B continuous open |
| **Now** | **Crop mono extract (`rr-ds3`)** | `crop_planter` ~4.9k — P1 arch tax (review) |
| **Now** | **Pathfinder promote (`rr-fjbk`)** | Out of `farm_clearer` |
| Next | A4 Coop skill composition + multi-adult fix | `rr-rbk` — stop growing `coop_task.py` |
| Done | A5 Contract preflight in day-plan probe | Soft notes when map/tool mismatch |
| Later | A6 D1 skill routes from power-on | `rr-7js5` — product path green; structure open |
| Later | A7 Festival + rainy-day distillation | `rr-1vc` |
| Later | A8 Promote Pathfinder / primitives shared | After second game consumer |

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
